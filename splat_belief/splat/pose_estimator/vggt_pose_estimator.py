"""
VGGTPoseEstimator — predicts camera poses by reusing the VGGT model already
loaded for alignment loss, and running its CameraHead on the aggregator output.

Usage:
    During training (alignment loss active):
        estimator = VGGTPoseEstimator(vggt_model=alignment_loss.vggt_model)
        # After alignment_loss.forward() has been called (which runs shortcut_forward),
        # reuse the cached aggregator tokens:
        poses_c2w, intrinsics = estimator.predict_from_tokens(
            aggregated_tokens_list, image_hw=(518, 518)
        )

    During inference (alignment loss not active):
        poses_c2w, intrinsics = estimator.predict_from_images(images, image_hw=(518, 518))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .coordinate_utils import camera_head_to_3dbelief, normalize_poses_to_first_frame


class VGGTPoseEstimator(nn.Module):
    """Predict camera poses using VGGT's CameraHead.

    Shares the VGGT model already loaded for alignment loss — no extra copy.
    Only the camera_head (~216M params) is used for pose prediction.
    """

    def __init__(
        self,
        vggt_model: nn.Module,
        freeze_camera_head: bool = True,
        num_refinement_iterations: int = 4,
    ):
        """
        Args:
            vggt_model: The VGGT model instance (from VGGTAlignmentLoss.vggt_model).
                        Must have .aggregator and .camera_head attributes.
            freeze_camera_head: If True, camera_head stays frozen (default).
                               Set False for finetuning.
            num_refinement_iterations: Number of iterative refinement steps in CameraHead.
        """
        super().__init__()
        self.vggt_model = vggt_model
        self.num_refinement_iterations = num_refinement_iterations

        if freeze_camera_head:
            for p in self.vggt_model.camera_head.parameters():
                p.requires_grad = False
        else:
            for p in self.vggt_model.camera_head.parameters():
                p.requires_grad = True

    def preprocess_images(self, images: torch.Tensor) -> torch.Tensor:
        """Preprocess images for VGGT input.

        Args:
            images: [B, S, C, H, W] in [0, 1] range (or [-1, 1] — will be clamped).

        Returns:
            [B, S, C, 518, 518] images clamped to [0, 1].
        """
        b, s, c, h, w = images.shape
        images = rearrange(images, "b s c h w -> (b s) c h w")
        images = F.interpolate(images, size=(518, 518), mode="bilinear", align_corners=False)
        images = torch.clamp(images, 0.0, 1.0)
        images = rearrange(images, "(b s) c h w -> b s c h w", b=b)
        return images

    @torch.no_grad()
    def run_aggregator(self, images: torch.Tensor):
        """Run VGGT aggregator on preprocessed images.

        Args:
            images: [B, S, C, 518, 518] in [0, 1] range.

        Returns:
            aggregated_tokens_list: list of token tensors from aggregator.
        """
        aggregated_tokens_list, patch_start_idx = self.vggt_model.aggregator(images)
        return aggregated_tokens_list

    def run_camera_head(
        self, aggregated_tokens_list: list
    ) -> torch.Tensor:
        """Run CameraHead on aggregator tokens.

        Args:
            aggregated_tokens_list: list of [B, S, P, 2048] tensors from aggregator.

        Returns:
            pose_encoding: [B, S, 9] from the last refinement iteration.
        """
        with torch.amp.autocast("cuda", enabled=False):
            pose_enc_list = self.vggt_model.camera_head(
                aggregated_tokens_list,
                num_iterations=self.num_refinement_iterations,
            )
        return pose_enc_list[-1]  # Last iteration = best prediction

    def predict_from_tokens(
        self,
        aggregated_tokens_list: list,
        image_hw: tuple[int, int] = (518, 518),
        normalize_to_first: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict poses from pre-computed aggregator tokens.

        Use this when the aggregator has already been run (e.g., for alignment loss).

        Args:
            aggregated_tokens_list: Aggregator output tokens.
            image_hw: (H, W) of images fed to VGGT.
            normalize_to_first: If True, normalize poses relative to first frame.

        Returns:
            c2w: [B, S, 4, 4] c2w matrices in 3d-belief convention.
            intrinsics_norm: [B, S, 3, 3] normalized intrinsics.
        """
        pose_encoding = self.run_camera_head(aggregated_tokens_list)
        c2w, intrinsics_norm = camera_head_to_3dbelief(pose_encoding, image_hw)

        if normalize_to_first:
            c2w = normalize_poses_to_first_frame(c2w)

        return c2w, intrinsics_norm

    def predict_from_images(
        self,
        images: torch.Tensor,
        normalize_to_first: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """End-to-end: preprocess images, run aggregator + CameraHead.

        Use this at inference time when alignment loss is not running.

        Args:
            images: [B, S, C, H, W] raw images (any resolution, [0,1] or [-1,1]).
            normalize_to_first: If True, normalize poses relative to first frame.

        Returns:
            c2w: [B, S, 4, 4] c2w matrices in 3d-belief convention.
            intrinsics_norm: [B, S, 3, 3] normalized intrinsics.
        """
        images_preprocessed = self.preprocess_images(images)
        aggregated_tokens_list = self.run_aggregator(images_preprocessed)
        return self.predict_from_tokens(
            aggregated_tokens_list,
            image_hw=(518, 518),
            normalize_to_first=normalize_to_first,
        )

    def predict_raw_encoding(
        self,
        aggregated_tokens_list: list,
    ) -> torch.Tensor:
        """Get raw 9D pose encoding without conversion.

        Useful for computing losses during CameraHead finetuning.

        Returns:
            pose_encoding: [B, S, 9] raw CameraHead output.
        """
        return self.run_camera_head(aggregated_tokens_list)
