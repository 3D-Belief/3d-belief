"""
Coordinate conversion utilities between CameraHead (VGGT/AnySplat) output
and 3d-belief's expected pose format.

CameraHead output:
  - 9D pose encoding: [T(3), quaternion(4), FOV(2)]
  - Decoded to: w2c extrinsics [B,S,3,4] OpenCV convention, pixel-space intrinsics [B,S,3,3]

3d-belief expects:
  - c2w 4x4 matrices in modified OpenCV (Y,Z flipped via diag([1,-1,-1,1]))
  - Normalized intrinsics (0-1 range: fx/W, fy/H, cx/W, cy/H)
  - Poses normalized relative to first context frame
"""

import torch
import torch.nn.functional as F

from splat_belief.splat.alignment.vggt.utils.pose_enc import (
    pose_encoding_to_extri_intri,
    extri_intri_to_pose_encoding,
)


# Coordinate flip matrix: bridges OpenCV (y-down, z-forward)
# to 3d-belief's convention (y-up, z-backward).
_COORD_FLIP = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0]))


def camera_head_to_3dbelief(
    pose_encoding: torch.Tensor,
    image_hw: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert CameraHead 9D pose encoding to 3d-belief c2w + normalized intrinsics.

    Args:
        pose_encoding: [B, S, 9] from CameraHead (last iteration output).
        image_hw: (H, W) of the images fed to VGGT.

    Returns:
        c2w: [B, S, 4, 4] camera-to-world matrices in 3d-belief convention.
        intrinsics_norm: [B, S, 3, 3] normalized intrinsics (fx/W, fy/H, etc.).
    """
    H, W = image_hw

    # Decode 9D → w2c [B,S,3,4] + pixel intrinsics [B,S,3,3]
    extrinsics_w2c, intrinsics_pixel = pose_encoding_to_extri_intri(
        pose_encoding,
        image_size_hw=(H, W),
        pose_encoding_type="absT_quaR_FoV",
        build_intrinsics=True,
    )

    # Build 4x4 w2c
    B, S = extrinsics_w2c.shape[:2]
    bottom = torch.tensor(
        [0.0, 0.0, 0.0, 1.0],
        device=extrinsics_w2c.device,
        dtype=extrinsics_w2c.dtype,
    ).expand(B, S, 1, 4)
    w2c_4x4 = torch.cat([extrinsics_w2c, bottom], dim=-2)  # [B, S, 4, 4]

    # Invert → c2w
    c2w_4x4 = torch.linalg.inv(w2c_4x4)

    # Apply coordinate flip: diag([1, -1, -1, 1]) @ c2w
    flip = _COORD_FLIP.to(device=c2w_4x4.device, dtype=c2w_4x4.dtype)
    c2w_4x4 = flip.unsqueeze(0).unsqueeze(0) @ c2w_4x4  # [B, S, 4, 4]

    # Normalize intrinsics: pixel → [0, 1]
    intrinsics_norm = intrinsics_pixel.clone()
    intrinsics_norm[..., 0, 0] /= W  # fx
    intrinsics_norm[..., 1, 1] /= H  # fy
    intrinsics_norm[..., 0, 2] /= W  # cx
    intrinsics_norm[..., 1, 2] /= H  # cy

    return c2w_4x4, intrinsics_norm


def normalize_poses_to_first_frame(
    c2w: torch.Tensor,
) -> torch.Tensor:
    """Normalize c2w poses relative to first frame (index 0 along dim 1).

    Args:
        c2w: [B, S, 4, 4] camera-to-world matrices.

    Returns:
        [B, S, 4, 4] normalized poses where first frame is identity.
    """
    inv_first = torch.linalg.inv(c2w[:, 0:1])  # [B, 1, 4, 4]
    return inv_first @ c2w  # [B, S, 4, 4]


def threeDbelief_to_camera_head(
    c2w: torch.Tensor,
    intrinsics_norm: torch.Tensor,
    image_hw: tuple[int, int],
) -> torch.Tensor:
    """Convert 3d-belief c2w + normalized intrinsics to CameraHead 9D encoding.

    Inverse of camera_head_to_3dbelief. Useful for computing supervised losses
    during CameraHead finetuning.

    Args:
        c2w: [B, S, 4, 4] camera-to-world in 3d-belief convention.
        intrinsics_norm: [B, S, 3, 3] or [B, 3, 3] normalized intrinsics.
        image_hw: (H, W) of images.

    Returns:
        pose_encoding: [B, S, 9] in CameraHead format.
    """
    H, W = image_hw

    # Undo coordinate flip: diag([1, -1, -1, 1]) is its own inverse
    flip = _COORD_FLIP.to(device=c2w.device, dtype=c2w.dtype)
    c2w_opencv = flip.unsqueeze(0).unsqueeze(0) @ c2w  # [B, S, 4, 4]

    # Invert c2w → w2c
    w2c_4x4 = torch.linalg.inv(c2w_opencv)
    extrinsics_w2c = w2c_4x4[..., :3, :]  # [B, S, 3, 4]

    # De-normalize intrinsics: [0, 1] → pixel space
    if intrinsics_norm.dim() == 3:
        # [B, 3, 3] → [B, 1, 3, 3]
        intrinsics_norm = intrinsics_norm.unsqueeze(1).expand_as(
            torch.zeros(*c2w.shape[:2], 3, 3)
        )
    intrinsics_pixel = intrinsics_norm.clone()
    intrinsics_pixel[..., 0, 0] *= W
    intrinsics_pixel[..., 1, 1] *= H
    intrinsics_pixel[..., 0, 2] *= W
    intrinsics_pixel[..., 1, 2] *= H

    # Encode to 9D
    pose_encoding = extri_intri_to_pose_encoding(
        extrinsics_w2c,
        intrinsics_pixel,
        image_size_hw=(H, W),
        pose_encoding_type="absT_quaR_FoV",
    )

    return pose_encoding
