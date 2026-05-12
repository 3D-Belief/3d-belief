from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

import torch
import wandb
from einops import pack, rearrange, repeat
import torch
from torch import Tensor, nn
import torchvision
from ..utils.vision_utils import *

from .decoder.decoder import Decoder, DepthRenderingMode
from .encoder import Encoder
from .encoder.visualization.encoder_visualizer import EncoderVisualizer
from ..embodied.semantic_mapper import SemanticMapper
from ..utils.model_utils import forward_2d_model_batch, preprocess_raw_image

class SplatBelief(nn.Module):
    encoder: Encoder
    encoder_visualizer: Optional[EncoderVisualizer]
    decoder: Decoder
    losses: nn.ModuleList
    semantic_mapper: Optional[SemanticMapper]
    clip_model: Optional[nn.Module]
    dino_model: Optional[nn.Module]
    repa_encoder: Optional[nn.Module]
    clip_process = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ]
    )

    def __init__(
        self,
        encoder: Encoder,
        encoder_visualizer: Optional[EncoderVisualizer],
        decoder: Decoder,
        semantic_mapper: Optional[SemanticMapper],
        depth_mode: DepthRenderingMode | None,
        extended_visualization: bool | None,
        clip_model: Optional[nn.Module] = None,
        dino_model: Optional[nn.Module] = None,
        repa_encoder: Optional[Encoder] = None,
        repa_encoder_name: Optional[str] = None,
        repa_encoder_resolution: Optional[int] = None,
        inference_mode: bool = False,
        use_semantic: bool = False,
        semantic_mode: str = "embed",
        use_depth_mask: bool = False,
        channels: int = 3,
        viz_type="interpolation",
        feature_patch_min_scale: float = 0.05,
        feature_patch_max_scale: float = 0.5,
        feature_patch_num_samples: int = 15,
        # Per-pixel class-text supervision (mode == "class_text_only").
        semantic_supervision_mode: str = "clip_patch_sampled",
        class_text_table: Optional[Tensor] = None,
    ) -> None:
        super().__init__()

        # Set up the model.
        self.channels = channels
        self.out_dim = channels
        self.viz_type = viz_type
        self.encoder = encoder
        self.encoder_visualizer = encoder_visualizer
        self.decoder = decoder
        self.use_semantic = use_semantic
        self.semantic_mode = semantic_mode
        if self.use_semantic:
            self.semantic_mapper = semantic_mapper
        self.clip_model = clip_model
        self.dino_model = dino_model
        self.repa_encoder = repa_encoder
        self.repa_encoder_name = repa_encoder_name
        self.repa_encoder_resolution = repa_encoder_resolution
        self.feature_patch_min_scale = feature_patch_min_scale
        self.feature_patch_max_scale = feature_patch_max_scale
        self.feature_patch_num_samples = feature_patch_num_samples
        self.use_depth_mask = use_depth_mask
        self.depth_mode = depth_mode
        self.extended_visualization = extended_visualization
        self.inference_mode = inference_mode
        self.normalize = normalize_to_neg_one_to_one
        self.unnormalize = unnormalize_to_zero_to_one
        self.gaussians = None

        # Class-text supervision plumbing (mode == "class_text_only").
        self.semantic_supervision_mode = semantic_supervision_mode
        # Frozen lookup table mapping class id (0 = ignore) -> D-dim text embed.
        if class_text_table is not None:
            self.register_buffer("class_text_table", class_text_table.float(), persistent=False)
        else:
            self.class_text_table = None

    def forward(self, model_input, t, global_step=100000):
        b, num_context, _, h, w = model_input["ctxt_rgb"].shape

        context_gaussians, target_gaussians = self.encoder(
            model_input, 
            t=t,
            global_step=global_step, 
            deterministic=self.inference_mode,
        )

        repa_pred = None
        if "repa_pred" in model_input:
            repa_pred = model_input["repa_pred"]
        repa_pred_ctxt = None
        if "repa_pred_ctxt" in model_input:
            repa_pred_ctxt = model_input["repa_pred_ctxt"]

        latents = None
        if "latents" in model_input:
            latents = model_input["latents"]

        repa_gt_ctxt = None
        repa_gt_trgt = None
        if self.repa_encoder is not None:
            raw_ctxt = self.unnormalize(model_input["ctxt_rgb"]).squeeze(1)
            raw_ctxt = preprocess_raw_image(
                raw_ctxt, self.repa_encoder_name, resolution=self.repa_encoder_resolution
            )
            repa_gt_ctxt = self.repa_encoder.forward_features(raw_ctxt)
            if 'dino' in self.repa_encoder_name: repa_gt_ctxt = repa_gt_ctxt['x_norm_patchtokens']

            raw_trgt = self.unnormalize(model_input["trgt_rgb"]).squeeze(1)
            raw_trgt = preprocess_raw_image(
                raw_trgt, self.repa_encoder_name, resolution=self.repa_encoder_resolution
            )
            repa_gt_trgt = self.repa_encoder.forward_features(raw_trgt)
            if 'dino' in self.repa_encoder_name: repa_gt_trgt = repa_gt_trgt['x_norm_patchtokens']

        # Augmented gaussians
        gaussians = context_gaussians + target_gaussians
        self.gaussians = gaussians
        output = self.decoder.forward(
            gaussians.float(),
            model_input["trgt_c2w"],
            model_input["intrinsics"].unsqueeze(1),
            model_input["near"].float().unsqueeze(1),
            model_input["far"].float().unsqueeze(1),
            (h, w),
            depth_mode=self.depth_mode,
        )

        # Re-render
        rerender_output = self.decoder.forward(
            gaussians.float(),
            model_input["ctxt_c2w"],
            model_input["intrinsics"].unsqueeze(1).repeat(1, num_context, 1, 1),
            model_input["near"].float().unsqueeze(1).repeat(1, num_context),
            model_input["far"].float().unsqueeze(1).repeat(1, num_context),
            (h, w),
            depth_mode=self.depth_mode,
        )

        target_rendered_color = torch.clamp(output.color, 0.0, 1.0)
        target_rendered_depth = output.depth
        target_rendered_features = output.features

        context_rendered_color = torch.clamp(rerender_output.color, 0.0, 1.0)
        context_rendered_depth = rerender_output.depth
        context_rendered_features = rerender_output.features

        if self.use_depth_mask:
            target_rendered_depth_mask = self.encoder.get_depth_mask(target_rendered_features)
            context_rendered_depth_mask = self.encoder.get_depth_mask(context_rendered_features)

        misc = {
            "rendered_ctxt_rgb": context_rendered_color,
            "rendered_trgt_rgb": target_rendered_color,
            "rendered_ctxt_depth": context_rendered_depth,
            "rendered_trgt_depth": target_rendered_depth,
            "repa_pred": repa_pred,
            "repa_pred_ctxt": repa_pred_ctxt,
            "repa_gt": repa_gt_trgt,
            "repa_gt_ctxt": repa_gt_ctxt,
            "latents": latents,
        }

        if self.use_depth_mask:
            misc.update({
                "rendered_ctxt_depth_mask": context_rendered_depth_mask,
                "rendered_trgt_depth_mask": target_rendered_depth_mask,
            })

        # Render intermediate
        if "intm_c2w" in model_input:
            b, num_intm, _, h, w = model_input["intm_rgb"].shape

            intm_output = self.decoder.forward(
                gaussians.float(),
                model_input["intm_c2w"],
                model_input["intrinsics"].unsqueeze(1).repeat(1, num_intm, 1, 1),
                model_input["near"].float().unsqueeze(1).repeat(1, num_intm),
                model_input["far"].float().unsqueeze(1).repeat(1, num_intm),
                (h, w),
                depth_mode=self.depth_mode,
            )
            intm_rendered_color = torch.clamp(intm_output.color, 0.0, 1.0)
            intm_rendered_depth = intm_output.depth
            intm_rendered_features = intm_output.features

            if self.use_depth_mask:
                intm_rendered_depth_mask = self.encoder.get_depth_mask(intm_rendered_features)
                misc.update({
                    "rendered_intm_depth_mask": intm_rendered_depth_mask,
                })
        
            misc.update({
                "rendered_intm_rgb": intm_rendered_color,
                "rendered_intm_depth": intm_rendered_depth,
            })

        # ---- Per-pass semantic supervision targets ----
        if self.use_semantic and not self.inference_mode:
            self._build_semantic_pass(
                pass_key="trgt_semantic",
                rendered_features=target_rendered_features,
                rgb_normed=model_input["trgt_rgb"],
                class_map=model_input.get("trgt_class_map"),
                misc=misc,
            )
            self._build_semantic_pass(
                pass_key="ctxt_semantic",
                rendered_features=context_rendered_features,
                rgb_normed=model_input["ctxt_rgb"],
                class_map=model_input.get("ctxt_class_map"),
                misc=misc,
            )
            if "intm_c2w" in model_input:
                self._build_semantic_pass(
                    pass_key="intm_semantic",
                    rendered_features=intm_rendered_features,
                    rgb_normed=model_input["intm_rgb"],
                    class_map=model_input.get("intm_class_map"),
                    misc=misc,
                )

            # CLIP forward needed when any per-pass dict carries rgb_patches —
            # used by clip_patch_sampled and (the image-side of) class_text_image.
            if self.semantic_supervision_mode in ("clip_patch_sampled", "class_text_image"):
                self._run_clip_on_collected_patches(misc)

        return self.normalize(target_rendered_color), target_rendered_depth, misc
    
    def render(self, gaussians, extrinsics, intrinsics, near, far, h, w):
        output = self.decoder.forward(
            gaussians,
            extrinsics,
            intrinsics,
            near,
            far,
            (h, w),
            depth_mode=self.depth_mode,
        )
        output.color = torch.clamp(output.color, 0.0, 1.0)
        return output

    @torch.no_grad()
    def render_video(
        self,
        model_input,
        t,
        n,
        num_videos=None,
        render_high_res=False,
        global_step=100000,
    ):
        if "render_poses" not in model_input.keys():
            render_poses = self.compute_poses(self.viz_type, model_input, n,)
            print("using computed poses")
        else:
            render_poses = model_input["render_poses"][0]
            n = len(render_poses)
            print("using provided poses", render_poses.shape)
        intrinsics = model_input["intrinsics"]

        if num_videos is None:
            num_videos = model_input["trgt_rgb"].shape[1]
        
        b, num_context, _, h, w = model_input["ctxt_rgb"].shape

        context_gaussians, target_gaussians = self.encoder(model_input, t, global_step=global_step)

        # Augmented gaussians
        gaussians = context_gaussians + target_gaussians
        self.gaussians = gaussians
        frames = []
        depth_frames = []
        semantics = []

        h = 128 if render_high_res else h
        w = 128 if render_high_res else w

        print(f"render_poses {render_poses.shape}")

        for i in range(n):
            output = self.decoder.forward(
                gaussians.float(),
                render_poses[i : i + 1][:, None, ...].cuda(),
                intrinsics[:num_videos].unsqueeze(1),
                model_input["near"].float().unsqueeze(1),
                model_input["far"].float().unsqueeze(1),
                (h, w),
                depth_mode=self.depth_mode,
            )
            rgb = output.color[:, 0, ...]
            depth = output.depth[:, 0, ...]
            rgb = rearrange(rgb, "b c h w -> b h w c")
            rgb = torch.clamp(rgb, 0.0, 1.0) 
            rgb = rgb * 255.0
            frames.append(rgb.float().cpu().detach().numpy().astype(np.uint8))
            depth_frames.append(depth.float().cpu().detach())

            if self.use_semantic:
                features = output.features[:, 0:1, ...]
                if self.semantic_mode == "embed":
                    features = self.encoder.get_semantic_features(features)
                    # features = self.encoder.get_semantic_reg_features(features)
                features = features.squeeze(1)  # [b, c, h, w]
                semantic = self.semantic_mapper.forward(features)
                semantics.append(semantic.float().cpu().detach().numpy().astype(np.uint8))

        print(f"frames {len(frames)}")
        return frames, depth_frames, semantics, render_poses

    @torch.no_grad()
    def compute_poses(
        self, type, model_input, n, radius=None, max_angle=None, canonical=False
    ):
        near = model_input["near"]
        far = model_input["far"]

        if type == "spherical":
            if radius is None:
                radius = (near + far) * 0.5
            if max_angle is None:
                max_angle = 60

            render_poses = []
            for angle in np.linspace(0, max_angle, n + 1)[:-1]:
                pose = pose_spherical(0, -angle, radius).cpu()
                if canonical:
                    pose = torch.einsum(
                        "ij, jk -> ik", model_input["inv_ctxt_c2w"][0, 0].cpu(), pose,
                    )
                else:
                    pose[2, -1] += radius
                render_poses.append(pose)
            render_poses = torch.stack(render_poses, 0)
        elif type == "interpolation":
            render_poses = torch.stack(
                [
                    interpolate_pose_wobble(
                        model_input["ctxt_c2w"][0][0],
                        model_input["trgt_c2w"][0][0],
                        t / n,
                        wobble=False,
                    )
                    for t in range(n)
                ],
                0,
            )
        else:
            raise ValueError("Unknown video type", type)
        print(f"render_poses: {render_poses.shape}")
        return render_poses

    def _compute_dense_dino_targets(self, rgb_maps):
        """Run the (frozen) DINO encoder on rgb_maps and return dense feature maps.

        Inputs are expected at rendered resolution; outputs are bilinearly
        upsampled back to that resolution so they can be compared pixel-wise
        against the encoder's dense reg head output.

        Returns: tensor of shape (batch, views, dino_c, height, width).
        """
        batch, views, _, height, width = rgb_maps.shape
        rgb_flat = rgb_maps.reshape(batch * views, 3, height, width)
        # DINOv2 uses patch_size=14, DINOv3 uses patch_size=16. Read it off the model.
        dino_patch_size = self.dino_model.patch_embed.patch_size
        if isinstance(dino_patch_size, (tuple, list)):
            dino_patch_size = dino_patch_size[0]
        # Run DINO at a multiple-of-patch-size resolution so reshape is well-defined.
        dino_h0 = max(dino_patch_size, height // 4 * dino_patch_size)
        dino_w0 = max(dino_patch_size, width // 4 * dino_patch_size)
        rgb_resized = torch.nn.functional.interpolate(
            rgb_flat, size=(dino_h0, dino_w0), mode='bilinear', align_corners=False
        )
        with torch.no_grad():
            feats = forward_2d_model_batch(rgb_resized, self.dino_model)
            _, dino_c, dh, dw = feats.shape
            feats = torch.nn.functional.interpolate(
                feats, size=(height, width), mode='bilinear', align_corners=False
            ).reshape(batch, views, dino_c, height, width)
        return feats

    def _build_semantic_pass(self, pass_key, rendered_features, rgb_normed, class_map, misc):
        """Populate misc[pass_key] with the targets needed for the active
        supervision mode. Always includes `semantic_dense` (the raw projection
        used for inference queries). Also populates dense DINO reg targets when
        the reg head is enabled.
        """
        rendered_semantic = self.encoder.get_semantic_features(rendered_features)
        unnorm = self.unnormalize(rgb_normed)
        mode = self.semantic_supervision_mode

        if mode in ("clip_patch_sampled", "class_text_image"):
            samples = self.sample_patches_and_get_clip_embeddings(
                rendered_semantic, unnorm,
                num_samples=self.feature_patch_num_samples,
                min_scale=self.feature_patch_min_scale,
                max_scale=self.feature_patch_max_scale,
            )
        else:
            samples = {}

        samples["semantic_dense"] = rendered_semantic

        # Dense DINO reg head — orthogonal supervision, kept under all modes
        # whenever use_reg_model is enabled (signaled by dino_model presence).
        if self.dino_model is not None:
            samples["semantic_reg_dense"] = self.encoder.get_semantic_reg_features(rendered_features)
            samples["dino_dense"] = self._compute_dense_dino_targets(unnorm)

        # Class-text targets (modes that consume per-pixel class-text targets).
        if mode in ("class_text_only", "class_text_image") and self.class_text_table is not None:
            if class_map is None:
                raise RuntimeError(
                    f"semantic_supervision_mode={mode!r} requires {pass_key.replace('semantic', 'class_map')} "
                    "in model_input; enable load_class_maps in the dataset."
                )
            samples["class_text_target"] = self._compute_class_text_targets(class_map.long())
            samples["class_map"] = class_map

        misc[pass_key] = samples

    def _compute_class_text_targets(self, class_map):
        """Build a dense per-pixel class-text target tensor.

        Inputs:
            class_map: (batch, views, H, W) int64 of class IDs (0 = ignore).
        Returns:
            text_target: (batch, views, D, H, W) float, the class-text vector
                at every non-ignore pixel; zero where class_id == 0.
        """
        assert self.class_text_table is not None, "class_text_table is required"
        b, v, h, w = class_map.shape
        d = self.class_text_table.shape[-1]
        flat_ids = class_map.reshape(-1).clamp_min(0).long()
        n_classes = self.class_text_table.shape[0]
        flat_ids = torch.where(flat_ids < n_classes, flat_ids, torch.zeros_like(flat_ids))
        text_target = (
            self.class_text_table[flat_ids]
            .reshape(b, v, h, w, d)
            .permute(0, 1, 4, 2, 3)
            .contiguous()
        )
        ignore_mask = (class_map == 0).unsqueeze(2)
        text_target = text_target * (~ignore_mask).float()
        return text_target

    # CLIP image normalization constants (OpenAI CLIP / open_clip ViT-B/16)
    _CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
    _CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

    def sample_patches_and_get_clip_embeddings(
        self,
        semantic_maps,
        rgb_maps,
        num_samples=10,
        min_scale=0.03125,  # 1/32 of the image size
        max_scale=1.0,
        seed=None,
    ):
        """
        Vectorized random-patch sampler that supervises the CLIP-style semantic head.
        Returns a dict with `center_features` (predicted features at sampled pixel
        centers) and `rgb_patches` (224x224 ImageNet-normalised fp16 patches, ready
        for CLIP). The caller is expected to run CLIP separately on `rgb_patches`
        — typically batched once across all passes (trgt/ctxt/intm).

        Shapes:
            semantic_maps: (B, V, C, H, W)
            rgb_maps:      (B, V, 3, H, W)
        Returns:
            center_features: (B, V, N, C)
            rgb_patches:     (B, V, N, 3, 224, 224)
        """
        if seed is not None:
            torch.manual_seed(seed)

        B, V, C, H, W = semantic_maps.shape
        N = num_samples
        device = semantic_maps.device

        # Random pixel coords + log-uniform scales, all on device, no CPU syncs.
        ys = torch.randint(0, H, (B, V, N), device=device)
        xs = torch.randint(0, W, (B, V, N), device=device)
        log_scale = torch.empty((B, V, N), device=device).uniform_(
            float(np.log(min_scale)), float(np.log(max_scale))
        )
        scales = log_scale.exp()  # (B, V, N)

        # Center features via gather: semantic_maps[b, v, :, ys[b,v,n], xs[b,v,n]].
        sm_flat = semantic_maps.reshape(B, V, C, H * W)
        flat_idx = (ys * W + xs).unsqueeze(2).expand(-1, -1, C, -1)  # (B, V, C, N)
        centers = torch.gather(sm_flat, 3, flat_idx)                  # (B, V, C, N)
        centers = centers.permute(0, 1, 3, 2).contiguous()            # (B, V, N, C)

        # Variable-scale 224x224 RGB patch extraction via grid_sample.
        # Half-extent of the patch in pixels (clamped to >= 1 px).
        half_pix = (min(H, W) * scales / 2.0).clamp(min=1.0)  # (B, V, N)

        lin = torch.linspace(-1.0, 1.0, 224, device=device)
        gy, gx = torch.meshgrid(lin, lin, indexing="ij")  # (224, 224) each
        base_x = gx[None, None, None]  # (1, 1, 1, 224, 224)
        base_y = gy[None, None, None]
        cx = xs.float()[..., None, None]            # (B, V, N, 1, 1)
        cy = ys.float()[..., None, None]
        hp = half_pix[..., None, None]              # (B, V, N, 1, 1)
        sample_x = cx + base_x * hp                  # pixel coords
        sample_y = cy + base_y * hp
        sample_x = (sample_x / max(W - 1, 1)) * 2.0 - 1.0
        sample_y = (sample_y / max(H - 1, 1)) * 2.0 - 1.0
        grid = torch.stack([sample_x, sample_y], dim=-1)  # (B, V, N, 224, 224, 2)

        rgb_flat = rgb_maps.reshape(B * V, 3, H, W)
        grid_per_bv = grid.reshape(B * V, N * 224, 224, 2)
        rgb_patches = torch.nn.functional.grid_sample(
            rgb_flat, grid_per_bv,
            mode="bilinear", padding_mode="border", align_corners=True,
        )  # (B*V, 3, N*224, 224)
        rgb_patches = rgb_patches.reshape(B * V, 3, N, 224, 224).permute(0, 2, 1, 3, 4)
        rgb_patches = rgb_patches.reshape(B, V, N, 3, 224, 224)

        # CLIP image normalisation, then cast to fp16 to match clip_model precision.
        mean = torch.tensor(self._CLIP_MEAN, device=device, dtype=rgb_patches.dtype).view(1, 1, 1, 3, 1, 1)
        std = torch.tensor(self._CLIP_STD, device=device, dtype=rgb_patches.dtype).view(1, 1, 1, 3, 1, 1)
        rgb_patches = ((rgb_patches - mean) / std).half()

        return {"center_features": centers, "rgb_patches": rgb_patches}

    def _run_clip_on_collected_patches(self, misc):
        """Single CLIP forward across trgt/ctxt/intm patches stashed in misc.
        Replaces each pass's `rgb_patches` entry with `clip_embeddings`."""
        if self.clip_model is None:
            return
        keys = []
        chunks = []
        for k in ("trgt_semantic", "ctxt_semantic", "intm_semantic"):
            if k not in misc or "rgb_patches" not in misc[k]:
                continue
            patches = misc[k]["rgb_patches"]  # (B, V, N, 3, 224, 224)
            B_k, V_k, N_k = patches.shape[:3]
            chunks.append(patches.reshape(-1, 3, 224, 224))
            keys.append((k, B_k, V_k, N_k))
        if not chunks:
            return
        all_patches = torch.cat(chunks, dim=0)
        with torch.no_grad():
            all_clip = self.clip_model.encode_image(all_patches)  # (total, dim)
        offset = 0
        for k, B_k, V_k, N_k in keys:
            count = B_k * V_k * N_k
            misc[k]["clip_embeddings"] = all_clip[offset:offset + count].reshape(B_k, V_k, N_k, -1)
            offset += count
            del misc[k]["rgb_patches"]
