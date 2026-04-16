"""
Scene-graph-conditioned encoder: passes model_input (with SG keys) to the backbone.
"""
from dataclasses import dataclass
from typing import Literal

from torch import Tensor

from .encoder_uvitmvsplat import EncoderUViTMVSplat, EncoderUViTMVSplatCfg
from ..types import Gaussians


@dataclass
class EncoderUViTMVSplatSGCfg(EncoderUViTMVSplatCfg):
    name: Literal["uvitmvsplat_sg"]


class EncoderUViTMVSplatSG(EncoderUViTMVSplat):
    """
    Thin wrapper that passes model_input through to the SG-aware backbone.
    Everything else (depth prediction, gaussian conversion) is inherited.
    """

    def __init__(self, cfg: EncoderUViTMVSplatSGCfg) -> None:
        super().__init__(cfg)

    def forward(
        self,
        model_input: dict,
        t: Tensor,
        global_step: int,
        deterministic: bool = False,
        filter_border_gaussians: bool = None,
        depth_inference_min: float = None,
        depth_inference_max: float = None,
    ) -> Gaussians:
        """
        Same as parent forward, but passes model_input to backbone for SG conditioning.
        """
        import torch
        from einops import rearrange, repeat
        from ..geometry.projection import sample_image_grid

        device = model_input["ctxt_rgb"].device
        b, v, c, h, w = model_input["ctxt_rgb"].shape

        extrinsics = torch.cat([model_input["ctxt_c2w"], model_input["trgt_c2w"]], dim=1)
        intrinsics = torch.cat([model_input["intrinsics"].unsqueeze(1), model_input["intrinsics"].unsqueeze(1)], dim=1)
        near = model_input["near"].unsqueeze(1).repeat(1, 2*v)
        far = model_input["far"].unsqueeze(1).repeat(1, 2*v)

        image = torch.cat([model_input["ctxt_rgb"], model_input["noisy_trgt_rgb"].unsqueeze(1)], dim=1)

        external_cond = None
        if self.use_camera_pose:
            fx = intrinsics[..., 0, 0]
            fy = intrinsics[..., 1, 1]
            cx = intrinsics[..., 0, 2]
            cy = intrinsics[..., 1, 2]
            intr_vec = torch.stack([fx, fy, cx, cy], dim=-1)
            extr_trimmed = extrinsics[..., :3, :]
            extr_vec = extr_trimmed.reshape(*extr_trimmed.shape[:2], -1)
            camera_info = torch.cat([intr_vec, extr_vec], dim=-1)
            external_cond = self.process_conditions(camera_info)

        # KEY DIFFERENCE: pass model_input to SG backbone
        output = self.backbone(
            image, t,
            external_cond=external_cond,
            model_input=model_input,
        )
        features = output['features']
        repa_pred = output.get('repa_pred', None)
        latents = output.get('latents', None)

        if repa_pred is not None:
            model_input["repa_pred"] = repa_pred[:, v:, :, :]
            model_input["repa_pred_ctxt"] = repa_pred[:, :v, :, :]

        if latents is not None:
            model_input["latents"] = latents

        # Propagate layout reconstruction outputs for auxiliary loss
        for _key in ("layout_recon_cls_emb", "layout_recon_cls_logits",
                     "layout_recon_depth_pred",
                     "layout_cls_gt", "layout_depth_gt", "clip_type_embeddings"):
            if _key in output:
                model_input[_key] = output[_key]

        features = rearrange(features, "b v c h w -> b v h w c")
        features = self.backbone_projection(features)
        features = rearrange(features, "b v h w c -> b v c h w")

        # Re-compute camera params for depth predictor
        extrinsics = torch.cat([model_input["ctxt_c2w"], model_input["trgt_c2w"]], dim=1)
        intrinsics = torch.cat([model_input["intrinsics"].unsqueeze(1), model_input["intrinsics"].unsqueeze(1)], dim=1)
        near = model_input["near"].unsqueeze(1).repeat(1, 2*v)
        far = model_input["far"].unsqueeze(1).repeat(1, 2*v)

        skip = rearrange(image, "b v c h w -> (b v) c h w")
        skip = self.high_resolution_skip(skip)
        features = features + rearrange(skip, "(b v) c h w -> b v c h w", b=b, v=2*v)

        extra_info = {}
        extra_info['images'] = rearrange(image.clone().detach(), "b v c h w -> (v b) c h w")

        gpp = self.cfg.gaussians_per_pixel

        if self.render_features or self.use_semantic:
            semantic_features = rearrange(features, "b v c h w -> b v (h w) c")
            semantic_features = rearrange(
                semantic_features,
                "... (srf c) -> ... srf 1 c",
                srf=self.cfg.num_surfaces,
            )
            semantic_features = repeat(semantic_features, '... 1 c -> ... gpp c', gpp=gpp)

        depths, densities, raw_gaussians, mask_exp = self.depth_predictor(
            features,
            intrinsics,
            extrinsics,
            near,
            far,
            gaussians_per_pixel=gpp,
            deterministic=deterministic,
            extra_info=extra_info,
            t=t,
            inference_mode=self.inference_mode,
            return_mask=True,
            depth_inference_min=depth_inference_min,
            depth_inference_max=depth_inference_max,
        )

        filter_border_gaussians = self.filter_border_gaussians if filter_border_gaussians is None else filter_border_gaussians

        if self.inference_mode and filter_border_gaussians:
            border = int(self.border_px)
            if border > 0:
                r = h * w
                idx = torch.arange(r, device=device)
                x = idx % w
                y = idx // w
                border_mask_r = (
                    (x < border) | (x >= (w - border)) |
                    (y < border) | (y >= (h - border))
                )
                border_mask = border_mask_r.view(1, 1, r, 1, 1).expand_as(mask_exp)
                mask_exp = mask_exp & (~border_mask)

        # Convert features and depths into Gaussians (same as parent)
        xy_ray, _ = sample_image_grid((h, w), device)
        xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")
        gaussians = rearrange(
            raw_gaussians,
            "... (srf c) -> ... srf c",
            srf=self.cfg.num_surfaces,
        )
        offset_xy = gaussians[..., :2].sigmoid()
        pixel_size = 1 / torch.tensor((w, h), dtype=torch.float32, device=device)
        xy_ray = xy_ray + (offset_xy - 0.5) * pixel_size

        gaussians = self.gaussian_adapter.forward(
            rearrange(extrinsics.double(), "b v i j -> b v () () () i j"),
            rearrange(intrinsics, "b v i j -> b v () () () i j"),
            rearrange(xy_ray.double(), "b v r srf xy -> b v r srf () xy"),
            depths.double(),
            self.map_pdf_to_opacity(densities, global_step) / gpp,
            rearrange(
                gaussians[..., 2:].double(),
                "b v r srf c -> b v r srf () c",
            ),
            (h, w),
        )

        opacity_multiplier = 1

        context_semantic = None
        target_semantic = None
        if self.render_features or self.use_semantic:
            context_semantic = rearrange(
                semantic_features[:, :v], "b v r srf spp c -> b (v r srf spp) c"
            )
            target_semantic = rearrange(
                semantic_features[:, v:], "b v r srf spp c -> b (v r srf spp) c"
            )

        if not self.inference_mode:
            from ..types import Gaussians as G
            context_gaussians = G(
                rearrange(gaussians.means[:, :v], "b v r srf spp xyz -> b (v r srf spp) xyz"),
                rearrange(gaussians.covariances[:, :v], "b v r srf spp i j -> b (v r srf spp) i j"),
                rearrange(gaussians.harmonics[:, :v], "b v r srf spp c d_sh -> b (v r srf spp) c d_sh"),
                rearrange(opacity_multiplier * gaussians.opacities[:, :v], "b v r srf spp -> b (v r srf spp)"),
                context_semantic
            )
            target_gaussians = G(
                rearrange(gaussians.means[:, v:], "b v r srf spp xyz -> b (v r srf spp) xyz"),
                rearrange(gaussians.covariances[:, v:], "b v r srf spp i j -> b (v r srf spp) i j"),
                rearrange(gaussians.harmonics[:, v:], "b v r srf spp c d_sh -> b (v r srf spp) c d_sh"),
                rearrange(opacity_multiplier * gaussians.opacities[:, v:], "b v r srf spp -> b (v r srf spp)"),
                target_semantic
            )
        else:
            mask = mask_exp[0]
            mask_ctx = mask[:v]
            mask_tgt = mask[v:]
            mask_ctx_f = mask_ctx.reshape(-1)
            mask_tgt_f = mask_tgt.reshape(-1)

            means0 = gaussians.means[0]
            covs0 = gaussians.covariances[0]
            harms0 = gaussians.harmonics[0]
            ops0 = gaussians.opacities[0]

            means_ctx = rearrange(means0[:v], "v r srf spp xyz -> (v r srf spp) xyz")
            means_tgt = rearrange(means0[v:], "v r srf spp xyz -> (v r srf spp) xyz")
            covs_ctx = rearrange(covs0[:v], "v r srf spp i j -> (v r srf spp) i j")
            covs_tgt = rearrange(covs0[v:], "v r srf spp i j -> (v r srf spp) i j")
            harms_ctx = rearrange(harms0[:v], "v r srf spp c d_sh -> (v r srf spp) c d_sh")
            harms_tgt = rearrange(harms0[v:], "v r srf spp c d_sh -> (v r srf spp) c d_sh")
            ops_ctx = opacity_multiplier * ops0.reshape(-1)[: mask_ctx_f.numel()]
            ops_tgt = opacity_multiplier * ops0.reshape(-1)[mask_ctx_f.numel():]

            if self.render_features or self.use_semantic:
                feats_ctx = context_semantic[0]
                feats_tgt = target_semantic[0]

            valid_means_ctx = means_ctx[mask_ctx_f].unsqueeze(0)
            valid_covs_ctx = covs_ctx[mask_ctx_f].unsqueeze(0)
            valid_harms_ctx = harms_ctx[mask_ctx_f].unsqueeze(0)
            valid_ops_ctx = ops_ctx[mask_ctx_f].unsqueeze(0)
            if self.render_features or self.use_semantic:
                valid_feats_ctx = feats_ctx[mask_ctx_f].unsqueeze(0)
            else:
                valid_feats_ctx = None

            valid_means_tgt = means_tgt[mask_tgt_f].unsqueeze(0)
            valid_covs_tgt = covs_tgt[mask_tgt_f].unsqueeze(0)
            valid_harms_tgt = harms_tgt[mask_tgt_f].unsqueeze(0)
            valid_ops_tgt = ops_tgt[mask_tgt_f].unsqueeze(0)
            if self.render_features or self.use_semantic:
                valid_feats_tgt = feats_tgt[mask_tgt_f].unsqueeze(0)
            else:
                valid_feats_tgt = None

            context_gaussians = Gaussians(
                valid_means_ctx, valid_covs_ctx, valid_harms_ctx, valid_ops_ctx, valid_feats_ctx,
            )
            target_gaussians = Gaussians(
                valid_means_tgt, valid_covs_tgt, valid_harms_tgt, valid_ops_tgt, valid_feats_tgt,
            )

        return context_gaussians, target_gaussians
