from dataclasses import dataclass
from typing import Literal, Optional

import torch
from einops import rearrange, repeat, reduce
from jaxtyping import Float
from torch import Tensor, nn

from ..geometry.projection import sample_image_grid
from ..geometry.geometry_utils import CameraPose
from ..types import Gaussians
from .backbone import Backbone, BackboneCfg, get_backbone
from .common.gaussian_adapter import GaussianAdapter, GaussianAdapterCfg
from .encoder import Encoder
from .epipolar.depth_predictor_monocular import DepthPredictorMonocular
from .epipolar.epipolar_transformer import EpipolarTransformer, EpipolarTransformerCfg

@dataclass
class OpacityMappingCfg:
    initial: float
    final: float
    warm_up: int


@dataclass
class EncoderUViTDiffusionCfg:
    name: Literal["uvitdiff"]
    d_feature: int
    num_monocular_samples: int
    num_surfaces: int
    predict_opacity: bool
    backbone: BackboneCfg
    near_disparity: float
    gaussian_adapter: GaussianAdapterCfg
    epipolar_transformer: EpipolarTransformerCfg
    opacity_mapping: OpacityMappingCfg
    gaussians_per_pixel: int
    d_semantic: int
    d_semantic_reg: int
    use_semantic: bool
    use_reg_model: bool
    use_transmittance: bool
    depth_predictor_time_embed: bool
    depth_inference_min: float
    depth_inference_max: float
    use_epipolar_transformer: bool
    use_image_condition: bool
    use_camera_pose: bool
    evolve_ctxt: bool
    inference_mode: bool
    use_depth_mask: bool
    conditioning_type: Literal["ray_encoding", "plucker"] = "ray_encoding"
    render_features: bool = True


class EncoderUViTDiffusion(Encoder[EncoderUViTDiffusionCfg]):
    backbone: Backbone
    backbone_projection: nn.Sequential
    epipolar_transformer: EpipolarTransformer | None
    depth_predictor: DepthPredictorMonocular
    to_gaussians: nn.Sequential
    to_semantic: nn.Sequential | None
    gaussian_adapter: GaussianAdapter
    high_resolution_skip: nn.Sequential

    def __init__(self, cfg: EncoderUViTDiffusionCfg) -> None:
        super().__init__(cfg)
        self.inference_mode = cfg.inference_mode
        self.use_semantic = cfg.use_semantic
        self.use_reg_model = cfg.use_reg_model
        self.evolve_ctxt = cfg.evolve_ctxt
        self.use_depth_mask = cfg.use_depth_mask
        self.use_camera_pose = cfg.use_camera_pose
        self.conditioning_type = cfg.conditioning_type
        self.render_features = cfg.render_features
        self.use_image_condition = cfg.use_image_condition
        self.backbone = get_backbone(cfg.backbone, 3)
        self.backbone_projection = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.backbone.d_out, cfg.d_feature),
        )
        self.x_shape = cfg.backbone.input_size
        if cfg.use_epipolar_transformer:
            self.epipolar_transformer = EpipolarTransformer(
                cfg.epipolar_transformer,
                cfg.d_feature,
            )
        else:
            self.epipolar_transformer = None

        self.depth_predictor = DepthPredictorMonocular(
            cfg.d_feature,
            cfg.num_monocular_samples,
            cfg.num_surfaces,
            cfg.use_transmittance,
            cfg.depth_predictor_time_embed,
        )
        self.gaussian_adapter = GaussianAdapter(cfg.gaussian_adapter)
        if cfg.predict_opacity:
            self.to_opacity = nn.Sequential(
                nn.ReLU(),
                nn.Linear(cfg.d_feature, 1),
                nn.Sigmoid(),
            )
        self.to_gaussians = nn.Sequential(
            nn.ReLU(),
            nn.Linear(
                cfg.d_feature,
                cfg.num_surfaces * (2 + self.gaussian_adapter.d_in),
            ),
        )
        if self.use_semantic:
            self.to_semantic = nn.Sequential(
                nn.ReLU(),
                nn.Linear(cfg.d_feature, cfg.d_feature*2),
                nn.ReLU(),
                nn.Linear(cfg.d_feature*2, cfg.d_feature*2),
                nn.ReLU(),
                nn.Linear(cfg.d_feature*2, cfg.d_feature*2),
                nn.ReLU(),
                nn.Linear(cfg.d_feature*2, cfg.num_surfaces * cfg.d_semantic)
            )
            if self.use_reg_model:
                self.to_semantic_reg = nn.Sequential(
                    nn.ReLU(),
                    nn.Linear(cfg.d_feature, cfg.d_feature*2),
                    nn.ReLU(),
                    nn.Linear(cfg.d_feature*2, cfg.d_feature*2),
                    nn.ReLU(),
                    nn.Linear(cfg.d_feature*2, cfg.d_feature*2),
                    nn.ReLU(),
                    nn.Linear(cfg.d_feature*2, cfg.num_surfaces * cfg.d_semantic_reg)
                )
        
        if self.use_depth_mask:
            self.depth_mask_predictor = nn.Sequential(
            nn.Conv2d(cfg.d_feature, cfg.d_feature, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.Conv2d(cfg.d_feature, 1, kernel_size=1),
        )

        self.high_resolution_skip = nn.Sequential(
            nn.Conv2d(3, cfg.d_feature, 7, 1, 3),
            nn.ReLU(),
        )

    def map_pdf_to_opacity(
        self,
        pdf: Float[Tensor, " *batch"],
        global_step: int,
    ) -> Float[Tensor, " *batch"]:
        # https://www.desmos.com/calculator/opvwti3ba9

        # Figure out the exponent.
        cfg = self.cfg.opacity_mapping
        x = cfg.initial + min(global_step / cfg.warm_up, 1) * (cfg.final - cfg.initial)
        exponent = 2**x

        # Map the probability density to an opacity.
        return 0.5 * (1 - (1 - pdf) ** exponent + pdf ** (1 / exponent))

    @torch.no_grad()
    @torch.autocast(
        device_type="cuda", enabled=False
    )  # force 32-bit precision for camera pose processing
    def process_conditions(
        self, conditions: Tensor) -> Tensor:
        """
        Process conditions (raw camera poses) to desired format for the model
        Args:
            conditions (Tensor): raw camera poses (B, T, 16)
        """
        camera_poses = CameraPose.from_vectors(conditions)

        # if self.camera_pose_conditioning.bound is not None:
        #     camera_poses.scale_within_bounds(self.camera_pose_conditioning.bound)

        rays = camera_poses.rays(resolution=self.x_shape[1])
        if self.conditioning_type == "ray_encoding":
            rays = rays.to_pos_encoding()[0]
        else:
            rays = rays.to_tensor(
                use_plucker=self.conditioning_type == "plucker"
            )
        return rearrange(rays, "b t h w c -> b t c h w")

    def forward(
        self,
        model_input: dict,
        t: Tensor,
        global_step: int,
        deterministic: bool = False,
    ) -> Gaussians:
        device = model_input["ctxt_rgb"].device
        b, v, c, h, w = model_input["ctxt_rgb"].shape
        # Concate camera params
        extrinsics = torch.cat([model_input["ctxt_c2w"], model_input["trgt_c2w"]], dim=1)
        intrinsics = torch.cat([model_input["intrinsics"].unsqueeze(1), model_input["intrinsics"].unsqueeze(1)], dim=1)
        near = model_input["near"].unsqueeze(1).repeat(1, 2*v)
        far = model_input["far"].unsqueeze(1).repeat(1, 2*v)

        image = torch.cat([model_input["ctxt_rgb"], model_input["noisy_trgt_rgb"].unsqueeze(1)], dim=1)

        external_cond = None
        if self.use_camera_pose:
            fx = intrinsics[..., 0, 0]   # (b, v)
            fy = intrinsics[..., 1, 1]   # (b, v)
            cx = intrinsics[..., 0, 2]   # (b, v)
            cy = intrinsics[..., 1, 2]   # (b, v)
            intr_vec = torch.stack([fx, fy, cx, cy], dim=-1)  # (b, v, 4)
            extr_trimmed = extrinsics[..., :3, :]  
            extr_vec = extr_trimmed.reshape(*extr_trimmed.shape[:2], -1)  # (b, v, 12)

            camera_info = torch.cat([intr_vec, extr_vec], dim=-1)  # (b, v, 16)
            external_cond = self.process_conditions(camera_info)

        output = self.backbone(image, t, external_cond=external_cond) # (B, V, C, H, W)
        features = output['features']
        repa_pred = output.get('repa_pred', None)
        if repa_pred is not None:
            model_input["repa_pred"] = repa_pred[:, v:, :, :]
            model_input["repa_pred_ctxt"] = repa_pred[:, :v, :, :]

        features = rearrange(features, "b v c h w -> b v h w c")
        features = self.backbone_projection(features)
        features = rearrange(features, "b v h w c -> b v c h w")

        # Run the epipolar transformer.
        if self.cfg.use_epipolar_transformer:
            features, sampling = self.epipolar_transformer(
                features,
                extrinsics,
                intrinsics,
                near.float(),
                far.float(),
                t,
            )
        
        # Add the high-resolution skip connection.
        skip = rearrange(image, "b v c h w -> (b v) c h w")
        skip = self.high_resolution_skip(skip)
        features = features + rearrange(skip, "(b v) c h w -> b v c h w", b=b, v=2*v)

        features = rearrange(features, "b v c h w -> b v (h w) c")
        
        gpp = self.cfg.gaussians_per_pixel

        # To semantic features
        if self.render_features or self.use_semantic:
            semantic_features = rearrange(
                features,
                "... (srf c) -> ... srf 1 c",
                srf=self.cfg.num_surfaces,
            )
            semantic_features = repeat(semantic_features, '... 1 c -> ... gpp c', gpp=gpp) 
            # TODO sample semantic features from a distribution

        # Sample depths from the resulting features.
        # near = near + (0.5 if self.cfg.inference_mode else 0.0)
        depths, densities, mask_exp = self.depth_predictor.forward(
            features,
            near,
            far,
            deterministic,
            1 if deterministic else self.cfg.gaussians_per_pixel,
            t=t,
            inference_mode=self.inference_mode,
        )

        # Convert the features and depths into Gaussians.
        xy_ray, _ = sample_image_grid((h, w), device)
        xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")
        gaussians = rearrange(
            self.to_gaussians(features),
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
            rearrange(gaussians[..., 2:].double(), "b v r srf c -> b v r srf () c"),
            (h, w),
        )

        # Optionally apply a per-pixel opacity.
        opacity_multiplier = (
            rearrange(self.to_opacity(features), "b v r () -> b v r () ()")
            if self.cfg.predict_opacity
            else 1
        )

        def split_opacity_multiplier(opacity_multiplier, first_half=True):
            if isinstance(opacity_multiplier, torch.Tensor):
                return opacity_multiplier[:, :v] if first_half else opacity_multiplier[:, v:]
            else:
                return 1  # fallback if it's scalar 1

        # Split the gaussians
        context_semantic = None
        target_semantic  = None
        if self.render_features or self.use_semantic:
            # [1, V, r, srf, spp, C] -> [1, G, C], where G = V·r·srf·spp
            context_semantic = rearrange(
                semantic_features[:, :v], "b v r srf spp c -> b (v r srf spp) c"
            )
            target_semantic = rearrange(
                semantic_features[:,  v:], "b v r srf spp c -> b (v r srf spp) c"
            )

        if not self.inference_mode:
            context_gaussians = Gaussians(
                rearrange(gaussians.means[:, :v], "b v r srf spp xyz -> b (v r srf spp) xyz"),
                rearrange(gaussians.covariances[:, :v], "b v r srf spp i j -> b (v r srf spp) i j"),
                rearrange(gaussians.harmonics[:, :v], "b v r srf spp c d_sh -> b (v r srf spp) c d_sh"),
                rearrange(split_opacity_multiplier(opacity_multiplier, first_half=True) * gaussians.opacities[:, :v], "b v r srf spp -> b (v r srf spp)"),
                context_semantic
            )

            target_gaussians = Gaussians(
                rearrange(gaussians.means[:, v:], "b v r srf spp xyz -> b (v r srf spp) xyz"),
                rearrange(gaussians.covariances[:, v:], "b v r srf spp i j -> b (v r srf spp) i j"),
                rearrange(gaussians.harmonics[:, v:], "b v r srf spp c d_sh -> b (v r srf spp) c d_sh"),
                rearrange(split_opacity_multiplier(opacity_multiplier, first_half=False) * gaussians.opacities[:, v:], "b v r srf spp -> b (v r srf spp)"),
                target_semantic
            )
        else:
            # Extract the boolean validity masks
            mask       = mask_exp[0]        # [V, r, srf, spp]
            mask_ctx   = mask[:v]           # first v views
            mask_tgt   = mask[v:]           # remaining V−v views
            mask_ctx_f = mask_ctx.reshape(-1)  # [(v·r·srf·spp)]
            mask_tgt_f = mask_tgt.reshape(-1)  # [((V−v)·r·srf·spp)]

            # Flatten each field for batch‐0
            means0 = gaussians.means[0]          # [V, r, srf, spp, 3]
            covs0  = gaussians.covariances[0]    # [V, r, srf, spp, 3,3]
            harms0 = gaussians.harmonics[0]      # [V, r, srf, spp, C, D_sh]
            ops0   = gaussians.opacities[0]      # [V, r, srf, spp]

            # reshape & split
            means_ctx = rearrange(means0[:v], "v r srf spp xyz -> (v r srf spp) xyz")
            means_tgt = rearrange(means0[v:], "v r srf spp xyz -> (v r srf spp) xyz")

            covs_ctx  = rearrange(covs0[:v],  "v r srf spp i j -> (v r srf spp) i j")
            covs_tgt  = rearrange(covs0[v:],  "v r srf spp i j -> (v r srf spp) i j")

            harms_ctx = rearrange(harms0[:v], "v r srf spp c d_sh -> (v r srf spp) c d_sh")
            harms_tgt = rearrange(harms0[v:], "v r srf spp c d_sh -> (v r srf spp) c d_sh")

            # Opacities, using split_opacity_multiplier
            ops_ctx   = (split_opacity_multiplier(opacity_multiplier, True)
                        * ops0.reshape(-1)[: mask_ctx_f.numel()])
            ops_tgt   = (split_opacity_multiplier(opacity_multiplier, False)
                        * ops0.reshape(-1)[mask_ctx_f.numel():])

            # Semantic features, if any
            if self.render_features or self.use_semantic:
                feats_ctx = context_semantic[0]  # [G_ctx, C]
                feats_tgt = target_semantic[0]   # [G_tgt, C]

            # Apply the masks to remove invalid slots
            valid_means_ctx = means_ctx[mask_ctx_f].unsqueeze(0)   # [1, N_ctx, 3]
            valid_covs_ctx  = covs_ctx[mask_ctx_f].unsqueeze(0)    # [1, N_ctx, 3,3]
            valid_harms_ctx = harms_ctx[mask_ctx_f].unsqueeze(0)   # [1, N_ctx, C, D_sh]
            valid_ops_ctx   = ops_ctx[mask_ctx_f].unsqueeze(0)     # [1, N_ctx]
            if self.render_features or self.use_semantic:
                valid_feats_ctx = feats_ctx[mask_ctx_f].unsqueeze(0)   # [1, N_ctx, C_sem] if used
            else:
                valid_feats_ctx = None

            valid_means_tgt = means_tgt[mask_tgt_f].unsqueeze(0)   # [1, N_tgt, 3]
            valid_covs_tgt  = covs_tgt[mask_tgt_f].unsqueeze(0)    # [1, N_tgt, 3,3]
            valid_harms_tgt = harms_tgt[mask_tgt_f].unsqueeze(0)   # [1, N_tgt, C, D_sh]
            valid_ops_tgt   = ops_tgt[mask_tgt_f].unsqueeze(0)     # [1, N_tgt]
            if self.render_features or self.use_semantic:
                valid_feats_tgt = feats_tgt[mask_tgt_f].unsqueeze(0)   # [1, N_tgt, C_sem] if used
            else:
                valid_feats_tgt = None

            # Wrap into Gaussians
            context_gaussians = Gaussians(
                means       = valid_means_ctx,
                covariances = valid_covs_ctx,
                harmonics   = valid_harms_ctx,
                opacities   = valid_ops_ctx,
                features    = valid_feats_ctx
            )

            target_gaussians = Gaussians(
                means       = valid_means_tgt,
                covariances = valid_covs_tgt,
                harmonics   = valid_harms_tgt,
                opacities   = valid_ops_tgt,
                features    = valid_feats_tgt
            )
        return context_gaussians, target_gaussians

    def get_semantic_features(
        self,
        rendered_features: Tensor,
    ) -> Optional[Tensor]:
        if self.use_semantic:
            b, v, c, h, w = rendered_features.shape
            # b v c h w -> b v (h w) c
            features = rearrange(rendered_features, "b v c h w -> b v (h w) c")
            semantic_features = self.to_semantic(features)
            # b v (h w) c -> b v c h w
            semantic_features = rearrange(semantic_features, "b v (h w) c -> b v c h w", h=h, w=w)
            return semantic_features
        return None

    def get_semantic_reg_features(
        self,
        rendered_features: Tensor,
    ) -> Optional[Tensor]:
        if self.use_semantic and self.use_reg_model:
            b, v, c, h, w = rendered_features.shape
            # b v c h w -> b v (h w) c
            features = rearrange(rendered_features, "b v c h w -> b v (h w) c")
            semantic_reg_features = self.to_semantic_reg(features)
            # b v (h w) c -> b v c h w
            semantic_reg_features = rearrange(semantic_reg_features, "b v (h w) c -> b v c h w", h=h, w=w)
            return semantic_reg_features
        return None
    
    def get_depth_mask(
        self,
        rendered_features: Tensor,
    ) -> Optional[Tensor]:
        if self.use_depth_mask:
            b, v, c, h, w = rendered_features.shape
            # b v c h w -> (b v) c h w
            features = rearrange(rendered_features, "b v c h w -> (b v) c h w")
            depth_mask = self.depth_mask_predictor(features)
            # (b v) c h w -> b v c h w
            depth_mask = rearrange(depth_mask, "(b v) c h w -> b v c h w", b=b, v=v)
            return depth_mask
        return None