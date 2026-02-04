from dataclasses import dataclass
from typing import Literal, Optional, List

import torch
from einops import rearrange, repeat
from jaxtyping import Float
from torch import Tensor, nn
from collections import OrderedDict
import itertools

from ..geometry.projection import sample_image_grid
from ..geometry.geometry_utils import CameraPose
from ..types import Gaussians
from .backbone import Backbone, BackboneCfg, get_backbone
from .common.gaussian_adapter import GaussianAdapter, GaussianAdapterCfg
from .encoder import Encoder
from .costvolume.depth_predictor_multiview import DepthPredictorMultiView
from .epipolar.epipolar_transformer import EpipolarTransformer, EpipolarTransformerCfg

@dataclass
class OpacityMappingCfg:
    initial: float
    final: float
    warm_up: int


@dataclass
class EncoderDiTMVSplatCfg:
    name: Literal["ditmvsplat"]
    d_feature: int
    num_depth_candidates: int
    num_surfaces: int
    gaussian_adapter: GaussianAdapterCfg
    opacity_mapping: OpacityMappingCfg
    backbone: BackboneCfg
    gaussians_per_pixel: int
    d_semantic: int
    d_semantic_reg: int
    use_semantic: bool
    use_reg_model: bool
    use_transmittance: bool
    depth_predictor_time_embed: bool
    depth_inference_min: float
    depth_inference_max: float
    downscale_factor: int
    costvolume_unet_feat_dim: int
    costvolume_unet_channel_mult: List[int]
    costvolume_unet_attn_res: List[int]
    depth_unet_feat_dim: int
    depth_unet_attn_res: List[int]
    depth_unet_channel_mult: List[int]
    wo_depth_refine: bool
    wo_cost_volume: bool
    wo_backbone_cross_attn: bool
    wo_cost_volume_refine: bool
    legacy_2views: bool
    grid_sample_disable_cudnn: bool
    use_image_condition: bool
    use_camera_pose: bool
    evolve_ctxt: bool
    inference_mode: bool
    use_depth_mask: bool
    render_features: bool = True
    freeze_depth_predictor: bool = False
    use_prope_attn: bool = False
    conditioning_type: Literal["ray_encoding", "plucker"] = "ray_encoding"
    encoder_ckpt: Optional[str] = None
    costvolume_nearest_n_views: Optional[int] = None
    multiview_trans_nearest_n_views: Optional[int] = None
    fit_ckpt: Optional[bool] = False
    depth_upscale_factor: Optional[int] = None


class EncoderDiTMVSplat(Encoder[EncoderDiTMVSplatCfg]):
    backbone: Backbone
    depth_predictor:  DepthPredictorMultiView
    gaussian_adapter: GaussianAdapter

    def __init__(self, cfg: EncoderDiTMVSplatCfg) -> None:
        super().__init__(cfg)

        self.inference_mode = cfg.inference_mode
        self.use_semantic = cfg.use_semantic
        self.use_reg_model = cfg.use_reg_model
        self.evolve_ctxt = cfg.evolve_ctxt
        self.use_depth_mask = cfg.use_depth_mask
        self.use_camera_pose = cfg.use_camera_pose
        self.use_prope_attn = cfg.use_prope_attn
        self.render_features = cfg.render_features
        self.conditioning_type = cfg.conditioning_type
        self.freeze_depth_predictor = cfg.freeze_depth_predictor
        self.use_image_condition = cfg.use_image_condition
        self.backbone = get_backbone(cfg.backbone, 3)
        self.backbone_projection = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.backbone.d_out, cfg.d_feature),
        )
        self.x_shape = cfg.backbone.input_size

        # gaussians convertor
        self.gaussian_adapter = GaussianAdapter(cfg.gaussian_adapter)

        # cost volume based depth predictor
        self.depth_predictor = DepthPredictorMultiView(
            feature_channels=cfg.d_feature,
            upscale_factor=(
                cfg.downscale_factor
                if cfg.depth_upscale_factor is None
                else cfg.depth_upscale_factor
            ),
            num_depth_candidates=cfg.num_depth_candidates,
            costvolume_unet_feat_dim=cfg.costvolume_unet_feat_dim,
            costvolume_unet_channel_mult=tuple(cfg.costvolume_unet_channel_mult),
            costvolume_unet_attn_res=tuple(cfg.costvolume_unet_attn_res),
            gaussian_raw_channels=cfg.num_surfaces * (self.gaussian_adapter.d_in + 2),
            gaussians_per_pixel=cfg.gaussians_per_pixel,
            num_views=2, # always use 2 views
            depth_unet_feat_dim=cfg.depth_unet_feat_dim,
            depth_unet_attn_res=cfg.depth_unet_attn_res,
            depth_unet_channel_mult=cfg.depth_unet_channel_mult,
            wo_depth_refine=cfg.wo_depth_refine,
            wo_cost_volume=cfg.wo_cost_volume,
            wo_cost_volume_refine=cfg.wo_cost_volume_refine,
            legacy_2views=cfg.legacy_2views,
            grid_sample_disable_cudnn=cfg.grid_sample_disable_cudnn,
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

        ckpt_path = cfg.encoder_ckpt if hasattr(cfg, "encoder_ckpt") else None
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)
            if self.freeze_depth_predictor:
                print("Freezing depth predictor weights.")
                for param in self.depth_predictor.parameters():
                    param.requires_grad = False

    def init_from_ckpt(self, path: str) -> None:
        print("Loading encoder pretrained weight from", path)
        if path.endswith("ckpt"):
            enc_dict = torch.load(path, map_location="cpu")["state_dict"]
        else:
            raise NotImplementedError
        if "state_dict" in list(enc_dict.keys()):
            enc_dict = enc_dict["state_dict"]
        # remove the 'encoder.' prefix
        enc_dict = OrderedDict(
            {k[8:]: v for k, v in enc_dict.items() if k.startswith("encoder")}
        )
        # only keep depth_predictor related weights
        enc_dict = OrderedDict(
            {k: v for k, v in enc_dict.items() if k.startswith("depth_predictor")}
        )
        # need to update the weight if predict features
        if self.cfg.fit_ckpt:
            print("Fitting pretrained mvsplat encoder weights to new models...")
            for name, param in self.state_dict().items():
                new_shape = param.shape
                if name not in enc_dict:
                    raise Exception(f"Found unknown new weight {name}")
                old_shape = enc_dict[name].shape
                assert len(old_shape) == len(new_shape)
                if len(new_shape) > 2:
                    # we only modify first two axes
                    assert new_shape[2:] == old_shape[2:]
                # assumes first axis corresponds to output dim
                if new_shape != old_shape:
                    print(
                        f"Manual init:{name} with new shape {new_shape} "
                        f"and old shape {old_shape}"
                    )
                    new_param = param.clone().zero_()
                    old_param = enc_dict[name]
                    if len(new_shape) == 1:
                        index_size = min(new_param.shape[0], old_param.shape[0])
                        new_param[:index_size] = old_param[:index_size]
                    elif len(new_shape) >= 2:
                        index_o_size = min(new_param.shape[0], old_param.shape[0])
                        index_i_size = min(new_param.shape[1], old_param.shape[1])
                        new_param[:index_o_size, :index_i_size] = old_param[
                            :index_o_size, :index_i_size
                        ]
                    enc_dict[name] = new_param

        missing, unexpected = self.load_state_dict(enc_dict, strict=False)
        print(
            f"Restored from {path} with {len(missing)} missing and "
            f"{len(unexpected)} unexpected keys "
        )
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

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

        viewmats = None
        Ks = None
        patches_x = None
        patches_y = None

        if self.use_prope_attn:
            viewmats = extrinsics  # [N, 2, 4, 4]
            Ks = intrinsics.expand(-1, 2, -1, -1)  # [N, 2, 3, 3]
            image_width, image_height = w, h
            patches_x, patches_y = image_width // self.backbone.patch_size, image_height // self.backbone.patch_size

        output = self.backbone(
            image, 
            t, 
            external_cond=external_cond, 
            viewmats=viewmats, 
            Ks=Ks, 
            patches_x=patches_x, 
            patches_y=patches_y
        ) # (B, V, C, H, W)
        
        features = output['features']
        repa_pred = output.get('repa_pred', None)
        latents = output.get('latents', None)

        if repa_pred is not None:
            repa_pred = rearrange(repa_pred, "b (v p) c -> b v p c", v=2*v)
            model_input["repa_pred"] = repa_pred[:, v:, :, :]
            model_input["repa_pred_ctxt"] = repa_pred[:, :v, :, :]

        if latents is not None:
            model_input["latents"] = latents

        features = rearrange(features, "b v c h w -> b v h w c")
        features = self.backbone_projection(features)
        features = rearrange(features, "b v h w c -> b v c h w")
        
        # Concate camera params
        extrinsics = torch.cat([model_input["ctxt_c2w"], model_input["trgt_c2w"]], dim=1)
        intrinsics = torch.cat([model_input["intrinsics"].unsqueeze(1), model_input["intrinsics"].unsqueeze(1)], dim=1)
        near = model_input["near"].unsqueeze(1).repeat(1, 2*v)
        far = model_input["far"].unsqueeze(1).repeat(1, 2*v)

        # Add the high-resolution skip connection.
        skip = rearrange(image, "b v c h w -> (b v) c h w")
        skip = self.high_resolution_skip(skip)
        # skip = self.high_resolution_skip_norm(skip)
        features = features + rearrange(skip, "(b v) c h w -> b v c h w", b=b, v=2*v)

        extra_info = {}
        extra_info['images'] = rearrange(image.clone().detach(), "b v c h w -> (v b) c h w")

        gpp = self.cfg.gaussians_per_pixel

        # To semantic features
        if self.render_features or self.use_semantic:
            semantic_features = rearrange(features, "b v c h w -> b v (h w) c")
            semantic_features = rearrange(
                semantic_features,
                "... (srf c) -> ... srf 1 c",
                srf=self.cfg.num_surfaces,
            )
            semantic_features = repeat(semantic_features, '... 1 c -> ... gpp c', gpp=gpp) 
            # TODO sample semantic features from a distribution

        depths, densities, raw_gaussians = self.depth_predictor(
            features,
            intrinsics,
            extrinsics,
            near,
            far,
            gaussians_per_pixel=gpp,
            deterministic=deterministic,
            extra_info=extra_info,
            t=t,
        )

        # Convert the features and depths into Gaussians.
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
    
        # Optionally apply a per-pixel opacity.
        opacity_multiplier = 1

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

        context_gaussians = Gaussians(
            rearrange(gaussians.means[:, :v], "b v r srf spp xyz -> b (v r srf spp) xyz"),
            rearrange(gaussians.covariances[:, :v], "b v r srf spp i j -> b (v r srf spp) i j"),
            rearrange(gaussians.harmonics[:, :v], "b v r srf spp c d_sh -> b (v r srf spp) c d_sh"),
            rearrange(opacity_multiplier * gaussians.opacities[:, :v], "b v r srf spp -> b (v r srf spp)"),
            context_semantic
        )

        target_gaussians = Gaussians(
            rearrange(gaussians.means[:, v:], "b v r srf spp xyz -> b (v r srf spp) xyz"),
            rearrange(gaussians.covariances[:, v:], "b v r srf spp i j -> b (v r srf spp) i j"),
            rearrange(gaussians.harmonics[:, v:], "b v r srf spp c d_sh -> b (v r srf spp) c d_sh"),
            rearrange(opacity_multiplier * gaussians.opacities[:, v:], "b v r srf spp -> b (v r srf spp)"),
            target_semantic
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