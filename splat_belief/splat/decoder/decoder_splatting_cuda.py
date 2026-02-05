from dataclasses import dataclass
from typing import Literal

import torch
from einops import rearrange, repeat
from jaxtyping import Float
from torch import Tensor

from ..types import Gaussians
from .cuda_splatting import DepthRenderingMode, render_cuda, RenderOutput, render_depth_and_features_cuda
from .decoder import Decoder, DecoderOutput


@dataclass
class DecoderSplattingCUDACfg:
    name: Literal["splatting_cuda"]


class DecoderSplattingCUDA(Decoder[DecoderSplattingCUDACfg]):
    background_color: Float[Tensor, "3"]

    def __init__(
        self,
        cfg: DecoderSplattingCUDACfg,
    ) -> None:
        super().__init__(cfg)
        self.register_buffer(
            "background_color",
            torch.tensor([0,0, 0,0, 0,0], dtype=torch.float32),
            persistent=False,
        )

    def forward(
        self,
        gaussians: Gaussians,
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        image_shape: tuple[int, int],
        depth_mode: DepthRenderingMode,
        return_colors: bool = True,
        return_features: bool = True
    ) -> DecoderOutput:
        b, v, _, _ = extrinsics.shape
        color_sh = repeat(gaussians.harmonics, "b g c d_sh -> (b v) g c d_sh", v=v) \
            if return_colors and gaussians.harmonics is not None else None
       
        rendered: RenderOutput = render_cuda(
            rearrange(extrinsics, "b v i j -> (b v) i j"),
            rearrange(intrinsics, "b v i j -> (b v) i j"),
            rearrange(near, "b v -> (b v)"),
            rearrange(far, "b v -> (b v)"),
            image_shape,
            repeat(self.background_color, "c -> (b v) c", b=b, v=v),
            repeat(gaussians.means, "b g xyz -> (b v) g xyz", v=v),
            repeat(gaussians.covariances, "b g i j -> (b v) g i j", v=v),
            repeat(gaussians.opacities, "b g -> (b v) g", v=v),
            color_sh,
        )
        color = rearrange(rendered.color, "(b v) c h w -> b v c h w", b=b, v=v) if rendered.color is not None else None
        
        depth_and_features = self.render_depth_and_features(
            gaussians, extrinsics, intrinsics, near, far, image_shape, depth_mode
        )

        depth = depth_and_features.depth
        features = depth_and_features.features if return_features else None
    
        depth = rearrange(depth, "(b v) h w -> b v h w", b=b, v=v)
        features = rearrange(features, "(b v) c h w -> b v c h w", b=b, v=v) if features is not None else None
        return DecoderOutput(
            color,
            depth,
            features,
        )

    def render_depth_and_features(
        self,
        gaussians: Gaussians,
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        image_shape: tuple[int, int],
        mode: DepthRenderingMode = "depth",
    ) -> Float[Tensor, "batch view height width"]:
        b, v, _, _ = extrinsics.shape
        features = repeat(gaussians.features, "b g c -> (b v) g c", v=v) if gaussians.features is not None \
            else None
        result = render_depth_and_features_cuda(
            rearrange(extrinsics, "b v i j -> (b v) i j"),
            rearrange(intrinsics, "b v i j -> (b v) i j"),
            rearrange(near, "b v -> (b v)"),
            rearrange(far, "b v -> (b v)"),
            image_shape,
            repeat(gaussians.means, "b g xyz -> (b v) g xyz", v=v),
            repeat(gaussians.covariances, "b g i j -> (b v) g i j", v=v),
            repeat(gaussians.opacities, "b g -> (b v) g", v=v),
            gaussian_features=features,
            mode=mode,
        )
        return result
