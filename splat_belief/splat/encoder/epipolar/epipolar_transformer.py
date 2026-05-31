from dataclasses import dataclass
from functools import partial
from typing import Optional

import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn

from ...geometry.epipolar_lines import get_depth
from ...encodings.positional_encoding import PositionalEncoding
from ...transformer.transformer import Transformer
from .conversions import depth_to_relative_disparity
from .epipolar_sampler import EpipolarSampler, EpipolarSampling
from .image_self_attention import ImageSelfAttention, ImageSelfAttentionCfg
from ..common.time_embedder import TimestepEmbedder


@dataclass
class EpipolarTransformerCfg:
    self_attention: ImageSelfAttentionCfg
    num_octaves: int
    num_layers: int
    num_heads: int
    num_samples: int
    d_dot: int
    d_mlp: int
    downscale: int
    use_time_embedder: bool


class EpipolarTransformer(nn.Module):
    cfg: EpipolarTransformerCfg
    epipolar_sampler: EpipolarSampler
    depth_encoding: nn.Sequential
    transformer: Transformer
    downscaler: Optional[nn.Conv2d]
    upscaler: Optional[nn.ConvTranspose2d]
    upscale_refinement: Optional[nn.Sequential]
    time_embedder: Optional[TimestepEmbedder]

    def __init__(
        self,
        cfg: EpipolarTransformerCfg,
        d_in: int,
    ) -> None:
        super().__init__()
        num_context_views = 2

        self.cfg = cfg
        self.epipolar_sampler = EpipolarSampler(num_context_views, cfg.num_samples)
        if self.cfg.num_octaves > 0:
            self.depth_encoding = nn.Sequential(
                (pe := PositionalEncoding(cfg.num_octaves)),
                nn.Linear(pe.d_out(1), d_in),
            )
        feed_forward_layer = partial(ImageSelfAttentionWrapper, cfg.self_attention)
        self.transformer = Transformer(
            d_in,
            cfg.num_layers,
            cfg.num_heads,
            cfg.d_dot,
            cfg.d_mlp,
            selfatt=False,
            kv_dim=d_in,
            feed_forward_layer=feed_forward_layer,
        )

        if cfg.downscale:
            self.downscaler = nn.Conv2d(d_in, d_in, cfg.downscale, cfg.downscale)
            self.upscaler = nn.ConvTranspose2d(d_in, d_in, cfg.downscale, cfg.downscale)
            self.upscale_refinement = nn.Sequential(
                nn.Conv2d(d_in, d_in * 2, 7, 1, 3),
                nn.GELU(),
                nn.Conv2d(d_in * 2, d_in, 7, 1, 3),
            )

        if num_context_views > 2:
            self.view_embeddings = nn.Embedding(num_context_views, d_in)
        
        self.use_time_embedder = cfg.use_time_embedder

        if self.use_time_embedder:
            self.time_embedder = TimestepEmbedder(d_in)
        else:
            self.time_embedder = None

    def forward(
        self,
        features: Float[Tensor, "batch view channel height width"],
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        t: Float[Tensor, "batch"] = None,
    ) -> tuple[Float[Tensor, "batch view channel height width"], EpipolarSampling]:
        b, v, c, h, w = features.shape
        # Optionally add time embedding.
        if self.use_time_embedder:
            assert t is not None
            # Compute time embedding
            time_emb = self.time_embedder(t)  # shape (b, c)
            # features = features + time_emb[:, None, :, None, None]
            half = v // 2
            zeros_emb = torch.zeros(b, half, c, device=features.device, dtype=features.dtype)
            time_emb_expanded = time_emb.unsqueeze(1).expand(b, v - half, c)
            time_emb_full = torch.cat([zeros_emb, time_emb_expanded], dim=1)
            features = features + time_emb_full.unsqueeze(-1).unsqueeze(-1)

        # If needed, apply downscaling.
        if self.downscaler is not None:
            features = rearrange(features, "b v c h w -> (b v) c h w")
            features = self.downscaler(features)
            features = rearrange(features, "(b v) c h w -> b v c h w", b=b, v=v)

        # Get the samples used for epipolar attention.
        sampling = self.epipolar_sampler.forward(
            features, extrinsics, intrinsics, near, far
        )

        if self.cfg.num_octaves > 0:
            # Compute positionally encoded depths for the features.
            collect = self.epipolar_sampler.collect
            depths = get_depth(
                rearrange(sampling.origins, "b v r xyz -> b v () r () xyz"),
                rearrange(sampling.directions, "b v r xyz -> b v () r () xyz"),
                sampling.xy_sample,
                rearrange(collect(extrinsics), "b v ov i j -> b v ov () () i j"),
                rearrange(collect(intrinsics), "b v ov i j -> b v ov () () i j"),
            )

            # Clip the depths. This is necessary for edge cases where the context views
            # are extremely close together (or possibly oriented the same way).
            depths = depths.maximum(near[..., None, None, None])
            depths = depths.minimum(far[..., None, None, None])
            depths = depth_to_relative_disparity(
                depths,
                rearrange(near, "b v -> b v () () ()"),
                rearrange(far, "b v -> b v () () ()"),
            )
            depths = self.depth_encoding(depths[..., None])
            kv = sampling.features + depths
        else:
            kv = sampling.features

        # Add randomly permuted per-view embeddings to the other views.
        if v > 2:
            shuffle = torch.randperm(v - 1, device=kv.device)
            view_embeddings = rearrange(
                self.view_embeddings(shuffle), "ov c -> () () ov () () c"
            )
            kv = kv + view_embeddings

        # Run the transformer.
        q = rearrange(features, "b v c h w -> (b v h w) () c")
        features = self.transformer.forward(
            q,
            rearrange(kv, "b v ov r s c -> (b v r) (s ov) c"),
            b=b,
            v=v,
            h=h // self.cfg.downscale,
            w=w // self.cfg.downscale,
        )
        features = rearrange(
            features,
            "(b v h w) () c -> b v c h w",
            b=b,
            v=v,
            h=h // self.cfg.downscale,
            w=w // self.cfg.downscale,
        )

        # If needed, apply upscaling.
        if self.upscaler is not None:
            features = rearrange(features, "b v c h w -> (b v) c h w")
            features = self.upscaler(features)
            features = self.upscale_refinement(features) + features
            features = rearrange(features, "(b v) c h w -> b v c h w", b=b, v=v)

        return features, sampling


class ImageSelfAttentionWrapper(nn.Module):
    def __init__(
        self,
        self_attention_cfg: ImageSelfAttentionCfg,
        d_in: int,
        d_hidden: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_attention = ImageSelfAttention(self_attention_cfg, d_in, d_in)

    def forward(
        self,
        x: Float[Tensor, "batch token dim"],
        b: int,
        v: int,
        h: int,
        w: int,
    ) -> Float[Tensor, "batch token dim"]:
        x = rearrange(x, "(b v h w) () c -> (b v) c h w", b=b, v=v, h=h, w=w)
        x = self.self_attention(x) + x
        return rearrange(x, "(b v) c h w -> (b v h w) () c", b=b, v=v, h=h, w=w)
