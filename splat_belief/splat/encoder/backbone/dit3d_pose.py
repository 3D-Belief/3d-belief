from dataclasses import dataclass
from typing import Literal, Tuple, Optional, Union
import torch
from torch import nn
from omegaconf import DictConfig
from einops import rearrange, repeat
from ..common.embeddings import (
    RandomDropoutPatchEmbed,
    RandomEmbeddingDropout,
)
from .dit3d import DiT3D
from .backbone import Backbone

@dataclass
class BackboneDiT3DPoseCfg:
    name: Literal["dit3d_pose"] = "dit3d_pose"
    variant: Literal["full", "factorized_encoder", "factorized_attention"] = "full"
    pos_emb_type: Literal["rope_3d", "learned_1d", "sinusoidal_1d", "sinusoidal_3d", "sinusoidal_factorized"] = "rope_3d"
    input_size: Tuple[int, int] = (64, 64)
    patch_size: int = 2
    hidden_size: int = 384
    max_tokens: int = 16
    depth: int = 12
    num_heads: int = 6
    mlp_ratio: float = 4.0
    use_gradient_checkpointing: bool = False
    conditioning_modeling: Literal["concat", "film"] = "film"
    conditioning_type: Literal["global", "ray_encoding"] = "ray_encoding"
    conditioning_dim: int = 180
    use_causal_mask: bool = False
    external_cond_dropout: float = 0.1
    use_repa: bool = False
    repa_z_dim: int = 768
    use_vggt_alignment: bool = False

class DiT3DPose(Backbone[BackboneDiT3DPoseCfg], DiT3D):

    def __init__(self, cfg: BackboneDiT3DPoseCfg, d_in: int):
        assert d_in == 3, "Input d_in must be 3"
        self.conditioning_modeling = cfg.conditioning_modeling
        self.conditioning_type = cfg.conditioning_type
        self.conditioning_dropout = cfg.external_cond_dropout
        self.input_size = cfg.input_size
        x_shape = torch.Size((cfg.hidden_size, cfg.input_size[0], cfg.input_size[1]))
        max_tokens = cfg.max_tokens
        conditioning_dim = cfg.conditioning_dim
        use_causal_mask = cfg.use_causal_mask
        self.use_repa = cfg.use_repa
        self.patch_size = cfg.patch_size

        super().__init__(
            cfg=cfg,
            x_shape=x_shape,
            max_tokens=max_tokens,
            external_cond_dim=conditioning_dim,
            use_causal_mask=use_causal_mask,
        )

    @property
    def in_channels(self) -> int:
        return (
            self.x_shape[0] + self.external_cond_dim
            if self.conditioning_modeling == "concat"
            else self.x_shape[0]
        )

    @property
    def d_out(self) -> int:
        return self.in_channels

    @property
    def external_cond_emb_dim(self) -> int:
        return self.cfg.hidden_size

    def _build_external_cond_embedding(self) -> Optional[nn.Module]:
        if self.conditioning_type == "global":
            return super()._build_external_cond_embedding()
        match self.conditioning_modeling:
            case "concat":
                return RandomEmbeddingDropout(
                    p=self.conditioning_dropout,
                )
            case "film":
                return RandomDropoutPatchEmbed(
                    dropout_prob=self.conditioning_dropout,
                    img_size=self.x_shape[1],
                    patch_size=self.cfg.patch_size,
                    in_chans=self.external_cond_dim,
                    embed_dim=self.external_cond_emb_dim,
                    bias=True,
                )
            case _:
                raise ValueError(
                    f"Unknown external condition modeling: {self.conditioning_modeling}"
                )

    def initialize_weights(self) -> None:
        super().initialize_weights()
        if self.conditioning_type != "global" and self.conditioning_modeling == "film":
            self._patch_embedder_init(self.external_cond_embedding.patch_embedder)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        external_cond: Optional[torch.Tensor] = None,
        external_cond_mask: Optional[torch.Tensor] = None,
        viewmats: Optional[torch.Tensor] = None,
        Ks: Optional[torch.Tensor] = None,
        patches_x: Optional[float] = None,
        patches_y: Optional[float] = None
    ) -> torch.Tensor:
        assert (
            external_cond is not None
        ), "External condition (camera pose) is required for DiT3DPose model."
        input_batch_size = x.shape[0]
        external_cond_emb = self.external_cond_embedding(
            external_cond, external_cond_mask
        )
        if self.conditioning_modeling == "concat":
            x = torch.cat(
                [x, external_cond_emb],
                dim=2,
            )
        b, v = x.shape[:2]
        assert v == 2, "Expects exactly 2 views (ctxt and trgt)." # TODO different noise levels for multiple views
        x = rearrange(x, "b v c h w -> (b v) c h w")
        x = self.patch_embedder(x)
        x = rearrange(x, "(b v) p c -> b (v p) c", b=input_batch_size)

        t0 = torch.zeros_like(t)
        t1 = t
        # get embeddings for each view
        emb0 = self.noise_level_pos_embedding(t0)
        emb1 = self.noise_level_pos_embedding(t1)

        # squeeze to (b, c) if the module returns (b, 1, c)
        if emb0.dim() == 3 and emb0.size(1) == 1:
            emb0 = emb0.squeeze(1)
            emb1 = emb1.squeeze(1)

        # stack to (b, v=2, c), then repeat patches
        emb = torch.stack([emb0, emb1], dim=1)              # (b, 2, c)
        emb = repeat(emb, "b v c -> b (v p) c", p=self.num_patches)

        if self.conditioning_modeling == "film":
            if self.conditioning_type == "global":
                external_cond_emb = repeat(
                    external_cond_emb, "b v c -> b (v p) c", p=self.num_patches
                )
            else:
                external_cond_emb = rearrange(external_cond_emb, "b v p c -> b (v p) c")
            emb = emb + external_cond_emb

        output_dict = {}
        x = self.dit_base(
            x,
            emb,
            viewmats=viewmats,
            Ks=Ks,
            patches_x=patches_x,
            patches_y=patches_y
        )  # (B, N, C)
        if isinstance(x, tuple):
            x, repa_pred = x
        x = self.unpatchify(
            rearrange(x, "b (v p) c -> (b v) p c", p=self.num_patches)
        )  # (B * V, H, W, C)
        x = rearrange(
            x, "(b v) h w c -> b v c h w", b=input_batch_size
        )  # (B, V, C, H, W)
        output_dict['features'] = x
        if self.use_repa:
            output_dict['repa_pred'] = repa_pred
        if self.use_vggt_alignment:
            output_dict['latents'] = [x.clone()]
        return output_dict