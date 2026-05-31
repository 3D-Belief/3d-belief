from dataclasses import dataclass
from typing import Literal, Tuple, Optional, Union
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from omegaconf import DictConfig
from einops import rearrange
from ..common.embeddings import (
    RandomDropoutPatchEmbed,
)
from .u_vit3d import UViT3D
from .backbone import Backbone

@dataclass
class BackboneUViT3DPoseCfg:
    name: Literal["u_vit3d_pose"] = "u_vit3d_pose"
    input_size: Tuple[int, int] = (64, 64)
    channels: tuple[int, int, int, int] = (128, 256, 512, 1024)
    max_tokens: int = 16
    emb_channels: int = 1024
    patch_size: int = 2
    block_types: tuple[str, str, str, str] = ("ResBlock", "ResBlock", "TransformerBlock", "TransformerBlock")
    block_dropouts: tuple[float, float, float, float] = (0.0, 0.0, 0.1, 0.1)
    num_updown_blocks: tuple[int, int, int] = (3, 3, 3)
    num_mid_blocks: int = 16
    num_heads: int = 4
    pos_emb_type: Literal["rope", "learned_1d", "sinusoidal_1d", "sinusoidal_3d", "sinusoidal_factorized"] = "rope"
    use_checkpointing: tuple[bool, bool, bool, bool] = (False, False, True, True)
    conditioning_type: Literal["global", "ray_encoding"] = "ray_encoding"
    conditioning_dim: int = 180
    external_cond_dropout: float = 0.1
    use_causal_mask: bool = False
    external_cond_dropout: float = 0.1
    use_repa: bool = False
    repa_z_dim: int = 768
    repa_size: int = 32
    use_vggt_alignment: bool = False

class UViT3DPose(Backbone[BackboneUViT3DPoseCfg], UViT3D):
    """
    U-ViT with pose embedding.
    """

    def __init__(self, cfg: BackboneUViT3DPoseCfg, d_in: int):
        assert d_in == 3, "Input d_in must be 3"
        self.conditioning_type = cfg.conditioning_type
        self.conditioning_dropout = cfg.external_cond_dropout
        self.input_size = cfg.input_size
        self.emb_channels = cfg.emb_channels
        x_shape = torch.Size((d_in, cfg.input_size[0], cfg.input_size[1]))
        max_tokens = cfg.max_tokens
        conditioning_dim = cfg.conditioning_dim
        use_causal_mask = cfg.use_causal_mask
        self.patch_size = cfg.patch_size
        super().__init__(
            cfg=cfg,
            x_shape=x_shape,
            max_tokens=max_tokens,
            external_cond_dim=conditioning_dim,
            use_causal_mask=use_causal_mask,
        )
    
    @property
    def d_out(self) -> int:
        return 1024

    def _build_external_cond_embedding(self) -> Optional[nn.Module]:
        return RandomDropoutPatchEmbed(
            dropout_prob=self.conditioning_dropout,
            img_size=self.x_shape[1],
            patch_size=self.cfg.patch_size,
            in_chans=self.external_cond_dim,
            embed_dim=self.external_cond_emb_dim,
            bias=True,
            flatten=False,
        )

    def _rearrange_and_add_pos_emb_if_transformer(
        self, x: Tensor, emb: Tensor, i_level: int
    ) -> Tuple[Tensor, Tensor]:
        is_transformer = self.is_transformers[i_level]
        if not is_transformer:
            return x, emb
        x, emb = map(
            lambda y: rearrange(
                y, "(b t) c h w -> b (t h w) c", t=self.temporal_length
            ),
            (x, emb),
        )
        if self.pos_emb_type == "learned_1d":
            x = self.pos_embs[f"{i_level}"](x)
        return x, emb

    def forward(
        self,
        x: Tensor,
        t: Tensor,
        external_cond: Optional[Tensor] = None,
        external_cond_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass of the U-ViT backbone, with pose conditioning.
        Args:
            x: Input tensor of shape (B, T, C, H, W).
            t: Noise level tensor of shape (B, T).
            external_cond: External conditioning tensor of shape (B, T, C', H, W).
        Returns:
            Output tensor of shape (B, T, C, H, W).
        """
        assert (
            x.shape[1] == self.temporal_length
        ), f"Temporal length of U-ViT is set to {self.temporal_length}, but input has temporal length {x.shape[1]}."

        assert (
            external_cond is not None
        ), "External condition (camera pose) is required for U-ViT3DPose model."

        h_in, w_in = x.shape[-2], x.shape[-1]

        x = rearrange(x, "b t c h w -> (b t) c h w")
        x = self.embed_input(x)

        # Embeddings
        external_cond = self.external_cond_embedding(external_cond, external_cond_mask)
        t0 = torch.zeros_like(t)
        t1 = t
        # get embeddings for each view
        emb0 = self.noise_level_pos_embedding(t0)
        emb1 = self.noise_level_pos_embedding(t1)

        # squeeze to (b, c) if the module returns (b, 1, c)
        if emb0.dim() == 3 and emb0.size(1) == 1:
            emb0 = emb0.squeeze(1)
            emb1 = emb1.squeeze(1)

        # stack to (b, v=2, c)
        emb = torch.stack([emb0, emb1], dim=1)              # (b, 2, c)
        emb = rearrange(
            rearrange(emb, "b t c -> b t c 1 1") + external_cond,
            "b t c h w -> (b t) c h w",
        )

        # Down-sample embeddings for each level
        embs = [
            (
                emb
                if i_level == 0
                # pylint: disable-next=not-callable
                else F.avg_pool2d(emb, kernel_size=2**i_level, stride=2**i_level)
            )
            for i_level in range(self.num_levels)
        ]
        hs_before = []  # hidden states before downsampling
        hs_after = []  # hidden states after downsampling

        return_latents = self.use_vggt_alignment
        if return_latents:
            latents_list = []

        # Down-sampling blocks
        for i_level, down_block in enumerate(
            self.down_blocks,
        ):
            x = self._run_level(x, embs[i_level], i_level)
            hs_before.append(x)
            if return_latents: 
                latents_list.append(x)
            x = down_block[-1](x)
            hs_after.append(x)

        # Middle blocks
        x = self._run_level(x, embs[-1], self.num_levels - 1)
        if return_latents: 
            latents_list.append(x)

        # Up-sampling blocks
        for _i_level, up_block in enumerate(self.up_blocks):
            i_level = self.num_levels - 2 - _i_level
            x = x - hs_after.pop()
            x = up_block[0](x) + hs_before.pop()
            x = self._run_level(x, embs[i_level], i_level, is_up=True)
            if return_latents: 
                latents_list.append(x)

        x = self.project_output(x)
        out = rearrange(x, "(b t) c h w -> b t c h w", t=self.temporal_length)
        output_dict = {"features": out}
        if return_latents:
            latents_list = [rearrange(latent, "(b t) c h w -> b t c h w", t=self.temporal_length) for latent in latents_list]
            output_dict["latents"] = latents_list

        if self.use_repa:
            out_features = F.interpolate(
                out.view(-1, out.size(2), out.size(3), out.size(4)),
                size=(self.repa_size, self.repa_size),
                mode="bilinear",
                align_corners=False,
            ).view(out.size(0), out.size(1), out.size(2), self.repa_size, self.repa_size)
            out_features = rearrange(out_features, "b t c h w -> b (t h w) c")
            batch, seq_len, dim = out_features.shape
            repa_pred = self.repa_mlp(out_features.reshape(-1, dim)).reshape(batch, seq_len, -1)
            assert repa_pred.shape[-1] == self.repa_z_dim
            repa_pred = rearrange(repa_pred, "b (t hw) c -> b t hw c", t=self.temporal_length)
            output_dict["repa_pred"] = repa_pred
        return output_dict
