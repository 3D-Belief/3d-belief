"""
Scene-graph-conditioned UViT3DPose backbone.

Subclasses UViT3DPose and injects scene graph conditioning via:
    - SGCrossAttention at TransformerBlock levels (levels 2, 3/mid)
    - SGFiLMProjection at ResBlock levels (levels 0, 1)
"""
from dataclasses import dataclass, field
from typing import Literal, Tuple, Optional
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from einops import rearrange, repeat

from .u_vit3d_pose import UViT3DPose, BackboneUViT3DPoseCfg
from .sg_modules import (
    SceneGraphEncoder,
    SGCrossAttention,
    SGFiLMProjection,
)


@dataclass
class BackboneUViT3DPoseSGCfg(BackboneUViT3DPoseCfg):
    name: Literal["u_vit3d_pose_sg"] = "u_vit3d_pose_sg"
    # Scene graph encoder params
    n_object_types: int = 201          # 200 types + 1 padding (id=0)
    n_edge_types: int = 3              # CONTAINS, NEAR, SAME_ROOM
    sg_use_gcn: bool = True            # False = skip GCN, encode objects independently
    sg_dim: int = 256                  # D_sg: scene graph feature dimension
    sg_clip_dim: int = 512             # CLIP embedding dim (512 for openai/clip-vit-base-patch32)
    sg_type_embeddings_path: str = ""  # path to pre-computed sg_type_embeddings.pt
    sg_gcn_layers: int = 3
    sg_n_heads: int = 8                # for cross-attention
    sg_d_head: int = 32
    sg_dropout_prob: float = 0.1       # SG dropout for classifier-free guidance
    sg_spatial_mode: str = "center"    # "center", "bbox", or "bbox_surface"
    # Wall conditioning
    include_walls: bool = False        # parse walls from seen_object_ids
    wall_height_default: float = 2.5   # fallback ceiling height if not in data
    wall_thickness: float = 0.15       # assumed wall thickness for AABB


class UViT3DPoseSG(UViT3DPose):
    """
    UViT3DPose with scene graph conditioning.

    Injects scene graph information at every UViT level:
    - ResBlock levels: global SG embedding added to conditioning via FiLM
    - TransformerBlock levels: cross-attention from patches to SG node tokens
    """

    def __init__(self, cfg: BackboneUViT3DPoseSGCfg, d_in: int):
        super().__init__(cfg, d_in)
        self.sg_cfg = cfg

        # Scene graph encoder
        self.sg_encoder = SceneGraphEncoder(
            n_object_types=cfg.n_object_types,
            n_edge_types=cfg.n_edge_types,
            sg_dim=cfg.sg_dim,
            clip_dim=cfg.sg_clip_dim,
            clip_embeddings_path=cfg.sg_type_embeddings_path,
            n_gcn_layers=cfg.sg_gcn_layers,
            use_gcn=cfg.sg_use_gcn,
        )

        # Per-level conditioning modules
        channels = cfg.channels
        block_types = cfg.block_types
        self.sg_film_projections = nn.ModuleDict()
        self.sg_cross_attentions = nn.ModuleDict()

        for i_level, (ch, btype) in enumerate(zip(channels, block_types)):
            if btype == "ResBlock":
                # FiLM: global SG emb → added to noise embedding
                self.sg_film_projections[str(i_level)] = SGFiLMProjection(
                    cfg.sg_dim, cfg.emb_channels
                )
            else:
                # Cross-attention: image patches attend to SG tokens
                self.sg_cross_attentions[str(i_level)] = SGCrossAttention(
                    query_dim=ch,
                    context_dim=cfg.sg_dim,
                    n_heads=cfg.sg_n_heads,
                    d_head=cfg.sg_d_head,
                    spatial_mode=cfg.sg_spatial_mode,
                )

    def _encode_scene_graph(self, model_input: dict) -> tuple[Tensor, Tensor]:
        """
        Extract SG keys from model_input and run through SceneGraphEncoder.
        Returns node_tokens (B, N, D_sg) and global_emb (B, D_sg).
        """
        node_tokens, global_emb = self.sg_encoder(
            sg_node_types=model_input["sg_node_types"],
            sg_node_positions=model_input["sg_node_positions"],
            sg_node_rotations=model_input["sg_node_rotations"],
            sg_node_sizes=model_input["sg_node_sizes"],
            sg_edge_index=model_input["sg_edge_index"],
            sg_edge_types=model_input["sg_edge_types"],
            sg_node_mask=model_input["sg_node_mask"],
            sg_edge_mask=model_input["sg_edge_mask"],
        )
        return node_tokens, global_emb

    def _apply_sg_dropout(
        self, node_tokens: Tensor, global_emb: Tensor, sg_node_mask: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Zero out SG conditioning with probability sg_dropout_prob during training."""
        if self.training and self.sg_cfg.sg_dropout_prob > 0:
            B = node_tokens.shape[0]
            drop_mask = (
                torch.rand(B, 1, 1, device=node_tokens.device) < self.sg_cfg.sg_dropout_prob
            )
            node_tokens = node_tokens.masked_fill(drop_mask, 0.0)
            global_emb = global_emb.masked_fill(drop_mask.squeeze(-1), 0.0)
            # Also mask out node_mask so cross-attention sees nothing
            sg_node_mask = sg_node_mask & ~drop_mask.squeeze(-1).bool()
        return node_tokens, global_emb, sg_node_mask

    def _run_level_with_sg(
        self,
        x: Tensor,
        emb: Tensor,
        i_level: int,
        sg_node_tokens: Tensor,
        sg_global_emb: Tensor,
        sg_node_mask: Tensor,
        is_up: bool = False,
        sg_node_positions: Optional[Tensor] = None,
        sg_node_sizes: Optional[Tensor] = None,
        camera_c2w: Optional[Tensor] = None,
        intrinsics: Optional[Tensor] = None,
        sg_wall_endpoints: Optional[Tensor] = None,
        sg_wall_heights: Optional[Tensor] = None,
        sg_node_is_wall: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Run a UViT level with scene graph conditioning injected.

        For ResBlock levels: SG FiLM is added to emb before running blocks.
        For TransformerBlock levels: cross-attention is applied after running blocks,
        with optional 3D→2D spatial Gaussian bias.
        """
        is_transformer = self.is_transformers[i_level]

        if not is_transformer and str(i_level) in self.sg_film_projections:
            # Add SG FiLM to embedding: (B, emb_channels) → expand to (B*T, emb_channels)
            sg_film = self.sg_film_projections[str(i_level)](sg_global_emb)  # (B, emb_channels)
            sg_film = repeat(sg_film, "b c -> (b t) c 1 1", t=self.temporal_length)
            emb = emb + sg_film

        # Run standard level (rearrange → blocks → unrearrange)
        x = self._run_level(x, emb, i_level, is_up)

        if is_transformer and str(i_level) in self.sg_cross_attentions:
            # Apply cross-attention: needs (B*T, N_patches, C) format
            # After _run_level, x is back in (B*T, C, H, W) format
            h, w = x.shape[-2:]
            x_seq = rearrange(x, "(b t) c h w -> (b t) (h w) c", t=self.temporal_length)
            x_seq = self.sg_cross_attentions[str(i_level)](
                x_seq, sg_node_tokens, sg_node_mask,
                temporal_length=self.temporal_length,
                sg_node_positions=sg_node_positions,
                sg_node_sizes=sg_node_sizes,
                camera_c2w=camera_c2w,
                intrinsics=intrinsics,
                patch_grid_size=(h, w),
                sg_wall_endpoints=sg_wall_endpoints,
                sg_wall_heights=sg_wall_heights,
                sg_node_is_wall=sg_node_is_wall,
            )
            x = rearrange(x_seq, "(b t) (h w) c -> (b t) c h w", t=self.temporal_length, h=h, w=w)

        return x

    def forward(
        self,
        x: Tensor,
        t: Tensor,
        external_cond: Optional[Tensor] = None,
        external_cond_mask: Optional[Tensor] = None,
        model_input: Optional[dict] = None,
    ) -> dict:
        """
        Forward pass with scene graph conditioning.

        Same signature as UViT3DPose.forward() but with additional `model_input` dict
        containing scene graph keys (sg_node_types, sg_edge_index, etc.).
        """
        assert model_input is not None, "model_input with SG keys is required"
        assert x.shape[1] == self.temporal_length
        assert external_cond is not None

        # ---- Encode scene graph ----
        sg_node_tokens, sg_global_emb = self._encode_scene_graph(model_input)
        sg_node_mask = model_input["sg_node_mask"]
        sg_node_tokens, sg_global_emb, sg_node_mask = self._apply_sg_dropout(
            sg_node_tokens, sg_global_emb, sg_node_mask
        )

        # ---- Camera info for spatial cross-attention bias ----
        sg_node_positions = model_input.get("sg_node_positions")   # (B, M, 3)
        sg_node_sizes = model_input.get("sg_node_sizes")           # (B, M, 3)
        intrinsics = model_input.get("intrinsics")                 # (B, 3, 3)
        ctxt_abs = model_input.get("ctxt_abs_camera_poses")        # (B, V_c, 4, 4)
        trgt_abs = model_input.get("trgt_abs_camera_poses")        # (B, V_t, 4, 4)
        if ctxt_abs is not None and trgt_abs is not None:
            camera_c2w = torch.cat([ctxt_abs, trgt_abs], dim=1)    # (B, T, 4, 4)
        else:
            camera_c2w = None

        # ---- Wall conditioning tensors (only present when include_walls=True) ----
        sg_wall_endpoints = model_input.get("sg_wall_endpoints")   # (B, M, 2, 3)
        sg_wall_heights = model_input.get("sg_wall_heights")       # (B, M)
        sg_node_is_wall = model_input.get("sg_node_is_wall")       # (B, M) bool

        # ---- Standard UViT setup (from UViT3DPose.forward) ----
        h_in, w_in = x.shape[-2], x.shape[-1]
        x = rearrange(x, "b t c h w -> (b t) c h w")
        x = self.embed_input(x)

        # Embeddings
        external_cond = self.external_cond_embedding(external_cond, external_cond_mask)
        t0 = torch.zeros_like(t)
        t1 = t
        emb0 = self.noise_level_pos_embedding(t0)
        emb1 = self.noise_level_pos_embedding(t1)
        if emb0.dim() == 3 and emb0.size(1) == 1:
            emb0 = emb0.squeeze(1)
            emb1 = emb1.squeeze(1)
        emb = torch.stack([emb0, emb1], dim=1)
        emb = rearrange(
            rearrange(emb, "b t c -> b t c 1 1") + external_cond,
            "b t c h w -> (b t) c h w",
        )

        # Down-sample embeddings for each level
        embs = [
            (
                emb
                if i_level == 0
                else F.avg_pool2d(emb, kernel_size=2**i_level, stride=2**i_level)
            )
            for i_level in range(self.num_levels)
        ]

        hs_before = []
        hs_after = []

        return_latents = self.use_vggt_alignment
        if return_latents:
            latents_list = []

        # ---- Down-sampling with SG conditioning ----
        for i_level, down_block in enumerate(self.down_blocks):
            x = self._run_level_with_sg(
                x, embs[i_level], i_level,
                sg_node_tokens, sg_global_emb, sg_node_mask,
                sg_node_positions=sg_node_positions,
                sg_node_sizes=sg_node_sizes,
                camera_c2w=camera_c2w,
                intrinsics=intrinsics,
                sg_wall_endpoints=sg_wall_endpoints,
                sg_wall_heights=sg_wall_heights,
                sg_node_is_wall=sg_node_is_wall,
            )
            hs_before.append(x)
            if return_latents:
                latents_list.append(x)
            x = down_block[-1](x)  # Downsample
            hs_after.append(x)

        # ---- Middle blocks with SG conditioning ----
        x = self._run_level_with_sg(
            x, embs[-1], self.num_levels - 1,
            sg_node_tokens, sg_global_emb, sg_node_mask,
            sg_node_positions=sg_node_positions,
            sg_node_sizes=sg_node_sizes,
            camera_c2w=camera_c2w,
            intrinsics=intrinsics,
            sg_wall_endpoints=sg_wall_endpoints,
            sg_wall_heights=sg_wall_heights,
            sg_node_is_wall=sg_node_is_wall,
        )
        if return_latents:
            latents_list.append(x)

        # ---- Up-sampling with SG conditioning ----
        for _i_level, up_block in enumerate(self.up_blocks):
            i_level = self.num_levels - 2 - _i_level
            x = x - hs_after.pop()
            x = up_block[0](x) + hs_before.pop()
            x = self._run_level_with_sg(
                x, embs[i_level], i_level,
                sg_node_tokens, sg_global_emb, sg_node_mask,
                is_up=True,
                sg_node_positions=sg_node_positions,
                sg_node_sizes=sg_node_sizes,
                camera_c2w=camera_c2w,
                intrinsics=intrinsics,
                sg_wall_endpoints=sg_wall_endpoints,
                sg_wall_heights=sg_wall_heights,
                sg_node_is_wall=sg_node_is_wall,
            )
            if return_latents:
                latents_list.append(x)

        # ---- Output ----
        x = self.project_output(x)
        out = rearrange(x, "(b t) c h w -> b t c h w", t=self.temporal_length)
        output_dict = {"features": out}

        if return_latents:
            latents_list = [
                rearrange(lat, "(b t) c h w -> b t c h w", t=self.temporal_length)
                for lat in latents_list
            ]
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
