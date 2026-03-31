"""
Scene graph conditioning modules for 3D-belief UViT backbone.

Implements:
    - TripletGCNLayer: Triplet message-passing GCN (CommonScenes/EchoScene)
    - SceneGraphEncoder: CLIP embeddings + geometry MLP + GCN stack
    - SGCrossAttention: Cross-attention from image patches to SG tokens (GLIGEN-style gating)
    - SGFiLMProjection: Global SG embedding → scale/shift for ResBlock FiLM
"""
from typing import Optional

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from einops import rearrange, repeat

from ..common.zero_module import zero_module


# ---------------------------------------------------------------------------
# Triplet-GCN layer (EchoScene / CommonScenes pattern)
# ---------------------------------------------------------------------------
class TripletGCNLayer(nn.Module):
    """
    Triplet message-passing: for edge (i→j) with edge feature e_ij,
    compute updated node and edge representations via
        triplet = MLP([node_i || edge_ij || node_j])
    then scatter-aggregate back to nodes.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.triplet_mlp = nn.Sequential(
            nn.Linear(3 * dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.node_update = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.edge_update = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        node_feat: Tensor,     # (B, N, D)
        edge_feat: Tensor,     # (B, E, D)
        edge_index: Tensor,    # (B, 2, E) — [src, dst]
        node_mask: Tensor,     # (B, N) bool
        edge_mask: Tensor,     # (B, E) bool
    ) -> tuple[Tensor, Tensor]:
        B, N, D = node_feat.shape
        E = edge_feat.shape[1]

        # Gather source and destination node features per edge
        src_idx = edge_index[:, 0, :]  # (B, E)
        dst_idx = edge_index[:, 1, :]  # (B, E)

        src_feat = torch.gather(node_feat, 1, src_idx.unsqueeze(-1).expand(-1, -1, D))  # (B, E, D)
        dst_feat = torch.gather(node_feat, 1, dst_idx.unsqueeze(-1).expand(-1, -1, D))  # (B, E, D)

        # Triplet message
        triplet = torch.cat([src_feat, edge_feat, dst_feat], dim=-1)  # (B, E, 3D)
        msg = self.triplet_mlp(triplet) * edge_mask.unsqueeze(-1).float()  # (B, E, D)

        # Update edges
        edge_feat = edge_feat + self.edge_update(msg)

        # Aggregate messages to destination nodes (scatter-add)
        agg = torch.zeros_like(node_feat)
        dst_expanded = dst_idx.unsqueeze(-1).expand(-1, -1, D)
        agg.scatter_add_(1, dst_expanded, msg)

        # Count incoming edges per node for mean aggregation
        counts = torch.zeros(B, N, 1, device=node_feat.device)
        counts.scatter_add_(1, dst_idx.unsqueeze(-1), edge_mask.unsqueeze(-1).float())
        counts = counts.clamp(min=1.0)
        agg = agg / counts

        # Update nodes
        node_feat = node_feat + self.node_update(agg) * node_mask.unsqueeze(-1).float()
        node_feat = self.norm(node_feat)

        return node_feat, edge_feat


# ---------------------------------------------------------------------------
# Scene Graph Encoder
# ---------------------------------------------------------------------------
class SceneGraphEncoder(nn.Module):
    """
    Encode ProcTHOR scene graphs into per-node tokens and a global embedding.

    Pipeline:
        1. Lookup pre-computed CLIP embeddings for object types → project to D_sg
        2. MLP encode geometry (pos, rot, size) → D_sg
        3. Fuse type + geometry → D_sg
        4. n_gcn_layers of TripletGCNLayer
        5. Mean-pool over valid nodes → global embedding
    """

    def __init__(
        self,
        n_object_types: int,
        n_edge_types: int,
        sg_dim: int,
        clip_dim: int,
        clip_embeddings_path: str,
        n_gcn_layers: int = 3,
        use_gcn: bool = True,
    ):
        super().__init__()
        self.sg_dim = sg_dim
        self.use_gcn = use_gcn

        # Load pre-computed CLIP type embeddings as frozen buffer
        clip_embs = torch.load(clip_embeddings_path, map_location="cpu")  # (n_types, clip_dim)
        self.register_buffer("clip_type_embeddings", clip_embs)

        self.type_proj = nn.Linear(clip_dim, sg_dim)
        self.node_geom_mlp = nn.Sequential(
            nn.Linear(9, sg_dim),
            nn.SiLU(),
            nn.Linear(sg_dim, sg_dim),
        )
        self.node_fuse = nn.Linear(2 * sg_dim, sg_dim)

        if self.use_gcn:
            self.edge_type_embed = nn.Embedding(n_edge_types + 1, sg_dim, padding_idx=0)
            # edge type 0 is padding; real types start at 1
            # shift real edge types by +1 in forward

            self.gcn_layers = nn.ModuleList(
                [TripletGCNLayer(sg_dim) for _ in range(n_gcn_layers)]
            )

    def forward(
        self,
        sg_node_types: Tensor,      # (B, N) int64
        sg_node_positions: Tensor,   # (B, N, 3)
        sg_node_rotations: Tensor,   # (B, N, 3)
        sg_node_sizes: Tensor,       # (B, N, 3)
        sg_edge_index: Tensor,       # (B, 2, E)
        sg_edge_types: Tensor,       # (B, E) int64
        sg_node_mask: Tensor,        # (B, N) bool
        sg_edge_mask: Tensor,        # (B, E) bool
    ) -> tuple[Tensor, Tensor]:
        """
        Returns:
            node_tokens: (B, N, D_sg)
            global_emb: (B, D_sg)
        """
        # 1. Type embedding via frozen CLIP lookup + learned projection
        type_embs = self.clip_type_embeddings[sg_node_types]  # (B, N, clip_dim)
        type_feat = self.type_proj(type_embs)                 # (B, N, D_sg)

        # 2. Geometry embedding
        geom_input = torch.cat([sg_node_positions, sg_node_rotations, sg_node_sizes], dim=-1)
        geom_feat = self.node_geom_mlp(geom_input)  # (B, N, D_sg)

        # 3. Fuse
        node_feat = self.node_fuse(torch.cat([type_feat, geom_feat], dim=-1))  # (B, N, D_sg)
        node_feat = node_feat * sg_node_mask.unsqueeze(-1).float()

        # 4-5. GCN message-passing (skipped when use_gcn=False)
        if self.use_gcn:
            edge_feat = self.edge_type_embed(sg_edge_types + 1)  # (B, E, D_sg)
            edge_feat = edge_feat * sg_edge_mask.unsqueeze(-1).float()

            for gcn in self.gcn_layers:
                node_feat, edge_feat = gcn(
                    node_feat, edge_feat, sg_edge_index, sg_node_mask, sg_edge_mask
                )

        # 6. Global pooling
        masked_sum = (node_feat * sg_node_mask.unsqueeze(-1).float()).sum(dim=1)
        node_count = sg_node_mask.float().sum(dim=1, keepdim=True).clamp(min=1.0)
        global_emb = masked_sum / node_count  # (B, D_sg)

        return node_feat, global_emb


# ---------------------------------------------------------------------------
# Cross-attention: image patches → SG tokens  (for TransformerBlock levels)
# ---------------------------------------------------------------------------
class SGCrossAttention(nn.Module):
    """
    Cross-attention from image features (query) to scene graph node tokens (key/value).
    Uses GLIGEN-style zero-initialized gating for stable training.

    Optionally adds a spatial attention bias: each object's 3D center is projected
    into each camera view and a Gaussian bias encourages image patches to attend
    to nearby SG tokens.  The bias is gated (zero-init) so it has no effect at
    the start of training.
    """

    def __init__(
        self,
        query_dim: int,
        context_dim: int,
        n_heads: int = 8,
        d_head: int = 32,
    ):
        super().__init__()
        inner_dim = n_heads * d_head
        self.n_heads = n_heads
        self.d_head = d_head
        self.scale = d_head ** -0.5

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim, bias=False)

        # Zero-init gate (GLIGEN pattern)
        self.gate = nn.Parameter(torch.zeros(1))

        # Layer norms
        self.norm_q = nn.LayerNorm(query_dim)
        self.norm_k = nn.LayerNorm(context_dim)

        # Spatial bias: learnable log-sigma per head + zero-init gate
        self.spatial_log_sigma = nn.Parameter(torch.zeros(n_heads))
        self.spatial_gate = nn.Parameter(torch.zeros(1))

    # ------------------------------------------------------------------
    def _compute_spatial_bias(
        self,
        sg_node_positions: Tensor,  # (B, M, 3) world-space centers
        camera_c2w: Tensor,         # (B, T, 4, 4) absolute camera-to-world
        intrinsics: Tensor,         # (B, 3, 3) normalised intrinsics
        patch_h: int,
        patch_w: int,
    ) -> Tensor:
        """
        Returns spatial Gaussian bias of shape (B*T, n_heads, P, M)
        where P = patch_h * patch_w, M = num SG nodes.
        """
        B, M, _ = sg_node_positions.shape
        T = camera_c2w.shape[1]
        device = sg_node_positions.device
        dtype = sg_node_positions.dtype

        # --- world → camera transform ---
        w2c = torch.inverse(camera_c2w)                     # (B, T, 4, 4)
        R = w2c[:, :, :3, :3]                               # (B, T, 3, 3)
        t = w2c[:, :, :3, 3]                                # (B, T, 3)

        # (B, 1, M, 3) @ (B, T, 3, 3)^T + (B, T, 1, 3) → (B, T, M, 3)
        pos_exp = sg_node_positions.unsqueeze(1)             # (B, 1, M, 3)
        p_cam = torch.einsum("btij,bpmj->bpmi", R, pos_exp.expand(-1, T, -1, -1)) \
                + t.unsqueeze(2)                             # (B, T, M, 3)

        z = p_cam[..., 2].clamp(min=1e-4)                   # (B, T, M)

        # --- project to normalised 2D ---
        fx = intrinsics[:, 0, 0]                             # (B,)
        fy = intrinsics[:, 1, 1]
        cx = intrinsics[:, 0, 2]
        cy = intrinsics[:, 1, 2]

        u = fx[:, None, None] * (p_cam[..., 0] / z) + cx[:, None, None]  # (B, T, M)
        v = fy[:, None, None] * (p_cam[..., 1] / z) + cy[:, None, None]

        # --- patch centre grid in [0, 1] ---
        grid_y = (torch.arange(patch_h, device=device, dtype=dtype) + 0.5) / patch_h
        grid_x = (torch.arange(patch_w, device=device, dtype=dtype) + 0.5) / patch_w
        gy, gx = torch.meshgrid(grid_y, grid_x, indexing="ij")  # (ph, pw)
        gx = gx.reshape(-1)                                      # (P,)
        gy = gy.reshape(-1)

        # --- squared distance: (B, T, P, M) ---
        sq_dist = (gx[None, None, :, None] - u[:, :, None, :]) ** 2 \
                + (gy[None, None, :, None] - v[:, :, None, :]) ** 2

        # --- depth-adaptive sigma per head: sigma_h / z ---
        sigma = self.spatial_log_sigma.exp()                 # (H,)
        # (H,) / (B, T, 1, M) → (B, T, H, M)  — broadcast later
        inv_two_sigma_sq = z.unsqueeze(2) ** 2 / (2.0 * sigma[None, None, :, None] ** 2 + 1e-8)
        # inv_two_sigma_sq: (B, T, H, M)

        # bias: -(sq_dist * z²) / (2 * sigma²)  →  (B, T, H, P, M)
        bias = -sq_dist.unsqueeze(2) * inv_two_sigma_sq.unsqueeze(3)

        # Zero out objects behind the camera (original z before clamp)
        behind = (p_cam[..., 2] <= 0.0)                      # (B, T, M)
        bias.masked_fill_(behind[:, :, None, None, :], 0.0)

        # Flatten (B, T) → (B*T)
        bias = rearrange(bias, "b t h p m -> (b t) h p m")  # (BT, H, P, M)
        return bias

    # ------------------------------------------------------------------
    def forward(
        self,
        x: Tensor,                                 # (B*T, P, C)
        sg_tokens: Tensor,                         # (B, M, D_sg)
        sg_node_mask: Tensor,                      # (B, M) bool
        temporal_length: int = 2,
        sg_node_positions: Optional[Tensor] = None,  # (B, M, 3)
        camera_c2w: Optional[Tensor] = None,         # (B, T, 4, 4)
        intrinsics: Optional[Tensor] = None,         # (B, 3, 3)
        patch_grid_size: Optional[tuple] = None,     # (h, w)
    ) -> Tensor:
        BT, N, C = x.shape
        B = BT // temporal_length
        T = temporal_length
        M = sg_tokens.shape[1]

        # Expand SG tokens for all views: (B, M, D_sg) → (B*T, M, D_sg)
        sg_tokens_exp = repeat(sg_tokens, "b n d -> (b t) n d", t=T)
        mask_exp = repeat(sg_node_mask, "b n -> (b t) n", t=T)

        # Normalize
        x_normed = self.norm_q(x)
        sg_normed = self.norm_k(sg_tokens_exp)

        # Project
        q = self.to_q(x_normed)
        k = self.to_k(sg_normed)
        v = self.to_v(sg_normed)

        # Reshape for multi-head attention
        q = rearrange(q, "bt n (h d) -> bt h n d", h=self.n_heads)
        k = rearrange(k, "bt m (h d) -> bt h m d", h=self.n_heads)
        v = rearrange(v, "bt m (h d) -> bt h m d", h=self.n_heads)

        # Attention bias: mask out padded nodes
        attn_bias = torch.zeros(BT, 1, 1, M, device=x.device, dtype=x.dtype)
        attn_bias.masked_fill_(~mask_exp.unsqueeze(1).unsqueeze(2), float("-inf"))

        # Spatial Gaussian bias (optional)
        if (
            sg_node_positions is not None
            and camera_c2w is not None
            and intrinsics is not None
            and patch_grid_size is not None
        ):
            ph, pw = patch_grid_size
            spatial_bias = self._compute_spatial_bias(
                sg_node_positions, camera_c2w, intrinsics, ph, pw,
            )                                                    # (BT, H, P, M)
            attn_bias = attn_bias + self.spatial_gate * spatial_bias

        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
        attn = rearrange(attn, "bt h n d -> bt n (h d)")
        out = self.to_out(attn)

        return x + self.gate * out


# ---------------------------------------------------------------------------
# FiLM projection: global SG embedding → scale/shift for ResBlock
# ---------------------------------------------------------------------------
class SGFiLMProjection(nn.Module):
    """
    Projects global scene graph embedding to be added to the noise conditioning
    embedding before FiLM modulation in ResBlocks.

    Output is (B, C_level) which gets added to existing emb of same shape.
    """

    def __init__(self, sg_dim: int, out_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(sg_dim, out_dim),
            nn.SiLU(),
            # Zero-init for stable training start
            zero_module(nn.Linear(out_dim, out_dim)),
        )

    def forward(self, global_emb: Tensor) -> Tensor:
        """
        Args:
            global_emb: (B, D_sg)
        Returns:
            (B, C_level)
        """
        return self.mlp(global_emb)
