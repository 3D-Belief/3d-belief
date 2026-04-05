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

    Optionally adds a spatial attention bias: each object's 3D geometry is projected
    into each camera view and a Gaussian bias encourages image patches to attend
    to nearby SG tokens.  The bias is gated (zero-init) so it has no effect at
    the start of training.

    spatial_mode:
        "center" — isotropic Gaussian from projected 3D center (depth-adaptive σ)
        "bbox"   — anisotropic Gaussian from projected 8-corner AABB
        "bbox_surface" — bbox for objects + surface-aware bias for walls
    """

    def __init__(
        self,
        query_dim: int,
        context_dim: int,
        n_heads: int = 8,
        d_head: int = 32,
        spatial_mode: str = "center",
    ):
        super().__init__()
        assert spatial_mode in ("center", "bbox", "bbox_surface"), f"Unknown spatial_mode: {spatial_mode}"
        self.spatial_mode = spatial_mode
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

        # Depth-dependent bias: closer objects get higher bias (occlusion prior)
        # Zero-init → no effect at start; model learns to use depth weighting
        self.depth_log_alpha = nn.Parameter(torch.zeros(n_heads))

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

        # Depth-dependent bias: -alpha * log(z) → closer objects get higher bias
        depth_bonus = -torch.log(z)                          # (B, T, M)
        depth_term = self.depth_log_alpha[None, None, :, None, None] \
                     * depth_bonus[:, :, None, None, :]      # (B, T, H, 1, M)
        bias = bias + depth_term

        # Zero out objects behind the camera (original z before clamp)
        behind = (p_cam[..., 2] <= 0.0)                      # (B, T, M)
        bias.masked_fill_(behind[:, :, None, None, :], 0.0)

        # Flatten (B, T) → (B*T)
        bias = rearrange(bias, "b t h p m -> (b t) h p m")  # (BT, H, P, M)
        return bias

    # ------------------------------------------------------------------
    @staticmethod
    def _compute_bbox_corners(
        positions: Tensor,  # (B, M, 3) world-space centres
        sizes: Tensor,      # (B, M, 3) full AABB dimensions (dx, dy, dz)
    ) -> Tensor:
        """Compute 8 AABB corners in world space → (B, M, 8, 3)."""
        half = sizes / 2                                     # (B, M, 3)
        # 8 sign combinations: (8, 3) with ±1 per axis
        signs = positions.new_tensor(
            [[-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
             [ 1, -1, -1], [ 1, -1, 1], [ 1, 1, -1], [ 1, 1, 1]],
        )                                                    # (8, 3)
        # (B, M, 1, 3) * (8, 3) → (B, M, 8, 3)
        offsets = half.unsqueeze(2) * signs                  # broadcast
        corners = positions.unsqueeze(2) + offsets           # (B, M, 8, 3)
        return corners

    # ------------------------------------------------------------------
    def _compute_spatial_bias_bbox(
        self,
        sg_node_positions: Tensor,  # (B, M, 3)
        sg_node_sizes: Tensor,      # (B, M, 3)
        camera_c2w: Tensor,         # (B, T, 4, 4)
        intrinsics: Tensor,         # (B, 3, 3)
        patch_h: int,
        patch_w: int,
    ) -> Tensor:
        """
        Anisotropic Gaussian bias from projected 3D AABBs.

        Projects all 8 AABB corners into each camera view, derives a 2D bounding
        rectangle, then computes a Gaussian whose σ_x and σ_y match the bbox
        half-widths.  Returns (B*T, n_heads, P, M).
        """
        B, M, _ = sg_node_positions.shape
        T = camera_c2w.shape[1]
        device = sg_node_positions.device
        dtype = sg_node_positions.dtype

        # --- 8 world-space AABB corners → (B, M, 8, 3) ---
        corners = self._compute_bbox_corners(sg_node_positions, sg_node_sizes)

        # --- world → camera ---
        w2c = torch.inverse(camera_c2w)                      # (B, T, 4, 4)
        R = w2c[:, :, :3, :3]                                # (B, T, 3, 3)
        t_vec = w2c[:, :, :3, 3]                             # (B, T, 3)

        # Transform corners: (B, M, 8, 3) → (B, T, M, 8, 3)
        corners_exp = corners.unsqueeze(1).expand(-1, T, -1, -1, -1)  # (B, T, M, 8, 3)
        # Batched matmul: R @ corner + t
        p_cam = torch.einsum("btij,btmcj->btmci", R, corners_exp) \
                + t_vec[:, :, None, None, :]                 # (B, T, M, 8, 3)

        z_corners = p_cam[..., 2]                            # (B, T, M, 8)
        visible = z_corners > 0                              # (B, T, M, 8)
        z_safe = z_corners.clamp(min=1e-4)

        # --- perspective project ---
        fx = intrinsics[:, 0, 0]                             # (B,)
        fy = intrinsics[:, 1, 1]
        cx = intrinsics[:, 0, 2]
        cy = intrinsics[:, 1, 2]

        u_corners = fx[:, None, None, None] * (p_cam[..., 0] / z_safe) \
                    + cx[:, None, None, None]                # (B, T, M, 8)
        v_corners = fy[:, None, None, None] * (p_cam[..., 1] / z_safe) \
                    + cy[:, None, None, None]                # (B, T, M, 8)

        # --- derive 2D bounding rect from visible corners ---
        INF = torch.tensor(float("inf"), device=device, dtype=dtype)
        u_for_min = torch.where(visible, u_corners, INF)
        u_for_max = torch.where(visible, u_corners, -INF)
        v_for_min = torch.where(visible, v_corners, INF)
        v_for_max = torch.where(visible, v_corners, -INF)

        u_min = u_for_min.min(dim=-1).values                # (B, T, M)
        u_max = u_for_max.max(dim=-1).values
        v_min = v_for_min.min(dim=-1).values
        v_max = v_for_max.max(dim=-1).values

        # Replace INF/-INF with safe defaults for fully-behind-camera objects
        # to prevent NaN in cu/cv which poisons the backward pass.
        any_visible = visible.any(dim=-1)                    # (B, T, M)
        fully_behind = ~any_visible
        u_min = torch.where(fully_behind, torch.zeros_like(u_min), u_min)
        u_max = torch.where(fully_behind, torch.ones_like(u_max), u_max)
        v_min = torch.where(fully_behind, torch.zeros_like(v_min), v_min)
        v_max = torch.where(fully_behind, torch.ones_like(v_max), v_max)

        # bbox centre & half-extents (clamped to avoid degenerate boxes)
        cu = (u_min + u_max) / 2                             # (B, T, M)
        cv = (v_min + v_max) / 2
        EPS = 0.01
        half_w = ((u_max - u_min) / 2).clamp(min=EPS)       # (B, T, M)
        half_h = ((v_max - v_min) / 2).clamp(min=EPS)

        # --- patch centre grid in [0, 1] ---
        grid_y = (torch.arange(patch_h, device=device, dtype=dtype) + 0.5) / patch_h
        grid_x = (torch.arange(patch_w, device=device, dtype=dtype) + 0.5) / patch_w
        gy, gx = torch.meshgrid(grid_y, grid_x, indexing="ij")
        gx = gx.reshape(-1)                                 # (P,)
        gy = gy.reshape(-1)

        # --- normalised squared distance: (B, T, P, M) ---
        du = (gx[None, None, :, None] - cu[:, :, None, :]) / half_w[:, :, None, :]
        dv = (gy[None, None, :, None] - cv[:, :, None, :]) / half_h[:, :, None, :]
        sq_dist_norm = du ** 2 + dv ** 2                     # (B, T, P, M)

        # --- per-head Gaussian bias ---
        sigma = self.spatial_log_sigma.exp()                 # (H,)
        inv_two_sigma_sq = 1.0 / (2.0 * sigma ** 2 + 1e-8)  # (H,)
        # (B, T, P, M) * (H,) → (B, T, H, P, M)
        bias = -sq_dist_norm.unsqueeze(2) * inv_two_sigma_sq[None, None, :, None, None]

        # Depth-dependent bias: project centers to get camera-space z
        pos_exp = sg_node_positions.unsqueeze(1).expand(-1, T, -1, -1)  # (B, T, M, 3)
        center_cam = torch.einsum("btij,btmj->btmi", R, pos_exp) \
                     + t_vec.unsqueeze(2)                    # (B, T, M, 3)
        z_center = center_cam[..., 2].clamp(min=1e-4)       # (B, T, M)
        depth_bonus = -torch.log(z_center)                   # (B, T, M)
        depth_term = self.depth_log_alpha[None, None, :, None, None] \
                     * depth_bonus[:, :, None, None, :]      # (B, T, H, 1, M)
        bias = bias + depth_term

        # Clamp to avoid extreme negative values that cause NaN gradients in softmax
        bias = bias.clamp(min=-50.0)

        # Objects fully behind camera (no visible corner) → neutral bias
        # (any_visible already computed above)
        bias.masked_fill_(~any_visible[:, :, None, None, :], 0.0)

        # Flatten (B, T) → (B*T)
        bias = rearrange(bias, "b t h p m -> (b t) h p m")  # (BT, H, P, M)
        return bias

    # ------------------------------------------------------------------
    @staticmethod
    def _compute_wall_quad_corners(
        wall_endpoints: Tensor,  # (B, M, 2, 3) — bottom endpoints in world space
        wall_heights: Tensor,    # (B, M) — ceiling height per wall
    ) -> Tensor:
        """Compute 4 wall quad corners → (B, M, 4, 3).

        Order: bottom_p1, bottom_p2, top_p1, top_p2
        """
        p1 = wall_endpoints[:, :, 0, :]                     # (B, M, 3)
        p2 = wall_endpoints[:, :, 1, :]                     # (B, M, 3)
        h = wall_heights.unsqueeze(-1)                       # (B, M, 1)
        y_offset = torch.zeros_like(p1)
        y_offset[..., 1] = h.squeeze(-1)                    # (B, M) → y component
        top_p1 = p1 + y_offset                              # (B, M, 3)
        top_p2 = p2 + y_offset                              # (B, M, 3)
        corners = torch.stack([p1, p2, top_p1, top_p2], dim=2)  # (B, M, 4, 3)
        return corners

    # ------------------------------------------------------------------
    def _compute_spatial_bias_surface(
        self,
        wall_endpoints: Tensor,     # (B, M_w, 2, 3) bottom endpoints
        wall_heights: Tensor,       # (B, M_w) ceiling heights
        camera_c2w: Tensor,         # (B, T, 4, 4)
        intrinsics: Tensor,         # (B, 3, 3)
        patch_h: int,
        patch_w: int,
    ) -> Tensor:
        """
        Surface-aware spatial bias for wall tokens.

        Projects 4 wall quad corners, finds the projected bounding rect,
        and computes a **box-distance** bias: zero inside the rect (uniform
        high bias), Gaussian falloff outside.

        Returns (B*T, n_heads, P, M_w).
        """
        B, M_w, _, _ = wall_endpoints.shape
        T = camera_c2w.shape[1]
        device = wall_endpoints.device
        dtype = wall_endpoints.dtype

        if M_w == 0:
            P = patch_h * patch_w
            return torch.zeros(B * T, self.n_heads, P, 0, device=device, dtype=dtype)

        # --- 4 wall quad corners: (B, M_w, 4, 3) ---
        corners = self._compute_wall_quad_corners(wall_endpoints, wall_heights)

        # --- world → camera ---
        w2c = torch.inverse(camera_c2w)                      # (B, T, 4, 4)
        R = w2c[:, :, :3, :3]                                # (B, T, 3, 3)
        t_vec = w2c[:, :, :3, 3]                             # (B, T, 3)

        # Transform corners: (B, M_w, 4, 3) → (B, T, M_w, 4, 3)
        corners_exp = corners.unsqueeze(1).expand(-1, T, -1, -1, -1)
        p_cam = torch.einsum("btij,btmcj->btmci", R, corners_exp) \
                + t_vec[:, :, None, None, :]                 # (B, T, M_w, 4, 3)

        z_corners = p_cam[..., 2]                            # (B, T, M_w, 4)
        visible = z_corners > 0                              # (B, T, M_w, 4)
        z_safe = z_corners.clamp(min=1e-4)

        # --- perspective project ---
        fx = intrinsics[:, 0, 0]
        fy = intrinsics[:, 1, 1]
        cx = intrinsics[:, 0, 2]
        cy = intrinsics[:, 1, 2]

        u_corners = fx[:, None, None, None] * (p_cam[..., 0] / z_safe) \
                    + cx[:, None, None, None]                # (B, T, M_w, 4)
        v_corners = fy[:, None, None, None] * (p_cam[..., 1] / z_safe) \
                    + cy[:, None, None, None]

        # --- derive 2D bounding rect from visible corners ---
        INF = torch.tensor(float("inf"), device=device, dtype=dtype)
        u_for_min = torch.where(visible, u_corners, INF)
        u_for_max = torch.where(visible, u_corners, -INF)
        v_for_min = torch.where(visible, v_corners, INF)
        v_for_max = torch.where(visible, v_corners, -INF)

        u_min = u_for_min.min(dim=-1).values                # (B, T, M_w)
        u_max = u_for_max.max(dim=-1).values
        v_min = v_for_min.min(dim=-1).values
        v_max = v_for_max.max(dim=-1).values

        any_visible = visible.any(dim=-1)                    # (B, T, M_w)
        fully_behind = ~any_visible
        u_min = torch.where(fully_behind, torch.zeros_like(u_min), u_min)
        u_max = torch.where(fully_behind, torch.ones_like(u_max), u_max)
        v_min = torch.where(fully_behind, torch.zeros_like(v_min), v_min)
        v_max = torch.where(fully_behind, torch.ones_like(v_max), v_max)

        cu = (u_min + u_max) / 2                             # (B, T, M_w)
        cv = (v_min + v_max) / 2
        EPS = 0.01
        half_w = ((u_max - u_min) / 2).clamp(min=EPS)
        half_h = ((v_max - v_min) / 2).clamp(min=EPS)

        # --- patch centre grid in [0, 1] ---
        grid_y = (torch.arange(patch_h, device=device, dtype=dtype) + 0.5) / patch_h
        grid_x = (torch.arange(patch_w, device=device, dtype=dtype) + 0.5) / patch_w
        gy, gx = torch.meshgrid(grid_y, grid_x, indexing="ij")
        gx = gx.reshape(-1)                                 # (P,)
        gy = gy.reshape(-1)

        # --- box-distance: 0 inside rect, positive outside ---
        # du_raw = |gx - cu| / half_w → 0 at center, 1 at edge
        du_raw = (gx[None, None, :, None] - cu[:, :, None, :]).abs() / half_w[:, :, None, :]
        dv_raw = (gy[None, None, :, None] - cv[:, :, None, :]).abs() / half_h[:, :, None, :]
        # Clamp to zero inside rect (du_raw <= 1 means inside)
        du_clamped = (du_raw - 1.0).clamp(min=0.0)          # (B, T, P, M_w)
        dv_clamped = (dv_raw - 1.0).clamp(min=0.0)
        sq_dist_box = du_clamped ** 2 + dv_clamped ** 2      # 0 inside, >0 outside

        # --- per-head Gaussian bias ---
        sigma = self.spatial_log_sigma.exp()                 # (H,)
        inv_two_sigma_sq = 1.0 / (2.0 * sigma ** 2 + 1e-8)  # (H,)
        bias = -sq_dist_box.unsqueeze(2) * inv_two_sigma_sq[None, None, :, None, None]

        # --- depth-dependent bias: use midpoint depth ---
        mid = (wall_endpoints[:, :, 0, :] + wall_endpoints[:, :, 1, :]) / 2  # (B, M_w, 3)
        mid[..., 1] = wall_heights / 2  # midpoint at half height
        mid_exp = mid.unsqueeze(1).expand(-1, T, -1, -1)    # (B, T, M_w, 3)
        mid_cam = torch.einsum("btij,btmj->btmi", R, mid_exp) \
                  + t_vec.unsqueeze(2)                       # (B, T, M_w, 3)
        z_center = mid_cam[..., 2].clamp(min=1e-4)          # (B, T, M_w)
        depth_bonus = -torch.log(z_center)
        depth_term = self.depth_log_alpha[None, None, :, None, None] \
                     * depth_bonus[:, :, None, None, :]      # (B, T, H, 1, M_w)
        bias = bias + depth_term

        bias = bias.clamp(min=-50.0)
        bias.masked_fill_(~any_visible[:, :, None, None, :], 0.0)

        bias = rearrange(bias, "b t h p m -> (b t) h p m")
        return bias

    # ------------------------------------------------------------------
    def forward(
        self,
        x: Tensor,                                 # (B*T, P, C)
        sg_tokens: Tensor,                         # (B, M, D_sg)
        sg_node_mask: Tensor,                      # (B, M) bool
        temporal_length: int = 2,
        sg_node_positions: Optional[Tensor] = None,  # (B, M, 3)
        sg_node_sizes: Optional[Tensor] = None,      # (B, M, 3) — needed for bbox mode
        camera_c2w: Optional[Tensor] = None,         # (B, T, 4, 4)
        intrinsics: Optional[Tensor] = None,         # (B, 3, 3)
        patch_grid_size: Optional[tuple] = None,     # (h, w)
        sg_wall_endpoints: Optional[Tensor] = None,  # (B, M, 2, 3) — for bbox_surface
        sg_wall_heights: Optional[Tensor] = None,    # (B, M) — for bbox_surface
        sg_node_is_wall: Optional[Tensor] = None,    # (B, M) bool — for bbox_surface
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
        has_spatial = (
            sg_node_positions is not None
            and camera_c2w is not None
            and intrinsics is not None
            and patch_grid_size is not None
        )
        if has_spatial:
            ph, pw = patch_grid_size

            if self.spatial_mode == "bbox_surface" and sg_node_is_wall is not None \
                    and sg_wall_endpoints is not None and sg_wall_heights is not None \
                    and sg_node_sizes is not None:
                # Dispatch: bbox bias for objects, surface bias for walls
                is_wall = sg_node_is_wall                    # (B, M) bool
                is_obj = ~is_wall & sg_node_mask             # (B, M) bool

                # Compute full bbox bias for all nodes (cheap, then overwrite wall positions)
                spatial_bias = self._compute_spatial_bias_bbox(
                    sg_node_positions, sg_node_sizes,
                    camera_c2w, intrinsics, ph, pw,
                )                                            # (BT, H, P, M)

                # Check if there are any walls
                if is_wall.any():
                    # Compute surface bias for wall nodes
                    # We need wall-specific endpoints/heights for wall nodes only.
                    # But wall_endpoints and wall_heights are stored at the same M positions.
                    surface_bias = self._compute_spatial_bias_surface(
                        sg_wall_endpoints, sg_wall_heights,
                        camera_c2w, intrinsics, ph, pw,
                    )                                        # (BT, H, P, M)

                    # Replace wall positions in the full bias tensor
                    wall_mask_exp = repeat(is_wall, "b m -> (b t) 1 1 m", t=T)
                    spatial_bias = torch.where(wall_mask_exp, surface_bias, spatial_bias)

            elif self.spatial_mode in ("bbox", "bbox_surface") and sg_node_sizes is not None:
                spatial_bias = self._compute_spatial_bias_bbox(
                    sg_node_positions, sg_node_sizes,
                    camera_c2w, intrinsics, ph, pw,
                )
            else:
                spatial_bias = self._compute_spatial_bias(
                    sg_node_positions, camera_c2w, intrinsics, ph, pw,
                )
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
