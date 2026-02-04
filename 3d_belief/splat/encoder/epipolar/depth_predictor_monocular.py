import math
import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn

from .conversions import relative_disparity_to_depth
from .distribution_sampler import DistributionSampler
from ..common.time_embedder import TimestepEmbedder


class DepthPredictorMonocular(nn.Module):
    projection: nn.Sequential
    time_embedding: TimestepEmbedder | None
    sampler: DistributionSampler
    num_samples: int
    num_surfaces: int
    use_time_embedder: bool
    depth_inference_min: float
    depth_inference_max: float

    def __init__(
        self,
        d_in: int,
        num_samples: int,
        num_surfaces: int,
        use_transmittance: bool,
        use_time_embedder: bool = True,
        depth_inference_min: float = 0.2,
        depth_inference_max: float = 10.0,
    ) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.ReLU(),
            nn.Linear(d_in, 2 * num_samples * num_surfaces),
        )
        self.use_time_embedder = use_time_embedder
        if self.use_time_embedder:
            self.time_embedding = TimestepEmbedder(hidden_size=2 * num_samples * num_surfaces)
        else:
            self.time_embedding = None

        self.depth_inference_min = depth_inference_min
        self.depth_inference_max = depth_inference_max

        self.sampler = DistributionSampler()
        self.num_samples = num_samples
        self.num_surfaces = num_surfaces
        self.use_transmittance = use_transmittance

        # This exists for hooks to latch onto.
        self.to_pdf = nn.Softmax(dim=-1)
        self.to_offset = nn.Sigmoid()

    def forward(
        self,
        features: Float[Tensor, "batch view ray channel"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        deterministic: bool,
        gaussians_per_pixel: int,
        t: Float[Tensor, "batch"] = None,
        inference_mode: bool = False,
    ) -> tuple[
        Float[Tensor, "batch view ray surface sample"],  # depth
        Float[Tensor, "batch view ray surface sample"],  # pdf
    ]:
        s = self.num_samples

        # Process spatial features.
        features_proj = self.projection(features)  # shape: (B, view, ray, 2*num_samples*num_surfaces)
        b, v, r, c = features_proj.shape
        # Conditionally add time embedding.
        if self.use_time_embedder:
            # print("use time embedding in depth predictor")
            assert t is not None
            time_emb = self.time_embedding(t)  # shape: (B, 2*num_samples*num_surfaces)
            half = v // 2
            zeros_emb = torch.zeros(b, half, c, device=features_proj.device, dtype=features_proj.dtype)
            # import ipdb; ipdb.set_trace()
            time_emb_expanded = time_emb.unsqueeze(1).expand(b, v - half, c)
            time_emb_full = torch.cat([zeros_emb, time_emb_expanded], dim=1)  # shape: (B, v, c)
            time_emb_full = time_emb_full[:, :, None, :]  # shape: (B, v, 1, c)
            features_proj = features_proj + time_emb_full

        # Convert the features into a depth distribution plus intra-bucket offsets.
        pdf_raw, offset_raw = rearrange(
            features_proj, "... (dpt srf c) -> c ... srf dpt", c=2, srf=self.num_surfaces
        )
        pdf = self.to_pdf(pdf_raw)
        offset = self.to_offset(offset_raw)

        # Sample from the depth distribution.
        index, pdf_i = self.sampler.sample(pdf, deterministic, gaussians_per_pixel)
        offset = self.sampler.gather(index, offset)

        # Convert the sampled bucket and offset to a depth.
        relative_disparity = (index + offset) / s
        depth = relative_disparity_to_depth(
            relative_disparity,
            rearrange(near, "b v -> b v () () ()"),
            rearrange(far, "b v -> b v () () ()"),
        )
        # The mask are all ones if not in inference mode.
        mask_exp = torch.ones_like(depth)
        if inference_mode:
            min_depth = self.depth_inference_min
            max_depth = self.depth_inference_max

            B, V, N, S, C = depth.shape

            dvals = depth[..., 0]  # [B, V, N, S]
            mask  = (dvals >= min_depth) & (dvals <= max_depth)

            mask_exp = mask.unsqueeze(-1).expand(B, V, N, S, C)  # [B, V, N, S, 3]

        # Compute opacity from PDF.
        if self.use_transmittance:
            partial = pdf.cumsum(dim=-1)
            partial = torch.cat(
                (torch.zeros_like(partial[..., :1]), partial[..., :-1]), dim=-1
            )
            opacity = pdf / (1 - partial + 1e-10)
            opacity = self.sampler.gather(index, opacity)
        else:
            opacity = pdf_i

        return depth, opacity, mask_exp
