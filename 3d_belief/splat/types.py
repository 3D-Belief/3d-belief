from dataclasses import dataclass

from jaxtyping import Float

import torch
from torch import Tensor
from typing import Optional


@dataclass
class Gaussians:
    means: Float[Tensor, "batch gaussian dim"]
    covariances: Float[Tensor, "batch gaussian dim dim"]
    harmonics: Float[Tensor, "batch gaussian 3 d_sh"]
    opacities: Float[Tensor, "batch gaussian"]
    features: Float[Tensor, "batch gaussian channels"] | None = None

    def __add__(self, other: "Gaussians") -> "Gaussians":
        if not isinstance(other, Gaussians):
            return NotImplemented
        # Concatenate along the gaussian dimension (assumed to be dim=1)
        assert (self.features is not None and other.features is not None) or \
            (self.features is None and other.features is None)
        features = torch.cat([self.features, other.features], dim=1) \
            if self.features is not None and other.features is not None else None
        return Gaussians(
            means=torch.cat([self.means, other.means], dim=1),
            covariances=torch.cat([self.covariances, other.covariances], dim=1),
            harmonics=torch.cat([self.harmonics, other.harmonics], dim=1),
            opacities=torch.cat([self.opacities, other.opacities], dim=1),
            features=features,
        )
    
    def float(self) -> "Gaussians":
        features = self.features.float() if self.features is not None else None
        return Gaussians(
            means=self.means.float(),
            covariances=self.covariances.float(),
            harmonics=self.harmonics.float(),
            opacities=self.opacities.float(),
            features=features,
        )

    def double(self) -> "Gaussians":
        features = self.features.double() if self.features is not None else None
        return Gaussians(
            means=self.means.double(),
            covariances=self.covariances.double(),
            harmonics=self.harmonics.double(),
            opacities=self.opacities.double(),
            features=features,
        )

    def clone(self) -> "Gaussians":
        features = self.features.clone() if self.features is not None else None
        return Gaussians(
            means=self.means.clone(),
            covariances=self.covariances.clone(),
            harmonics=self.harmonics.clone(),
            opacities=self.opacities.clone(),
            features=features,
        )
    
    def detach(self) -> "Gaussians":
        features = self.features.detach() if self.features is not None else None
        return Gaussians(
            means=self.means.detach(),
            covariances=self.covariances.detach(),
            harmonics=self.harmonics.detach(),
            opacities=self.opacities.detach(),
            features=features,
        )
    
    def transform(self, c2w: Float[Tensor, "batch 4 4"]) -> "Gaussians":
        B, N, D = self.means.shape
        assert D == 3, "Means must be 3D points"
        assert c2w.shape[0] == B, "Batch size of c2w must match batch size of Gaussians"
        assert c2w.shape[1] == 4 and c2w.shape[2] == 4, "c2w must be of shape [B, 4, 4]"

        device = self.means.device
        dtype  = self.means.dtype

        # Convert means to homogeneous coordinates
        means_h = torch.cat([self.means, torch.ones((B, N, 1), device=device, dtype=dtype)], dim=-1)  # [B, N, 4]
        # Transform means
        means_transformed_h = torch.einsum("bij,bnj->bni", c2w.float(), means_h.float())  # [B, N, 4]
        means_transformed = means_transformed_h[..., :3] / means_transformed_h[..., 3:]  # [B, N, 3]
        # Transform covariances
        R = c2w[:, :3, :3]  # [B, 3, 3]
        covariances_transformed = torch.einsum("bij,bnjk,bkl->bnil", R.float(), self.covariances.float(), R.transpose(1, 2).float())  # [B, N, 3, 3]
        return Gaussians(
            means=means_transformed,
            covariances=covariances_transformed,
            harmonics=self.harmonics,
            opacities=self.opacities,
            features=self.features,
        )

    def __del__(self):
        del self.means
        del self.covariances
        del self.harmonics
        del self.opacities
        if self.features is not None:
            del self.features

    def filter_ceiling(self, ceiling_threshold: float = 2.5) -> "Gaussians":
        B, N, _ = self.means.shape
        device = self.means.device
        dtype  = self.means.dtype

        new_means       = []
        new_covars      = []
        new_harms       = []
        new_opacs       = []
        new_features    = [] if self.features is not None else None

        for b in range(B):
            keep = (self.means[b, :, 1] < ceiling_threshold)  # [N] bool
            # slice out the survivors
            new_means .append(self.means[b     , keep, :])
            new_covars.append(self.covariances[b, keep, :, :])
            new_harms .append(self.harmonics[b , keep, :])
            new_opacs .append(self.opacities[b , keep])
            if self.features is not None:
                new_features.append(self.features[b, keep, :])

        # stack back; this will preserve dtype/device of the inputs
        means_f  = torch.stack(new_means,    dim=0).to(device=device, dtype=dtype)
        covs_f   = torch.stack(new_covars,   dim=0).to(device=device, dtype=dtype)
        harms_f  = torch.stack(new_harms,    dim=0).to(device=device, dtype=dtype)
        opacs_f  = torch.stack(new_opacs,    dim=0).to(device=device, dtype=dtype)
        feats_f  = (torch.stack(new_features, dim=0)
                    .to(device=device, dtype=dtype)) if self.features is not None else None

        return Gaussians(
            means       = means_f,
            covariances = covs_f,
            harmonics   = harms_f,
            opacities   = opacs_f,
            features    = feats_f,
        )