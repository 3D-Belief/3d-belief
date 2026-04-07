"""
Online Gaussian refinement module inspired by Splat-SLAM.

After the diffusion model predicts Gaussians from observations, this module
refines Gaussian parameters (means, scales, rotations, harmonics, opacities)
via gradient-based photometric loss against observed images.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float
from torch import Tensor

from .encoder.common.gaussians import build_covariance, quaternion_to_matrix
from .types import Gaussians


@dataclass
class GaussianRefinerCfg:
    enabled: bool = False
    num_iterations: int = 30
    lr_means: float = 1.6e-4
    lr_harmonics: float = 2.5e-3
    lr_opacity: float = 0.05
    lr_scaling: float = 1e-3
    lr_rotation: float = 1e-3
    rgb_loss_weight: float = 1.0
    use_ssim: bool = False
    ssim_weight: float = 0.2
    window_size: int = 10
    # Regularization
    prior_weight: float = 0.1
    freeze_geometry: bool = False
    harmonics_dc_only: bool = True  # only optimize 0th-order SH (view-independent color)
    min_observations: int = 2
    # LR decay for means (Splat-SLAM-style exponential decay)
    use_lr_decay: bool = True
    lr_means_final: float = 1.6e-6
    # Visibility masking — only optimize Gaussians visible from ≥1 observation
    use_visibility_masking: bool = True
    visibility_margin: float = 0.05  # frustum margin in normalized coords
    # Densification (only triggers if num_iterations >= densify_every)
    densify_every: int = 150
    densify_grad_threshold: float = 0.0002
    opacity_prune_threshold: float = 0.7
    size_prune_threshold: float = 0.1


def matrix_to_quaternion_xyzw(matrix: Float[Tensor, "*batch 3 3"]) -> Float[Tensor, "*batch 4"]:
    """Convert rotation matrix to quaternion in XYZW format (matching GaussianAdapter)."""
    # Adapted from pytorch3d
    batch_shape = matrix.shape[:-2]
    m = matrix.reshape(-1, 3, 3)
    
    trace = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    
    q = torch.zeros(m.shape[0], 4, device=m.device, dtype=m.dtype)
    
    # Case 1: trace > 0
    s = torch.sqrt(torch.clamp(trace + 1.0, min=1e-10)) * 2  # s = 4*w
    mask1 = trace > 0
    q[mask1, 3] = 0.25 * s[mask1]  # w
    q[mask1, 0] = (m[mask1, 2, 1] - m[mask1, 1, 2]) / s[mask1]  # x
    q[mask1, 1] = (m[mask1, 0, 2] - m[mask1, 2, 0]) / s[mask1]  # y
    q[mask1, 2] = (m[mask1, 1, 0] - m[mask1, 0, 1]) / s[mask1]  # z
    
    # Case 2: m[0,0] is max diagonal
    mask2 = (~mask1) & (m[:, 0, 0] > m[:, 1, 1]) & (m[:, 0, 0] > m[:, 2, 2])
    s2 = torch.sqrt(torch.clamp(1.0 + m[:, 0, 0] - m[:, 1, 1] - m[:, 2, 2], min=1e-10)) * 2
    q[mask2, 0] = 0.25 * s2[mask2]
    q[mask2, 1] = (m[mask2, 0, 1] + m[mask2, 1, 0]) / s2[mask2]
    q[mask2, 2] = (m[mask2, 0, 2] + m[mask2, 2, 0]) / s2[mask2]
    q[mask2, 3] = (m[mask2, 2, 1] - m[mask2, 1, 2]) / s2[mask2]
    
    # Case 3: m[1,1] is max diagonal
    mask3 = (~mask1) & (~mask2) & (m[:, 1, 1] > m[:, 2, 2])
    s3 = torch.sqrt(torch.clamp(1.0 + m[:, 1, 1] - m[:, 0, 0] - m[:, 2, 2], min=1e-10)) * 2
    q[mask3, 0] = (m[mask3, 0, 1] + m[mask3, 1, 0]) / s3[mask3]
    q[mask3, 1] = 0.25 * s3[mask3]
    q[mask3, 2] = (m[mask3, 1, 2] + m[mask3, 2, 1]) / s3[mask3]
    q[mask3, 3] = (m[mask3, 0, 2] - m[mask3, 2, 0]) / s3[mask3]
    
    # Case 4: m[2,2] is max diagonal
    mask4 = (~mask1) & (~mask2) & (~mask3)
    s4 = torch.sqrt(torch.clamp(1.0 + m[:, 2, 2] - m[:, 0, 0] - m[:, 1, 1], min=1e-10)) * 2
    q[mask4, 0] = (m[mask4, 0, 2] + m[mask4, 2, 0]) / s4[mask4]
    q[mask4, 1] = (m[mask4, 1, 2] + m[mask4, 2, 1]) / s4[mask4]
    q[mask4, 2] = 0.25 * s4[mask4]
    q[mask4, 3] = (m[mask4, 1, 0] - m[mask4, 0, 1]) / s4[mask4]
    
    # Normalize
    q = q / (q.norm(dim=-1, keepdim=True) + 1e-10)
    
    return q.reshape(*batch_shape, 4)


def decompose_covariance(covariances: Float[Tensor, "batch gaussian 3 3"]):
    """Eigendecompose covariance matrices into scales and quaternion rotations.
    
    Returns:
        scales: [batch, gaussian, 3] — raw scale values
        rotations: [batch, gaussian, 4] — XYZW quaternions
    """
    B, G = covariances.shape[:2]
    if G == 0:
        return (torch.zeros(B, 0, 3, device=covariances.device),
                torch.zeros(B, 0, 4, device=covariances.device))

    covs = covariances.reshape(B * G, 3, 3).float()

    # Replace NaN/Inf with zeros, then add small identity for PSD guarantee
    covs = torch.nan_to_num(covs, nan=0.0, posinf=1e6, neginf=-1e6)
    # Symmetrize to handle numerical asymmetry
    covs = (covs + covs.transpose(-1, -2)) / 2
    # Add diagonal regularization so every matrix is strictly positive-definite
    covs = covs + torch.eye(3, device=covs.device, dtype=covs.dtype).unsqueeze(0) * 1e-7

    # Eigendecomposition — process in chunks to avoid cusolver batch-size limits
    CHUNK = 65536
    all_eigenvalues = []
    all_eigenvectors = []
    for start in range(0, covs.shape[0], CHUNK):
        chunk = covs[start:start + CHUNK]
        try:
            evals, evecs = torch.linalg.eigh(chunk)
        except torch._C._LinAlgError:
            # Fallback: return identity rotations and unit scales for this chunk
            n = chunk.shape[0]
            evals = torch.ones(n, 3, device=covs.device, dtype=covs.dtype) * 1e-6
            evecs = torch.eye(3, device=covs.device, dtype=covs.dtype).unsqueeze(0).expand(n, -1, -1)
        all_eigenvalues.append(evals)
        all_eigenvectors.append(evecs)

    eigenvalues = torch.cat(all_eigenvalues, dim=0)
    eigenvectors = torch.cat(all_eigenvectors, dim=0)
    
    # Clamp eigenvalues to be positive
    eigenvalues = eigenvalues.clamp(min=1e-12)
    
    # scales = sqrt(eigenvalues) (these are the diagonal scale values)
    scales = eigenvalues.sqrt()
    
    # Ensure proper rotation (det > 0)
    det = torch.linalg.det(eigenvectors)
    eigenvectors[det < 0, :, 0] *= -1
    
    rotations = matrix_to_quaternion_xyzw(eigenvectors)
    
    scales = scales.reshape(B, G, 3)
    rotations = rotations.reshape(B, G, 4)
    
    return scales, rotations


def reconstruct_gaussians(
    means: Tensor,
    scales: Tensor,
    rotations: Tensor,
    harmonics: Tensor,
    opacities: Tensor,
    features: Optional[Tensor] = None,
) -> Gaussians:
    """Reconstruct Gaussians dataclass from optimizable parameters.
    
    Args:
        means: [B, G, 3]
        scales: [B, G, 3] — raw (unconstrained) scale values
        rotations: [B, G, 4] — XYZW quaternions (will be normalized)
        harmonics: [B, G, 3, d_sh]
        opacities: [B, G] — raw (pre-sigmoid) opacities
        features: [B, G, C] or None
    """
    # Normalize quaternions
    rot_normalized = rotations / (rotations.norm(dim=-1, keepdim=True) + 1e-10)
    
    # Ensure positive scales
    scales_pos = scales.abs().clamp(min=1e-8)
    
    # Build covariance: Σ = R @ diag(s²) @ R^T
    covariances = build_covariance(scales_pos, rot_normalized)
    
    # Sigmoid for opacities
    opacities_activated = torch.sigmoid(opacities)
    
    return Gaussians(
        means=means,
        covariances=covariances,
        harmonics=harmonics,
        opacities=opacities_activated,
        features=features,
    )


@dataclass
class ObservedFrame:
    """An observed frame stored for refinement."""
    rgb: Float[Tensor, "1 3 height width"]  # [0, 1] range
    c2w: Float[Tensor, "1 4 4"]
    intrinsics: Float[Tensor, "1 3 3"]
    near: Float[Tensor, "1"]
    far: Float[Tensor, "1"]


class GaussianRefiner:
    """Online Gaussian refinement via photometric optimization.
    
    After the diffusion model predicts Gaussians, this module refines them
    by rendering from observed views and minimizing L1 RGB loss.
    """
    
    def __init__(self, cfg: GaussianRefinerCfg, decoder):
        self.cfg = cfg
        self.decoder = decoder
        self.observed_frames: List[ObservedFrame] = []
    
    def add_observation(
        self,
        rgb: Float[Tensor, "1 3 height width"],
        c2w: Float[Tensor, "1 4 4"],
        intrinsics: Float[Tensor, "1 3 3"],
        near: Float[Tensor, "1"],
        far: Float[Tensor, "1"],
    ):
        """Store an observed frame for refinement."""
        frame = ObservedFrame(
            rgb=rgb.detach(),
            c2w=c2w.detach(),
            intrinsics=intrinsics.detach(),
            near=near.detach(),
            far=far.detach(),
        )
        self.observed_frames.append(frame)
        # Keep only the most recent window_size observations
        if len(self.observed_frames) > self.cfg.window_size:
            self.observed_frames = self.observed_frames[-self.cfg.window_size:]
    
    def reset(self):
        """Clear all stored observations."""
        self.observed_frames = []
    
    def _compute_visibility_mask(
        self,
        means: Float[Tensor, "B G 3"],
    ) -> Float[Tensor, "B G"]:
        """Compute which Gaussians are visible from at least one observation.
        
        Uses frustum projection: a Gaussian is "visible" if its mean projects
        into the image plane (with margin) and is in front of the camera for
        at least one stored observation.
        
        Args:
            means: Gaussian centres in world space [B, G, 3].
            
        Returns:
            Boolean mask [B, G] — True for visible Gaussians.
        """
        B, G, _ = means.shape
        device = means.device
        margin = self.cfg.visibility_margin
        visible = torch.zeros(B, G, dtype=torch.bool, device=device)
        
        for obs in self.observed_frames:
            # w2c = inv(c2w), obs.c2w is [B, 4, 4]
            w2c = torch.linalg.inv(obs.c2w)  # [B, 4, 4]
            
            # Transform means to camera space: [B, G, 3]
            ones = torch.ones(B, G, 1, device=device, dtype=means.dtype)
            pts_h = torch.cat([means, ones], dim=-1)  # [B, G, 4]
            pts_cam = torch.einsum("bij,bgj->bgi", w2c, pts_h)  # [B, G, 4]
            z = pts_cam[..., 2]  # [B, G]
            
            # Must be in front of camera
            in_front = z > obs.near.view(B, 1) * 0.5  # generous near-plane
            
            # Project to normalised image coords using intrinsics [B, 3, 3]
            # K @ [x, y, z]^T => [u*z, v*z, z] in normalised coords
            K = obs.intrinsics  # [B, 3, 3]
            pts_cam_3 = pts_cam[..., :3]  # [B, G, 3]
            proj = torch.einsum("bij,bgj->bgi", K, pts_cam_3)  # [B, G, 3]
            u = proj[..., 0] / (z + 1e-8)  # [B, G]
            v = proj[..., 1] / (z + 1e-8)  # [B, G]
            
            # Check bounds (normalised [0, 1] with margin)
            in_bounds = (
                (u > -margin) & (u < 1.0 + margin) &
                (v > -margin) & (v < 1.0 + margin)
            )
            
            visible = visible | (in_front & in_bounds)
        
        return visible
    
    def refine(
        self,
        gaussians: Gaussians,
        image_shape: tuple[int, int],
    ) -> Gaussians:
        """Refine Gaussians by optimizing against observed frames.
        
        Args:
            gaussians: The predicted Gaussians to refine.
            image_shape: (H, W) for rendering.
            
        Returns:
            Refined Gaussians dataclass.
        """
        if not self.cfg.enabled:
            return None
        
        if len(self.observed_frames) < self.cfg.min_observations:
            return None
        
        B, G, _ = gaussians.means.shape
        device = gaussians.means.device
        
        # Compute visibility mask — which Gaussians to actually update
        if self.cfg.use_visibility_masking:
            vis_mask = self._compute_visibility_mask(gaussians.means.detach())  # [B, G]
            num_visible = int(vis_mask.sum().item())
            if num_visible == 0:
                return None
        else:
            vis_mask = None
        
        # Decompose covariance matrices → optimizable scales + rotations
        scales, rotations = decompose_covariance(gaussians.covariances.detach())
        
        # Free unused memory before starting optimization loop
        torch.cuda.empty_cache()
        
        # Create optimizable parameters (replace NaN/Inf with zeros to avoid optimizer issues)
        means_det = gaussians.means.detach().clone()
        means_det = torch.where(torch.isfinite(means_det), means_det, torch.zeros_like(means_det))
        means_param = means_det.requires_grad_(True)
        scales_param = scales.clone().requires_grad_(True)
        rotations_param = rotations.clone().requires_grad_(True)
        harmonics_det = gaussians.harmonics.detach().clone()
        harmonics_det = torch.where(torch.isfinite(harmonics_det), harmonics_det, torch.zeros_like(harmonics_det))
        if self.cfg.harmonics_dc_only:
            # Split into DC (index 0, optimizable) and higher-order (frozen)
            harmonics_dc_param = harmonics_det[..., :1].clone().requires_grad_(True)  # [B, G, 3, 1]
            harmonics_rest = harmonics_det[..., 1:].clone()  # [B, G, 3, d_sh-1] — frozen
            harmonics_param = None  # not used in dc_only mode
        else:
            harmonics_dc_param = None
            harmonics_rest = None
            harmonics_param = harmonics_det.requires_grad_(True)
        # Store opacities in pre-sigmoid (logit) space for unconstrained optimization
        opacities_clamped = gaussians.opacities.detach().clamp(1e-5, 1 - 1e-5)
        opacities_clamped = torch.where(torch.isfinite(opacities_clamped), opacities_clamped, torch.full_like(opacities_clamped, 0.5))
        opacities_param = torch.logit(opacities_clamped).requires_grad_(True)
        
        features_detached = gaussians.features.detach() if gaussians.features is not None else None
        
        # Store initial (prior) values for proximity regularization
        means_init = means_det.detach().clone()
        scales_init = scales.detach().clone()
        rotations_init = rotations.detach().clone()
        if self.cfg.harmonics_dc_only:
            harmonics_dc_init = harmonics_det[..., :1].detach().clone()
        else:
            harmonics_init = harmonics_det.detach().clone()
        opacities_init = opacities_clamped.detach().clone()
        opacities_logit_init = torch.logit(opacities_init)
        
        # Re-enable gradient tracking — refine is called inside @torch.no_grad()
        prev_grad_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(True)
        
        # Build optimizer param groups — optionally freeze geometry
        param_groups = []
        if not self.cfg.freeze_geometry:
            param_groups.append({"params": [means_param], "lr": self.cfg.lr_means})
            param_groups.append({"params": [scales_param], "lr": self.cfg.lr_scaling})
            param_groups.append({"params": [rotations_param], "lr": self.cfg.lr_rotation})
        if self.cfg.harmonics_dc_only:
            param_groups.append({"params": [harmonics_dc_param], "lr": self.cfg.lr_harmonics})
        else:
            param_groups.append({"params": [harmonics_param], "lr": self.cfg.lr_harmonics})
        param_groups.append({"params": [opacities_param], "lr": self.cfg.lr_opacity})
        
        optimizer = torch.optim.Adam(param_groups)
        
        # Densification accumulators
        grad_accum = torch.zeros(B, G, 1, device=device)
        grad_count = torch.zeros(B, G, 1, device=device)
        
        num_obs = len(self.observed_frames)
        num_iters = self.cfg.num_iterations
        
        for iteration in range(num_iters):
            optimizer.zero_grad()
            
            # LR decay for means (exponential, Splat-SLAM-style)
            if self.cfg.use_lr_decay and not self.cfg.freeze_geometry:
                t = iteration / max(num_iters - 1, 1)
                lr_means_cur = self.cfg.lr_means * (
                    self.cfg.lr_means_final / self.cfg.lr_means
                ) ** t
                # means is always the first param group when geometry is not frozen
                optimizer.param_groups[0]["lr"] = lr_means_cur
            
            # Render from each observed view, backward per-view for memory efficiency
            for obs in self.observed_frames:
                # Assemble full harmonics (DC may be optimizable, rest frozen)
                if self.cfg.harmonics_dc_only:
                    harmonics_full = torch.cat([harmonics_dc_param, harmonics_rest], dim=-1)
                else:
                    harmonics_full = harmonics_param
                
                # Reconstruct gaussians from current params
                refined = reconstruct_gaussians(
                    means_param, scales_param, rotations_param,
                    harmonics_full, opacities_param, features_detached,
                )
                
                # Render
                output = self.decoder.forward(
                    refined.float(),
                    obs.c2w.unsqueeze(0),  # [1, 1, 4, 4]
                    obs.intrinsics.unsqueeze(0),  # [1, 1, 3, 3]
                    obs.near.unsqueeze(0),  # [1, 1]
                    obs.far.unsqueeze(0),  # [1, 1]
                    image_shape,
                    depth_mode=None,
                )
                
                rendered_rgb = output.color[:, 0]  # [B, 3, H, W]
                gt_rgb = obs.rgb  # [B, 3, H, W]
                
                # L1 photometric loss
                loss = F.l1_loss(rendered_rgb, gt_rgb)
                
                if self.cfg.use_ssim:
                    ssim_loss = 1.0 - _ssim(rendered_rgb, gt_rgb)
                    loss = (1.0 - self.cfg.ssim_weight) * loss + self.cfg.ssim_weight * ssim_loss
                
                # Backward per-view to free graph immediately (gradient accumulation)
                (loss / num_obs).backward()
            
            # Prior-proximity regularization: L2 penalty to initial (diffusion-predicted) values
            if self.cfg.prior_weight > 0:
                if self.cfg.harmonics_dc_only:
                    loss_prior = (
                        F.mse_loss(harmonics_dc_param, harmonics_dc_init)
                        + F.mse_loss(opacities_param, opacities_logit_init)
                    )
                else:
                    loss_prior = (
                        F.mse_loss(harmonics_param, harmonics_init)
                        + F.mse_loss(opacities_param, opacities_logit_init)
                    )
                if not self.cfg.freeze_geometry:
                    loss_prior = loss_prior + (
                        F.mse_loss(means_param, means_init)
                        + F.mse_loss(scales_param, scales_init)
                        + F.mse_loss(rotations_param, rotations_init)
                    )
                (self.cfg.prior_weight * loss_prior).backward()
            
            # Zero out gradients for non-visible Gaussians so they stay frozen
            if vis_mask is not None:
                inv_mask = ~vis_mask  # [B, G]
                for p in [means_param, scales_param, rotations_param]:
                    if p.grad is not None:
                        p.grad[inv_mask] = 0.0
                if self.cfg.harmonics_dc_only:
                    if harmonics_dc_param.grad is not None:
                        harmonics_dc_param.grad[inv_mask] = 0.0
                else:
                    if harmonics_param.grad is not None:
                        harmonics_param.grad[inv_mask] = 0.0
                if opacities_param.grad is not None:
                    opacities_param.grad[inv_mask] = 0.0
            
            # Accumulate gradients for densification
            if means_param.grad is not None:
                visible_mask = means_param.grad.abs().sum(dim=-1) > 0  # [B, G]
                grad_norm = means_param.grad.norm(dim=-1, keepdim=True)  # [B, G, 1]
                grad_accum[visible_mask] += grad_norm[visible_mask]
                grad_count[visible_mask] += 1
            
            optimizer.step()
            
            # Densification
            if (self.cfg.num_iterations >= self.cfg.densify_every and
                iteration > 0 and 
                iteration % self.cfg.densify_every == 0):
                if self.cfg.harmonics_dc_only:
                    harmonics_full_d = torch.cat([harmonics_dc_param, harmonics_rest], dim=-1)
                else:
                    harmonics_full_d = harmonics_param
                means_param, scales_param, rotations_param, harmonics_full_d, \
                    opacities_param, features_detached, optimizer, grad_accum, grad_count = \
                    self._densify_and_prune(
                        means_param, scales_param, rotations_param,
                        harmonics_full_d, opacities_param, features_detached,
                        optimizer, grad_accum, grad_count,
                    )
                if self.cfg.harmonics_dc_only:
                    harmonics_dc_param = harmonics_full_d[..., :1].requires_grad_(True)
                    harmonics_rest = harmonics_full_d[..., 1:].clone()
                else:
                    harmonics_param = harmonics_full_d
        
        # Final reconstruction 
        with torch.no_grad():
            if self.cfg.harmonics_dc_only:
                harmonics_final = torch.cat([harmonics_dc_param, harmonics_rest], dim=-1)
            else:
                harmonics_final = harmonics_param
            result = reconstruct_gaussians(
                means_param, scales_param, rotations_param,
                harmonics_final, opacities_param, features_detached,
            )
            # Deep-copy to free optimizer state and computation graph
            result = Gaussians(
                means=result.means.detach().clone(),
                covariances=result.covariances.detach().clone(),
                harmonics=result.harmonics.detach().clone(),
                opacities=result.opacities.detach().clone(),
                features=result.features.detach().clone() if result.features is not None else None,
            )
        
        # Aggressively free optimizer, params, and GPU cache
        del optimizer, means_param, scales_param, rotations_param
        if self.cfg.harmonics_dc_only:
            del harmonics_dc_param, harmonics_rest
        else:
            del harmonics_param
        del opacities_param, grad_accum, grad_count
        torch.cuda.empty_cache()
        
        # Restore previous gradient tracking state
        torch.set_grad_enabled(prev_grad_enabled)
        
        return result
    
    def _densify_and_prune(
        self,
        means: Tensor, scales: Tensor, rotations: Tensor,
        harmonics: Tensor, opacities: Tensor, features: Optional[Tensor],
        optimizer: torch.optim.Adam,
        grad_accum: Tensor, grad_count: Tensor,
    ):
        """Clone small high-gradient Gaussians, split large ones, prune transparent ones."""
        B, G, _ = means.shape
        device = means.device
        cfg = self.cfg
        
        # Average gradient
        avg_grad = grad_accum / (grad_count + 1e-8)
        avg_grad = avg_grad.squeeze(-1)  # [B, G]
        
        # For simplicity, operate on batch index 0 (batch_size=1 at inference)
        b = 0
        high_grad = avg_grad[b] >= cfg.densify_grad_threshold  # [G]
        
        scale_max = scales[b].abs().clamp(min=1e-8).max(dim=-1).values  # [G]
        
        # Clone: small Gaussians with high gradient
        clone_mask = high_grad & (scale_max <= cfg.size_prune_threshold)
        # Split: large Gaussians with high gradient  
        split_mask = high_grad & (scale_max > cfg.size_prune_threshold)
        
        new_means_list = [means.detach()]
        new_scales_list = [scales.detach()]
        new_rotations_list = [rotations.detach()]
        new_harmonics_list = [harmonics.detach()]
        new_opacities_list = [opacities.detach()]
        new_features_list = [features.detach()] if features is not None else None
        
        # Clone
        if clone_mask.any():
            new_means_list.append(means[:, clone_mask].detach())
            new_scales_list.append(scales[:, clone_mask].detach())
            new_rotations_list.append(rotations[:, clone_mask].detach())
            new_harmonics_list.append(harmonics[:, clone_mask].detach())
            new_opacities_list.append(opacities[:, clone_mask].detach())
            if features is not None and new_features_list is not None:
                new_features_list.append(features[:, clone_mask].detach())
        
        # Split  
        if split_mask.any():
            n_split = split_mask.sum().item()
            split_scales = scales[:, split_mask].detach()
            split_means = means[:, split_mask].detach()
            split_rots = rotations[:, split_mask].detach()
            split_harms = harmonics[:, split_mask].detach()
            split_opacs = opacities[:, split_mask].detach()
            
            # Create 2 children with perturbed positions and reduced scale
            rot_matrices = quaternion_to_matrix(split_rots.reshape(-1, 4))  # [N, 3, 3]
            scale_vals = split_scales.reshape(-1, 3).abs().clamp(min=1e-8)
            stds = scale_vals
            
            samples = torch.randn(2, stds.shape[0], 3, device=device)  # [2, N, 3]
            offsets = torch.bmm(
                rot_matrices.expand(2, -1, -1, -1).reshape(-1, 3, 3),
                (samples.reshape(-1, 3) * stds.repeat(2, 1)).unsqueeze(-1),
            ).squeeze(-1).reshape(2, 1, n_split, 3)  # [2, B, N, 3]
            
            for child in range(2):
                new_means_list.append(split_means + offsets[child])
                new_scales_list.append(split_scales * 0.6)  # reduce scale
                new_rotations_list.append(split_rots.clone())
                new_harmonics_list.append(split_harms.clone())
                new_opacities_list.append(split_opacs.clone())
                if features is not None and new_features_list is not None:
                    new_features_list.append(features[:, split_mask].detach())
        
        # Concatenate
        means_new = torch.cat(new_means_list, dim=1).requires_grad_(True)
        scales_new = torch.cat(new_scales_list, dim=1).requires_grad_(True)
        rotations_new = torch.cat(new_rotations_list, dim=1).requires_grad_(True)
        harmonics_new = torch.cat(new_harmonics_list, dim=1).requires_grad_(True)
        opacities_new = torch.cat(new_opacities_list, dim=1).requires_grad_(True)
        features_new = torch.cat(new_features_list, dim=1) if new_features_list is not None else None
        
        # Prune low-opacity Gaussians
        with torch.no_grad():
            opacities_activated = torch.sigmoid(opacities_new)
            keep_mask = opacities_activated[0] >= cfg.opacity_prune_threshold  # [G']
            
            if keep_mask.any() and keep_mask.sum() > 100:  # don't prune to nothing
                means_new = means_new[:, keep_mask].detach().requires_grad_(True)
                scales_new = scales_new[:, keep_mask].detach().requires_grad_(True)
                rotations_new = rotations_new[:, keep_mask].detach().requires_grad_(True)
                harmonics_new = harmonics_new[:, keep_mask].detach().requires_grad_(True)
                opacities_new = opacities_new[:, keep_mask].detach().requires_grad_(True)
                if features_new is not None:
                    features_new = features_new[:, keep_mask].detach()
        
        # Rebuild optimizer
        new_optimizer = torch.optim.Adam([
            {"params": [means_new], "lr": self.cfg.lr_means},
            {"params": [scales_new], "lr": self.cfg.lr_scaling},
            {"params": [rotations_new], "lr": self.cfg.lr_rotation},
            {"params": [harmonics_new], "lr": self.cfg.lr_harmonics},
            {"params": [opacities_new], "lr": self.cfg.lr_opacity},
        ])
        
        G_new = means_new.shape[1]
        new_grad_accum = torch.zeros(B, G_new, 1, device=device)
        new_grad_count = torch.zeros(B, G_new, 1, device=device)
        
        return (means_new, scales_new, rotations_new, harmonics_new,
                opacities_new, features_new, new_optimizer, 
                new_grad_accum, new_grad_count)


def _ssim(img1: Tensor, img2: Tensor, window_size: int = 11) -> Tensor:
    """Simple SSIM computation for loss."""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    # Create Gaussian window
    sigma = 1.5
    coords = torch.arange(window_size, dtype=torch.float32, device=img1.device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window = g.unsqueeze(0) * g.unsqueeze(1)
    window = window.unsqueeze(0).unsqueeze(0)  # [1, 1, k, k]
    
    channels = img1.shape[1]
    window = window.expand(channels, 1, -1, -1)
    
    mu1 = F.conv2d(img1, window, groups=channels, padding=window_size // 2)
    mu2 = F.conv2d(img2, window, groups=channels, padding=window_size // 2)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv2d(img1 * img1, window, groups=channels, padding=window_size // 2) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, groups=channels, padding=window_size // 2) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, groups=channels, padding=window_size // 2) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean()
