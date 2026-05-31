from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

import torch
import wandb
from einops import pack, rearrange, repeat
from jaxtyping import Float
import torch
from torch import Tensor, nn
import torchvision
from splat_belief.utils.vision_utils import *

from .decoder.decoder import Decoder, DepthRenderingMode
from .encoder import Encoder
from .encoder.visualization.encoder_visualizer import EncoderVisualizer
from splat_belief.embodied.semantic_mapper import SemanticMapper
from splat_belief.utils.model_utils import forward_2d_model_batch
from .gaussian_refiner import GaussianRefiner, GaussianRefinerCfg

class SplatBeliefState(nn.Module):
    encoder: Encoder
    encoder_visualizer: Optional[EncoderVisualizer]
    decoder: Decoder
    losses: nn.ModuleList
    semantic_mapper: Optional[SemanticMapper]
    clip_model: Optional[nn.Module]
    dino_model: Optional[nn.Module]
    clip_process = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ]
    )

    def __init__(
        self,
        encoder: Encoder,
        encoder_visualizer: Optional[EncoderVisualizer],
        decoder: Decoder,
        semantic_mapper: Optional[SemanticMapper],
        depth_mode: DepthRenderingMode | None,
        extended_visualization: bool | None,
        clip_model: Optional[nn.Module] = None,
        dino_model: Optional[nn.Module] = None,
        inference_mode: bool = False,
        use_semantic: bool = False,
        semantic_mode: str = "embed",
        channels: int = 3,
        viz_type="interpolation",
        use_history: bool = True,
        feature_patch_min_scale: float = 0.05,
        feature_patch_max_scale: float = 0.5,
        feature_patch_num_samples: int = 15,
        use_depth_mask: bool = False,
        refiner_cfg: Optional[GaussianRefinerCfg] = None,
        obj_permanence_mode: str = "none",
        obj_permanence_observed_mode: str = "live",
        obj_permanence_state_t_min: int = 0,
        obj_permanence_mask_blur: int = 0,
        obj_permanence_mask_threshold: float = 0.5,
        obj_permanence_erode_kernel: int = 21,
        obj_permanence_mask_binarize_after_blur: bool = False,
        dps_guidance_scale: float = 1.0,
        dps_pos_weight: float = 1.0,
        dps_opacity_weight: float = 0.5,
    ) -> None:
        super().__init__()

        # Set up the model.
        self.channels = channels
        self.out_dim = channels
        self.viz_type = viz_type
        self.use_history = use_history
        self.encoder = encoder
        self.encoder_visualizer = encoder_visualizer
        self.decoder = decoder
        self.use_depth_mask = use_depth_mask
        self.use_semantic = use_semantic
        self.semantic_mode = semantic_mode
        if self.use_semantic:
            self.semantic_mapper = semantic_mapper
        self.clip_model = clip_model
        self.dino_model = dino_model
        self.feature_patch_min_scale = feature_patch_min_scale
        self.feature_patch_max_scale = feature_patch_max_scale
        self.feature_patch_num_samples = feature_patch_num_samples
        self.depth_mode = depth_mode
        self.extended_visualization = extended_visualization
        self.inference_mode = inference_mode
        self.normalize = normalize_to_neg_one_to_one
        self.unnormalize = unnormalize_to_zero_to_one

        # Online Gaussian refinement (Splat-SLAM-inspired)
        # Always create the refiner so that code paths (add_observation, etc.)
        # are identical regardless of the enabled flag.  The enabled flag only
        # controls whether the actual optimisation loop runs.
        if refiner_cfg is not None:
            self.refiner = GaussianRefiner(refiner_cfg, decoder)
        else:
            self.refiner = None

        # Internal counter for time steps.
        self.current_timestep = None
        # When True, sample() calls are autoregressive *imagination* steps (no
        # new ground-truth observation): a finished keyframe's belief is kept
        # (accumulated into imagined_gaussians) instead of being re-encoded into
        # observed history. Set externally per call; defaults to observation.
        self.imagine_mode = False
        # Containers for Gaussians
        self.history_gaussians = None
        self.incremental_gaussians = None
        self.belief_gaussians = None
        # Accumulated belief from earlier keyframes of the current autoregressive
        # imagination rollout (dropped when a real observation arrives).
        self.imagined_gaussians = None
        # The current keyframe's context Gaussians kept "live" (re-encoded and
        # re-masked every diffusion step) under obj_permanence_observed_mode
        # == "live"; committed into history_gaussians when the next keyframe
        # arrives. Stays None in "none"/"oneshot" mode.
        self.context_gaussians_live = None
        # Container for history
        self.history_rgb = None
        self.history_pose = None
        self.first_ctxt_shift = None

        # Object permanence guidance configuration. See docs/object_permanence_guidance.md.
        # mode: "none" | "opacity" | "dps"
        assert obj_permanence_mode in ("none", "opacity", "dps"), \
            f"unknown obj_permanence_mode: {obj_permanence_mode!r}"
        self.obj_permanence_mode = obj_permanence_mode
        # Observed-side object permanence — suppress an incoming keyframe's
        # context where already-observed history covers it:
        #   none    -- disabled
        #   oneshot -- mask once, when the keyframe is folded into history
        #   live    -- keep the context live, re-masked every diffusion step
        assert obj_permanence_observed_mode in ("none", "oneshot", "live"), \
            f"unknown obj_permanence_observed_mode: {obj_permanence_observed_mode!r}"
        self.obj_permanence_observed_mode = obj_permanence_observed_mode
        self.obj_permanence_state_t_min = int(obj_permanence_state_t_min)
        self.obj_permanence_mask_blur = int(obj_permanence_mask_blur)
        self.obj_permanence_mask_threshold = float(obj_permanence_mask_threshold)
        self.obj_permanence_mask_binarize_after_blur = bool(obj_permanence_mask_binarize_after_blur)
        # When > 0, the binary depth-validity mask is eroded by this kernel
        # size (odd) before being blurred. This shifts the "fully suppress
        # belief" core inward and frees the silhouette band to receive both
        # history (sparse-edge) and partial belief, smoothing 3D seams.
        self.obj_permanence_erode_kernel = int(obj_permanence_erode_kernel)
        self.dps_guidance_scale = float(dps_guidance_scale)
        self.dps_pos_weight = float(dps_pos_weight)
        self.dps_opacity_weight = float(dps_opacity_weight)
        # Cache from the most recent forward(): used by samplers (e.g. DPS).
        self._last_history_render = None  # dict with rgb, depth, mask, mask_per_g, P_ref
    
    @property
    def augmented_gaussians(self):
        """Full renderable Gaussian set: observed history + the live current
        keyframe context (obj_permanence_observed_mode == "live") + accumulated
        autoregressive imagination + the current belief."""
        parts = [g for g in (self.history_gaussians, self.imagined_gaussians,
                              self.context_gaussians_live, self.belief_gaussians)
                 if g is not None]
        if not parts:
            return None
        combined = parts[0]
        for g in parts[1:]:
            combined = combined + g
        return combined

    @property
    def observed_gaussians(self):
        """Real observed keyframes only: committed history plus the live
        current keyframe context. Excludes autoregressive imagination and the
        current belief. Equals history_gaussians in "none"/"oneshot" mode."""
        parts = [g for g in (self.history_gaussians, self.context_gaussians_live)
                 if g is not None]
        if not parts:
            return None
        combined = parts[0]
        for g in parts[1:]:
            combined = combined + g
        return combined

    def _obj_permanence_active(self, state_t):
        """Whether the object-permanence constraint should run for this step."""
        if self.obj_permanence_mode == "none":
            return False
        if self.history_gaussians is None:
            return False
        if state_t is None:
            return False
        if int(state_t) < self.obj_permanence_state_t_min:
            return False
        return True

    def _render_history_only(self, model_input, h, w, pose_key="trgt_c2w"):
        """Render history_gaussians alone at the `pose_key` pose (trgt_c2w by
        default; ctxt_c2w for the observed-side constraint). Returns rgb
        [B,3,H,W], depth [B,1,H,W], mask [B,1,H,W] in [0,1]; mask is 1 where
        history covers."""
        if self.history_gaussians is None:
            return None
        out = self.decoder.forward(
            self.history_gaussians.float(),
            model_input[pose_key],
            model_input["intrinsics"].unsqueeze(1),
            model_input["near"].float().unsqueeze(1),
            model_input["far"].float().unsqueeze(1),
            (h, w),
            depth_mode=self.depth_mode,
        )
        rgb = torch.clamp(out.color[:, 0], 0.0, 1.0)         # [B, 3, H, W]
        depth = out.depth[:, 0].unsqueeze(1)                  # [B, 1, H, W]
        b = rgb.shape[0]
        near = model_input["near"].float().view(b, 1, 1, 1)
        far = model_input["far"].float().view(b, 1, 1, 1)
        valid = (depth > (near + 1e-3)) & (depth < (far - 1e-3))
        mask = valid.float()
        # Optional binary erosion of the depth-validity mask. Shrinks the
        # "fully covered" core inward so the silhouette band has both history
        # (sparse-edge) and belief (with partial opacity) compositing
        # together — eliminates the thin seam that appears when both are
        # being suppressed near the boundary.
        if self.obj_permanence_erode_kernel and self.obj_permanence_erode_kernel > 1:
            ek = int(self.obj_permanence_erode_kernel)
            if ek % 2 == 0:
                ek += 1
            # Erosion of a [0,1] mask = -max_pool(-mask) over a KxK window.
            mask = -torch.nn.functional.max_pool2d(
                -mask, kernel_size=ek, stride=1, padding=ek // 2,
            )
            mask = mask.clamp(0.0, 1.0)
        if self.obj_permanence_mask_blur and self.obj_permanence_mask_blur > 1:
            k = int(self.obj_permanence_mask_blur)
            if k % 2 == 0:
                k += 1
            mask = torchvision.transforms.functional.gaussian_blur(mask, k)
        mask = mask.clamp(0.0, 1.0)
        if getattr(self, "obj_permanence_mask_binarize_after_blur", False):
            mask = (mask >= float(self.obj_permanence_mask_threshold)).to(mask.dtype)
        return {"rgb": rgb, "depth": depth, "mask": mask}

    def _build_per_gaussian_mask(self, mask_pix, num_gaussians, h, w):
        """Expand a per-pixel mask [B, 1, H, W] into a per-gaussian mask [B, G]
        matching the order produced by the encoder: V * (H*W) * srf * spp.
        Assumes V == 1 (single target view), which is the case in this stack."""
        b = mask_pix.shape[0]
        per_pixel = num_gaussians // (h * w)  # = srf * spp (assuming V==1)
        if per_pixel * h * w != num_gaussians:
            return None  # layout doesn't match expectation; bail out
        m = mask_pix.view(b, h * w, 1).expand(-1, -1, per_pixel).reshape(b, -1)
        return m

    def _project_gaussians_to_trgt(self, means_world, model_input, h, w,
                                   pose_key="trgt_c2w"):
        """Project gaussian world means [B, G, 3] into the `pose_key` camera.
        Returns:
          uv_norm: [B, G, 2] in pixel coords ([0, w], [0, h])
          z_cam:  [B, G] depth in camera frame (>0 = in front of camera)
        Intrinsics here are in normalised image coords (fx is (focal/W)).
        """
        b, G, _ = means_world.shape
        device = means_world.device
        c2w = model_input[pose_key][:, 0].to(torch.float32)               # [B, 4, 4]
        # invert (single-target view)
        w2c = torch.linalg.inv(c2w)                                       # [B, 4, 4]
        ones = torch.ones((b, G, 1), device=device, dtype=torch.float32)
        means_h = torch.cat([means_world.float(), ones], dim=-1)          # [B, G, 4]
        cam_h = torch.einsum("bij,bnj->bni", w2c, means_h)
        cam = cam_h[..., :3] / cam_h[..., 3:].clamp(min=1e-8)             # [B, G, 3]
        intr = model_input["intrinsics"].float()                          # [B, 3, 3]
        fx = intr[:, 0, 0].view(b, 1)
        fy = intr[:, 1, 1].view(b, 1)
        cx = intr[:, 0, 2].view(b, 1)
        cy = intr[:, 1, 2].view(b, 1)
        z = cam[..., 2].clamp(min=1e-6)
        u_norm = fx * cam[..., 0] / z + cx
        v_norm = fy * cam[..., 1] / z + cy
        u_pix = u_norm * w
        v_pix = v_norm * h
        uv = torch.stack([u_pix, v_pix], dim=-1)                          # [B, G, 2]
        return uv, cam[..., 2]

    def _sample_mask_at_gaussians(self, mask_pix, uv_pix, z_cam, h, w):
        """Bilinear sample mask_pix [B,1,H,W] at projected gaussian pixel
        coordinates uv_pix [B, G, 2]. Out-of-frame / behind-camera gaussians
        get mask 0 (treated as not covered). Returns [B, G]."""
        b, G, _ = uv_pix.shape
        # Convert to grid_sample coords in [-1, 1] with align_corners=False
        # (matching pixel-center convention used in _build_3d_position_refs).
        u_n = (uv_pix[..., 0] / w) * 2.0 - 1.0
        v_n = (uv_pix[..., 1] / h) * 2.0 - 1.0
        in_frame = (u_n.abs() <= 1.0) & (v_n.abs() <= 1.0) & (z_cam > 0)
        grid = torch.stack([u_n, v_n], dim=-1).view(b, G, 1, 2)           # [B, G, 1, 2]
        sampled = torch.nn.functional.grid_sample(
            mask_pix, grid, mode="bilinear", padding_mode="zeros", align_corners=False,
        )                                                                  # [B, 1, G, 1]
        sampled = sampled.view(b, G)
        sampled = sampled * in_frame.float()
        return sampled.clamp(0.0, 1.0)

    def _build_3d_position_refs(self, hist_render, model_input, h, w):
        """Backproject the history depth at trgt_c2w into 3D points (one per pixel),
        used as DPS-style position targets. Returns [B, H*W, 3] in world frame."""
        depth = hist_render["depth"].squeeze(1)               # [B, H, W]
        b = depth.shape[0]
        device = depth.device
        intr = model_input["intrinsics"].float()              # [B, 3, 3]
        c2w = model_input["trgt_c2w"][:, 0].float()           # [B, 4, 4]

        ys, xs = torch.meshgrid(
            torch.arange(h, device=device, dtype=torch.float32),
            torch.arange(w, device=device, dtype=torch.float32),
            indexing="ij",
        )
        # The decoder treats intrinsics as normalised coordinates ([0,1]); rays
        # are produced from the (h, w) grid via xy = (x+.5)/w, (y+.5)/h.  We
        # mirror that convention here so the back-projected points line up with
        # what the renderer projected.
        u = (xs + 0.5) / w                                    # [H, W]
        v = (ys + 0.5) / h
        # Pinhole back-projection in normalised coords.
        fx = intr[:, 0, 0].view(b, 1, 1)
        fy = intr[:, 1, 1].view(b, 1, 1)
        cx = intr[:, 0, 2].view(b, 1, 1)
        cy = intr[:, 1, 2].view(b, 1, 1)
        x_cam = (u.unsqueeze(0) - cx) / fx                    # [B, H, W]
        y_cam = (v.unsqueeze(0) - cy) / fy
        z_cam = depth                                          # [B, H, W]
        cam_pts = torch.stack([x_cam * z_cam, y_cam * z_cam, z_cam], dim=-1)  # [B, H, W, 3]
        cam_pts = cam_pts.view(b, h * w, 3)
        # Convert to homogeneous and apply c2w.
        ones = torch.ones((b, h * w, 1), device=device, dtype=cam_pts.dtype)
        cam_h = torch.cat([cam_pts, ones], dim=-1)            # [B, HW, 4]
        world = torch.einsum("bij,bnj->bni", c2w, cam_h)
        world = world[..., :3] / world[..., 3:].clamp(min=1e-8)
        return world                                          # [B, H*W, 3]

    def _apply_opacity_constraint(self, belief, model_input, h, w, hist_render=None,
                                  pose_key="trgt_c2w"):
        """Zero opacity of `belief` gaussians whose projected location in the
        `pose_key` view falls inside the history-covered region. Returns
        (constrained_belief, hist_render_dict). Uses projection so it works
        regardless of inference_mode pre-filtering of gaussians.

        pose_key="trgt_c2w" is the belief-side constraint; "ctxt_c2w" is the
        observed-side constraint (suppress an incoming keyframe's context where
        prior observed history already covers it)."""
        if hist_render is None:
            hist_render = self._render_history_only(model_input, h, w, pose_key=pose_key)
        if hist_render is None:
            return belief, None
        mask_pix = hist_render["mask"]                                 # [B,1,H,W]
        with torch.no_grad():
            uv, z = self._project_gaussians_to_trgt(belief.means.detach(), model_input, h, w,
                                                    pose_key=pose_key)
            mask_per_g = self._sample_mask_at_gaussians(mask_pix, uv, z, h, w)
        # Soft scaling: in fully covered regions belief contributes 0; in
        # partially covered regions it contributes (1 - mask).
        scale = (1.0 - mask_per_g).to(belief.opacities.dtype)
        belief.opacities = belief.opacities * scale
        return belief, hist_render

    def refine_current_gaussians(self, image_shape: tuple[int, int]):
        """Run online Gaussian refinement against stored observations.

        Only refines history_gaussians (from real observations), leaving
        belief_gaussians (imagined future) untouched so imagination renders
        are not affected by refinement artifacts.

        Returns:
            Augmented (refined history + belief) Gaussians if refinement
            actually ran, else None.
        """
        if self.history_gaussians is None:
            return None
        if self.refiner is not None:
            refined_history = self.refiner.refine(self.history_gaussians, image_shape)
            if refined_history is not None:
                self.history_gaussians = refined_history
                torch.cuda.empty_cache()
                # Return the full augmented set for the final re-render
                return self.augmented_gaussians
        # No refinement happened — return None so caller knows not to re-render
        return None

    def forward(
        self, model_input, t, global_step=100000, state_t=None,
        filter_border_gaussians: bool = False, 
        depth_inference_min: float = 0.2, 
        depth_inference_max: float = 10.0
    ):
        b, num_context, _, h, w = model_input["ctxt_rgb"].shape

        if self.current_timestep is None:
            # record the first ctxt c2w
            first_ctxt_c2w = model_input["ctxt_c2w"].clone()
            # make first_ctxt_shift to be the inverse of first_ctxt_c2w's position, but keep orientation identity
            first_ctxt_shift = torch.eye(4, device=first_ctxt_c2w.device).unsqueeze(0).unsqueeze(0).repeat(b, 1, 1, 1)
            first_ctxt_shift[:, 0:1, 3, 0:1] = -first_ctxt_c2w[:, 0:1, 3, 0:1]
            self.first_ctxt_shift = first_ctxt_shift

        # Build context Gaussians for this step
        if not (self.current_timestep==state_t): # do nothing if already updated history for this step
            # Append current context to history rgb for image conditioning
            if self.use_history:
                if self.history_rgb is None:
                    self.history_rgb = model_input["ctxt_rgb"]
                    self.history_pose = model_input["ctxt_c2w"]
                else:
                    self.history_rgb = torch.cat([self.history_rgb, model_input["ctxt_rgb"]], dim=1)
                    self.history_pose = torch.cat([self.history_pose, model_input["ctxt_c2w"]], dim=1)
                model_input["cond_rgb"] = self.history_rgb
                model_input["cond_pose"] = self.history_pose
            else:
                ctxt_c2w_raw = model_input["ctxt_c2w"].clone() 
                trgt_c2w_raw = model_input["trgt_c2w"].clone() 
                # Make ctxt to be identity and adjust trgt accordingly
                model_input["ctxt_c2w"] = torch.eye(4, device=ctxt_c2w_raw.device).unsqueeze(0).unsqueeze(0).repeat(b, num_context, 1, 1)
                # linalg.inv requires fp32; cast to float for the inverse under bf16 autocast.
                _orig_dtype = trgt_c2w_raw.dtype
                model_input["trgt_c2w"] = torch.einsum(
                    "bijk, bikl -> bijl",
                    torch.linalg.inv(ctxt_c2w_raw.float()),
                    trgt_c2w_raw.float(),
                ).to(_orig_dtype)
            context_gaussians, target_gaussians = self.encoder(
                model_input, 
                t=t,
                global_step=global_step, 
                deterministic=False,
                filter_border_gaussians=filter_border_gaussians,
                depth_inference_min=depth_inference_min,
                depth_inference_max=depth_inference_max,
            )
            if not self.use_history:
                # Shift the means of context and target gaussians back to original coordinates
                context_gaussians = context_gaussians.transform(ctxt_c2w_raw.squeeze(1))
                target_gaussians = target_gaussians.transform(ctxt_c2w_raw.squeeze(1))
                # Keep shifting the target gaussians using first_ctxt_c2w's position, but not orientation
                context_gaussians = context_gaussians.transform(self.first_ctxt_shift.squeeze(1))
                target_gaussians = target_gaussians.transform(self.first_ctxt_shift.squeeze(1))
                # Restore original coordinates
                model_input["ctxt_c2w"] = ctxt_c2w_raw
                model_input["trgt_c2w"] = trgt_c2w_raw
                # Shift ctxt_c2w and trgt_c2w using first_ctxt_shift with matrix multiplication
                model_input["ctxt_c2w"] = torch.einsum("bijk, bikl -> bijl", self.first_ctxt_shift, model_input["ctxt_c2w"])
                model_input["trgt_c2w"] = torch.einsum("bijk, bikl -> bijl", self.first_ctxt_shift, model_input["trgt_c2w"])

            # Update history Gaussians.
            # obj_permanence_observed_mode controls observed-side object
            # permanence: "oneshot" masks the incoming keyframe once at the
            # fold; "live" keeps it re-encoded + re-masked every diffusion step.
            live_obs = (self.obj_permanence_observed_mode == "live")
            mask_obs = self.obj_permanence_observed_mode in ("oneshot", "live")
            if self.current_timestep is None:
                self.current_timestep = 0
                if live_obs:
                    # First keyframe: no prior history to mask against; keep it live.
                    self.context_gaussians_live = context_gaussians
                else:
                    self.history_gaussians = context_gaussians
            elif self.current_timestep<state_t:
                assert self.current_timestep+1==state_t # can only differ by 1
                self.current_timestep+=1
                # Commit the previous, now-finalised keyframe's live context
                # into frozen observed history before this step's update.
                if self.context_gaussians_live is not None:
                    self.history_gaussians = (
                        self.context_gaussians_live if self.history_gaussians is None
                        else self.context_gaussians_live + self.history_gaussians.detach()
                    )
                    self.context_gaussians_live = None
                if self.imagine_mode:
                    # Autoregressive imagination step: the keyframe that just
                    # finished is ungrounded imagination, not a real observation.
                    # Keep its belief (accumulate into imagined_gaussians) rather
                    # than re-encoding it into observed history -- re-encoding it
                    # from the model's own generated image loses the joint
                    # multi-view consistency it had with the observed frames.
                    prev_belief = self.belief_gaussians.detach()
                    self.imagined_gaussians = (
                        prev_belief if self.imagined_gaussians is None
                        else self.imagined_gaussians + prev_belief
                    )
                else:
                    # A real new observation arrived: drop the imagined rollout
                    # and incorporate the newly-observed keyframe.
                    # Observed-side object permanence: suppress the incoming
                    # keyframe's context Gaussians where committed history
                    # already covers them (rendered at the context pose). The
                    # encoder input stays a clean RGB image -- no OOD.
                    if mask_obs and self.history_gaussians is not None:
                        context_gaussians, _ = self._apply_opacity_constraint(
                            context_gaussians, model_input, h, w, pose_key="ctxt_c2w")
                    if live_obs:
                        # Keep the new keyframe's context live so it is
                        # re-encoded + re-masked every diffusion step.
                        self.context_gaussians_live = context_gaussians
                    else:
                        self.history_gaussians=context_gaussians+self.history_gaussians.detach()
                    self.imagined_gaussians = None
            if self.refiner is not None:
                self.refiner.add_source_rays(
                    context_gaussians,
                    model_input["ctxt_c2w"][:, 0],
                    prepend=self.current_timestep > 0,
                )
            
            # Update inc obs Gaussians
            self.incremental_gaussians = context_gaussians
            # Update belief Gaussians
            self.belief_gaussians = target_gaussians

            # Object-permanence guidance: opacity-mode short-circuits belief
            # contribution in regions that history already covers. DPS-mode
            # leaves belief untouched here and relies on sampler-level grad.
            self._last_history_render = None
            if self._obj_permanence_active(state_t):
                hist_render = self._render_history_only(model_input, h, w)
                if hist_render is not None:
                    if self.obj_permanence_mode == "opacity":
                        self.belief_gaussians, _ = self._apply_opacity_constraint(
                            self.belief_gaussians, model_input, h, w, hist_render=hist_render,
                        )
                    self._last_history_render = hist_render

            # Store observed frame for online refinement (once per new state_t)
            if self.refiner is not None:
                obs_rgb = self.unnormalize(model_input["ctxt_rgb"][:, 0])  # [B, 3, H, W]
                obs_c2w = model_input["ctxt_c2w"][:, 0]  # [B, 4, 4]
                obs_intrinsics = model_input["intrinsics"]  # [B, 3, 3]
                obs_near = model_input["near"].float()  # [B]
                obs_far = model_input["far"].float()  # [B]
                self.refiner.add_observation(obs_rgb, obs_c2w, obs_intrinsics, obs_near, obs_far)
        else:
            assert self.current_timestep==state_t
            if self.use_history:
                model_input["cond_rgb"] = self.history_rgb
                model_input["cond_pose"] = self.history_pose
            else:
                ctxt_c2w_raw = model_input["ctxt_c2w"].clone() 
                trgt_c2w_raw = model_input["trgt_c2w"].clone() 
                # Make ctxt to be identity and adjust trgt accordingly
                model_input["ctxt_c2w"] = torch.eye(4, device=ctxt_c2w_raw.device).unsqueeze(0).unsqueeze(0).repeat(b, num_context, 1, 1)
                # linalg.inv requires fp32; cast to float for the inverse under bf16 autocast.
                _orig_dtype = trgt_c2w_raw.dtype
                model_input["trgt_c2w"] = torch.einsum(
                    "bijk, bikl -> bijl",
                    torch.linalg.inv(ctxt_c2w_raw.float()),
                    trgt_c2w_raw.float(),
                ).to(_orig_dtype)
            # TODO evolve context_gaussians as diffusion t goes
            context_gaussians, target_gaussians = self.encoder(
                model_input, 
                t=t,
                global_step=global_step, 
                deterministic=False,
                filter_border_gaussians=filter_border_gaussians,
                depth_inference_min=depth_inference_min,
                depth_inference_max=depth_inference_max,
            )
            if not self.use_history:
                # Shift the means of context and target gaussians back to original coordinates
                context_gaussians = context_gaussians.transform(ctxt_c2w_raw.squeeze(1))
                target_gaussians = target_gaussians.transform(ctxt_c2w_raw.squeeze(1))
                # Keep shifting the target gaussians using first_ctxt_c2w's position, but not orientation
                context_gaussians = context_gaussians.transform(self.first_ctxt_shift.squeeze(1))
                target_gaussians = target_gaussians.transform(self.first_ctxt_shift.squeeze(1))
                # Restore original coordinates
                model_input["ctxt_c2w"] = ctxt_c2w_raw
                model_input["trgt_c2w"] = trgt_c2w_raw
                # Shift ctxt_c2w and trgt_c2w using first_ctxt_shift with matrix multiplication
                model_input["ctxt_c2w"] = torch.einsum("bijk, bikl -> bijl", self.first_ctxt_shift, model_input["ctxt_c2w"])
                model_input["trgt_c2w"] = torch.einsum("bijk, bikl -> bijl", self.first_ctxt_shift, model_input["trgt_c2w"])
            # Update belief Gaussians
            self.belief_gaussians = target_gaussians

            # Live observed-side object permanence: re-encode and re-mask the
            # current keyframe's context every diffusion step, so the observed
            # seam is harmonised through the diffusion feedback loop.
            if self.obj_permanence_observed_mode == "live" and not self.imagine_mode:
                if self.history_gaussians is not None:
                    context_gaussians, _ = self._apply_opacity_constraint(
                        context_gaussians, model_input, h, w, pose_key="ctxt_c2w")
                self.context_gaussians_live = context_gaussians

            # Same constraint as the new-state branch — keeps every diffusion
            # step (state_t == current_timestep) consistent with the same rule.
            self._last_history_render = None
            if self._obj_permanence_active(state_t):
                hist_render = self._render_history_only(model_input, h, w)
                if hist_render is not None:
                    if self.obj_permanence_mode == "opacity":
                        self.belief_gaussians, _ = self._apply_opacity_constraint(
                            self.belief_gaussians, model_input, h, w, hist_render=hist_render,
                        )
                    self._last_history_render = hist_render

        assert state_t >= self.current_timestep, "state_t should not be smaller than the current timestep"

        # Augmented gaussians (observed history + imagined rollout + belief)
        gaussians = self.augmented_gaussians

        # Decode
        output = self.decoder.forward(
            gaussians.float(),
            model_input["trgt_c2w"],
            model_input["intrinsics"].unsqueeze(1),
            model_input["near"].float().unsqueeze(1),
            model_input["far"].float().unsqueeze(1),
            (h, w),
            depth_mode=self.depth_mode,
        )

        # Re-render
        rerender_output = self.decoder.forward(
            gaussians.float(),
            model_input["ctxt_c2w"],
            model_input["intrinsics"].unsqueeze(1).repeat(1, num_context, 1, 1),
            model_input["near"].float().unsqueeze(1).repeat(1, num_context),
            model_input["far"].float().unsqueeze(1).repeat(1, num_context),
            (h, w),
            depth_mode=self.depth_mode,
        )

        target_rendered_color = torch.clamp(output.color, 0.0, 1.0)
        target_rendered_depth = output.depth
        target_rendered_features = output.features # b v c h w

        context_rendered_color = torch.clamp(rerender_output.color, 0.0, 1.0)
        context_rendered_depth = rerender_output.depth
        context_rendered_features = rerender_output.features # b v c h w

        misc = {
            "rendered_ctxt_rgb": context_rendered_color,
            "rendered_trgt_rgb": target_rendered_color,
            "rendered_ctxt_depth": context_rendered_depth,
            "rendered_trgt_depth": target_rendered_depth,
        }

        # Render intermediate
        if "intm_c2w" in model_input:
            b, num_intm, _, h, w = model_input["intm_rgb"].shape

            intm_output = self.decoder.forward(
                gaussians.float(),
                model_input["intm_c2w"],
                model_input["intrinsics"].unsqueeze(1).repeat(1, num_intm, 1, 1),
                model_input["near"].float().unsqueeze(1).repeat(1, num_intm),
                model_input["far"].float().unsqueeze(1).repeat(1, num_intm),
                (h, w),
                depth_mode=self.depth_mode,
            )
            intm_rendered_color = torch.clamp(intm_output.color, 0.0, 1.0)
            intm_rendered_depth = intm_output.depth
            intm_rendered_features = intm_output.features
        
            misc.update({
                "rendered_intm_rgb": intm_rendered_color,
                "rendered_intm_depth": intm_rendered_depth,
            })
        
        # Sample from target rendered semantic feature maps
        if self.use_semantic and not self.inference_mode:
            target_rendered_semantic = self.encoder.get_semantic_features(target_rendered_features)
            target_samples = self.sample_patches_and_get_clip_embeddings(
                target_rendered_semantic,
                self.unnormalize(model_input["trgt_rgb"]),
                num_samples=self.feature_patch_num_samples,
                min_scale=self.feature_patch_min_scale,
                max_scale=self.feature_patch_max_scale,
            )
            if self.dino_model is not None:
                target_samples["semantic_reg_dense"] = self.encoder.get_semantic_reg_features(target_rendered_features)
                target_samples["dino_dense"] = self._compute_dense_dino_targets(
                    self.unnormalize(model_input["trgt_rgb"])
                )

            misc[f"trgt_semantic"] = target_samples

        # Sample from context rendered semantic feature maps
        if self.use_semantic and not self.inference_mode:
            context_rendered_semantic = self.encoder.get_semantic_features(context_rendered_features)
            context_samples = self.sample_patches_and_get_clip_embeddings(
                context_rendered_semantic,
                self.unnormalize(model_input["ctxt_rgb"]),
                num_samples=self.feature_patch_num_samples,
                min_scale=self.feature_patch_min_scale,
                max_scale=self.feature_patch_max_scale,
            )
            if self.dino_model is not None:
                context_samples["semantic_reg_dense"] = self.encoder.get_semantic_reg_features(context_rendered_features)
                context_samples["dino_dense"] = self._compute_dense_dino_targets(
                    self.unnormalize(model_input["ctxt_rgb"])
                )

            misc[f"ctxt_semantic"] = context_samples

        # If available, also sample from intermediate rendered semantic maps
        if "intm_c2w" in model_input and self.use_semantic and not self.inference_mode:
            intm_rendered_semantic = self.encoder.get_semantic_features(intm_rendered_features)
            intm_samples = self.sample_patches_and_get_clip_embeddings(
                intm_rendered_semantic,
                self.unnormalize(model_input["intm_rgb"]),
                num_samples=self.feature_patch_num_samples,
                min_scale=self.feature_patch_min_scale,
                max_scale=self.feature_patch_max_scale,
            )
            if self.dino_model is not None:
                intm_samples["semantic_reg_dense"] = self.encoder.get_semantic_reg_features(intm_rendered_features)
                intm_samples["dino_dense"] = self._compute_dense_dino_targets(
                    self.unnormalize(model_input["intm_rgb"])
                )

            # Update misc with the sampled patches and their center features
            misc[f"intm_semantic"] = intm_samples

        # One CLIP forward across all collected patches.
        if self.use_semantic and not self.inference_mode:
            self._run_clip_on_collected_patches(misc)

        return self.normalize(target_rendered_color), target_rendered_depth, misc

    def render(self, gaussians, extrinsics, intrinsics, near, far, h, w,
               filter_ceiling: bool = False, ceiling_threshold: float = 2.5,
               query_label: Optional[str] = None):
        
        # Shift extrinsics using first_ctxt_shift with matrix multiplication
        extrinsics = extrinsics.unsqueeze(0).unsqueeze(0)
        extrinsics = torch.einsum("bijk, bikl -> bijl", self.first_ctxt_shift, extrinsics)
        extrinsics = extrinsics.squeeze(0).squeeze(0)

        if filter_ceiling:
            gaussians = gaussians.filter_ceiling(ceiling_threshold)
        
        output = self.decoder.forward(
            gaussians.float(),
            extrinsics.float().unsqueeze(0)[:, None, ...].cuda(),
            intrinsics.float().unsqueeze(0)[:1].unsqueeze(1).cuda(),
            near.float().unsqueeze(0).unsqueeze(1).cuda(),
            far.float().unsqueeze(0).unsqueeze(1).cuda(),
            (h, w),
            depth_mode=self.depth_mode,
        )

        rgb = output.color[:, 0, ...]
        depth = output.depth[:, 0, ...]
        rgb = rearrange(rgb, "b c h w -> b h w c")
        rgb = torch.clamp(rgb, 0.0, 1.0) 
        rgb = rgb * 255.0
        rgb = rgb.float().cpu().detach().numpy().astype(np.uint8)
        depth = depth.float().cpu().detach()
        semantic = None
        if self.use_semantic:
            features = output.features[:, 0:1, ...]
            if self.semantic_mode == "embed":
                features = self.encoder.get_semantic_features(features)
                # features = self.encoder.get_semantic_reg_features(features)
            features = features.squeeze(1)  # [b, c, h, w]
            raw_intensity = True if query_label is not None else False
            semantic = self.semantic_mapper.forward(features, query_label, raw_intensity)
            semantic = semantic.float().cpu().detach().numpy()
            if not raw_intensity:
                semantic = semantic.astype(np.uint8)
        return rgb, depth, semantic

    @torch.no_grad()
    def render_video(
        self,
        model_input,
        t,
        n,
        num_videos=None,
        render_high_res=False,
        global_step=100000,
    ):
        b, num_context, _, h, w = model_input["ctxt_rgb"].shape
        model_input["ctxt_c2w"] = torch.einsum("bijk, bikl -> bijl", self.first_ctxt_shift, model_input["ctxt_c2w"])
        model_input["trgt_c2w"] = torch.einsum("bijk, bikl -> bijl", self.first_ctxt_shift, model_input["trgt_c2w"])

        ## Render videos using the updated belief gaussians and whatever history gaussians available
        if "render_poses" not in model_input.keys():
            render_poses = self.compute_poses(self.viz_type, model_input, n,)
            print("using computed poses")
        else:
            # shift each pose in render_poses (b, N, 4, 4) using first_ctxt_shift
            render_poses = torch.einsum("bijk, bikl -> bijl", self.first_ctxt_shift, model_input["render_poses"])[0]
            n = render_poses.shape[0]
            print("using provided poses", render_poses.shape)
        intrinsics = model_input["intrinsics"]

        if num_videos is None:
            num_videos = 1
        
        frames = []
        depth_frames = []
        semantics = []
        if self.use_depth_mask:
            depth_masks = []

        h = 128 if render_high_res else h
        w = 128 if render_high_res else w

        print(f"render_poses {render_poses.shape}")

        gaussians = self.augmented_gaussians

        for i in range(n):
            output = self.decoder.forward(
                gaussians.float(),
                render_poses[i : i + 1][:, None, ...].cuda(),
                intrinsics[:num_videos].unsqueeze(1),
                model_input["near"].float().unsqueeze(1),
                model_input["far"].float().unsqueeze(1),
                (h, w),
                depth_mode=self.depth_mode,
            )
            rgb = output.color[:, 0, ...]
            depth = output.depth[:, 0, ...]
            rgb = rearrange(rgb, "b c h w -> b h w c")
            rgb = torch.clamp(rgb, 0.0, 1.0) 
            if self.use_depth_mask:
                depth_mask = self.encoder.get_depth_mask(output.features)[:, 0, ...]
                depth_mask = rearrange(depth_mask, "b c h w -> b h w c")
                # apply sigmoid to the mask
                depth_mask = torch.sigmoid(depth_mask)
                # make the mask binary
                depth_mask = (depth_mask > 0.5).float()
                # apply the mask to the color
                rgb = rgb * depth_mask
                depth = depth * depth_mask.squeeze(3)

            rgb = rgb * 255.0
            frames.append(rgb.float().cpu().detach().numpy().astype(np.uint8))
            depth_frames.append(depth.float().cpu().detach())
            if self.use_depth_mask:
                depth_masks.append((depth_mask * 255.0).cpu().detach().numpy().astype(np.uint8))

            if self.use_semantic:
                features = output.features[:, 0:1, ...]
                if self.semantic_mode == "embed":
                    features = self.encoder.get_semantic_features(features)
                    # features = self.encoder.get_semantic_reg_features(features)
                features = features.squeeze(1)  # [b, c, h, w]
                semantic = self.semantic_mapper.forward(features)
                semantics.append(semantic.float().cpu().detach().numpy().astype(np.uint8))
            
        print(f"frames {len(frames)}")
        if self.use_depth_mask:
            return frames, depth_frames, semantics, render_poses, depth_masks
        else:
            return frames, depth_frames, semantics, render_poses

    @torch.no_grad()
    def compute_poses(
        self, type, model_input, n, radius=None, max_angle=None, canonical=False
    ):
        near = model_input["near"]
        far = model_input["far"]

        if type == "spherical":
            if radius is None:
                radius = (near + far) * 0.5
            if max_angle is None:
                max_angle = 60

            render_poses = []
            for angle in np.linspace(0, max_angle, n + 1)[:-1]:
                pose = pose_spherical(0, -angle, radius).cpu()
                if canonical:
                    pose = torch.einsum(
                        "ij, jk -> ik", model_input["inv_ctxt_c2w"][0, 0].cpu(), pose,
                    )
                else:
                    pose[2, -1] += radius
                render_poses.append(pose)
            render_poses = torch.stack(render_poses, 0)
        elif type == "interpolation":
            render_poses = torch.stack(
                [
                    interpolate_pose_wobble(
                        model_input["ctxt_c2w"][0][0],
                        model_input["trgt_c2w"][0][0],
                        t / n,
                        wobble=False,
                    )
                    for t in range(n)
                ],
                0,
            )
        else:
            raise ValueError("Unknown video type", type)
        print(f"render_poses: {render_poses.shape}")
        return render_poses

    def get_current_timestep(self):
        """Returns the current time step."""
        return self.current_timestep

    def reset_timestep(self):
        """Resets the internal time step counter and stored state."""
        self.current_timestep = None
        self.history_gaussians = None
        self.incremental_gaussians = None
        self.belief_gaussians = None
        self.imagined_gaussians = None
        self.context_gaussians_live = None
        self.history_rgb = None
        self.history_pose = None
        self.first_ctxt_c2w = None
        if self.refiner is not None:
            self.refiner.reset()

    def copy_states_to_ema(self, model: SplatBeliefState):
        model.current_timestep = self.current_timestep
        model.history_rgb = self.history_rgb
        model.history_pose = self.history_pose
        model.first_ctxt_shift = self.first_ctxt_shift
        if self.history_gaussians is not None:
            model.history_gaussians = self.history_gaussians.clone()
            model.history_gaussians.means = self.history_gaussians.means[:1]
            model.history_gaussians.covariances = self.history_gaussians.covariances[:1]
            model.history_gaussians.harmonics = self.history_gaussians.harmonics[:1]
            model.history_gaussians.opacities = self.history_gaussians.opacities[:1]
            model.history_gaussians.features = self.history_gaussians.features[:1] if self.history_gaussians.features is not None else None
        else:
            model.history_gaussians = None
        if self.incremental_gaussians is not None:
            model.incremental_gaussians = self.incremental_gaussians.clone()
            model.incremental_gaussians.means = self.incremental_gaussians.means[:1]
            model.incremental_gaussians.covariances = self.incremental_gaussians.covariances[:1]
            model.incremental_gaussians.harmonics = self.incremental_gaussians.harmonics[:1]
            model.incremental_gaussians.opacities = self.incremental_gaussians.opacities[:1]
            model.incremental_gaussians.features = self.incremental_gaussians.features[:1] if self.incremental_gaussians.features is not None else None
        else:
            model.incremental_gaussians = None
        if self.belief_gaussians is not None:
            model.belief_gaussians = self.belief_gaussians.clone()
            model.belief_gaussians.means = self.belief_gaussians.means[:1]
            model.belief_gaussians.covariances = self.belief_gaussians.covariances[:1]
            model.belief_gaussians.harmonics = self.belief_gaussians.harmonics[:1]
            model.belief_gaussians.opacities = self.belief_gaussians.opacities[:1]
            model.belief_gaussians.features = self.belief_gaussians.features[:1] if self.belief_gaussians.features is not None else None
        else:
            model.belief_gaussians = None
        if self.context_gaussians_live is not None:
            model.context_gaussians_live = self.context_gaussians_live.clone()
            model.context_gaussians_live.means = self.context_gaussians_live.means[:1]
            model.context_gaussians_live.covariances = self.context_gaussians_live.covariances[:1]
            model.context_gaussians_live.harmonics = self.context_gaussians_live.harmonics[:1]
            model.context_gaussians_live.opacities = self.context_gaussians_live.opacities[:1]
            model.context_gaussians_live.features = self.context_gaussians_live.features[:1] if self.context_gaussians_live.features is not None else None
        else:
            model.context_gaussians_live = None

    def _compute_dense_dino_targets(self, rgb_maps):
        """Run the (frozen) DINO encoder on rgb_maps and return dense feature maps.

        Outputs are bilinearly upsampled back to rendered resolution so they
        can be compared pixel-wise against the encoder's dense reg head output.

        Returns: tensor of shape (batch, views, dino_c, height, width).
        """
        batch, views, _, height, width = rgb_maps.shape
        rgb_flat = rgb_maps.reshape(batch * views, 3, height, width)
        dino_patch_size = self.dino_model.patch_embed.patch_size
        if isinstance(dino_patch_size, (tuple, list)):
            dino_patch_size = dino_patch_size[0]
        dino_h0 = max(dino_patch_size, height // 4 * dino_patch_size)
        dino_w0 = max(dino_patch_size, width // 4 * dino_patch_size)
        rgb_resized = torch.nn.functional.interpolate(
            rgb_flat, size=(dino_h0, dino_w0), mode='bilinear', align_corners=False
        )
        with torch.no_grad():
            feats = forward_2d_model_batch(rgb_resized, self.dino_model)
            _, dino_c, dh, dw = feats.shape
            feats = torch.nn.functional.interpolate(
                feats, size=(height, width), mode='bilinear', align_corners=False
            ).reshape(batch, views, dino_c, height, width)
        return feats

    # CLIP image normalization constants (OpenAI CLIP / open_clip ViT-B/16)
    _CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
    _CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

    def sample_patches_and_get_clip_embeddings(
        self,
        semantic_maps,
        rgb_maps,
        num_samples=10,
        min_scale=0.03125,  # 1/32 of the image size
        max_scale=1.0,
        seed=None,
    ):
        """Vectorized random-patch sampler. See SplatBelief for details."""
        if seed is not None:
            torch.manual_seed(seed)

        B, V, C, H, W = semantic_maps.shape
        N = num_samples
        device = semantic_maps.device

        ys = torch.randint(0, H, (B, V, N), device=device)
        xs = torch.randint(0, W, (B, V, N), device=device)
        log_scale = torch.empty((B, V, N), device=device).uniform_(
            float(np.log(min_scale)), float(np.log(max_scale))
        )
        scales = log_scale.exp()

        sm_flat = semantic_maps.reshape(B, V, C, H * W)
        flat_idx = (ys * W + xs).unsqueeze(2).expand(-1, -1, C, -1)
        centers = torch.gather(sm_flat, 3, flat_idx)
        centers = centers.permute(0, 1, 3, 2).contiguous()

        half_pix = (min(H, W) * scales / 2.0).clamp(min=1.0)

        lin = torch.linspace(-1.0, 1.0, 224, device=device)
        gy, gx = torch.meshgrid(lin, lin, indexing="ij")
        base_x = gx[None, None, None]
        base_y = gy[None, None, None]
        cx = xs.float()[..., None, None]
        cy = ys.float()[..., None, None]
        hp = half_pix[..., None, None]
        sample_x = cx + base_x * hp
        sample_y = cy + base_y * hp
        sample_x = (sample_x / max(W - 1, 1)) * 2.0 - 1.0
        sample_y = (sample_y / max(H - 1, 1)) * 2.0 - 1.0
        grid = torch.stack([sample_x, sample_y], dim=-1)

        rgb_flat = rgb_maps.reshape(B * V, 3, H, W)
        grid_per_bv = grid.reshape(B * V, N * 224, 224, 2)
        rgb_patches = torch.nn.functional.grid_sample(
            rgb_flat, grid_per_bv,
            mode="bilinear", padding_mode="border", align_corners=True,
        )
        rgb_patches = rgb_patches.reshape(B * V, 3, N, 224, 224).permute(0, 2, 1, 3, 4)
        rgb_patches = rgb_patches.reshape(B, V, N, 3, 224, 224)

        mean = torch.tensor(self._CLIP_MEAN, device=device, dtype=rgb_patches.dtype).view(1, 1, 1, 3, 1, 1)
        std = torch.tensor(self._CLIP_STD, device=device, dtype=rgb_patches.dtype).view(1, 1, 1, 3, 1, 1)
        rgb_patches = ((rgb_patches - mean) / std).half()

        return {"center_features": centers, "rgb_patches": rgb_patches}

    def _run_clip_on_collected_patches(self, misc):
        """Single CLIP forward across trgt/ctxt/intm patches stashed in misc."""
        if self.clip_model is None:
            return
        keys = []
        chunks = []
        for k in ("trgt_semantic", "ctxt_semantic", "intm_semantic"):
            if k not in misc or "rgb_patches" not in misc[k]:
                continue
            patches = misc[k]["rgb_patches"]
            B_k, V_k, N_k = patches.shape[:3]
            chunks.append(patches.reshape(-1, 3, 224, 224))
            keys.append((k, B_k, V_k, N_k))
        if not chunks:
            return
        all_patches = torch.cat(chunks, dim=0)
        with torch.no_grad():
            all_clip = self.clip_model.encode_image(all_patches)
        offset = 0
        for k, B_k, V_k, N_k in keys:
            count = B_k * V_k * N_k
            misc[k]["clip_embeddings"] = all_clip[offset:offset + count].reshape(B_k, V_k, N_k, -1)
            offset += count
            del misc[k]["rgb_patches"]
