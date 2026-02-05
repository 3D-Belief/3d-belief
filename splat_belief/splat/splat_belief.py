from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

import torch
import wandb
from einops import pack, rearrange, repeat
import torch
from torch import Tensor, nn
import torchvision
from ..utils.vision_utils import *

from .decoder.decoder import Decoder, DepthRenderingMode
from .encoder import Encoder
from .encoder.visualization.encoder_visualizer import EncoderVisualizer
from ..embodied.semantic_mapper import SemanticMapper
from ..utils.model_utils import forward_2d_model_batch, preprocess_raw_image

class SplatBelief(nn.Module):
    encoder: Encoder
    encoder_visualizer: Optional[EncoderVisualizer]
    decoder: Decoder
    losses: nn.ModuleList
    semantic_mapper: Optional[SemanticMapper]
    clip_model: Optional[nn.Module]
    dino_model: Optional[nn.Module]
    repa_encoder: Optional[nn.Module]
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
        repa_encoder: Optional[Encoder] = None,
        repa_encoder_name: Optional[str] = None,
        repa_encoder_resolution: Optional[int] = None,
        inference_mode: bool = False,
        use_semantic: bool = False,
        semantic_mode: str = "embed",
        use_depth_mask: bool = False,
        channels: int = 3,
        viz_type="interpolation",
        feature_patch_min_scale: float = 0.05,
        feature_patch_max_scale: float = 0.5,
        feature_patch_num_samples: int = 15,
    ) -> None:
        super().__init__()

        # Set up the model.
        self.channels = channels
        self.out_dim = channels
        self.viz_type = viz_type
        self.encoder = encoder
        self.encoder_visualizer = encoder_visualizer
        self.decoder = decoder
        self.use_semantic = use_semantic
        self.semantic_mode = semantic_mode
        if self.use_semantic:
            self.semantic_mapper = semantic_mapper
        self.clip_model = clip_model
        self.dino_model = dino_model
        self.repa_encoder = repa_encoder
        self.repa_encoder_name = repa_encoder_name
        self.repa_encoder_resolution = repa_encoder_resolution
        self.feature_patch_min_scale = feature_patch_min_scale
        self.feature_patch_max_scale = feature_patch_max_scale
        self.feature_patch_num_samples = feature_patch_num_samples
        self.use_depth_mask = use_depth_mask
        self.depth_mode = depth_mode
        self.extended_visualization = extended_visualization
        self.inference_mode = inference_mode
        self.normalize = normalize_to_neg_one_to_one
        self.unnormalize = unnormalize_to_zero_to_one
        self.gaussians = None

    def forward(self, model_input, t, global_step=100000):
        b, num_context, _, h, w = model_input["ctxt_rgb"].shape

        context_gaussians, target_gaussians = self.encoder(
            model_input, 
            t=t,
            global_step=global_step, 
            deterministic=self.inference_mode,
        )

        repa_pred = None
        if "repa_pred" in model_input:
            repa_pred = model_input["repa_pred"]
        repa_pred_ctxt = None
        if "repa_pred_ctxt" in model_input:
            repa_pred_ctxt = model_input["repa_pred_ctxt"]

        latents = None
        if "latents" in model_input:
            latents = model_input["latents"]

        repa_gt_ctxt = None
        repa_gt_trgt = None
        if self.repa_encoder is not None:
            raw_ctxt = self.unnormalize(model_input["ctxt_rgb"]).squeeze(1)
            raw_ctxt = preprocess_raw_image(
                raw_ctxt, self.repa_encoder_name, resolution=self.repa_encoder_resolution
            )
            repa_gt_ctxt = self.repa_encoder.forward_features(raw_ctxt)
            if 'dinov2' in self.repa_encoder_name: repa_gt_ctxt = repa_gt_ctxt['x_norm_patchtokens']

            raw_trgt = self.unnormalize(model_input["trgt_rgb"]).squeeze(1)
            raw_trgt = preprocess_raw_image(
                raw_trgt, self.repa_encoder_name, resolution=self.repa_encoder_resolution
            )
            repa_gt_trgt = self.repa_encoder.forward_features(raw_trgt)
            if 'dinov2' in self.repa_encoder_name: repa_gt_trgt = repa_gt_trgt['x_norm_patchtokens']

        # Augmented gaussians
        gaussians = context_gaussians + target_gaussians
        self.gaussians = gaussians
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
        target_rendered_features = output.features

        context_rendered_color = torch.clamp(rerender_output.color, 0.0, 1.0)
        context_rendered_depth = rerender_output.depth
        context_rendered_features = rerender_output.features

        if self.use_depth_mask:
            target_rendered_depth_mask = self.encoder.get_depth_mask(target_rendered_features)
            context_rendered_depth_mask = self.encoder.get_depth_mask(context_rendered_features)

        misc = {
            "rendered_ctxt_rgb": context_rendered_color,
            "rendered_trgt_rgb": target_rendered_color,
            "rendered_ctxt_depth": context_rendered_depth,
            "rendered_trgt_depth": target_rendered_depth,
            "repa_pred": repa_pred,
            "repa_pred_ctxt": repa_pred_ctxt,
            "repa_gt": repa_gt_trgt,
            "repa_gt_ctxt": repa_gt_ctxt,
            "latents": latents,
        }

        if self.use_depth_mask:
            misc.update({
                "rendered_ctxt_depth_mask": context_rendered_depth_mask,
                "rendered_trgt_depth_mask": target_rendered_depth_mask,
            })

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

            if self.use_depth_mask:
                intm_rendered_depth_mask = self.encoder.get_depth_mask(intm_rendered_features)
                misc.update({
                    "rendered_intm_depth_mask": intm_rendered_depth_mask,
                })
        
            misc.update({
                "rendered_intm_rgb": intm_rendered_color,
                "rendered_intm_depth": intm_rendered_depth,
            })

        extract_dino = False if self.dino_model is None else True
        # Sample from target rendered semantic feature maps
        if self.use_semantic and not self.inference_mode:
            target_rendered_semantic = self.encoder.get_semantic_features(target_rendered_features)
            target_rendered_semantic_reg = None
            if self.dino_model is not None:
                target_rendered_semantic_reg = self.encoder.get_semantic_reg_features(target_rendered_features)
            target_samples = self.sample_patches_and_get_clip_embeddings(
                target_rendered_semantic,
                self.unnormalize(model_input["trgt_rgb"]),
                num_samples=self.feature_patch_num_samples,
                min_scale=self.feature_patch_min_scale,
                max_scale=self.feature_patch_max_scale,
                extract_dino=extract_dino,
                semantic_reg_maps=target_rendered_semantic_reg
            )
            
            misc[f"trgt_semantic"] = target_samples
        
        # Sample from context rendered semantic feature maps
        if self.use_semantic and not self.inference_mode:
            context_rendered_semantic = self.encoder.get_semantic_features(context_rendered_features)
            context_rendered_semantic_reg = None
            if self.dino_model is not None:
                context_rendered_semantic_reg = self.encoder.get_semantic_reg_features(context_rendered_features)
            context_samples = self.sample_patches_and_get_clip_embeddings(
                context_rendered_semantic,
                self.unnormalize(model_input["ctxt_rgb"]),
                num_samples=self.feature_patch_num_samples,
                min_scale=self.feature_patch_min_scale,
                max_scale=self.feature_patch_max_scale,
                extract_dino=extract_dino,
                semantic_reg_maps=context_rendered_semantic_reg
            )
    
            misc[f"ctxt_semantic"] = context_samples
        
        # If available, also sample from intermediate rendered semantic maps
        if "intm_c2w" in model_input and self.use_semantic and not self.inference_mode:
            intm_rendered_semantic = self.encoder.get_semantic_features(intm_rendered_features)
            intm_rendered_semantic_reg = None
            if self.dino_model is not None:
                intm_rendered_semantic_reg = self.encoder.get_semantic_reg_features(intm_rendered_features)
            intm_samples = self.sample_patches_and_get_clip_embeddings(
                intm_rendered_semantic,
                self.unnormalize(model_input["intm_rgb"]),
                num_samples=self.feature_patch_num_samples,
                min_scale=self.feature_patch_min_scale,
                max_scale=self.feature_patch_max_scale,
                extract_dino=extract_dino,
                semantic_reg_maps=intm_rendered_semantic_reg
            )
            
            # Update misc with the sampled patches and their center features
            misc[f"intm_semantic"] = intm_samples
        
        return self.normalize(target_rendered_color), target_rendered_depth, misc
    
    def render(self, gaussians, extrinsics, intrinsics, near, far, h, w):
        output = self.decoder.forward(
            gaussians,
            extrinsics,
            intrinsics,
            near,
            far,
            (h, w),
            depth_mode=self.depth_mode,
        )
        output.color = torch.clamp(output.color, 0.0, 1.0)
        return output

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
        if "render_poses" not in model_input.keys():
            render_poses = self.compute_poses(self.viz_type, model_input, n,)
            print("using computed poses")
        else:
            render_poses = model_input["render_poses"][0]
            n = len(render_poses)
            print("using provided poses", render_poses.shape)
        intrinsics = model_input["intrinsics"]

        if num_videos is None:
            num_videos = model_input["trgt_rgb"].shape[1]
        
        b, num_context, _, h, w = model_input["ctxt_rgb"].shape

        context_gaussians, target_gaussians = self.encoder(model_input, t, global_step=global_step)

        # Augmented gaussians
        gaussians = context_gaussians + target_gaussians
        self.gaussians = gaussians
        frames = []
        depth_frames = []
        semantics = []

        h = 128 if render_high_res else h
        w = 128 if render_high_res else w

        print(f"render_poses {render_poses.shape}")

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
            rgb = rgb * 255.0
            frames.append(rgb.float().cpu().detach().numpy().astype(np.uint8))
            depth_frames.append(depth.float().cpu().detach())

            if self.use_semantic:
                features = output.features[:, 0:1, ...]
                if self.semantic_mode == "embed":
                    features = self.encoder.get_semantic_features(features)
                    # features = self.encoder.get_semantic_reg_features(features)
                features = features.squeeze(1)  # [b, c, h, w]
                semantic = self.semantic_mapper.forward(features)
                semantics.append(semantic.float().cpu().detach().numpy().astype(np.uint8))

        print(f"frames {len(frames)}")
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

    def sample_patches_and_get_clip_embeddings(
        self,
        semantic_maps,
        rgb_maps,
        num_samples=10,
        min_scale=0.03125,  # 1/32 of the image size
        max_scale=1.0,
        seed=None,
        extract_dino=False,
        reg_patch_size=4,
        semantic_reg_maps=None,
    ):
        """
        Randomly sample pixels from rendered semantic feature maps and get CLIP embeddings for patches
        centered at these pixels at random scales.
        
        Args:
            semantic_maps: Tensor of semantic feature maps [batch, views, channels, height, width]
            rgb_maps: Tensor of RGB maps [batch, views, 3, height, width] 
            num_samples: Number of random pixels to sample
            min_scale: Minimum scale factor (relative to image size)
            max_scale: Maximum scale factor (relative to image size)
            seed: Random seed for reproducibility
            extract_dino: Whether to extract DINO feature embeddings
            
        Returns:
            Dictionary containing stacked tensors of sampled features and embeddings
        """
        if seed is not None:
            torch.manual_seed(seed)
            
        # Get dimensions
        batch, views, channels, height, width = semantic_maps.shape
        
        # Extract DINO features for all images at once if requested and model is available
        dino_feature_maps = None
        if extract_dino and hasattr(self, 'dino_model') and self.dino_model is not None:
            # Reshape RGB maps to [batch*views, 3, height, width] for batch processing
            rgb_flat = rgb_maps.reshape(batch * views, 3, height, width)
            dino_h0 = height // reg_patch_size * 14 # dinov2 uses patch size of 14
            dino_w0 = width // reg_patch_size * 14
            # Resize images for DINO model
            rgb_resized = torch.nn.functional.interpolate(
                rgb_flat, size=(dino_h0, dino_w0), mode='bilinear', align_corners=False
            )
            with torch.no_grad():
                dino_feature_maps = forward_2d_model_batch(rgb_resized, self.dino_model)
                # Reshape back to [batch, views, feature_dim, feature_height, feature_width]
                _, dino_c, dino_h, dino_w = dino_feature_maps.shape
                dino_feature_maps = dino_feature_maps.reshape(batch, views, dino_c, dino_h, dino_w)
                
                # Interpolate DINO features to match semantic maps size
                dino_feature_maps = torch.nn.functional.interpolate(
                    dino_feature_maps.flatten(0, 1),  # [batch*views, dino_c, dino_h, dino_w]
                    size=(height, width),
                    mode='bilinear',
                    align_corners=False
                ).reshape(batch, views, dino_c, height, width)
                
        # Initialize output containers
        all_center_features = []
        all_center_features_reg = []
        all_rgb_patches = []
        all_batch_indices = []
        all_dino_features = [] if extract_dino else None
        
        # For each batch
        for b in range(batch):
            batch_rgb_patches = []
            batch_center_features = []
            if extract_dino and semantic_reg_maps is not None:
                batch_center_features_reg = []
            batch_pixel_positions = []
            batch_scales = []
            batch_view_indices = []
            batch_dino_features = [] if extract_dino else None
            
            # For each view
            for v in range(views):
                view_center_features = []
                view_center_features_reg = [] if extract_dino and semantic_reg_maps is not None else None
                view_rgb_patches = []
                view_pixel_positions = []
                view_scales = []
                view_dino_features = [] if extract_dino else None
                
                # Generate random pixel coordinates
                rand_pixels = []
                for _ in range(num_samples):
                    # Random x, y positions
                    y = torch.randint(0, height, (1,)).item()
                    x = torch.randint(0, width, (1,)).item()
                    
                    # Log-uniform sampling between min_scale and max_scale
                    log_min = np.log(min_scale)
                    log_max = np.log(max_scale)
                    log_scale = log_min + torch.rand(1).item() * (log_max - log_min)
                    scale = np.exp(log_scale)
                    
                    # Calculate patch size based on scale
                    patch_size = max(1, int(min(height, width) * scale))
                    
                    # Calculate patch boundaries with clamping to image boundaries
                    half_size = patch_size // 2
                    y_min = max(0, y - half_size)
                    y_max = min(height, y + half_size)
                    x_min = max(0, x - half_size)
                    x_max = min(width, x + half_size)
                    
                    rand_pixels.append((y, x, scale, y_min, y_max, x_min, x_max))
                
                # Extract patches for this view
                for y, x, scale, y_min, y_max, x_min, x_max in rand_pixels:
                    # Extract features at the center point
                    center_feature = semantic_maps[b, v, :, y, x]
                    # center_feature = center_feature / center_feature.norm(dim=-1, keepdim=True)
                    view_center_features.append(center_feature)
                    # Extract regularized semantic features if available
                    if extract_dino and semantic_reg_maps is not None:
                        center_feature_reg = semantic_reg_maps[b, v, :, y, x]
                        view_center_features_reg.append(center_feature_reg)
                    
                    # Extract DINO features at the center point if available
                    if extract_dino and dino_feature_maps is not None:
                        # Directly get DINO feature at center pixel using same coordinates
                        dino_feature = dino_feature_maps[b, v, :, y, x]
                        view_dino_features.append(dino_feature)
                    
                    # Sample RGB for the same patch
                    rgb_patch = rgb_maps[b, v, :, y_min:y_max, x_min:x_max]
                    
                    rgb_patch_resized = self.clip_process(rgb_patch).half()
                    
                    # Store resized patch for later batch processing with CLIP
                    view_rgb_patches.append(rgb_patch_resized)
                    view_pixel_positions.append((y, x))
                    view_scales.append(scale)
                
                # Add view data to batch containers
                batch_center_features.append(torch.stack(view_center_features))
                if extract_dino and view_center_features_reg:
                    batch_center_features_reg.append(torch.stack(view_center_features_reg))
                batch_rgb_patches.extend(view_rgb_patches)
                batch_pixel_positions.extend(view_pixel_positions)
                batch_scales.extend(view_scales)
                batch_view_indices.extend([v] * len(view_rgb_patches))
                if extract_dino and view_dino_features:
                    batch_dino_features.append(torch.stack(view_dino_features))
            
            # Stack view features for this batch
            all_center_features.append(torch.stack(batch_center_features))  # [views, num_samples, channels]
            if extract_dino and batch_center_features_reg:
                all_center_features_reg.append(torch.stack(batch_center_features_reg)) # [views, num_samples, reg_channels]
            if extract_dino and batch_dino_features:
                all_dino_features.append(torch.stack(batch_dino_features))  # [views, num_samples, dino_dim]
            
            # Process all RGB patches for this batch through CLIP at once
            if hasattr(self, 'clip_model') and self.clip_model is not None and batch_rgb_patches:
                # Stack all patches for batch inference
                stacked_rgb_patches = torch.stack(batch_rgb_patches)  # [views*num_samples, 3, 224, 224]
                
                with torch.no_grad():
                    batch_clip_embeddings = self.clip_model.encode_image(stacked_rgb_patches)
                    # batch_clip_embeddings /= batch_clip_embeddings.norm(dim=-1, keepdim=True)
                
                # Reshape to match the views and samples dimensions
                batch_clip_embeddings = batch_clip_embeddings.reshape(views, num_samples, -1)
                all_rgb_patches.append(stacked_rgb_patches.reshape(views, num_samples, 3, 224, 224))
            else:
                batch_clip_embeddings = None
                if batch_rgb_patches:
                    all_rgb_patches.append(torch.stack(batch_rgb_patches).reshape(views, num_samples, 3, 224, 224))
            
            # Store batch clip embeddings
            if batch_clip_embeddings is not None:
                all_batch_indices.append(torch.full((views * num_samples,), b, dtype=torch.long))
            
        # Create final return dictionary with stacked tensors
        result = {}
        
        # Stack center features across batches: [batch, views, num_samples, channels]
        if all_center_features:
            result["center_features"] = torch.stack(all_center_features)
        
        # Stack regularized center features if available: [batch, views, num_samples, reg_channels]
        if extract_dino and all_center_features_reg is not None:
            result["center_features_reg"] = torch.stack(all_center_features_reg)
        
        # Stack DINO features if extracted: [batch, views, num_samples, dino_dim]
        if extract_dino and all_dino_features:
            result["dino_embeddings"] = torch.stack(all_dino_features)
        
        # Stack RGB patches: [batch, views, num_samples, 3, 224, 224]
        if all_rgb_patches:
            result["rgb_patches"] = torch.stack(all_rgb_patches)
        
        # Stack CLIP embeddings if available: [batch, views, num_samples, embedding_dim]
        if hasattr(self, 'clip_model') and self.clip_model is not None and 'batch_clip_embeddings' in locals() and batch_clip_embeddings is not None:
            clip_embeddings = []
            for b in range(batch):
                # Get all embeddings for this batch
                clip_embeddings.append(batch_clip_embeddings)
            
            if clip_embeddings:
                result["clip_embeddings"] = torch.stack(clip_embeddings) # [batch, views, num_samples, embedding_dim]
        
        return result
