import sys
import os 
import torch
import torch.nn as nn
import torch.nn.functional as F
from .vggt import VGGT
from einops import rearrange
from torch import Tensor
from .alignment_projector import ConvProjector

def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))


class VGGTAlignmentLoss(nn.Module):
    def __init__(self,
                 alignment_context_length: int = 16,
                 apply_unnormalize_recon: bool = False,
                 unormalize_lambda: float =  1, # lambda for unormalize recon loss
                 latents_info: list = None, # the shape of generative model latents
                 encoder_info: list = [2048, 37, 37], # (D, H, W) of VGGT patch-token grid for 518x518 input, patch_size 14
                 mid_channels: int = 128,
                 vggt_layer_index=None, # -1 means last layer, -2 means second last layer, etc.
                ):
        super().__init__()
        self.alignment_context_length = alignment_context_length
        self.unormalize_lambda = unormalize_lambda 
        # === 1. 初始化冻结的 VGGT 模型 ===
        self.vggt_model = VGGT.from_pretrained("facebook/VGGT-1B")
        self.vggt_model.eval()
        for p in self.vggt_model.parameters():
            p.requires_grad = False
        # import pdb; pdb.set_trace()
        # === 2. 构建连接器 ===
        # for realestate 10k uvit setting: totally 7 blocks 
        if latents_info is None:
            latents_info = [
                [576, 32, 32],
                [256, 64, 64],
                [128, 128, 128]
            ]
        elif latents_info ==-1 or latents_info ==0 :
            latents_info = [
                [128, 128, 128]
            ]
        elif latents_info == -2 or latents_info == 1: 
            # 倒数第二
            latents_info = [
                [256, 64, 64]
            ]
        elif latents_info == -3 or latents_info == 2:
            latents_info = [
               [576, 32, 32]
            ]
        elif latents_info == -4 or latents_info==3:
            latents_info = [
                [1152, 16, 16],
            ]
        elif latents_info == "dit_128":
            latents_info = [
                [384, 128, 128],
            ]
        elif latents_info == "dit_64":
            latents_info = [
                [384, 64, 64],
            ]
        self.latents_info = latents_info
        self.encoder_info = encoder_info
        self.mid_channels = mid_channels
        self.projectors = nn.ModuleList()
        out_channels, out_h, out_w = self.encoder_info  # (2048, 37, 37) by default
        for c, h, w in self.latents_info:
            projector = ConvProjector(
                in_channels=c,
                in_h=h,
                in_w=w,
                out_h=out_h,
                out_w=out_w,
                mid_channels=mid_channels,
                out_channels=out_channels,
            )
            self.projectors.append(projector)
        ## this module aims to recon normalized latent feature to unnormalized feature
        self.apply_unnormalize_recon = apply_unnormalize_recon
        if self.apply_unnormalize_recon:
            self.feature_unormalizer = nn.Sequential(
                nn.Conv2d(out_channels, mid_channels, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(mid_channels, out_channels, kernel_size=1)
            )
    def load_projector_from_ckpt(self,ckpt_path):
        ckpt = torch.load(ckpt_path,map_location='cpu',weights_only=False)
        new_state_dict = {k.replace("alignment_loss.",""):v for k,v in ckpt['state_dict'].items()}
        missing, unexpected = self.load_state_dict(new_state_dict,strict=False)
        # filter missing and unexpected keys 
        # 1. vggt is loaded from pretrained model
        missing = [k for k in missing if not k.startswith("vggt_model.")]
        # 2. model is saved in ckpt but not in here 
        unexpected = [k for k in unexpected if not k.startswith("diffusion_model.")]
        print(f"Load projector from {ckpt_path} successfully. with  missing keys: {missing}, unexpected keys: {unexpected}")
        
    def latent_to_aggregated_token(self,latent, layer_index=-1):
        """
        Args:
            latent: list[Tensor], shape [BxT,C,H,W]
        Returns:
            aggregated_tokens_lists: list[Tensor], shape [B,T,1374,2048]
        """
        # assert self.apply_unnormalize_recon is True, "Please set apply_unnormalize_recon to True to enable unnormalize operation" 
        aggregated_tokens_lists = []
        projector = self.projectors[layer_index] 
        if latent.dim() == 5:
            b, t, c, h, w = latent.shape
            latent = rearrange(latent, 'b t c h w -> (b t) c h w')
        else:
            b, c, h, w = latent.shape
            t = 16
            latent = latent
        projected_latent = projector(latent)  # [BT, c, h , w , ]
        ph =  projected_latent.shape[2]
        # projected_latent = rearrange(projected_latent, '(b t) c h w -> b t c h w', t = self.temporal_length)
        projected_latent_flat = rearrange(projected_latent, 'b c h w -> b c (h w)')  # [B, C, H*W]
        projected_latent_flat_norm = F.normalize(projected_latent_flat, p=2, dim=-1)  # [B, C, H*W]
        projected_latent_norm = rearrange(projected_latent_flat_norm, 'b c (h w) -> b c h w', h=ph)  # [B, C, H, W]
        # unnormalize 
        if self.apply_unnormalize_recon:
            print(f"[external.vggt_alignment_loss][VGGTAlignmentLoss] Applying unnormalize recon operation.")
            pred_gt_latent = self.feature_unormalizer(projected_latent_norm)  # [B,T,24,512,512] 
        else:
            print(f"[external.vggt_alignment_loss][VGGTAlignmentLoss] without unnormalize recon operation.")
            pred_gt_latent = projected_latent_norm
        # uninterpolate from 512 512 -> 1374 2048
        pred_gt_latent = F.interpolate(pred_gt_latent, size=(1374, 2048), mode='bilinear', align_corners=False)
        pred_gt_latent = rearrange(pred_gt_latent, '(b t) c h w -> b t c h w', h=1374,t=t)  # [B, C, H, W]
        aggregated_tokens_list = [pred_gt_latent[:,:,i,:,:] for i in range(pred_gt_latent.shape[2])] # each item is [B,T,1374,2048] 
        aggregated_tokens_lists.append(aggregated_tokens_list)
        return aggregated_tokens_lists 
    
    def forward_vggt_prediction(self, images,latent_list,layer_index=-1,return_original_predictions=False): 
        """
        Args:
            images: original images, shape [B, T, C, H, W]
            latent_list: list[Tensor], shape [B,T,24,1374,2048]  #
        intermidiate output:
            aggregated_tokens_list: List[B,N,1374,2048],len(latents)=24
        Returns:
            point_cloud 
        """
        images = self.vggt_processor(images)  # [B, T, C, H, W]
        latent = latent_list[-1]
        # import pdb; pdb.set_trace()
        aggregated_tokens_lists = self.latent_to_aggregated_token(latent,layer_index)  # list[Tensor], shape [B,T,1374,2048]
        # print(f"[external.vggt_alignment_loss][VGGTAlignmentLoss][forward_vggt_prediction] Default Using Last Layer Output")
        aggregated_tokens_list = aggregated_tokens_lists[0]
        predictions,original_predictions = self.vggt_model.forward_with_external_feature(
            images,
            aggregated_tokens_list=aggregated_tokens_list,
            return_original_predictions=return_original_predictions
        )
        return predictions ,original_predictions
        
    def vggt_processor(self, images: Tensor): 
        # images: b t c h w 
        b = images.shape[0] 
        images = rearrange(images, 'b f c h w -> (b f) c h w')
        # print(f"[external.vggt_alignment_loss][VGGTAlignmentLoss][vggt_processor] using imagesize 224x224")
        # using a  224x224 will cause drop from 317->327(fvd)
        images_resized = F.interpolate(images, size=(518, 518), mode="bilinear", align_corners=False)
        images_resized = rearrange(images_resized, '(b f) c h w -> b f c h w', b=b)
        images_resized = torch.clamp(images_resized, 0.0, 1.0)
        return images_resized
    
    def forward(self, latents, images, return_depth=False):
        """REPA-style cosine alignment between UViT latents and VGGT patch-token features.

        Args:
            latents: list[Tensor], each [B, T, C, H, W] from UViT levels
            images: [B, T_total, 3, H_in, W_in] in [0, 1]
            return_depth: if True, also return VGGT's predicted depth + confidence.
        Returns:
            alignment_loss (scalar in [-1, 1]); optionally (loss, vggt_depth, vggt_conf)
        """
        images = images[:, :self.alignment_context_length, :, :, :]
        images = self.vggt_processor(images)  # [B, T, 3, 518, 518]

        with torch.no_grad():
            aggregated_tokens_list, patch_start_idx = self.vggt_model.shortcut_forward(images)
            if return_depth:
                # depth: [B, T, 1, H, W]; conf: [B, T, H, W]; H=W=518.
                vggt_depth, vggt_conf = self.vggt_model.depth_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
            # Use the last transformer layer only.
            tokens = aggregated_tokens_list[-1]  # [B, T, P, D=2048]
            # Drop special tokens (camera + register) which are not spatial.
            patch_tokens = tokens[:, :, patch_start_idx:, :]  # [B, T, P_patch, D]
            n_patches = patch_tokens.shape[2]
            grid = int(round(n_patches ** 0.5))
            assert grid * grid == n_patches, (
                f"VGGT patch count {n_patches} is not a perfect square; "
                f"check patch_start_idx={patch_start_idx} and image size."
            )
            # Reshape to a 2D spatial grid: [BT, D, grid, grid].
            target_feat = rearrange(
                patch_tokens, 'b t (h w) d -> (b t) d h w', h=grid, w=grid
            )

        alignment_loss = 0.0
        unormalize_loss = 0.0
        assert len(latents) == len(self.projectors), (
            f"latents length {len(latents)} should match projectors length {len(self.projectors)}"
        )

        for latent, projector in zip(latents, self.projectors):
            latent = latent[:, :self.alignment_context_length, ...]
            latent = rearrange(latent, 'b t c h w -> (b t) c h w')
            latent_proj = projector(latent)  # [BT, D, grid, grid]

            # Cosine similarity along the feature dim, averaged over spatial positions.
            l = F.normalize(latent_proj, p=2, dim=1)
            t = F.normalize(target_feat, p=2, dim=1)
            cos_per_pos = (l * t).sum(dim=1)  # [BT, grid, grid] in [-1, 1]
            alignment_loss = alignment_loss + (-cos_per_pos.mean())

            if self.apply_unnormalize_recon:
                unormalized_latent_proj = self.feature_unormalizer(l)
                unormalize_loss = unormalize_loss + F.mse_loss(unormalized_latent_proj, target_feat)

        alignment_loss = alignment_loss / len(latents)
        if self.apply_unnormalize_recon:
            unormalize_loss = unormalize_loss / len(latents)
            loss = alignment_loss + self.unormalize_lambda * unormalize_loss
        else:
            loss = alignment_loss

        if return_depth:
            return loss, vggt_depth, vggt_conf
        return loss
