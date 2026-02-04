# Adapted from https://github.com/ywyue/FiT3D/blob/main/utils/model_utils.py

import numpy as np
import timm
import torch
import types
import math

from PIL import Image
from sklearn.decomposition import PCA
from torchvision import transforms
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import Normalize


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


timm_model_card = {
    "dinov2_small": "vit_small_patch14_dinov2.lvd142m",
    "dinov2_base": "vit_base_patch14_dinov2.lvd142m",
    "dinov2_reg_small": "vit_small_patch14_reg4_dinov2.lvd142m",
    "clip_base": "vit_base_patch16_clip_384.laion2b_ft_in12k_in1k",
    "mae_base": "vit_base_patch16_224.mae",
    "deit3_base": "deit3_base_patch16_224.fb_in1k"
}

def viz_feat(feat, file_path):

    _, _, h, w = feat.shape
    feat = feat.squeeze(0).permute((1,2,0))
    projected_featmap = feat.reshape(-1, feat.shape[-1]).cpu()

    pca = PCA(n_components=3)
    pca.fit(projected_featmap)
    pca_features = pca.transform(projected_featmap)
    pca_features = (pca_features - pca_features.min()) / (pca_features.max() - pca_features.min())
    pca_features = pca_features * 255
    res_pred = Image.fromarray(pca_features.reshape(h, w, 3).astype(np.uint8))
    res_pred.save(file_path)
    print("... saved to: ", file_path)

def _custom_get_intermediate_layers(
    self,
    x: torch.Tensor,
    n=1,
    reshape: bool = False,
    return_prefix_tokens: bool = False,
    norm: bool = True,
):
    """
    - Call self._orig_get_intermediate_layers(x, n) to get a list `outputs` of length n.
    - Each element out ∈ outputs has shape [B_local, 1 + num_prefix_tokens + num_patches, D_embed].
    - We normalize (if requested), strip off prefix tokens (CLS), then optionally reshape
      into a [B_local, D_embed, h_feat, w_feat] feature map.
    """
    outputs = self._orig_get_intermediate_layers(x, n, 
                                                 return_prefix_tokens=return_prefix_tokens,
                                                 norm=norm)

    # each out is [B_local, num_patches, D_embed]

    if reshape:
        reshaped_list = []
        for out in outputs:
            # out.shape == (B_local, N_tokens, D_embed)
            B_local, N_tokens, D = out.shape

            # We assume a square grid of patches (h_feat × w_feat = N_tokens)
            h_feat = int(math.sqrt(N_tokens))
            w_feat = h_feat
            if h_feat * w_feat != N_tokens:
                # Fallback: try deriving h_feat from the actual image H and patch_size
                H_img, W_img = x.shape[2], x.shape[3]
                p = self.patch_embed.patch_size[0]
                s = self.patch_embed.proj.stride[0]
                h_feat2 = (H_img - p) // s + 1
                w_feat2 = (W_img - p) // s + 1
                if h_feat2 * w_feat2 == N_tokens:
                    h_feat, w_feat = h_feat2, w_feat2
                else:
                    raise RuntimeError(
                        f"Cannot reshape {N_tokens} tokens into a grid. "
                        f"Calculated sqrt(N_tokens)={h_feat} → {h_feat}x{h_feat} != {N_tokens}, "
                        f"nor (H-p)//s+1 = {h_feat2} x {(W_img-p)//s+1} = {h_feat2*(W_img-p)//s+1}."
                    )

            # [B_local, D_embed, h_feat, w_feat]
            out_feat = (
                out
                .reshape(B_local, h_feat, w_feat, D)  # [B_local, h_feat, w_feat, D_embed]
                .permute(0, 3, 1, 2)                  # → [B_local, D_embed, h_feat, w_feat]
                .contiguous()
            )
            reshaped_list.append(out_feat)

        return tuple(reshaped_list)

    return tuple(outputs)

def build_2d_model(model_name="dinov2_small"):

    assert model_name in timm_model_card.keys(), "invalid model name"
    model = timm.create_model(
        timm_model_card[model_name],
        pretrained=True,
        num_classes=0,
        dynamic_img_size=True,
        dynamic_img_pad=False,
    )
    model.eval()

    model._orig_get_intermediate_layers = model.get_intermediate_layers
    model.get_intermediate_layers = types.MethodType(_custom_get_intermediate_layers, model)

    return model

def forward_2d_model_batch(images, feature_extractor):

    B, _, height, width = images.shape
    device = images.device
    
    stride = feature_extractor.patch_embed.patch_size[0]
    width_int = (width // stride)*stride
    height_int = (height // stride)*stride

    resized = torch.nn.functional.interpolate(images, size=(height_int, width_int), mode='bilinear')
    normalize = transforms.Normalize(mean=list(MEAN), std=list(STD))
    normalized = normalize(resized)
    batch = normalized.to(device)

    featmap = feature_extractor.get_intermediate_layers(
        batch,
        n=[len(feature_extractor.blocks)-1],
        reshape=True,
        return_prefix_tokens=False,
        norm=True,
    )[-1]

    return featmap

@torch.no_grad()
def load_repa_encoder(enc_name, device, resolution=256):
    
    encoder_type, architecture, model_config = enc_name.split('-')

    if 'dinov2' in encoder_type:
        if 'reg' in encoder_type:
            encoder = torch.hub.load('facebookresearch/dinov2', f'dinov2_vit{model_config}14_reg')
        else:
            encoder = torch.hub.load('facebookresearch/dinov2', f'dinov2_vit{model_config}14')
        del encoder.head
        patch_resolution = 16 * (resolution // 256)
        encoder.pos_embed.data = timm.layers.pos_embed.resample_abs_pos_embed(
            encoder.pos_embed.data, [patch_resolution, patch_resolution],
        )
        encoder.head = torch.nn.Identity()
        encoder = encoder.to(device)
        encoder.eval()
        
    return encoder, encoder_type, architecture

def preprocess_raw_image(x, enc_type, resolution=256):
    if 'dinov2' in enc_type:
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')
    return x