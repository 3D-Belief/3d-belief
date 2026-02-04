import functools
from dataclasses import dataclass
from typing import Literal, Tuple, Optional, Union

import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from diffusers.models.attention_processor import Attention as DiffAttention
from ..common.attention import attn_forward, LinearAttention
from ..common.mlp import GLUMBConv
from ..common.harmonic_embedding import HarmonicEmbedding
import torch.nn.functional as F
from .backbone import Backbone


@dataclass
class BackboneDiTCfg:
    name: Literal["dit"] = "dit"
    model: Literal[
        "DiT-XL/2", "DiT-XL/4", "DiT-XL/8",
        "DiT-L/2", "DiT-L/4", "DiT-L/8",
        "DiT-B/2", "DiT-B/4", "DiT-B/8",
        "DiT-S/2", "DiT-S/4", "DiT-S/8"
    ] = "DiT-XL/2"
    input_size: Tuple[int, int] = (64, 64)
    in_channels: int = 64
    out_channels: int = 512
    class_dropout_prob: float = 0.1
    num_classes: int = 1000
    cond_feats_dim: int = 0
    mlp_ratio: float = 4.0
    learn_sigma: bool = True
    d_out: int = 512
    use_image_condition: bool = False
    use_camera_pose: bool = False
    pose_condition_type: Literal["mlp", "prope"] = "prope"
    union_cond_attn: bool = True
    use_diff_pos_embed: bool = True
    use_cond_res_mlp: bool = False
    use_repa: bool = False
    repa_z_dim: int = 768
    view_attn_n_heads: int = 4
    view_attn_n_layers: int = 1
    view_attn_ff_mult: int = 4
    view_attn_pool_type: str = "attn"
    attn_type: Literal['mma', 'mmla'] = 'mma' # 'mma' for DiffAttention, 'mmla' for LinearAttention
    mlp_type: Literal['mlp', 'sana_mlp'] = 'mlp'  # 'sana_mlp' for GLUMBConv

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(
            num_classes + use_cfg_embedding, hidden_size
        )
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = (
                torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
            )
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core DiT Model                                #
#################################################################################


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(
            self, hidden_size, num_heads, mlp_ratio=4.0, 
            img_cond=False, union_cond_attn=True, 
            attn_type='mma', # attn_type: 'mma', 'mmla'
            mlp_type='mlp', # mlp_type: 'mlp', 'sana_mlp'
            **block_kwargs
        ):
        super().__init__()
        self.img_cond = img_cond
        self.union_cond_attn = union_cond_attn
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs
        )
        if self.img_cond: 
            if attn_type=='mma':
                self.attn = DiffAttention(
                    query_dim=hidden_size,
                    cross_attention_dim=hidden_size,  # the same dim for conditioning tokens
                    heads=num_heads,
                    dim_head=hidden_size // num_heads,
                    dropout=0.0,
                    bias=True,
                )
            elif attn_type=='mmla':
                self.attn = LinearAttention(
                in_channels=hidden_size,
                out_channels=hidden_size,
                num_attention_heads=num_heads,
                attention_head_dim=hidden_size // num_heads,
                mult=1.0,
                norm_type="layer_norm",
                kernel_sizes=(5,),
                eps=1e-15,
                residual_connection=False,
            )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        if mlp_type=='mlp':
            self.mlp = Mlp(
                in_features=hidden_size,
                hidden_features=mlp_hidden_dim,
                act_layer=approx_gelu,
                drop=0,
            )
        elif mlp_type=='sana_mlp':
            self.mlp = GLUMBConv(
                in_features=hidden_size,
                hidden_features=mlp_hidden_dim,
                use_bias=(True, True, False),
                norm=(None, None, None),
                act=("silu", "silu", None)
            )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, cond_tokens: Optional[torch.Tensor] = None, cond_c: Optional[torch.Tensor] = None,
                 viewmats: Optional[torch.Tensor] = None, Ks: Optional[torch.Tensor] = None,
                 patches_x: Optional[float] = None, patches_y: Optional[float] = None):
        """
        x: (B, T_main, D) main tokens.
        c: (B, D) conditioning vector.
        cond_tokens: (B, T_cond, D) image condition tokens (optional).
        cond_c: (B, D) conditioning vector for img_cond (optional).
        """
        # Compute modulation parameters from conditioning vector.
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN_modulation(c).chunk(6, dim=1)

        if self.img_cond and cond_tokens is not None:
            (
                cond_shift_msa,
                cond_scale_msa,
                cond_gate_msa,
                cond_shift_mlp,
                cond_scale_mlp,
                cond_gate_mlp,
            ) = self.adaLN_modulation(cond_c).chunk(6, dim=1) # same modulation as main
        
        if self.img_cond: 
            assert isinstance(self.attn, Union[DiffAttention, LinearAttention])
            if cond_tokens is not None:
                assert cond_c is not None
                # Normalize and modulate cond tokens.
                cond_tokens = modulate(self.norm1(cond_tokens), cond_shift_msa, cond_scale_msa)
            # Normalize and modulate main tokens.
            x = modulate(self.norm1(x), shift_msa, scale_msa)
            # Call attn_forward with condition_latents as cond_tokens.
            attn_out = attn_forward(
                self.attn, x, condition_latents=cond_tokens, 
                union_cond_attn=self.union_cond_attn,
                viewmats=viewmats, Ks=Ks,
                patches_x=patches_x, patches_y=patches_y
            )
            # If condition tokens are provided, attn_forward returns a tuple (main tokens, condition tokens).
            if cond_tokens is not None:
                x_main, x_cond = attn_out
                cond_tokens = cond_tokens + cond_gate_msa.unsqueeze(1) * x_cond
            else:
                x_main = attn_out
            # Residual connection.
            x = x + gate_msa.unsqueeze(1) * x_main
        else:
            assert isinstance(self.attn, Attention)
            assert cond_tokens is None
            # No image condition tokens are provided.
            x = x + gate_msa.unsqueeze(1) * self.attn(
                modulate(self.norm1(x), shift_msa, scale_msa)
            )
        # MLP branch.
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        if self.img_cond and cond_tokens is not None:
            cond_tokens = cond_tokens + cond_gate_mlp.unsqueeze(1) * self.mlp(
                modulate(self.norm2(cond_tokens), cond_shift_mlp, cond_scale_mlp)
            )
            result = (x, cond_tokens)
        else:
            result = x
        
        return result


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(Backbone[BackboneDiTCfg]):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(self, cfg: BackboneDiTCfg, d_in: int) -> None:
        super().__init__(cfg=cfg)
        assert d_in == 3, "Input d_in must be 3"
        self.learn_sigma = cfg.learn_sigma
        self.in_channels = cfg.in_channels
        self.out_channels = cfg.out_channels
        self.input_size = cfg.input_size
        self.use_image_condition = cfg.use_image_condition
        self.use_camera_pose = cfg.use_camera_pose
        self.pose_condition_type = cfg.pose_condition_type
        self.use_cond_res_mlp = cfg.use_cond_res_mlp
        self.union_cond_attn = cfg.union_cond_attn
        self.use_diff_pos_embed = cfg.use_diff_pos_embed
        self.attn_type = cfg.attn_type
        self.mlp_type = cfg.mlp_type
        self.use_repa = cfg.use_repa
        self.repa_z_dim = cfg.repa_z_dim

        # Lookup model-specific parameters using the constant.
        model_params = DI_T_MODEL_PARAMS[cfg.model]
        depth = model_params["depth"]
        hidden_size = model_params["hidden_size"]
        patch_size = model_params["patch_size"]
        num_heads = model_params["num_heads"]

        self.patch_size = patch_size
        self.num_heads = num_heads

        self.x_embedder = PatchEmbed(
            cfg.input_size, (patch_size, patch_size), cfg.in_channels, hidden_size
        )
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(cfg.num_classes, hidden_size, cfg.class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        
        if self.use_image_condition and self.use_diff_pos_embed:
            self.cond_pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        else:
            self.cond_pos_embed = None

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_size, num_heads, mlp_ratio=cfg.mlp_ratio, 
                      img_cond=self.use_image_condition, 
                      union_cond_attn=self.union_cond_attn,
                      attn_type=self.attn_type,
                      mlp_type=self.mlp_type) for _ in range(depth)]
        )
        if self.use_camera_pose and self.pose_condition_type == "mlp":
            self.pose_embed = HarmonicEmbedding()
            self.pose_mlp = nn.Sequential(
                nn.Linear(156, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
            )
            self.fuse_mlp = nn.Sequential(
                nn.Linear(2*hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
            )
            self.fuse_norm = nn.LayerNorm(hidden_size)
            if self.use_cond_res_mlp:
                self.cond_res_mlp = nn.Sequential(
                    nn.Linear(2*hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, hidden_size),
                )

        if self.use_repa:
            self.repa_mlp = nn.Sequential(
                nn.Linear(hidden_size, 2*hidden_size),
                nn.SiLU(),
                nn.Linear(2*hidden_size, 2*hidden_size),
                nn.SiLU(),
                nn.Linear(2*hidden_size, self.repa_z_dim),
            )

        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            self.input_size[0] // self.patch_size,
            self.input_size[1] // self.patch_size,
        )

        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # If using a different positional embedding for conditioning, initialize it too.
        if self.cond_pos_embed is not None:
            print("Use diff pos embed in DiT")
            cond_pos_embed = get_2d_sincos_pos_embed(
                self.cond_pos_embed.shape[-1],
                self.input_size[0] // self.patch_size,
                self.input_size[1] // self.patch_size,
                offset_h=1.0,
                offset_w=1.0,
            )
            self.cond_pos_embed.data.copy_(torch.from_numpy(cond_pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        from collections import OrderedDict

        """
        pretrained = torch.load("/home/ubuntu/jhu-scai-lab/yyin34/DFM/data/DiT-XL-2-256x256.pt", map_location='cpu')
        import ipdb; ipdb.set_trace()
        
        new_state_dict = OrderedDict()
        for k, v in pretrained.items():
            if (
                "conv1" not in k
                and "x_embedder.proj.weight" not in k
                and "final_layer" not in k
            ):
                new_state_dict[k] = v
        # self.load_state_dict(new_state_dict, strict=False)
        print("Loaded pretrained DiT weights")
        # """
        print(f"NOT LOADING DIT WEIGHTS")

    @property
    def d_out(self) -> int:
        return self.cfg.d_out

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        # h = w = int(x.shape[1] ** 0.5)
        h = self.input_size[0] // p
        w = self.input_size[1] // p
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def forward(self, x, t=None, y=None, pose=None, lang=None, intrinsics=None,
                cond_img: Optional[torch.Tensor]=None, cond_pose: Optional[torch.Tensor]=None,
                return_cond_tokens: Optional[bool]=False):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        pose: (N, 4, 4) tensor of camera poses
        intrinsics: (N, 3, 3) tensor of camera intrinsics
        cond_img: (N, C, H, W) conditioning image
        cond_pose: (N, 4, 4) conditioning camera pose
        """

        if t is None:
            t = torch.zeros(x.shape[0], device=x.device, dtype=torch.long)
        if y is None:
            y = torch.zeros(x.shape[0], device=x.device, dtype=torch.long)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.x_embedder(x) + self.pos_embed # (N, T, D)
        B, seq_len, D = x.shape
        # print("x shape seq_len", seq_len)

        t = self.t_embedder(t)  # (N, D)
        y = self.y_embedder(y, self.training)  # (N, D)
        c = t + y  # (N, D)

        if self.use_camera_pose and self.pose_condition_type == "mlp":
            assert pose is not None
            pose = pose.reshape(-1, 16)[:, :-4] # last row is always [0, 0, 0, 1]
            pose = self.pose_mlp(self.pose_embed(pose)).unsqueeze(1)
            # Fuse to x
            x = x.view(B, 1, seq_len, D)
            pose = pose.unsqueeze(2).expand(B, 1, seq_len, D) # [B, 1, 1, D] -> [B, 1, seq_len, D]
            x = torch.cat([x, pose], dim=-1) # [B, 1, seq_len, 2*D]
            x = self.fuse_mlp(x)  # [B, 1, seq_len, D]
            x = self.fuse_norm(x)
            x = x.contiguous().view(B, seq_len, D) # [B, seq_len, D]
        
        # Process conditioning image if enabled.
        cond_tokens = None
        cond_c = None
        if self.use_image_condition and cond_img is not None:
            cond_img = self.conv1(cond_img) # Same encoder.
            cond_img = self.relu(cond_img)
            # Use separate positional embedding if available.
            if self.cond_pos_embed is not None:
                cond_tokens = self.x_embedder(cond_img) + self.cond_pos_embed  # (N, T_cond, D)
            else:
                cond_tokens = self.x_embedder(cond_img) + self.pos_embed  # (N, T_cond, D)
            # Rearrange
            n_cond = cond_tokens.shape[0]
            cond_tokens = rearrange(cond_tokens, "(b v) c d -> b v c d", b=B, v=n_cond//B)
            cond_tokens = rearrange(cond_tokens, "b v c d -> b (v c) d")

            cond_t = torch.zeros(cond_tokens.shape[0], device=cond_tokens.device, dtype=torch.long)
            cond_y = torch.zeros(cond_tokens.shape[0], device=cond_tokens.device, dtype=torch.long)
            cond_t = self.t_embedder(cond_t)  # (N, D)
            cond_y = self.y_embedder(cond_y, self.training)  # (N, D)
            cond_c = cond_t + cond_y  # (N, D)
            total_len = cond_tokens.shape[1]
            n = total_len // seq_len
            if self.use_camera_pose:
                assert pose is not None and cond_pose is not None
                if self.pose_condition_type == "mlp":
                    cond_pose = cond_pose.reshape(-1, 16)[:, :-4] # last row is always [0, 0, 0, 1]
                    cond_pose = self.pose_mlp(self.pose_embed(cond_pose))
                    n_cond_pose = cond_pose.shape[0]
                    cond_pose = rearrange(cond_pose, "(b v) d -> b v d", b=B, v=n_cond_pose//B)
                    # Fuse to cond_tokens
                    assert n_cond_pose//B==n
                    cond_tokens = cond_tokens.view(B, n, seq_len, D) # [B, n, seq_len, D]
                    cond_pose = cond_pose.unsqueeze(2).expand(B, n, seq_len, D) # [B, n, 1, D] -> [B, n, seq_len, D]
                    cond_tokens = torch.cat([cond_tokens, cond_pose], dim=-1) # [B, n, seq_len, 2*D]
                    cond_tokens = self.fuse_mlp(cond_tokens)  # → [B, n, seq_len, D]
                    cond_tokens = self.fuse_norm(cond_tokens)
                    cond_tokens = cond_tokens.contiguous().view(B, total_len, D)
                elif self.pose_condition_type == "prope":
                    assert intrinsics is not None, "Intrinsics must be provided for prope condition type"
                    viewmats = torch.cat([pose, cond_pose], dim=1)  # [N, 2, 4, 4]
                    Ks = intrinsics.expand(-1, 2, -1, -1)  # [N, 2, 3, 3]
                    image_width, image_height = self.input_size
                    patches_x, patches_y = image_width // self.patch_size, image_height // self.patch_size

        for block in self.blocks:
            if self.use_camera_pose and cond_img is not None and self.pose_condition_type == "prope":
                result = block(
                        x, c, 
                        cond_tokens=cond_tokens, 
                        cond_c=cond_c, 
                        viewmats=viewmats, 
                        Ks=Ks, 
                        patches_x=patches_x, 
                        patches_y=patches_y
                    )  # (N, T, D)
            else:
                result = block(x, c, cond_tokens=cond_tokens, cond_c=cond_c)  # (N, T, D)
            if self.use_image_condition and cond_img is not None:
                x, cond_tokens = result
            else:
                x = result
        batch, seq_len, dim = x.shape

        output = {}
        repa_pred = None
        if self.use_repa:
            repa_pred = self.repa_mlp(x.reshape(-1, dim)).reshape(batch, seq_len, -1)
            assert repa_pred.shape[-1] == self.repa_z_dim
            output['repa_pred'] = repa_pred

        x = self.final_layer(x, c)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        if self.use_image_condition and cond_img is not None and return_cond_tokens:
            assert (cond_tokens is not None) and (cond_c is not None)
            cond_tokens = cond_tokens[:, -seq_len:, :]
            cond_tokens = self.final_layer(cond_tokens, cond_c)  # (N, T, patch_size ** 2 * out_channels)
            cond_tokens = self.unpatchify(cond_tokens)
            output["pred"] = x
            output["cond_tokens"] = cond_tokens
        else:
            output["pred"] = x

        return output

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py


def get_2d_sincos_pos_embed(
        embed_dim, grid_size_h, grid_size_w, cls_token=False, extra_tokens=0, offset_h=0.0, offset_w=0.0, 
):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size_h, dtype=np.float32) + offset_h
    grid_w = np.arange(grid_size_w, dtype=np.float32) + offset_w
    grid = np.meshgrid(grid_w, grid_h)  # (W, H)
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size_h, grid_size_w])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate(
            [np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0
        )
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   DiT Configs                                  #
#################################################################################

DI_T_MODEL_PARAMS = {
    "DiT-XL/2": {"depth": 28, "hidden_size": 1152, "patch_size": 2, "num_heads": 16},
    "DiT-XL/4": {"depth": 28, "hidden_size": 1152, "patch_size": 4, "num_heads": 16},
    "DiT-XL/8": {"depth": 28, "hidden_size": 1152, "patch_size": 8, "num_heads": 16},
    "DiT-L/2":  {"depth": 24, "hidden_size": 1024, "patch_size": 2, "num_heads": 16},
    "DiT-L/4":  {"depth": 24, "hidden_size": 1024, "patch_size": 4, "num_heads": 16},
    "DiT-L/8":  {"depth": 24, "hidden_size": 1024, "patch_size": 8, "num_heads": 16},
    "DiT-B/2":  {"depth": 12, "hidden_size": 768,  "patch_size": 2, "num_heads": 12},
    "DiT-B/4":  {"depth": 12, "hidden_size": 768,  "patch_size": 4, "num_heads": 12},
    "DiT-B/8":  {"depth": 12, "hidden_size": 768,  "patch_size": 8, "num_heads": 12},
    "DiT-S/2":  {"depth": 12, "hidden_size": 384,  "patch_size": 2, "num_heads": 6},
    "DiT-S/4":  {"depth": 12, "hidden_size": 384,  "patch_size": 4, "num_heads": 6},
    "DiT-S/8":  {"depth": 12, "hidden_size": 384,  "patch_size": 8, "num_heads": 6},
}
