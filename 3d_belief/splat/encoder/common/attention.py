import torch
from typing import List, Union, Optional, Dict, Any, Callable, Tuple
from diffusers.models.attention_processor import Attention, F, SanaMultiscaleLinearAttention
from .prope import prope_dot_product_attention


class LinearAttention(SanaMultiscaleLinearAttention):
    r"""Lightweight multi-scale linear attention"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_attention_heads: Optional[int] = None,
        attention_head_dim: int = 8,
        mult: float = 1.0,
        norm_type: str = "batch_norm",
        kernel_sizes: Tuple[int, ...] = (5,),
        eps: float = 1e-15,
        residual_connection: bool = False,
    ):  
        self.heads = num_attention_heads
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            mult=mult,
            norm_type=norm_type,
            kernel_sizes=kernel_sizes,
            eps=eps,
            residual_connection=residual_connection,
        )

# Adapted from https://github.com/Yuanshi9815/OminiControl/blob/main/omini/pipeline/flux_omini.py
def attn_forward(
    attn: Union[Attention, LinearAttention],
    hidden_states: torch.FloatTensor,
    condition_latents: torch.FloatTensor = None,
    union_cond_attn: bool = True,
    attention_mask: Optional[torch.FloatTensor] = None,
    image_rotary_emb: Optional[torch.Tensor] = None,
    cond_rotary_emb: Optional[torch.Tensor] = None,
    viewmats: Optional[torch.Tensor] = None,
    Ks: Optional[torch.Tensor] = None,
    patches_x: Optional[float] = None,
    patches_y: Optional[float] = None
) -> torch.FloatTensor:
    batch_size, _, _ = hidden_states.shape

    # `sample` projections.
    query = attn.to_q(hidden_states)
    key = attn.to_k(hidden_states)
    value = attn.to_v(hidden_states)

    inner_dim = key.shape[-1]
    head_dim = inner_dim // attn.heads

    query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

    if hasattr(attn, "norm_q") and attn.norm_q is not None:
        query = attn.norm_q(query)
    if hasattr(attn, "norm_k") and attn.norm_k is not None:
        key = attn.norm_k(key)

    if image_rotary_emb is not None:
        from diffusers.models.embeddings import apply_rotary_emb
        query = apply_rotary_emb(query, image_rotary_emb)
        key = apply_rotary_emb(key, image_rotary_emb)

    if condition_latents is not None:
        cond_query = attn.to_q(condition_latents)
        cond_key = attn.to_k(condition_latents)
        cond_value = attn.to_v(condition_latents)

        cond_query = cond_query.view(batch_size, -1, attn.heads, head_dim).transpose(
            1, 2
        )
        cond_key = cond_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        cond_value = cond_value.view(batch_size, -1, attn.heads, head_dim).transpose(
            1, 2
        )
        if hasattr(attn, "norm_q") and attn.norm_q is not None:
            cond_query = attn.norm_q(cond_query)
        if hasattr(attn, "norm_k") and attn.norm_k is not None:
            cond_key = attn.norm_k(cond_key)
    
    if cond_rotary_emb is not None:
        cond_query = apply_rotary_emb(cond_query, cond_rotary_emb)
        cond_key = apply_rotary_emb(cond_key, cond_rotary_emb)

    if condition_latents is not None:
        query = torch.cat([query, cond_query], dim=2)
        key = torch.cat([key, cond_key], dim=2)
        value = torch.cat([value, cond_value], dim=2)
    
    attention_mask = torch.ones(
        query.shape[2], key.shape[2], device=query.device, dtype=torch.bool
    )

    if not union_cond_attn:
        # If we don't want to use the union condition attention, we need to mask the attention
        # between the hidden states and the condition latents
        if condition_latents is not None:
            condition_n = cond_query.shape[2]
            attention_mask[-condition_n:, :-condition_n] = False
            attention_mask[:-condition_n, -condition_n:] = False
    else:
        if condition_latents is not None:
            condition_n = cond_query.shape[2]
            attention_mask[-condition_n:, :-condition_n] = False
    if hasattr(attn, "c_factor"):
        attention_mask = torch.zeros(
            query.shape[2], key.shape[2], device=query.device, dtype=query.dtype
        )
        condition_n = cond_query.shape[2]
        bias = torch.log(attn.c_factor[0])
        attention_mask[-condition_n:, :-condition_n] = bias
        attention_mask[:-condition_n, -condition_n:] = bias
    if viewmats is not None and Ks is not None:
        # Use prope_dot_product_attention if viewmats and Ks are provided
        hidden_states = prope_dot_product_attention(
            query, key, value, viewmats=viewmats, Ks=Ks, patches_x=patches_x, patches_y=patches_y,
            image_width=1, image_height=1, # intrinsics already normalized
        )
    else:
        # Use the standard scaled dot product attention
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0, is_causal=False, attn_mask=attention_mask
        )
    hidden_states = hidden_states.transpose(1, 2).reshape(
        batch_size, -1, attn.heads * head_dim
    )
    hidden_states = hidden_states.to(query.dtype)

    if condition_latents is not None:
        # if there are condition_latents, we need to separate the hidden_states and the condition_latents
        hidden_states, condition_latents = (
            hidden_states[:, : -condition_latents.shape[1]],
            hidden_states[:, -condition_latents.shape[1] :],
        )
        return hidden_states, condition_latents
    else:
        return hidden_states
