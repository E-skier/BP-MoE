# Copyright (c) Meta Platforms, Inc. and affiliates.
import math
import abc
import json
import logging
import os
from pathlib import Path
from enum import Enum
from typing import Callable, Optional, Tuple, Union

import torch
import torch.distributed as dist
from pydantic import BaseModel, ConfigDict
from torch import nn
from torch.distributed.nn.functional import all_to_all_single
from torch.nn import functional as F

_ALLOW_MISSING_FLEX_ATTENTION = (
    int(os.environ.get("BLT_ALLOW_MISSING_FLEX_ATTENTION", False)) != 0
)

try:
    from torch.nn.attention.flex_attention import (
        BlockMask,
        _mask_mod_signature,
        flex_attention,
    )

    _FLEX_ATTENTION_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    if not _ALLOW_MISSING_FLEX_ATTENTION:
        raise

    class BlockMask:
        pass

    _mask_mod_signature = Callable
    _FLEX_ATTENTION_AVAILABLE = False

    def flex_attention(*args, **kwargs):
        raise RuntimeError(
            "torch.nn.attention.flex_attention is unavailable. Install a Torch "
            "build with flex_attention or use an SDPA attention path."
        )


try:
    from xformers.ops import AttentionBias, fmha
except ImportError:
    AttentionBias = object
    fmha = None

from bytelatent.initialization import trunc_normal_
from bytelatent.tokenizers.constants import EOS_ID

logger = logging.getLogger()

PATCH_BYTE_TYPE_NAMES = (
    "alpha",
    "digit",
    "whitespace",
    "punctuation",
    "non_ascii",
)

try:
    from apex.normalization.fused_layer_norm import FusedRMSNorm

    RMSNorm = FusedRMSNorm
except (ImportError, ModuleNotFoundError):
    logging.debug("Apex not found. Using nn.RMSNorm")
    RMSNorm = nn.RMSNorm

if not _FLEX_ATTENTION_AVAILABLE:
    logger.warning(
        "BLT_ALLOW_MISSING_FLEX_ATTENTION is set and flex attention is "
        "unavailable; flex_attention calls will raise unless an SDPA path is used."
    )
    flex_attention_comp = flex_attention
elif not _ALLOW_MISSING_FLEX_ATTENTION:
    flex_attention_comp = torch.compile(flex_attention)
else:
    logger.warning(
        "BLT_ALLOW_MISSING_FLEX_ATTENTION is set, but flex attention is available; "
        "using uncompiled flex_attention instead of disabling it."
    )
    flex_attention_comp = flex_attention


class InitStdFactor(str, Enum):
    DISABLED = "disabled"  # Init std is divided by 1.0
    GLOBAL_DEPTH = "global_depth"  # Init std is divided by sqrt(2*n_layers)
    CURRENT_DEPTH = "current_depth"  # Init std is divided by sqrt(2*depth)
    DIM_RATIO = "dim_ratio"  # Init std is divided by model_dim/4096


class BaseTransformerArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dim: int = 512
    n_layers: int = 8
    head_dim: int | None = None
    n_heads: int | None = None
    n_kv_heads: int | None = None

    ffn_dim_multiplier: float | None = None

    # Optional sparse FFN routing. Disabled by default.
    moe_num_experts: int = 0
    moe_top_k: int = 2
    moe_layer_frequency: int = 1
    moe_ep_size: int = 1
    moe_ffn_dim_multiplier: float | None = None
    moe_balance_loss_weight: float = 0.0
    moe_balance_schedule: str = "constant"
    moe_balance_start_step: int = 0
    moe_balance_peak_step: int = 0
    moe_balance_decay_start_step: int = 0
    moe_balance_end_step: int = 0
    moe_balance_final_weight: float = 0.0
    moe_router_jitter: float = 0.0
    moe_router_congestion_weight: float = 0.0
    moe_router_z_loss_weight: float = 0.0
    moe_router_anticollapse_loss_weight: float = 0.0
    moe_router_dominance_loss_weight: float = 0.0
    moe_router_dominance_threshold: float = 0.35
    moe_router_use_hidden_state: bool = True
    moe_router_patch_feature_bias: bool = False
    moe_router_normalize_patch_entropy: bool = True
    moe_router_entropy_mlp_hidden_dim: int = 0
    moe_router_entropy_mean: float = 0.0
    moe_router_entropy_std: float = 1.0
    moe_router_entropy_clip: float = 4.0
    moe_router_use_patch_length: bool = False
    moe_router_use_patch_entropy: bool = False
    moe_router_use_patch_byte_features: bool = False
    moe_routing_granularity: str = "expert"
    moe_pair_size: int = 2
    moe_entropy_prior_mode: str = "none"
    moe_entropy_prior_scale: float = 1.0
    moe_entropy_prior_hidden_scale: float = 1.0
    moe_entropy_prior_calibration_path: str | None = None
    moe_entropy_prior_trainable: bool = False
    moe_pair_bias_mode: str = "none"
    moe_pair_bias_ema: float = 0.95
    moe_pair_bias_update_interval: int = 20
    moe_pair_bias_lr: float = 0.05
    moe_pair_bias_clip: float = 1.5
    moe_pair_min_byte_fraction: float = 0.05
    moe_pair_max_byte_fraction: float = 0.55
    moe_hidden_residual_ramp_start_step: int = 0
    moe_hidden_residual_ramp_end_step: int = 0
    moe_hidden_residual_final_scale: float = 1.0
    moe_balance_cost: str = "patch"

    multiple_of: int = 256

    norm_eps: float = 1e-5

    rope_theta: float = 10000.0
    rope_use_fp32_in_outer_product: bool = False

    init_base_std: float | None = None
    init_std_factor: InitStdFactor = InitStdFactor.DISABLED

    max_seqlen: int = 1024

    attn_impl: str | None = "sdpa"
    attn_bias_type: str | None = None
    # Special token config
    eos_id: int | None = EOS_ID


def cross_entropy(pred, target, **kwargs):
    return F.nll_loss(
        F.log_softmax(pred.flatten(end_dim=-2).float(), -1),
        target.flatten(end_dim=-1),
        **kwargs,
    )


def repeat_kv(x: torch.Tensor, n_rep: int, dim: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    assert dim == 2, "Only dim=2 is supported. Check the implementation for other dims."
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


def precompute_freqs_cis(
    dim: int,
    end: int,
    theta: float = 10000.0,
    rope_use_fp32_in_outer_product: bool = False,
):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    if rope_use_fp32_in_outer_product:
        t = t.to(torch.float32)

    freqs = torch.outer(t, freqs).float()

    cos, sin = freqs.cos(), freqs.sin()

    return torch.stack((cos, -sin, sin, cos), dim=-1).view(*freqs.size(), 2, 2)


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor, seq_dim: int):
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.
        seq_dim (int): Sequence dimension index.

    Returns:
        torch.Tensor: Reshaped frequency tensor.
    """
    ndim = x.ndim
    assert 0 <= seq_dim < ndim
    assert freqs_cis.shape == (
        x.shape[seq_dim],
        x.shape[-3],
        2,
        2,
    ), f"freqs_cis vs x: {(freqs_cis.shape, x.shape)}"
    shape = [
        d if i == seq_dim or i == ndim - 3 else 1 for i, d in enumerate(x.shape[:-2])
    ] + [2, 2]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    seq_dim: int,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = xq.reshape(*xq.shape[:-1], -1, 1, 2)  # B S H D -> B S H D/2 1 2
    xk_ = xk.reshape(*xk.shape[:-1], -1, 1, 2)  # B S H D -> B S H D/2 1 2
    freqs_cis = reshape_for_broadcast(
        freqs_cis, xq_, seq_dim
    ).float()  # S D/2 2 2 -> 1 S 1 D/2 2 2
    xq_out = (xq_ * freqs_cis).sum(5).flatten(3)
    xk_out = (xk_ * freqs_cis).sum(5).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


def lengths_to_start_ids(lengths):
    doc_start = lengths.cumsum(0)
    doc_start = doc_start.roll(1)
    doc_start[0] = 0
    return doc_start


def lengths_to_local_ids(lengths):
    assert lengths.ndim == 1
    nb_seqs = lengths.size(0)
    total_seqlen = lengths.sum()
    # This gives the document id of each token
    doc_id = torch.repeat_interleave(lengths)
    # Compute document start for each document
    doc_start = lengths_to_start_ids(lengths)
    # Compute document start for each token
    doc_start = doc_start[doc_id]
    # Compute the position of each token within each document
    tok_id = torch.arange(total_seqlen, device=lengths.device) - doc_start

    return doc_id, tok_id


def generate_doc_mask_mod(
    mask_mod: _mask_mod_signature,
    lengths: torch.Tensor,
    kv_lengths: Optional[torch.Tensor] = None,
) -> _mask_mod_signature:
    """Generates mask mods that apply to inputs to flex attention in the sequence stacked
    format.

    Args:
        mask_mod: The mask mod to apply to the documents
        lengths: Lengths of each document

    Note:
        What is the sequence stacked format? When assembling batches of inputs, we
        take multiple sequences and stack them together to form 1 large sequence. We then
        use masking to ensure that the attention scores are only applied to tokens within
        the same document.

    Example:

    - Square mask
      doc_mask         lengths
      a a b b b c c    2 3 2
    a 1 0 0 0 0 0 0
    a 1 1 0 0 0 0 0
    b 0 0 1 0 0 0 0
    b 0 0 1 1 0 0 0
    b 0 0 1 1 1 0 0
    c 0 0 0 0 0 1 0
    c 0 0 0 0 0 1 1

    """
    kv_lengths = kv_lengths if kv_lengths is not None else lengths
    q_document_id, q_token_id = lengths_to_local_ids(lengths)
    kv_document_id, kv_token_id = lengths_to_local_ids(kv_lengths)
    q_max_idx = lengths.sum() - 1
    kv_max_idx = kv_lengths.sum() - 1

    def doc_mask_mod(b, h, q_idx, kv_idx):
        q_idx_cap = torch.minimum(q_max_idx, q_idx)
        kv_idx_cap = torch.minimum(kv_max_idx, kv_idx)
        valid_idx = (q_idx <= q_max_idx) & (kv_idx <= kv_max_idx)
        same_doc = q_document_id[q_idx_cap] == kv_document_id[kv_idx_cap]
        q_logical = q_token_id[q_idx_cap]
        kv_logical = kv_token_id[kv_idx_cap]
        inner_mask = mask_mod(b, h, q_logical, kv_logical)
        return same_doc & inner_mask & valid_idx

    return doc_mask_mod


# Rotary embedding as in xformer, see if torchtrain implementation is not better. Also might be usefull to make it work with batch*seqlen collapsed.
class RotaryEmbedding(torch.nn.Module):
    """
    RotaryEmbedding Module
    """

    def __init__(
        self,
        theta: float,
        head_dim: int,
        max_seqlen: int = 1024,
        rope_use_fp32_in_outer_product: bool = False,
    ):
        super().__init__()

        self.theta = theta
        self.head_dim = head_dim
        self.max_seqlen = max_seqlen
        self.rope_use_fp32_in_outer_product = rope_use_fp32_in_outer_product

        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(
                dim=head_dim,
                end=max_seqlen,
                theta=theta,
                rope_use_fp32_in_outer_product=self.rope_use_fp32_in_outer_product,
            ),
            persistent=False,
        )

    def reset_parameters(self):
        self.freqs_cis[...] = precompute_freqs_cis(
            dim=self.head_dim,
            end=self.max_seqlen,
            theta=self.theta,
            rope_use_fp32_in_outer_product=self.rope_use_fp32_in_outer_product,
        )

    def forward(
        self, seqlen: Optional[int] = None, tok_idx: Optional[torch.Tensor] = None
    ):
        """
        Return freqs_cis corresponding to consecutive seqlen positions or the corresponding tok_idx positions
        Args:
            seqlen (int): Contiguous sequence length
            tok_idx (torch.Tensor[int]): Position indices of each token this overrides seqlen

        Returns:
            Tuple(torch.Tensor, torch.Tensor): Embedded input tensor and freqs_cis
        """
        test = (seqlen is not None) or (tok_idx is not None)
        assert test, "Should provide atleast seqlen or tok_idx"
        if tok_idx is not None:
            return self.freqs_cis[tok_idx]
        elif seqlen is not None:
            return self.freqs_cis[0:seqlen]


def _reshape_for_attn_bias(
    attn_bias: AttentionBias | None,
    *tensors: torch.Tensor,
) -> list[torch.Tensor]:
    to_transform = list(tensors)
    if isinstance(attn_bias, fmha.attn_bias.BlockDiagonalCausalMask):
        # could be `view` instead of reshape during training, but for inference
        # have to reshape due to strides mismatch
        to_transform = [t.reshape(1, -1, *t.shape[2:]) for t in to_transform]
    return to_transform


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        head_dim: int,
        n_heads: int,
        n_kv_heads: int,
        rope_theta: float,
    ):
        super().__init__()

        self.dim = dim
        self.head_dim = head_dim
        self.rope_theta = rope_theta

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.heads_per_group = self.n_heads // self.n_kv_heads

        self.wq = nn.Linear(
            dim,
            n_heads * head_dim,
            bias=False,
        )
        self.wk = nn.Linear(
            dim,
            n_kv_heads * head_dim,
            bias=False,
        )
        self.wv = nn.Linear(
            dim,
            n_kv_heads * head_dim,
            bias=False,
        )

        self.wo = nn.Linear(
            n_heads * head_dim,
            dim,
            bias=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        freq_cis: torch.Tensor,
        tok_idx: Optional[torch.Tensor] = None,
        mask: Optional[Union[BlockMask, AttentionBias, str]] = None,
        attn_impl: str = "sdpa",
    ) -> torch.Tensor:
        # B S D
        bsz, seq_len, dim = x.shape
        xq = self.wq(x.view_as(x))
        xk = self.wk(x.view_as(x))
        xv = self.wv(x.view_as(x))

        output_shape = xq.shape
        # B S D -> B S H D
        xq = xq.view(bsz, seq_len, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, 1, freq_cis[0:seq_len])

        # This condition helps us be easily compatible
        # with inference by adding a pluggable KVCache
        if hasattr(self, "kv_cache"):
            xk, xv = self.kv_cache.update(xk, xv, tok_idx)

        xk = repeat_kv(xk, self.heads_per_group, dim=2)
        xv = repeat_kv(xv, self.heads_per_group, dim=2)

        if attn_impl == "flex_attention":
            assert mask is None or isinstance(mask, BlockMask)
            xq, xk, xv = map(lambda e: e.transpose(1, 2), (xq, xk, xv))
            output = flex_attention_comp(xq, xk, xv, block_mask=mask)
            output = output.transpose(1, 2).contiguous()  # B H S D -> B S H D

        elif attn_impl == "xformers":
            if fmha is None:
                raise ImportError("xformers is required when attn_impl='xformers'")
            assert mask is None or isinstance(mask, AttentionBias)
            query_shape = xq.shape
            xq, xk, xv = _reshape_for_attn_bias(mask, xq, xk, xv)
            output = fmha.memory_efficient_attention(xq, xk, xv, attn_bias=mask)
            output = output.view(query_shape)
            # This uses B S H D instead of B H S D of pytorch

        elif attn_impl == "sdpa":
            xq, xk, xv = map(lambda e: e.transpose(1, 2), (xq, xk, xv))
            assert mask is None or isinstance(mask, (str, torch.Tensor))
            is_causal = (mask == "causal") if isinstance(mask, str) else False
            mask = mask if isinstance(mask, torch.Tensor) else None
            output = F.scaled_dot_product_attention(
                xq,
                xk,
                xv,
                is_causal=is_causal,
                attn_mask=mask,
            )
            output = output.transpose(1, 2).contiguous()  # B H S D -> B S H D
        else:
            raise NotImplementedError(
                f"Attention implementation {attn_impl} not supported"
            )

        output = self.wo(output.reshape(output_shape))

        return output

    def reset_parameters(self, init_std=None, factor=1.0):
        init_std = init_std or (self.dim ** (-0.5)) / factor

        for w in [self.wq, self.wk, self.wv]:
            trunc_normal_(
                w.weight,
                mean=0.0,
                std=init_std,
                a=-3 * init_std,
                b=3 * init_std,
            )

        trunc_normal_(
            self.wo.weight,
            mean=0.0,
            std=init_std,
            a=-3 * init_std,
            b=3 * init_std,
        )


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
        mp_size: int = 1,
    ):
        super().__init__()

        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        assert hidden_dim % mp_size == 0

        self.dim = dim
        self.hidden_dim = hidden_dim

        self.w1 = nn.Linear(
            dim,
            hidden_dim,
            bias=False,
        )
        self.w3 = nn.Linear(
            dim,
            hidden_dim,
            bias=False,
        )
        self.w2 = nn.Linear(
            hidden_dim,
            dim,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # B S D
        x1 = self.w1(x.view_as(x))
        x3 = self.w3(x.view_as(x))
        output = self.w2(F.silu(x1) * x3)
        return output

    def reset_parameters(self, init_std=None, factor=1.0):
        in_init_std = init_std or (self.dim ** (-0.5)) / factor
        out_init_std = init_std or (self.hidden_dim ** (-0.5)) / factor

        trunc_normal_(
            self.w1.weight,
            mean=0.0,
            std=in_init_std,
            a=-3 * in_init_std,
            b=3 * in_init_std,
        )
        trunc_normal_(
            self.w2.weight,
            mean=0.0,
            std=out_init_std,
            a=-3 * out_init_std,
            b=3 * out_init_std,
        )
        trunc_normal_(
            self.w3.weight,
            mean=0.0,
            std=in_init_std,
            a=-3 * in_init_std,
            b=3 * in_init_std,
        )


class SparseMoEFeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
        num_experts: int,
        top_k: int,
        expert_parallel_size: int = 1,
        router_jitter: float = 0.0,
        router_congestion_weight: float = 0.0,
        router_z_loss_weight: float = 0.0,
        router_anticollapse_loss_weight: float = 0.0,
        router_dominance_loss_weight: float = 0.0,
        router_dominance_threshold: float = 0.35,
        router_use_hidden_state: bool = True,
        router_patch_feature_bias: bool = False,
        router_normalize_patch_entropy: bool = True,
        router_entropy_mlp_hidden_dim: int = 0,
        router_entropy_mean: float = 0.0,
        router_entropy_std: float = 1.0,
        router_entropy_clip: float = 4.0,
        router_use_patch_length: bool = False,
        router_use_patch_entropy: bool = False,
        router_use_patch_byte_features: bool = False,
        routing_granularity: str = "expert",
        pair_size: int = 2,
        entropy_prior_mode: str = "none",
        entropy_prior_scale: float = 1.0,
        entropy_prior_hidden_scale: float = 1.0,
        entropy_prior_calibration_path: Optional[str] = None,
        entropy_prior_trainable: bool = False,
        pair_bias_mode: str = "none",
        pair_bias_ema: float = 0.95,
        pair_bias_update_interval: int = 20,
        pair_bias_lr: float = 0.05,
        pair_bias_clip: float = 1.5,
        pair_min_byte_fraction: float = 0.05,
        pair_max_byte_fraction: float = 0.55,
        hidden_residual_ramp_start_step: int = 0,
        hidden_residual_ramp_end_step: int = 0,
        hidden_residual_final_scale: float = 1.0,
        balance_cost: str = "patch",
    ):
        super().__init__()
        if num_experts <= 0:
            raise ValueError("num_experts must be positive for SparseMoEFeedForward")
        if top_k <= 0:
            raise ValueError("top_k must be positive for SparseMoEFeedForward")
        if expert_parallel_size <= 0:
            raise ValueError("expert_parallel_size must be positive")
        if num_experts % expert_parallel_size != 0:
            raise ValueError(
                "num_experts must be divisible by expert_parallel_size, got "
                f"{num_experts} experts and EP size {expert_parallel_size}"
            )
        if router_congestion_weight < 0:
            raise ValueError("router_congestion_weight must be non-negative")
        if router_z_loss_weight < 0:
            raise ValueError("router_z_loss_weight must be non-negative")
        if router_anticollapse_loss_weight < 0:
            raise ValueError("router_anticollapse_loss_weight must be non-negative")
        if router_dominance_loss_weight < 0:
            raise ValueError("router_dominance_loss_weight must be non-negative")
        if router_dominance_threshold <= 0 or router_dominance_threshold > 1:
            raise ValueError("router_dominance_threshold must be in (0, 1]")
        if router_entropy_mlp_hidden_dim < 0:
            raise ValueError("router_entropy_mlp_hidden_dim must be non-negative")
        if router_entropy_std <= 0:
            raise ValueError("router_entropy_std must be positive")
        if router_entropy_clip <= 0:
            raise ValueError("router_entropy_clip must be positive")
        if balance_cost not in {"patch", "byte", "entropy_byte"}:
            raise ValueError("balance_cost must be one of: patch, byte, entropy_byte")
        if routing_granularity not in {"expert", "pair"}:
            raise ValueError("routing_granularity must be one of: expert, pair")
        if entropy_prior_mode not in {"none", "linear_legacy", "gaussian_pairs"}:
            raise ValueError(
                "entropy_prior_mode must be one of: none, linear_legacy, gaussian_pairs"
            )
        if pair_bias_mode not in {"none", "ema_byte_floor"}:
            raise ValueError("pair_bias_mode must be one of: none, ema_byte_floor")
        if pair_bias_update_interval <= 0:
            raise ValueError("pair_bias_update_interval must be positive")
        if pair_bias_ema < 0 or pair_bias_ema >= 1:
            raise ValueError("pair_bias_ema must be in [0, 1)")
        if pair_bias_lr < 0:
            raise ValueError("pair_bias_lr must be non-negative")
        if pair_bias_clip <= 0:
            raise ValueError("pair_bias_clip must be positive")
        if pair_min_byte_fraction < 0 or pair_max_byte_fraction > 1:
            raise ValueError("pair byte fraction bounds must be within [0, 1]")
        if pair_min_byte_fraction > pair_max_byte_fraction:
            raise ValueError("pair_min_byte_fraction must be <= pair_max_byte_fraction")
        if hidden_residual_ramp_start_step < 0 or hidden_residual_ramp_end_step < 0:
            raise ValueError("hidden residual ramp steps must be non-negative")
        if hidden_residual_final_scale < 0:
            raise ValueError("hidden_residual_final_scale must be non-negative")
        if pair_size <= 0:
            raise ValueError("pair_size must be positive")
        if routing_granularity == "pair":
            if num_experts % pair_size != 0:
                raise ValueError("num_experts must be divisible by pair_size")
            if top_k != pair_size:
                raise ValueError("pair routing requires top_k == pair_size")
        if entropy_prior_mode == "gaussian_pairs":
            if routing_granularity != "pair":
                raise ValueError("gaussian_pairs entropy prior requires pair routing")
            if entropy_prior_calibration_path is None:
                raise ValueError(
                    "entropy_prior_calibration_path is required for gaussian_pairs"
                )
        if pair_bias_mode == "ema_byte_floor" and routing_granularity != "pair":
            raise ValueError("ema_byte_floor pair bias requires pair routing")

        self.dim = dim
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.routing_granularity = routing_granularity
        self.pair_size = pair_size
        self.num_routing_groups = (
            num_experts // pair_size if routing_granularity == "pair" else num_experts
        )
        self.requested_expert_parallel_size = expert_parallel_size
        self.expert_parallel_size = 1
        self.expert_parallel_rank = 0
        self.expert_parallel_group = None
        self.local_expert_ids = tuple(range(num_experts))
        self.router_jitter = router_jitter
        self.router_congestion_weight = router_congestion_weight
        self.router_z_loss_weight = router_z_loss_weight
        self.router_anticollapse_loss_weight = router_anticollapse_loss_weight
        self.router_dominance_loss_weight = router_dominance_loss_weight
        self.router_dominance_threshold = router_dominance_threshold
        self.router_use_hidden_state = router_use_hidden_state
        self.router_normalize_patch_entropy = router_normalize_patch_entropy
        self.router_entropy_mlp_hidden_dim = router_entropy_mlp_hidden_dim
        self.router_entropy_mean = float(router_entropy_mean)
        self.router_entropy_std = float(router_entropy_std)
        self.router_entropy_clip = float(router_entropy_clip)
        self.router_use_patch_length = router_use_patch_length
        self.router_use_patch_entropy = router_use_patch_entropy
        self.router_use_patch_byte_features = router_use_patch_byte_features
        self.use_entropy_mlp_router = (
            router_use_patch_entropy and router_entropy_mlp_hidden_dim > 0
        )
        self.entropy_prior_mode = entropy_prior_mode
        self.entropy_prior_scale = entropy_prior_scale
        self.entropy_prior_hidden_scale = entropy_prior_hidden_scale
        self.entropy_prior_trainable = entropy_prior_trainable
        self.pair_bias_mode = pair_bias_mode
        self.pair_bias_ema = pair_bias_ema
        self.pair_bias_update_interval = pair_bias_update_interval
        self.pair_bias_lr = pair_bias_lr
        self.pair_bias_clip = pair_bias_clip
        self.pair_min_byte_fraction = pair_min_byte_fraction
        self.pair_max_byte_fraction = pair_max_byte_fraction
        self.hidden_residual_ramp_start_step = hidden_residual_ramp_start_step
        self.hidden_residual_ramp_end_step = hidden_residual_ramp_end_step
        self.hidden_residual_final_scale = hidden_residual_final_scale
        self.balance_cost = balance_cost
        self.is_sparse_moe = True
        self.router_schedule_enabled = (
            hidden_residual_ramp_end_step > hidden_residual_ramp_start_step
        )
        if self.router_schedule_enabled:
            self.register_buffer("router_step", torch.zeros((), dtype=torch.long))
        else:
            self.register_buffer(
                "router_step", torch.zeros((), dtype=torch.long), persistent=False
            )

        self.router = nn.Linear(dim, self.num_routing_groups, bias=False)
        self.patch_feature_names = []
        self.linear_patch_feature_names = []
        if router_use_patch_length:
            self.patch_feature_names.append("length")
            self.linear_patch_feature_names.append("length")
        if router_use_patch_entropy:
            self.patch_feature_names.append("entropy")
            if not self.use_entropy_mlp_router:
                self.linear_patch_feature_names.append("entropy")
        if router_use_patch_byte_features:
            self.patch_feature_names.extend(
                f"byte_{name}" for name in PATCH_BYTE_TYPE_NAMES
            )
            self.linear_patch_feature_names.extend(
                f"byte_{name}" for name in PATCH_BYTE_TYPE_NAMES
            )
        self.patch_feature_router = (
            nn.Linear(
                len(self.linear_patch_feature_names),
                self.num_routing_groups,
                bias=router_patch_feature_bias,
            )
            if len(self.linear_patch_feature_names) > 0
            else None
        )
        self.entropy_router = (
            nn.Sequential(
                nn.Linear(1, router_entropy_mlp_hidden_dim, bias=True),
                nn.SiLU(),
                nn.Linear(router_entropy_mlp_hidden_dim, self.num_routing_groups, bias=True),
            )
            if self.use_entropy_mlp_router
            else None
        )
        self.patch_length_router = self.patch_feature_router
        if (
            not router_use_hidden_state
            and self.patch_feature_router is None
            and self.entropy_router is None
            and self.entropy_prior_mode == "none"
        ):
            raise ValueError(
                "At least one patch routing feature is required when hidden-state "
                "routing is disabled"
            )
        if self.entropy_prior_mode == "gaussian_pairs":
            centers, widths, static_bias = self._load_entropy_prior_calibration(
                entropy_prior_calibration_path
            )
            if entropy_prior_trainable:
                self.entropy_prior_centers = nn.Parameter(centers)
                self.entropy_prior_widths = nn.Parameter(widths)
                self.entropy_prior_static_bias = nn.Parameter(static_bias)
            else:
                self.register_buffer("entropy_prior_centers", centers)
                self.register_buffer("entropy_prior_widths", widths)
                self.register_buffer("entropy_prior_static_bias", static_bias)
        if self.pair_bias_mode == "ema_byte_floor":
            self.register_buffer(
                "dynamic_pair_bias",
                torch.zeros(self.num_routing_groups, dtype=torch.float32),
            )
            self.register_buffer(
                "pair_load_ema",
                torch.zeros(self.num_routing_groups, dtype=torch.float32),
            )
            self.register_buffer(
                "pair_bias_step", torch.zeros((), dtype=torch.long)
            )
        self.experts = nn.ModuleList(
            [
                FeedForward(
                    dim=dim,
                    hidden_dim=hidden_dim,
                    multiple_of=multiple_of,
                    ffn_dim_multiplier=ffn_dim_multiplier,
                )
                for _ in range(num_experts)
            ]
        )
        self.last_balance_loss: Optional[torch.Tensor] = None
        self.last_router_z_loss: Optional[torch.Tensor] = None
        self.last_router_anticollapse_loss: Optional[torch.Tensor] = None
        self.last_router_entropy_collapse_loss: Optional[torch.Tensor] = None
        self.last_router_dominance_loss: Optional[torch.Tensor] = None
        self.last_metrics: dict[str, float] = {}
        self.last_expert_parallel_metrics: dict[str, float] = {}

    def set_router_step(self, step: int) -> None:
        self.router_step.fill_(int(step))

    def _hidden_residual_scale(self) -> float:
        if self.hidden_residual_ramp_end_step <= self.hidden_residual_ramp_start_step:
            return float(self.entropy_prior_hidden_scale)
        step = int(self.router_step.item())
        if step <= self.hidden_residual_ramp_start_step:
            return float(self.entropy_prior_hidden_scale)
        if step >= self.hidden_residual_ramp_end_step:
            return float(self.hidden_residual_final_scale)
        progress = (step - self.hidden_residual_ramp_start_step) / (
            self.hidden_residual_ramp_end_step - self.hidden_residual_ramp_start_step
        )
        start = float(self.entropy_prior_hidden_scale)
        end = float(self.hidden_residual_final_scale)
        return start + progress * (end - start)

    def _load_entropy_prior_calibration(
        self, calibration_path: Optional[str]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if calibration_path is None:
            raise ValueError("Missing entropy prior calibration path")
        with Path(calibration_path).open() as f:
            calibration = json.load(f)
        expected_pairs = self.num_routing_groups
        expected_experts = self.num_experts
        expected_pair_size = self.pair_size
        if int(calibration.get("num_pairs", -1)) != expected_pairs:
            raise ValueError(
                f"calibration num_pairs must be {expected_pairs}, got "
                f"{calibration.get('num_pairs')}"
            )
        if int(calibration.get("num_experts", expected_experts)) != expected_experts:
            raise ValueError(
                f"calibration num_experts must be {expected_experts}, got "
                f"{calibration.get('num_experts')}"
            )
        if int(calibration.get("pair_size", expected_pair_size)) != expected_pair_size:
            raise ValueError(
                f"calibration pair_size must be {expected_pair_size}, got "
                f"{calibration.get('pair_size')}"
            )

        center_values = calibration["entropy_centers"]
        width_values = calibration["entropy_widths"]
        static_bias_values = calibration["static_pair_bias"]
        if len(center_values) != expected_pairs:
            raise ValueError("entropy_centers length does not match num_pairs")
        if len(width_values) != expected_pairs:
            raise ValueError("entropy_widths length does not match num_pairs")
        if len(static_bias_values) != expected_pairs:
            raise ValueError("static_pair_bias length does not match num_pairs")
        if any(float(width) <= 0 for width in width_values):
            raise ValueError("entropy_widths must be positive")
        centers = torch.tensor(center_values, dtype=torch.float32)
        widths = torch.tensor(width_values, dtype=torch.float32)
        static_bias = torch.tensor(static_bias_values, dtype=torch.float32)
        return centers, widths, static_bias

    def configure_expert_parallel(self, process_group=None) -> None:
        ep_size = self.requested_expert_parallel_size
        if ep_size == 1:
            return
        if not dist.is_initialized():
            raise RuntimeError("Expert parallelism requires torch.distributed")
        if process_group is None:
            raise ValueError("An expert-parallel process group is required")
        if self.expert_parallel_size != 1:
            raise RuntimeError("Expert parallelism is already configured")
        if dist.get_world_size(process_group) != ep_size:
            raise ValueError(
                "Expert-parallel process group size does not match model config: "
                f"{dist.get_world_size(process_group)} != {ep_size}"
            )

        self.expert_parallel_size = ep_size
        self.expert_parallel_rank = dist.get_rank(process_group)
        self.expert_parallel_group = process_group
        local_expert_count = self.num_experts // ep_size
        local_expert_start = self.expert_parallel_rank * local_expert_count
        self.local_expert_ids = tuple(
            range(local_expert_start, local_expert_start + local_expert_count)
        )
        local_expert_ids = set(self.local_expert_ids)
        for expert_id in range(self.num_experts):
            if expert_id not in local_expert_ids:
                self.experts[expert_id] = None

    def forward(
        self,
        x: torch.Tensor,
        patch_lengths: Optional[torch.Tensor] = None,
        patch_entropies: Optional[torch.Tensor] = None,
        patch_byte_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        flat_x = x.reshape(-1, x.shape[-1])
        flat_patch_lengths = self._prepare_patch_lengths(x, patch_lengths)
        flat_patch_entropies = self._prepare_patch_entropies(x, patch_entropies)
        flat_patch_byte_features = self._prepare_patch_byte_features(
            x, patch_byte_features
        )

        if flat_patch_lengths is None:
            valid_mask = torch.ones(
                flat_x.shape[0], device=flat_x.device, dtype=torch.bool
            )
        else:
            valid_mask = flat_patch_lengths > 0
        flat_output = torch.zeros_like(flat_x)
        if not valid_mask.any():
            zero = flat_x.sum() * 0.0
            self.last_balance_loss = zero.float()
            self.last_router_z_loss = zero.float()
            self.last_router_anticollapse_loss = zero.float()
            self.last_router_entropy_collapse_loss = zero.float()
            self.last_router_dominance_loss = zero.float()
            self.last_expert_parallel_metrics = {}
            self.last_metrics = {
                "hidden_state_routing": float(self.router_use_hidden_state),
                "hidden_residual_scale": (
                    self._hidden_residual_scale()
                    if self.router_use_hidden_state
                    else 0.0
                ),
                "routing_granularity_pair": float(
                    self.routing_granularity == "pair"
                ),
                "router_entropy": 0.0,
                "load_imbalance": 0.0,
                "max_load_fraction": 0.0,
                "min_load_fraction": 0.0,
                "active_experts": 0.0,
                "unit_active_experts": 0.0,
                "balance_loss": 0.0,
                "routed_units": 0.0,
                "total_units": float(flat_x.shape[0]),
                "valid_patch_fraction": 0.0,
                "invalid_positions_skipped": float(flat_x.shape[0]),
                "load_weight_mean": 0.0,
                "load_weight_max": 0.0,
                "side_feature_count": float(len(self.patch_feature_names)),
                "congestion_weight": self.router_congestion_weight,
                "router_z_loss": 0.0,
                "router_z_loss_weight": self.router_z_loss_weight,
            }
            return flat_output.reshape_as(x)

        valid_x = flat_x[valid_mask]
        valid_patch_lengths = (
            flat_patch_lengths[valid_mask] if flat_patch_lengths is not None else None
        )
        valid_patch_entropies = (
            flat_patch_entropies[valid_mask]
            if flat_patch_entropies is not None
            else None
        )
        valid_patch_byte_features = (
            flat_patch_byte_features[valid_mask]
            if flat_patch_byte_features is not None
            else None
        )
        load_weights = self._load_weights(
            valid_x, valid_patch_lengths, valid_patch_entropies
        )

        if self.router_use_hidden_state:
            hidden_router_logits = self.router(valid_x)
            hidden_residual_scale = self._hidden_residual_scale()
            router_logits = hidden_router_logits * hidden_residual_scale
        else:
            hidden_residual_scale = 0.0
            router_logits = valid_x.new_zeros(
                (valid_x.shape[0], self.num_routing_groups)
            )
        learned_router_logits = router_logits
        if self.patch_feature_router is not None:
            patch_features = self._patch_features(
                valid_patch_lengths,
                valid_patch_entropies,
                valid_patch_byte_features,
            )
            patch_features = patch_features.to(
                dtype=self.patch_feature_router.weight.dtype
            )
            router_logits = router_logits + self.patch_feature_router(
                patch_features
            ).to(router_logits.dtype)
            learned_router_logits = router_logits
        if self.entropy_router is not None:
            assert valid_patch_entropies is not None
            entropy_features = self._entropy_router_features(valid_patch_entropies)
            entropy_features = entropy_features.to(
                dtype=self.entropy_router[0].weight.dtype
            )
            router_logits = router_logits + self.entropy_router(
                entropy_features
            ).to(router_logits.dtype)
            learned_router_logits = router_logits
        if self.entropy_prior_mode == "gaussian_pairs":
            if valid_patch_entropies is None:
                raise ValueError(
                    "patch_entropies must be provided for gaussian_pairs entropy prior"
                )
            entropy_prior_logits = self._gaussian_entropy_prior_logits(
                valid_patch_entropies
            ).to(router_logits.dtype)
            router_logits = router_logits + entropy_prior_logits
            learned_router_logits = router_logits
        if self.pair_bias_mode == "ema_byte_floor":
            router_logits = router_logits + self.dynamic_pair_bias.to(
                device=router_logits.device, dtype=router_logits.dtype
            )
            learned_router_logits = router_logits

        self.last_router_z_loss = (
            torch.logsumexp(learned_router_logits.float(), dim=-1).square().mean()
        )

        if self.training and self.router_jitter > 0:
            router_logits = router_logits + torch.empty_like(router_logits).uniform_(
                -self.router_jitter, self.router_jitter
            )

        pre_congestion_group_probs = F.softmax(router_logits.float(), dim=-1)
        congestion_price = self._congestion_price(
            pre_congestion_group_probs, load_weights
        )
        if self.router_congestion_weight > 0:
            router_logits = router_logits - self.router_congestion_weight * (
                congestion_price.to(router_logits.dtype)
            )
            group_probs = F.softmax(router_logits.float(), dim=-1)
        else:
            group_probs = pre_congestion_group_probs

        self.last_router_anticollapse_loss = router_logits.new_zeros(()).float()
        self.last_router_entropy_collapse_loss = router_logits.new_zeros(()).float()
        self.last_router_dominance_loss = router_logits.new_zeros(()).float()
        if (
            self.training
            and (
                self.router_anticollapse_loss_weight > 0.0
                or self.router_dominance_loss_weight > 0.0
            )
        ):
            mean_group_probs = (
                group_probs.float() * load_weights.float().unsqueeze(-1)
            ).sum(dim=0)
            mean_group_probs = mean_group_probs / load_weights.float().sum().clamp_min(1.0)

            eps = 1e-8
            entropy = -(
                mean_group_probs * mean_group_probs.clamp_min(eps).log()
            ).sum()
            entropy_norm = entropy / math.log(float(mean_group_probs.numel()))
            entropy_collapse_loss = 1.0 - entropy_norm

            dominance = mean_group_probs.max()
            dominance_loss = torch.relu(
                dominance - self.router_dominance_threshold
            ).pow(2)

            self.last_router_entropy_collapse_loss = entropy_collapse_loss.to(
                router_logits.dtype
            )
            self.last_router_dominance_loss = dominance_loss.to(router_logits.dtype)
            self.last_router_anticollapse_loss = (
                self.router_anticollapse_loss_weight * entropy_collapse_loss
                + self.router_dominance_loss_weight * dominance_loss
            ).to(router_logits.dtype)

        if self.routing_granularity == "pair":
            router_probs = group_probs.repeat_interleave(self.pair_size, dim=-1)
            router_probs = router_probs / float(self.pair_size)
            pre_congestion_router_probs = pre_congestion_group_probs.repeat_interleave(
                self.pair_size, dim=-1
            )
            pre_congestion_router_probs = pre_congestion_router_probs / float(
                self.pair_size
            )
            top_pair_indices = group_probs.argmax(dim=-1)
            pair_offsets = torch.arange(
                self.pair_size, device=valid_x.device, dtype=top_pair_indices.dtype
            )
            top_indices = (
                top_pair_indices.unsqueeze(-1) * self.pair_size + pair_offsets
            )
            top_weights = torch.full(
                top_indices.shape,
                1.0 / self.pair_size,
                device=valid_x.device,
                dtype=group_probs.dtype,
            )
        else:
            router_probs = group_probs
            pre_congestion_router_probs = pre_congestion_group_probs
            top_weights, top_indices = torch.topk(
                router_probs, k=self.top_k, dim=-1, sorted=False
            )
            top_weights = top_weights / top_weights.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-9)

        self._maybe_update_dynamic_pair_bias(top_indices, load_weights)

        if self.requested_expert_parallel_size > 1:
            if self.expert_parallel_size == 1:
                raise RuntimeError(
                    "Expert parallelism was requested but has not been configured"
                )
            valid_output = self._forward_expert_parallel(
                valid_x, top_indices, top_weights
            )
        else:
            valid_output = self._forward_local(valid_x, top_indices, top_weights)
        flat_output[valid_mask] = valid_output

        balance_probs = group_probs
        balance_group_count = self.num_routing_groups
        prob_density = (balance_probs * load_weights.unsqueeze(-1)).sum(dim=0)
        prob_density = prob_density / load_weights.sum().clamp_min(1.0)
        uniform = torch.full_like(prob_density, 1.0 / balance_group_count)
        self.last_balance_loss = balance_group_count * torch.sum(
            (prob_density - uniform) ** 2
        )
        self._record_metrics(
            router_probs,
            top_indices,
            top_weights,
            load_weights,
            valid_patch_lengths,
            valid_patch_entropies,
            valid_patch_byte_features,
            congestion_price,
            pre_congestion_router_probs,
            pair_router_probs=(
                group_probs if self.routing_granularity == "pair" else None
            ),
            hidden_residual_scale=hidden_residual_scale,
            total_units=flat_x.shape[0],
            invalid_positions_skipped=(~valid_mask).sum().item(),
        )
        return flat_output.reshape_as(x)

    @staticmethod
    def _safe_corr(x: torch.Tensor, y: torch.Tensor) -> float:
        if x.numel() == 0 or y.numel() == 0:
            return 0.0
        x = x.float()
        y = y.float()
        x = x - x.mean()
        y = y - y.mean()
        denom = x.square().sum().sqrt() * y.square().sum().sqrt()
        if denom <= 0:
            return 0.0
        return (x * y).sum().div(denom).item()

    def _forward_local(
        self,
        flat_x: torch.Tensor,
        top_indices: torch.Tensor,
        top_weights: torch.Tensor,
    ) -> torch.Tensor:
        flat_output = torch.zeros_like(flat_x)
        empty_expert_dependency = flat_x.new_zeros(())
        for expert_rank in range(self.top_k):
            rank_expert_ids = top_indices[:, expert_rank]
            rank_weights = top_weights[:, expert_rank].to(flat_x.dtype)
            for expert_id, expert in enumerate(self.experts):
                assert expert is not None
                token_ids = torch.where(rank_expert_ids == expert_id)[0]
                if token_ids.numel() == 0:
                    if self.training:
                        empty_input = flat_x.new_empty((0, flat_x.shape[-1]))
                        empty_output = expert(empty_input)
                        # Keep FSDP expert collectives aligned across ranks.
                        empty_expert_dependency = (
                            empty_expert_dependency + empty_output.sum() * 0.0
                        )
                    continue
                expert_input = flat_x.index_select(0, token_ids)
                expert_output = expert(expert_input)
                expert_output = expert_output * rank_weights.index_select(
                    0, token_ids
                ).unsqueeze(-1)
                flat_output.index_add_(0, token_ids, expert_output)

        if self.training:
            flat_output = flat_output + empty_expert_dependency
        return flat_output

    def _forward_expert_parallel(
        self,
        flat_x: torch.Tensor,
        top_indices: torch.Tensor,
        top_weights: torch.Tensor,
    ) -> torch.Tensor:
        assert self.expert_parallel_group is not None
        local_expert_count = self.num_experts // self.expert_parallel_size
        flat_expert_ids = top_indices.reshape(-1)
        flat_token_ids = (
            torch.arange(flat_x.shape[0], device=flat_x.device)
            .unsqueeze(-1)
            .expand_as(top_indices)
            .reshape(-1)
        )
        expert_owners = torch.div(
            flat_expert_ids, local_expert_count, rounding_mode="floor"
        )
        send_order = torch.argsort(expert_owners)
        send_counts = torch.bincount(expert_owners, minlength=self.expert_parallel_size)
        gathered_send_counts = torch.empty(
            self.expert_parallel_size,
            self.expert_parallel_size,
            device=send_counts.device,
            dtype=send_counts.dtype,
        )
        dist.all_gather_into_tensor(
            gathered_send_counts,
            send_counts.contiguous(),
            group=self.expert_parallel_group,
        )
        recv_counts = gathered_send_counts[:, self.expert_parallel_rank].contiguous()
        send_count_list = send_counts.tolist()
        recv_count_list = recv_counts.tolist()

        ordered_token_ids = flat_token_ids.index_select(0, send_order)
        send_x = flat_x.index_select(0, ordered_token_ids).contiguous()
        recv_x = flat_x.new_empty((sum(recv_count_list), flat_x.shape[-1]))
        recv_x = all_to_all_single(
            recv_x,
            send_x,
            output_split_sizes=recv_count_list,
            input_split_sizes=send_count_list,
            group=self.expert_parallel_group,
        )

        send_expert_ids = flat_expert_ids.index_select(0, send_order).contiguous()
        recv_expert_ids = send_expert_ids.new_empty((sum(recv_count_list),))
        dist.all_to_all_single(
            recv_expert_ids,
            send_expert_ids,
            output_split_sizes=recv_count_list,
            input_split_sizes=send_count_list,
            group=self.expert_parallel_group,
        )

        recv_output = torch.zeros_like(recv_x)
        empty_expert_dependency = flat_x.new_zeros(())
        for expert_id in self.local_expert_ids:
            expert = self.experts[expert_id]
            assert expert is not None
            token_ids = torch.where(recv_expert_ids == expert_id)[0]
            expert_input = recv_x.index_select(0, token_ids)
            expert_output = expert(expert_input).to(recv_output.dtype)
            if token_ids.numel() == 0:
                # Expert-DP replicas must enter the same nested FSDP collectives.
                empty_expert_dependency = (
                    empty_expert_dependency + expert_output.sum() * 0.0
                )
            else:
                recv_output.index_copy_(0, token_ids, expert_output)
        recv_output = recv_output + empty_expert_dependency

        returned_output = send_x.new_empty(send_x.shape)
        returned_output = all_to_all_single(
            returned_output,
            recv_output,
            output_split_sizes=send_count_list,
            input_split_sizes=recv_count_list,
            group=self.expert_parallel_group,
        )
        flat_assignment_output = torch.empty_like(returned_output)
        flat_assignment_output.index_copy_(0, send_order, returned_output)
        flat_assignment_output = flat_assignment_output * top_weights.reshape(-1, 1).to(
            flat_x.dtype
        )
        flat_output = torch.zeros_like(flat_x)
        flat_output.index_add_(0, flat_token_ids, flat_assignment_output)

        self.last_expert_parallel_metrics = {
            "ep_size": float(self.expert_parallel_size),
            "ep_rank": float(self.expert_parallel_rank),
            "ep_dispatched_assignments": float(send_x.shape[0]),
            "ep_received_assignments": float(recv_x.shape[0]),
            "ep_max_send_assignments": float(send_counts.max().item()),
            "ep_max_recv_assignments": float(recv_counts.max().item()),
            "ep_all_to_all_bytes": float(
                2 * send_x.numel() * send_x.element_size()
                + send_expert_ids.numel() * send_expert_ids.element_size()
            ),
        }
        return flat_output

    def _prepare_patch_lengths(
        self, x: torch.Tensor, patch_lengths: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        if patch_lengths is None:
            if self.router_use_patch_length or self.balance_cost in {
                "byte",
                "entropy_byte",
            }:
                raise ValueError(
                    "patch_lengths must be provided when length-aware PatchMoE "
                    "routing, byte-cost balancing, or entropy-byte balancing "
                    "is enabled"
                )
            return None

        if patch_lengths.shape != x.shape[:-1]:
            raise ValueError(
                f"patch_lengths shape {tuple(patch_lengths.shape)} must match "
                f"MoE input prefix shape {tuple(x.shape[:-1])}"
            )
        return patch_lengths.reshape(-1).to(device=x.device)

    def _prepare_patch_entropies(
        self, x: torch.Tensor, patch_entropies: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        if patch_entropies is None:
            if (
                self.router_use_patch_entropy
                or self.balance_cost == "entropy_byte"
                or self.entropy_prior_mode == "gaussian_pairs"
            ):
                raise ValueError(
                    "patch_entropies must be provided when entropy-aware "
                    "PatchMoE routing or entropy-byte balancing is enabled"
                )
            return None

        if patch_entropies.shape != x.shape[:-1]:
            raise ValueError(
                f"patch_entropies shape {tuple(patch_entropies.shape)} must "
                f"match MoE input prefix shape {tuple(x.shape[:-1])}"
            )
        return patch_entropies.reshape(-1).to(device=x.device)

    def _gaussian_entropy_prior_logits(
        self, flat_patch_entropies: torch.Tensor
    ) -> torch.Tensor:
        centers = self.entropy_prior_centers.to(device=flat_patch_entropies.device)
        widths = self.entropy_prior_widths.to(device=flat_patch_entropies.device)
        static_bias = self.entropy_prior_static_bias.to(
            device=flat_patch_entropies.device
        )
        entropies = flat_patch_entropies.float().clamp_min(0.0).unsqueeze(-1)
        widths = widths.float().clamp_min(1e-6)
        logits = -((entropies - centers.float()) ** 2) / (2.0 * widths.square())
        logits = logits * float(self.entropy_prior_scale)
        return logits + static_bias.float()

    @torch.no_grad()
    def _maybe_update_dynamic_pair_bias(
        self, top_indices: torch.Tensor, load_weights: torch.Tensor
    ) -> None:
        if self.pair_bias_mode != "ema_byte_floor":
            return
        pair_ids = top_indices[:, 0].div(self.pair_size, rounding_mode="floor")
        pair_bytes = torch.zeros(
            self.num_routing_groups, device=load_weights.device, dtype=torch.float32
        )
        pair_bytes.scatter_add_(0, pair_ids, load_weights.float())
        total_bytes = load_weights.float().sum()
        if dist.is_initialized():
            local_stats = torch.cat([pair_bytes, total_bytes.reshape(1)]).to(
                dtype=torch.bfloat16
            )
            world_size = dist.get_world_size()
            gathered_stats = torch.empty(
                world_size * local_stats.numel(),
                device=local_stats.device,
                dtype=local_stats.dtype,
            )
            dist.all_gather_into_tensor(gathered_stats, local_stats.contiguous())
            global_stats = gathered_stats.view(world_size, -1).float().sum(dim=0)
            pair_bytes = global_stats[:-1]
            total_bytes = global_stats[-1]
        pair_share = pair_bytes / total_bytes.clamp_min(1.0)
        self.pair_bias_step.add_(1)
        if int(self.pair_bias_step.item()) % self.pair_bias_update_interval != 0:
            return

        local_share = pair_share.to(device=self.pair_load_ema.device)
        self.pair_load_ema.mul_(self.pair_bias_ema).add_(
            local_share, alpha=1.0 - self.pair_bias_ema
        )
        low_delta = torch.clamp(
            self.pair_min_byte_fraction - self.pair_load_ema, min=0.0
        )
        high_delta = torch.clamp(
            self.pair_load_ema - self.pair_max_byte_fraction, min=0.0
        )
        self.dynamic_pair_bias.add_(
            (low_delta - high_delta) * float(self.pair_bias_lr)
        )
        self.dynamic_pair_bias.clamp_(
            min=-float(self.pair_bias_clip), max=float(self.pair_bias_clip)
        )
        self.dynamic_pair_bias.sub_(self.dynamic_pair_bias.mean())

    def _prepare_patch_byte_features(
        self, x: torch.Tensor, patch_byte_features: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        if patch_byte_features is None:
            if self.router_use_patch_byte_features:
                raise ValueError(
                    "patch_byte_features must be provided when byte-type-aware "
                    "PatchMoE routing is enabled"
                )
            return None

        expected_shape = x.shape[:-1] + (len(PATCH_BYTE_TYPE_NAMES),)
        if patch_byte_features.shape != expected_shape:
            raise ValueError(
                f"patch_byte_features shape {tuple(patch_byte_features.shape)} "
                f"must match {tuple(expected_shape)}"
            )
        return patch_byte_features.reshape(-1, len(PATCH_BYTE_TYPE_NAMES)).to(
            device=x.device
        )

    def _entropy_router_features(self, flat_patch_entropies: torch.Tensor) -> torch.Tensor:
        normalized = (
            flat_patch_entropies.float() - self.router_entropy_mean
        ) / self.router_entropy_std
        normalized = normalized.clamp(
            min=-self.router_entropy_clip,
            max=self.router_entropy_clip,
        )
        return normalized.unsqueeze(-1)

    def _patch_features(
        self,
        flat_patch_lengths: Optional[torch.Tensor],
        flat_patch_entropies: Optional[torch.Tensor],
        flat_patch_byte_features: Optional[torch.Tensor],
    ) -> torch.Tensor:
        features = []
        if self.router_use_patch_length:
            assert flat_patch_lengths is not None
            length_feature = torch.log1p(flat_patch_lengths.float()).unsqueeze(-1)
            positive_lengths = flat_patch_lengths[flat_patch_lengths > 0].float()
            if positive_lengths.numel() > 0:
                length_feature = length_feature / torch.log1p(
                    positive_lengths.mean()
                ).clamp_min(1.0)
            features.append(length_feature)
        if self.router_use_patch_entropy and not self.use_entropy_mlp_router:
            assert flat_patch_entropies is not None
            entropy_feature = flat_patch_entropies.float().clamp_min(0.0)
            if flat_patch_lengths is None:
                active_entropies = entropy_feature[entropy_feature > 0]
            else:
                active_entropies = entropy_feature[flat_patch_lengths > 0]
            if self.router_normalize_patch_entropy and active_entropies.numel() > 0:
                entropy_feature = entropy_feature / active_entropies.mean().clamp_min(
                    1.0
                )
            if flat_patch_lengths is not None:
                entropy_feature = entropy_feature.masked_fill(
                    flat_patch_lengths <= 0, 0.0
                )
            features.append(entropy_feature.unsqueeze(-1))
        if self.router_use_patch_byte_features:
            assert flat_patch_byte_features is not None
            features.append(flat_patch_byte_features.float())
        if len(features) == 0:
            raise RuntimeError("Patch feature router has no enabled features")
        return torch.cat(features, dim=-1)

    def _load_weights(
        self,
        flat_x: torch.Tensor,
        flat_patch_lengths: Optional[torch.Tensor],
        flat_patch_entropies: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if flat_patch_lengths is None:
            return torch.ones(
                flat_x.shape[0], device=flat_x.device, dtype=torch.float32
            )

        patch_lengths = flat_patch_lengths.float().clamp_min(0.0)
        if self.balance_cost == "patch":
            return (patch_lengths > 0).float()
        if self.balance_cost == "byte":
            return patch_lengths
        if self.balance_cost == "entropy_byte":
            assert flat_patch_entropies is not None
            patch_entropies = flat_patch_entropies.float().clamp_min(0.0)
            return patch_lengths * patch_entropies
        raise ValueError(f"Unknown balance_cost: {self.balance_cost}")

    def _congestion_price(
        self,
        router_probs: torch.Tensor,
        load_weights: torch.Tensor,
    ) -> torch.Tensor:
        group_count = router_probs.shape[-1]
        total_load = load_weights.sum()
        load_density = (router_probs * load_weights.unsqueeze(-1)).sum(dim=0)
        load_density = load_density / total_load.clamp_min(1.0)
        has_load = (total_load > 0).to(load_density.dtype)
        return (group_count * load_density - 1.0).mul(has_load).detach()

    @torch.no_grad()
    def _record_metrics(
        self,
        router_probs: torch.Tensor,
        top_indices: torch.Tensor,
        top_weights: Optional[torch.Tensor],
        load_weights: torch.Tensor,
        flat_patch_lengths: Optional[torch.Tensor],
        flat_patch_entropies: Optional[torch.Tensor],
        flat_patch_byte_features: Optional[torch.Tensor] = None,
        congestion_price: Optional[torch.Tensor] = None,
        pre_congestion_router_probs: Optional[torch.Tensor] = None,
        pair_router_probs: Optional[torch.Tensor] = None,
        hidden_residual_scale: Optional[float] = None,
        total_units: Optional[int] = None,
        invalid_positions_skipped: int = 0,
    ) -> None:
        flat_top_indices = top_indices.reshape(-1)
        assignment_weights = (
            load_weights.unsqueeze(-1).expand_as(top_indices).reshape(-1)
        )
        assignments = torch.zeros(
            self.num_experts, device=router_probs.device, dtype=torch.float32
        )
        assignments.scatter_add_(0, flat_top_indices, assignment_weights)
        total_assignments = assignments.sum().clamp_min(1.0)
        load_fraction = assignments / total_assignments
        mean_load = load_fraction.mean().clamp_min(1e-9)
        router_entropy = -(router_probs * router_probs.clamp_min(1e-9).log()).sum(
            dim=-1
        )

        if flat_patch_lengths is None:
            active_unit_mask = torch.ones_like(load_weights, dtype=torch.bool)
        else:
            active_unit_mask = flat_patch_lengths > 0
        active_weights = load_weights[active_unit_mask]
        routed_units = active_unit_mask.sum().item()
        load_weight_mean = active_weights.mean().item() if routed_units > 0 else 0.0
        load_weight_max = active_weights.max().item() if routed_units > 0 else 0.0

        unit_assignment_weights = (
            active_unit_mask.float().unsqueeze(-1).expand_as(top_indices).reshape(-1)
        )
        unit_assignments = torch.zeros_like(assignments)
        unit_assignments.scatter_add_(0, flat_top_indices, unit_assignment_weights)
        total_unit_assignments = unit_assignments.sum().clamp_min(1.0)
        unit_assignment_fraction = unit_assignments / total_unit_assignments

        self.last_metrics = {
            "hidden_state_routing": float(self.router_use_hidden_state),
            "hidden_residual_scale": (
                self._hidden_residual_scale()
                if hidden_residual_scale is None
                else float(hidden_residual_scale)
            ),
            "routing_granularity_pair": float(self.routing_granularity == "pair"),
            "router_entropy": router_entropy.mean().item(),
            "load_imbalance": (load_fraction.max() / mean_load).item(),
            "max_load_fraction": load_fraction.max().item(),
            "min_load_fraction": load_fraction.min().item(),
            "active_experts": (assignments > 0).float().sum().item(),
            "unit_active_experts": (unit_assignments > 0).float().sum().item(),
            "balance_loss": (
                self.last_balance_loss.detach().item()
                if self.last_balance_loss is not None
                else 0.0
            ),
            "router_anticollapse_loss": (
                self.last_router_anticollapse_loss.detach().item()
                if self.last_router_anticollapse_loss is not None
                else 0.0
            ),
            "router_entropy_collapse_loss": (
                self.last_router_entropy_collapse_loss.detach().item()
                if self.last_router_entropy_collapse_loss is not None
                else 0.0
            ),
            "router_dominance_loss": (
                self.last_router_dominance_loss.detach().item()
                if self.last_router_dominance_loss is not None
                else 0.0
            ),
            "routed_units": float(routed_units),
            "total_units": float(
                load_weights.numel() if total_units is None else total_units
            ),
            "valid_patch_fraction": (
                float(routed_units)
                / max(
                    float(load_weights.numel() if total_units is None else total_units),
                    1.0,
                )
            ),
            "invalid_positions_skipped": float(invalid_positions_skipped),
            "load_weight_mean": load_weight_mean,
            "load_weight_max": load_weight_max,
            "side_feature_count": float(len(self.patch_feature_names)),
            "congestion_weight": self.router_congestion_weight,
            "router_z_loss": (
                self.last_router_z_loss.detach().item()
                if self.last_router_z_loss is not None
                else 0.0
            ),
            "router_z_loss_weight": self.router_z_loss_weight,
        }
        self.last_metrics.update(self.last_expert_parallel_metrics)
        if congestion_price is not None and pre_congestion_router_probs is not None:
            pre_prob_density = (
                pre_congestion_router_probs * load_weights.unsqueeze(-1)
            ).sum(dim=0)
            pre_prob_density = pre_prob_density / load_weights.sum().clamp_min(1.0)
            post_prob_density = (router_probs * load_weights.unsqueeze(-1)).sum(dim=0)
            post_prob_density = post_prob_density / load_weights.sum().clamp_min(1.0)
            self.last_metrics.update(
                {
                    "congestion_price_min": congestion_price.min().item(),
                    "congestion_price_max": congestion_price.max().item(),
                    "congestion_price_std": congestion_price.std(unbiased=False).item(),
                    "pre_congestion_prob_load_imbalance": (
                        pre_prob_density.max() / pre_prob_density.mean().clamp_min(1e-9)
                    ).item(),
                    "post_congestion_prob_load_imbalance": (
                        post_prob_density.max()
                        / post_prob_density.mean().clamp_min(1e-9)
                    ).item(),
                    "congestion_top1_reroute_fraction": (
                        pre_congestion_router_probs.argmax(dim=-1)
                        != router_probs.argmax(dim=-1)
                    )
                    .float()
                    .mean()
                    .item(),
                }
            )
        for expert_id, fraction in enumerate(load_fraction):
            self.last_metrics[f"expert_{expert_id}_load_fraction"] = fraction.item()
        for expert_id, fraction in enumerate(unit_assignment_fraction):
            self.last_metrics[f"expert_{expert_id}_unit_assignment_fraction"] = (
                fraction.item()
            )

        if self.num_experts % self.pair_size == 0:
            pair_count = self.num_experts // self.pair_size
            flat_pair_indices = flat_top_indices.div(
                self.pair_size, rounding_mode="floor"
            )
            pair_assignments = torch.zeros(
                pair_count, device=router_probs.device, dtype=torch.float32
            )
            pair_assignments.scatter_add_(0, flat_pair_indices, assignment_weights)
            total_pair_assignments = pair_assignments.sum().clamp_min(1.0)
            pair_load_fraction = pair_assignments / total_pair_assignments

            unit_pair_assignments = torch.zeros_like(pair_assignments)
            unit_pair_assignments.scatter_add_(
                0, flat_pair_indices, unit_assignment_weights
            )
            total_unit_pair_assignments = unit_pair_assignments.sum().clamp_min(1.0)
            pair_unit_fraction = unit_pair_assignments / total_unit_pair_assignments
            pair_mean = pair_load_fraction.mean().clamp_min(1e-9)

            self.last_metrics.update(
                {
                    "pair_load_cv": (
                        pair_load_fraction.std(unbiased=False) / pair_mean
                    ).item(),
                    "pair_max_byte_fraction": pair_load_fraction.max().item(),
                    "pair_min_byte_fraction": pair_load_fraction.min().item(),
                    "pair_dead_count": (
                        pair_load_fraction < 0.005
                    ).float().sum().item(),
                }
            )
            if pair_router_probs is None:
                pair_probs = router_probs.reshape(
                    router_probs.shape[0], pair_count, self.pair_size
                ).sum(dim=-1)
            else:
                pair_probs = pair_router_probs
            pair_router_entropy = -(
                pair_probs * pair_probs.clamp_min(1e-9).log()
            ).sum(dim=-1)
            self.last_metrics["pair_router_entropy"] = (
                pair_router_entropy.mean().item()
                if pair_router_entropy.numel() > 0
                else 0.0
            )
            if top_weights is not None and top_weights.numel() > 0:
                self.last_metrics["pair_top_weight_min"] = top_weights.min().item()
                self.last_metrics["pair_top_weight_max"] = top_weights.max().item()

            for pair_id, fraction in enumerate(pair_load_fraction):
                self.last_metrics[f"pair_{pair_id}_byte_fraction"] = fraction.item()
            for pair_id, fraction in enumerate(pair_unit_fraction):
                self.last_metrics[f"pair_{pair_id}_unit_fraction"] = fraction.item()
            if self.pair_bias_mode == "ema_byte_floor":
                for pair_id, value in enumerate(self.dynamic_pair_bias):
                    self.last_metrics[f"dynamic_pair_bias_{pair_id}"] = value.item()
                for pair_id, value in enumerate(self.pair_load_ema):
                    self.last_metrics[f"pair_load_ema_{pair_id}"] = value.item()
                self.last_metrics["pair_bias_step"] = float(
                    self.pair_bias_step.item()
                )

        if flat_patch_lengths is not None:
            active_lengths = flat_patch_lengths.float()[active_unit_mask]
            self.last_metrics.update(
                {
                    "patch_length_mean": (
                        active_lengths.mean().item()
                        if active_lengths.numel() > 0
                        else 0.0
                    ),
                    "patch_length_max": (
                        active_lengths.max().item()
                        if active_lengths.numel() > 0
                        else 0.0
                    ),
                }
            )
            self._record_per_expert_feature_metrics(
                "patch_length",
                flat_patch_lengths.float(),
                top_indices,
                flat_top_indices,
                active_unit_mask,
                unit_assignment_weights,
                unit_assignments,
            )
            self._record_bucket_usage_metrics(
                "length",
                (
                    ("short", flat_patch_lengths <= 4),
                    ("medium", (flat_patch_lengths > 4) & (flat_patch_lengths <= 8)),
                    ("long", flat_patch_lengths > 8),
                ),
                top_indices,
                flat_top_indices,
                active_unit_mask,
                unit_assignment_weights,
                unit_assignments,
                total_unit_assignments,
            )
            if self.num_experts % self.pair_size == 0:
                self._record_per_pair_feature_metrics(
                    "length",
                    flat_patch_lengths.float(),
                    top_indices,
                    flat_top_indices,
                    unit_assignment_weights,
                    unit_pair_assignments,
                )

        if flat_patch_entropies is not None:
            active_entropies = flat_patch_entropies.float()[active_unit_mask]
            self.last_metrics.update(
                {
                    "patch_entropy_mean": (
                        active_entropies.mean().item()
                        if active_entropies.numel() > 0
                        else 0.0
                    ),
                    "patch_entropy_max": (
                        active_entropies.max().item()
                        if active_entropies.numel() > 0
                        else 0.0
                    ),
                }
            )
            if active_entropies.numel() > 0:
                active_router_probs = router_probs[active_unit_mask].float()
                expert_ids = torch.arange(
                    self.num_experts,
                    device=router_probs.device,
                    dtype=torch.float32,
                )
                expected_expert_id = active_router_probs @ expert_ids
                selected_expert_id = top_indices[active_unit_mask].float().mean(dim=-1)
                self.last_metrics.update(
                    {
                        "entropy_expected_expert_corr": self._safe_corr(
                            active_entropies, expected_expert_id
                        ),
                        "entropy_selected_expert_corr": self._safe_corr(
                            active_entropies, selected_expert_id
                        ),
                    }
                )
                if self.num_experts % 2 == 0:
                    pair_ids = torch.arange(
                        self.num_experts,
                        device=router_probs.device,
                        dtype=torch.float32,
                    ).div(2, rounding_mode="floor")
                    expected_pair_id = active_router_probs @ pair_ids
                    selected_pair_id = (
                        top_indices[active_unit_mask]
                        .float()
                        .div(2, rounding_mode="floor")
                        .mean(dim=-1)
                    )
                    self.last_metrics.update(
                        {
                            "entropy_expected_pair_corr": self._safe_corr(
                                active_entropies, expected_pair_id
                            ),
                            "entropy_selected_pair_corr": self._safe_corr(
                                active_entropies, selected_pair_id
                            ),
                        }
                    )
                    if self.top_k == 2:
                        selected_pairs = top_indices[active_unit_mask].div(
                            2, rounding_mode="floor"
                        )
                        self.last_metrics["top2_same_pair_fraction"] = (
                            (selected_pairs[:, 0] == selected_pairs[:, 1])
                            .float()
                            .mean()
                            .item()
                        )
            self._record_per_expert_feature_metrics(
                "patch_entropy",
                flat_patch_entropies.float(),
                top_indices,
                flat_top_indices,
                active_unit_mask,
                unit_assignment_weights,
                unit_assignments,
            )
            self._record_bucket_usage_metrics(
                "entropy",
                (
                    ("low", flat_patch_entropies <= 1.0),
                    (
                        "medium",
                        (flat_patch_entropies > 1.0) & (flat_patch_entropies <= 2.0),
                    ),
                    ("high", flat_patch_entropies > 2.0),
                ),
                top_indices,
                flat_top_indices,
                active_unit_mask,
                unit_assignment_weights,
                unit_assignments,
                total_unit_assignments,
            )
            if self.num_experts % self.pair_size == 0:
                self._record_per_pair_feature_metrics(
                    "entropy",
                    flat_patch_entropies.float(),
                    top_indices,
                    flat_top_indices,
                    unit_assignment_weights,
                    unit_pair_assignments,
                )

        if flat_patch_byte_features is not None:
            active_byte_features = flat_patch_byte_features.float()[active_unit_mask]
            for feature_id, feature_name in enumerate(PATCH_BYTE_TYPE_NAMES):
                values = flat_patch_byte_features[:, feature_id].float()
                self.last_metrics[f"patch_byte_{feature_name}_fraction_mean"] = (
                    active_byte_features[:, feature_id].mean().item()
                    if active_byte_features.numel() > 0
                    else 0.0
                )
                self._record_per_expert_feature_metrics(
                    f"patch_byte_{feature_name}_fraction",
                    values,
                    top_indices,
                    flat_top_indices,
                    active_unit_mask,
                    unit_assignment_weights,
                    unit_assignments,
                )

            dominant_byte_types = flat_patch_byte_features.argmax(dim=-1)
            has_classified_bytes = flat_patch_byte_features.sum(dim=-1) > 0
            self._record_bucket_usage_metrics(
                "byte_type",
                tuple(
                    (
                        feature_name,
                        (dominant_byte_types == feature_id) & has_classified_bytes,
                    )
                    for feature_id, feature_name in enumerate(PATCH_BYTE_TYPE_NAMES)
                ),
                top_indices,
                flat_top_indices,
                active_unit_mask,
                unit_assignment_weights,
                unit_assignments,
                total_unit_assignments,
            )

    @torch.no_grad()
    def _record_per_expert_feature_metrics(
        self,
        feature_name: str,
        values: torch.Tensor,
        top_indices: torch.Tensor,
        flat_top_indices: torch.Tensor,
        active_unit_mask: torch.Tensor,
        unit_assignment_weights: torch.Tensor,
        unit_assignments: torch.Tensor,
    ) -> None:
        del active_unit_mask
        expanded_values = values.unsqueeze(-1).expand_as(top_indices).reshape(-1)
        feature_sums = torch.zeros(
            self.num_experts, device=values.device, dtype=torch.float32
        )
        feature_sums.scatter_add_(
            0,
            flat_top_indices,
            expanded_values * unit_assignment_weights,
        )
        feature_means = feature_sums / unit_assignments.clamp_min(1.0)
        for expert_id, mean_value in enumerate(feature_means):
            self.last_metrics[f"expert_{expert_id}_{feature_name}_mean"] = (
                mean_value.item()
            )

    @torch.no_grad()
    def _record_per_pair_feature_metrics(
        self,
        feature_name: str,
        values: torch.Tensor,
        top_indices: torch.Tensor,
        flat_top_indices: torch.Tensor,
        unit_assignment_weights: torch.Tensor,
        unit_pair_assignments: torch.Tensor,
    ) -> None:
        pair_count = self.num_experts // self.pair_size
        flat_pair_indices = flat_top_indices.div(self.pair_size, rounding_mode="floor")
        expanded_values = values.unsqueeze(-1).expand_as(top_indices).reshape(-1)
        feature_sums = torch.zeros(
            pair_count, device=values.device, dtype=torch.float32
        )
        feature_sums.scatter_add_(
            0,
            flat_pair_indices,
            expanded_values * unit_assignment_weights,
        )
        feature_means = feature_sums / unit_pair_assignments.clamp_min(1.0)
        for pair_id, mean_value in enumerate(feature_means):
            self.last_metrics[f"pair_{pair_id}_{feature_name}_mean"] = (
                mean_value.item()
            )

    @torch.no_grad()
    def _record_bucket_usage_metrics(
        self,
        feature_name: str,
        buckets: tuple[tuple[str, torch.Tensor], ...],
        top_indices: torch.Tensor,
        flat_top_indices: torch.Tensor,
        active_unit_mask: torch.Tensor,
        unit_assignment_weights: torch.Tensor,
        unit_assignments: torch.Tensor,
        total_unit_assignments: torch.Tensor,
    ) -> None:
        del unit_assignment_weights
        active_unit_count = active_unit_mask.float().sum().clamp_min(1.0)
        for bucket_name, bucket_mask in buckets:
            bucket_unit_mask = bucket_mask & active_unit_mask
            bucket_units = bucket_unit_mask.float().sum()
            bucket_assignment_weights = (
                bucket_unit_mask.float()
                .unsqueeze(-1)
                .expand_as(top_indices)
                .reshape(-1)
            )
            bucket_assignments = torch.zeros(
                self.num_experts, device=top_indices.device, dtype=torch.float32
            )
            bucket_assignments.scatter_add_(
                0, flat_top_indices, bucket_assignment_weights
            )
            bucket_total_assignments = bucket_assignments.sum()
            self.last_metrics[f"{feature_name}_bucket_{bucket_name}_unit_fraction"] = (
                bucket_units / active_unit_count
            ).item()
            self.last_metrics[
                f"{feature_name}_bucket_{bucket_name}_assignment_fraction"
            ] = (bucket_total_assignments / total_unit_assignments).item()
            bucket_distribution = (
                bucket_assignments / bucket_total_assignments.clamp_min(1.0)
            )
            for expert_id, fraction in enumerate(bucket_distribution):
                self.last_metrics[
                    f"{feature_name}_bucket_{bucket_name}_expert_{expert_id}_assignment_fraction"
                ] = fraction.item()
            expert_bucket_fraction = bucket_assignments / unit_assignments.clamp_min(
                1.0
            )
            for expert_id, fraction in enumerate(expert_bucket_fraction):
                self.last_metrics[
                    f"expert_{expert_id}_{feature_name}_bucket_{bucket_name}_assignment_fraction"
                ] = fraction.item()

    def reset_parameters(self, init_std=None, factor=1.0):
        router_init_std = init_std or (self.dim ** (-0.5)) / factor
        trunc_normal_(
            self.router.weight,
            mean=0.0,
            std=router_init_std,
            a=-3 * router_init_std,
            b=3 * router_init_std,
        )
        if self.patch_feature_router is not None:
            trunc_normal_(
                self.patch_feature_router.weight,
                mean=0.0,
                std=router_init_std,
                a=-3 * router_init_std,
                b=3 * router_init_std,
            )
            if self.patch_feature_router.bias is not None:
                nn.init.zeros_(self.patch_feature_router.bias)
        if self.entropy_router is not None:
            for layer in self.entropy_router:
                if isinstance(layer, nn.Linear):
                    trunc_normal_(
                        layer.weight,
                        mean=0.0,
                        std=router_init_std,
                        a=-3 * router_init_std,
                        b=3 * router_init_std,
                    )
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
        for expert in self.experts:
            if expert is not None:
                expert.reset_parameters(init_std, factor)


def _iter_sparse_moe_modules(module: nn.Module):
    for name, child in module.named_modules():
        if getattr(child, "is_sparse_moe", False):
            yield name, child


def get_moe_aux_loss(module: nn.Module) -> Optional[torch.Tensor]:
    losses = [
        child.last_balance_loss
        for _, child in _iter_sparse_moe_modules(module)
        if child.last_balance_loss is not None
    ]
    if len(losses) == 0:
        return None
    return torch.stack(losses).mean()


def get_moe_router_z_loss(module: nn.Module) -> Optional[torch.Tensor]:
    losses = [
        child.last_router_z_loss
        for _, child in _iter_sparse_moe_modules(module)
        if child.last_router_z_loss is not None
    ]
    if len(losses) == 0:
        return None
    return torch.stack(losses).mean()


def get_moe_router_anticollapse_loss(module: nn.Module) -> Optional[torch.Tensor]:
    losses = [
        child.last_router_anticollapse_loss
        for _, child in _iter_sparse_moe_modules(module)
        if child.last_router_anticollapse_loss is not None
    ]
    if len(losses) == 0:
        return None
    return torch.stack(losses).mean()


def get_moe_balance_loss_weight_for_step(args: BaseTransformerArgs, step: int) -> float:
    base_weight = float(args.moe_balance_loss_weight)
    if args.moe_balance_schedule == "constant":
        return base_weight
    if args.moe_balance_schedule != "linear_decay":
        raise ValueError(f"Unsupported MoE balance schedule: {args.moe_balance_schedule}")

    start = int(args.moe_balance_start_step)
    peak = int(args.moe_balance_peak_step)
    decay_start = int(args.moe_balance_decay_start_step)
    end = int(args.moe_balance_end_step)
    final_weight = float(args.moe_balance_final_weight)

    if step < start:
        return 0.0
    if peak > start and step < peak:
        return base_weight * (step - start) / (peak - start)
    if step < decay_start or end <= decay_start:
        return base_weight
    if step < end:
        progress = (step - decay_start) / (end - decay_start)
        return base_weight + progress * (final_weight - base_weight)
    return final_weight


def get_moe_metrics(module: nn.Module) -> dict[str, float]:
    metrics = [
        child.last_metrics
        for _, child in _iter_sparse_moe_modules(module)
        if child.last_metrics
    ]
    if len(metrics) == 0:
        return {}

    keys = sorted({key for item in metrics for key in item})
    out: dict[str, float] = {"num_layers": float(len(metrics))}
    for key in keys:
        values = [item[key] for item in metrics if key in item]
        out[f"{key}_mean"] = sum(values) / len(values)
    return out


def set_moe_router_step(module: nn.Module, step: int) -> None:
    for _, child in _iter_sparse_moe_modules(module):
        if hasattr(child, "set_router_step"):
            child.set_router_step(step)


class TransformerBlock(nn.Module):
    def __init__(self, args: BaseTransformerArgs, layer_idx: int | None = None):
        super().__init__()

        assert (args.head_dim is not None) or (
            args.n_heads is not None
        ), "Should specify at least head_dim or n_heads"
        self.head_dim = args.head_dim or args.dim // args.n_heads
        self.n_heads = args.n_heads or args.dim // args.head_dim
        self.n_kv_heads = args.n_kv_heads or self.n_heads

        assert args.n_heads % self.n_kv_heads == 0
        assert args.dim % args.n_heads == 0

        self.attention = Attention(
            dim=args.dim,
            head_dim=self.head_dim,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            rope_theta=args.rope_theta,
        )
        moe_frequency = max(args.moe_layer_frequency, 1)
        use_moe = args.moe_num_experts > 0 and (
            layer_idx is None or layer_idx % moe_frequency == 0
        )
        ffn_cls = SparseMoEFeedForward if use_moe else FeedForward
        ffn_kwargs = (
            dict(
                num_experts=args.moe_num_experts,
                top_k=args.moe_top_k,
                expert_parallel_size=args.moe_ep_size,
                ffn_dim_multiplier=(
                    args.moe_ffn_dim_multiplier
                    if args.moe_ffn_dim_multiplier is not None
                    else args.ffn_dim_multiplier
                ),
                router_jitter=args.moe_router_jitter,
                router_congestion_weight=args.moe_router_congestion_weight,
                router_z_loss_weight=args.moe_router_z_loss_weight,
                router_anticollapse_loss_weight=args.moe_router_anticollapse_loss_weight,
                router_dominance_loss_weight=args.moe_router_dominance_loss_weight,
                router_dominance_threshold=args.moe_router_dominance_threshold,
                router_use_hidden_state=args.moe_router_use_hidden_state,
                router_patch_feature_bias=args.moe_router_patch_feature_bias,
                router_normalize_patch_entropy=args.moe_router_normalize_patch_entropy,
                router_entropy_mlp_hidden_dim=args.moe_router_entropy_mlp_hidden_dim,
                router_entropy_mean=args.moe_router_entropy_mean,
                router_entropy_std=args.moe_router_entropy_std,
                router_entropy_clip=args.moe_router_entropy_clip,
                router_use_patch_length=args.moe_router_use_patch_length,
                router_use_patch_entropy=args.moe_router_use_patch_entropy,
                router_use_patch_byte_features=args.moe_router_use_patch_byte_features,
                routing_granularity=args.moe_routing_granularity,
                pair_size=args.moe_pair_size,
                entropy_prior_mode=args.moe_entropy_prior_mode,
                entropy_prior_scale=args.moe_entropy_prior_scale,
                entropy_prior_hidden_scale=args.moe_entropy_prior_hidden_scale,
                entropy_prior_calibration_path=args.moe_entropy_prior_calibration_path,
                entropy_prior_trainable=args.moe_entropy_prior_trainable,
                pair_bias_mode=args.moe_pair_bias_mode,
                pair_bias_ema=args.moe_pair_bias_ema,
                pair_bias_update_interval=args.moe_pair_bias_update_interval,
                pair_bias_lr=args.moe_pair_bias_lr,
                pair_bias_clip=args.moe_pair_bias_clip,
                pair_min_byte_fraction=args.moe_pair_min_byte_fraction,
                pair_max_byte_fraction=args.moe_pair_max_byte_fraction,
                hidden_residual_ramp_start_step=args.moe_hidden_residual_ramp_start_step,
                hidden_residual_ramp_end_step=args.moe_hidden_residual_ramp_end_step,
                hidden_residual_final_scale=args.moe_hidden_residual_final_scale,
                balance_cost=args.moe_balance_cost,
            )
            if use_moe
            else {}
        )
        self.feed_forward = ffn_cls(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=(
                args.ffn_dim_multiplier
                if not use_moe
                else ffn_kwargs.pop("ffn_dim_multiplier")
            ),
            **ffn_kwargs,
        )
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        freq_cis: torch.Tensor,
        tok_idx: Optional[torch.Tensor] = None,
        mask: Optional[Union[BlockMask, AttentionBias, str]] = None,
        attn_impl: str = "sdpa",
        patch_lengths: Optional[torch.Tensor] = None,
        patch_entropies: Optional[torch.Tensor] = None,
        patch_byte_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        attn_out = self.attention(
            self.attention_norm(x),
            freq_cis,
            tok_idx=tok_idx,
            mask=mask,
            attn_impl=attn_impl,
        )
        h = x + attn_out
        h_norm = self.ffn_norm(h)
        if isinstance(self.feed_forward, SparseMoEFeedForward):
            ffn_out = self.feed_forward(
                h_norm,
                patch_lengths=patch_lengths,
                patch_entropies=patch_entropies,
                patch_byte_features=patch_byte_features,
            )
        else:
            ffn_out = self.feed_forward(h_norm)
        out = h + ffn_out
        return out

    def init_weights(self, init_std=None, factor=1.0):
        self.attention.reset_parameters(init_std, factor)
        self.attention_norm.reset_parameters()

        self.feed_forward.reset_parameters(init_std, factor)
        self.ffn_norm.reset_parameters()


class SequenceModelWithOutput(abc.ABC):
    @abc.abstractmethod
    def get_output_seq_len(self) -> int:
        pass


class BaseTransformer(nn.Module, SequenceModelWithOutput):
    def __init__(self, args: BaseTransformerArgs):
        super().__init__()
        self.dim = args.dim
        self.init_base_std = args.init_base_std
        self.attn_impl = args.attn_impl
        self.attn_bias_type = args.attn_bias_type
        self.init_std_factor = InitStdFactor(args.init_std_factor)
        self.max_seqlen = args.max_seqlen
        self.rope_embeddings = RotaryEmbedding(
            theta=args.rope_theta,
            head_dim=args.head_dim or args.dim // args.n_heads,
            max_seqlen=args.max_seqlen,
            rope_use_fp32_in_outer_product=args.rope_use_fp32_in_outer_product,
        )
        self.eos_id = args.eos_id

        self.layers = nn.ModuleList()
        for layer_idx in range(args.n_layers):
            self.layers.append(TransformerBlock(args, layer_idx=layer_idx))

    def get_output_seq_len(self):
        return self.max_seqlen

    def forward(
        self,
        h,
        tok_idx: Optional[torch.Tensor] = None,
        mask: Optional[Union[BlockMask, AttentionBias, str]] = None,
        attn_impl: str = "sdpa",
        patch_lengths: Optional[torch.Tensor] = None,
        patch_entropies: Optional[torch.Tensor] = None,
        patch_byte_features: Optional[torch.Tensor] = None,
    ):

        freq_cis = self.rope_embeddings(seqlen=self.max_seqlen, tok_idx=tok_idx)

        for i, layer in enumerate(self.layers):
            h = layer(
                h,
                freq_cis,
                tok_idx=tok_idx,
                mask=mask,
                attn_impl=attn_impl,
                patch_lengths=patch_lengths,
                patch_entropies=patch_entropies,
                patch_byte_features=patch_byte_features,
            )
        return h

    def reset_rope_embeddings(self):
        self.rope_embeddings.reset_parameters()

    def init_weights(self):
        self.reset_rope_embeddings()
        for depth, layer in enumerate(self.layers):
            factor = {
                InitStdFactor.CURRENT_DEPTH: (2 * (depth + 1)) ** 0.5,
                InitStdFactor.GLOBAL_DEPTH: (2 * (len(self.layers) + 1)) ** 0.5,
                InitStdFactor.DIM_RATIO: self.dim / 4096,
                InitStdFactor.DISABLED: 1.0,
            }[self.init_std_factor]

            layer.init_weights(self.init_base_std, factor)
