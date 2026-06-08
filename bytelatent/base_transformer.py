# Copyright (c) Meta Platforms, Inc. and affiliates.
import abc
import logging
import os
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
    moe_assignment_balance_loss_weight: float = 0.0
    moe_router_jitter: float = 0.0
    moe_router_congestion_weight: float = 0.0
    moe_router_z_loss_weight: float = 0.0
    moe_router_use_patch_length: bool = False
    moe_router_use_patch_entropy: bool = False
    moe_router_use_patch_byte_features: bool = False
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
        assignment_balance_loss_weight: float = 0.0,
        router_jitter: float = 0.0,
        router_congestion_weight: float = 0.0,
        router_z_loss_weight: float = 0.0,
        router_use_patch_length: bool = False,
        router_use_patch_entropy: bool = False,
        router_use_patch_byte_features: bool = False,
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
        if assignment_balance_loss_weight < 0:
            raise ValueError("assignment_balance_loss_weight must be non-negative")
        if router_congestion_weight < 0:
            raise ValueError("router_congestion_weight must be non-negative")
        if router_z_loss_weight < 0:
            raise ValueError("router_z_loss_weight must be non-negative")
        if balance_cost not in {"patch", "byte", "entropy_byte"}:
            raise ValueError("balance_cost must be one of: patch, byte, entropy_byte")

        self.dim = dim
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.requested_expert_parallel_size = expert_parallel_size
        self.expert_parallel_size = 1
        self.expert_parallel_rank = 0
        self.expert_parallel_group = None
        self.local_expert_ids = tuple(range(num_experts))
        self.assignment_balance_loss_weight = assignment_balance_loss_weight
        self.router_jitter = router_jitter
        self.router_congestion_weight = router_congestion_weight
        self.router_z_loss_weight = router_z_loss_weight
        self.router_use_patch_length = router_use_patch_length
        self.router_use_patch_entropy = router_use_patch_entropy
        self.router_use_patch_byte_features = router_use_patch_byte_features
        self.balance_cost = balance_cost
        self.is_sparse_moe = True

        self.router = nn.Linear(dim, num_experts, bias=False)
        self.patch_feature_names = []
        if router_use_patch_length:
            self.patch_feature_names.append("length")
        if router_use_patch_entropy:
            self.patch_feature_names.append("entropy")
        if router_use_patch_byte_features:
            self.patch_feature_names.extend(
                f"byte_{name}" for name in PATCH_BYTE_TYPE_NAMES
            )
        self.patch_feature_router = (
            nn.Linear(len(self.patch_feature_names), num_experts, bias=False)
            if len(self.patch_feature_names) > 0
            else None
        )
        self.patch_length_router = self.patch_feature_router
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
        self.last_assignment_balance_loss: Optional[torch.Tensor] = None
        self.last_router_z_loss: Optional[torch.Tensor] = None
        self.last_metrics: dict[str, float] = {}
        self.last_expert_parallel_metrics: dict[str, float] = {}

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
        load_weights = self._load_weights(
            flat_x, flat_patch_lengths, flat_patch_entropies
        )

        router_logits = self.router(flat_x)
        learned_router_logits = router_logits
        if self.patch_feature_router is not None:
            patch_features = self._patch_features(
                flat_patch_lengths,
                flat_patch_entropies,
                flat_patch_byte_features,
            )
            patch_features = patch_features.to(
                dtype=self.patch_feature_router.weight.dtype
            )
            router_logits = router_logits + self.patch_feature_router(
                patch_features
            ).to(router_logits.dtype)
            learned_router_logits = router_logits

        self.last_router_z_loss = (
            torch.logsumexp(learned_router_logits.float(), dim=-1).square().mean()
        )

        if self.training and self.router_jitter > 0:
            router_logits = router_logits + torch.empty_like(router_logits).uniform_(
                -self.router_jitter, self.router_jitter
            )

        pre_congestion_router_probs = F.softmax(router_logits.float(), dim=-1)
        congestion_price = self._congestion_price(
            pre_congestion_router_probs, load_weights
        )
        if self.router_congestion_weight > 0:
            router_logits = router_logits - self.router_congestion_weight * (
                congestion_price.to(router_logits.dtype)
            )
            router_probs = F.softmax(router_logits.float(), dim=-1)
        else:
            router_probs = pre_congestion_router_probs
        top_weights, top_indices = torch.topk(
            router_probs, k=self.top_k, dim=-1, sorted=False
        )
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True).clamp_min(
            1e-9
        )

        if self.requested_expert_parallel_size > 1:
            if self.expert_parallel_size == 1:
                raise RuntimeError(
                    "Expert parallelism was requested but has not been configured"
                )
            flat_output = self._forward_expert_parallel(
                flat_x, top_indices, top_weights
            )
        else:
            flat_output = self._forward_local(flat_x, top_indices, top_weights)

        prob_density = (router_probs * load_weights.unsqueeze(-1)).sum(dim=0)
        prob_density = prob_density / load_weights.sum().clamp_min(1.0)
        uniform = torch.full_like(prob_density, 1.0 / self.num_experts)
        self.last_balance_loss = self.num_experts * torch.sum(
            (prob_density - uniform) ** 2
        )
        self.last_assignment_balance_loss = self._assignment_balance_loss(
            router_probs, top_indices, load_weights
        )
        self._record_metrics(
            router_probs,
            top_indices,
            load_weights,
            flat_patch_lengths,
            flat_patch_entropies,
            flat_patch_byte_features,
            congestion_price,
            pre_congestion_router_probs,
        )
        return flat_output.reshape_as(x)

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
            if self.router_use_patch_entropy or self.balance_cost == "entropy_byte":
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
        if self.router_use_patch_entropy:
            assert flat_patch_entropies is not None
            entropy_feature = flat_patch_entropies.float().clamp_min(0.0)
            if flat_patch_lengths is None:
                active_entropies = entropy_feature[entropy_feature > 0]
            else:
                active_entropies = entropy_feature[flat_patch_lengths > 0]
            if active_entropies.numel() > 0:
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
        total_load = load_weights.sum()
        load_density = (router_probs * load_weights.unsqueeze(-1)).sum(dim=0)
        load_density = load_density / total_load.clamp_min(1.0)
        has_load = (total_load > 0).to(load_density.dtype)
        return (self.num_experts * load_density - 1.0).mul(has_load).detach()

    def _assignment_balance_loss(
        self,
        router_probs: torch.Tensor,
        top_indices: torch.Tensor,
        load_weights: torch.Tensor,
    ) -> torch.Tensor:
        total_load = load_weights.sum()
        top_mask = torch.zeros_like(router_probs)
        top_mask.scatter_(1, top_indices, 1.0)
        selected_density = (router_probs * top_mask * load_weights.unsqueeze(-1)).sum(
            dim=0
        )
        selected_density = selected_density / total_load.clamp_min(1.0)
        target_density = selected_density.sum().detach() / self.num_experts
        has_load = (total_load > 0).to(selected_density.dtype)
        return (
            self.num_experts
            * torch.sum((selected_density - target_density) ** 2)
            * has_load
        )

    @torch.no_grad()
    def _record_metrics(
        self,
        router_probs: torch.Tensor,
        top_indices: torch.Tensor,
        load_weights: torch.Tensor,
        flat_patch_lengths: Optional[torch.Tensor],
        flat_patch_entropies: Optional[torch.Tensor],
        flat_patch_byte_features: Optional[torch.Tensor] = None,
        congestion_price: Optional[torch.Tensor] = None,
        pre_congestion_router_probs: Optional[torch.Tensor] = None,
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
            "assignment_balance_loss": (
                self.last_assignment_balance_loss.detach().item()
                if self.last_assignment_balance_loss is not None
                else 0.0
            ),
            "assignment_balance_loss_weight": self.assignment_balance_loss_weight,
            "routed_units": float(routed_units),
            "total_units": float(load_weights.numel()),
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


def get_moe_assignment_balance_loss(module: nn.Module) -> Optional[torch.Tensor]:
    losses = [
        child.last_assignment_balance_loss
        for _, child in _iter_sparse_moe_modules(module)
        if child.last_assignment_balance_loss is not None
    ]
    if len(losses) == 0:
        return None
    return torch.stack(losses).mean()


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
                assignment_balance_loss_weight=(
                    args.moe_assignment_balance_loss_weight
                ),
                ffn_dim_multiplier=(
                    args.moe_ffn_dim_multiplier
                    if args.moe_ffn_dim_multiplier is not None
                    else args.ffn_dim_multiplier
                ),
                router_jitter=args.moe_router_jitter,
                router_congestion_weight=args.moe_router_congestion_weight,
                router_z_loss_weight=args.moe_router_z_loss_weight,
                router_use_patch_length=args.moe_router_use_patch_length,
                router_use_patch_entropy=args.moe_router_use_patch_entropy,
                router_use_patch_byte_features=args.moe_router_use_patch_byte_features,
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
