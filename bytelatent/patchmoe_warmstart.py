# Copyright (c) Meta Platforms, Inc. and affiliates.

import json
import math
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.distributed.checkpoint as dcp
from torch import nn

from bytelatent.base_transformer import PATCH_BYTE_TYPE_NAMES

DENSE_GLOBAL_FFN_RE = re.compile(
    r"^global_transformer\.layers\.(?P<layer>\d+)\."
    r"feed_forward\.(?P<weight>w[123])\.weight$"
)
PATCH_FEATURE_CHOICES = ("length", "entropy", "byte_type")
EXPERT_INIT_MODES = ("replicated_prefix", "dense_copy", "paired_partition")
PATCH_FEATURE_INIT_MODES = ("random", "entropy_bands", "length_entropy_anchors")
MANIFEST_NAME = "patchmoe_warmstart_manifest.json"


@dataclass(frozen=True)
class PatchMoEWarmStartSpec:
    num_experts: int = 8
    top_k: int = 2
    layer_frequency: int = 1
    patch_features: tuple[str, ...] = ("entropy",)
    router_seed: int = 42
    init_std_factor: str = "current_depth"
    expert_ffn_dim_multiplier: float | None = None
    expert_init_mode: str = "replicated_prefix"
    patch_feature_bias: bool = False
    patch_feature_init: str = "random"
    # Keep patch-feature router rows independent even when FFN experts use
    # paired_partition initialization.
    pair_patch_feature_router: bool = False
    routing_granularity: str = "expert"
    entropy_band_logit_scale: float = 1.0
    entropy_prior_mode: str = "none"
    entropy_prior_calibration_path: str | None = None
    pair_bias_mode: str = "none"
    hidden_residual_ramp_start_step: int = 0
    hidden_residual_ramp_end_step: int = 0
    entropy_mlp_hidden_dim: int = 0

    def __post_init__(self):
        if self.num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if self.top_k <= 0 or self.top_k > self.num_experts:
            raise ValueError("top_k must be in [1, num_experts]")
        if self.layer_frequency <= 0:
            raise ValueError("layer_frequency must be positive")
        unknown_features = set(self.patch_features) - set(PATCH_FEATURE_CHOICES)
        if unknown_features:
            raise ValueError(f"Unknown patch features: {sorted(unknown_features)}")
        if len(set(self.patch_features)) != len(self.patch_features):
            raise ValueError("patch_features must not contain duplicates")
        if (
            self.expert_ffn_dim_multiplier is not None
            and self.expert_ffn_dim_multiplier <= 0
        ):
            raise ValueError("expert_ffn_dim_multiplier must be positive")
        if self.init_std_factor not in {
            "disabled",
            "current_depth",
            "global_depth",
            "dim_ratio",
        }:
            raise ValueError(f"Unsupported init_std_factor: {self.init_std_factor}")
        if self.expert_init_mode not in EXPERT_INIT_MODES:
            raise ValueError(
                f"expert_init_mode must be one of: {', '.join(EXPERT_INIT_MODES)}"
            )
        if self.patch_feature_init not in PATCH_FEATURE_INIT_MODES:
            raise ValueError(
                "patch_feature_init must be one of: "
                f"{', '.join(PATCH_FEATURE_INIT_MODES)}"
            )
        if self.routing_granularity not in {"expert", "pair"}:
            raise ValueError("routing_granularity must be one of: expert, pair")
        if self.entropy_prior_mode not in {"none", "gaussian_pairs"}:
            raise ValueError("entropy_prior_mode must be one of: none, gaussian_pairs")
        if self.pair_bias_mode not in {"none", "ema_byte_floor"}:
            raise ValueError("pair_bias_mode must be one of: none, ema_byte_floor")
        if self.entropy_band_logit_scale <= 0:
            raise ValueError("entropy_band_logit_scale must be positive")
        if self.patch_feature_init == "entropy_bands":
            if "entropy" not in self.patch_features:
                raise ValueError("entropy_bands patch_feature_init requires entropy")
            if self.top_k != 2:
                raise ValueError("entropy_bands patch_feature_init requires top_k=2")
            if self.num_experts % 2 != 0:
                raise ValueError(
                    "entropy_bands patch_feature_init requires an even num_experts"
                )
            if not self.patch_feature_bias:
                raise ValueError(
                    "entropy_bands patch_feature_init requires patch_feature_bias"
                )
        if self.patch_feature_init == "length_entropy_anchors":
            if self.routing_granularity != "pair":
                raise ValueError(
                    "length_entropy_anchors patch_feature_init requires pair routing"
                )
            if "length" not in self.patch_features or "entropy" not in self.patch_features:
                raise ValueError(
                    "length_entropy_anchors patch_feature_init requires length and entropy"
                )
            if self.top_k != 2:
                raise ValueError("length_entropy_anchors patch_feature_init requires top_k=2")
            if self.num_experts % 2 != 0:
                raise ValueError(
                    "length_entropy_anchors patch_feature_init requires an even num_experts"
                )
            if not self.patch_feature_bias:
                raise ValueError(
                    "length_entropy_anchors patch_feature_init requires patch_feature_bias"
                )
            if self.entropy_prior_calibration_path is None:
                raise ValueError(
                    "length_entropy_anchors requires entropy_prior_calibration_path"
                )

        if self.expert_init_mode == "paired_partition":
            if self.top_k != 2:
                raise ValueError("paired_partition requires top_k=2")
            if self.num_experts % 2 != 0:
                raise ValueError("paired_partition requires an even num_experts")
            if self.expert_ffn_dim_multiplier != 0.5:
                raise ValueError(
                    "paired_partition requires expert_ffn_dim_multiplier=0.5"
                )
        if self.routing_granularity == "pair":
            if self.num_experts % 2 != 0:
                raise ValueError("pair routing requires an even num_experts")
            if self.top_k != 2:
                raise ValueError("pair routing requires top_k=2")
            if self.expert_init_mode != "paired_partition":
                raise ValueError("pair routing warmstart requires paired_partition")
        if self.entropy_prior_mode == "gaussian_pairs":
            if self.routing_granularity != "pair":
                raise ValueError("gaussian_pairs prior warmstart requires pair routing")
            if self.entropy_prior_calibration_path is None:
                raise ValueError(
                    "entropy_prior_calibration_path is required for gaussian_pairs"
                )
        if self.pair_bias_mode == "ema_byte_floor" and self.routing_granularity != "pair":
            raise ValueError("ema_byte_floor warmstart requires pair routing")
        if self.hidden_residual_ramp_start_step < 0 or self.hidden_residual_ramp_end_step < 0:
            raise ValueError("hidden residual ramp steps must be non-negative")
        if self.entropy_mlp_hidden_dim < 0:
            raise ValueError("entropy_mlp_hidden_dim must be non-negative")

    @property
    def patch_feature_count(self) -> int:
        count = 0
        if "length" in self.patch_features:
            count += 1
        if "entropy" in self.patch_features:
            count += 1
        if "byte_type" in self.patch_features:
            count += len(PATCH_BYTE_TYPE_NAMES)
        return count


@dataclass(frozen=True)
class PatchMoEWarmStartReport:
    source_parameter_count: int
    target_parameter_count: int
    source_global_ffn_parameter_count: int
    selected_layers: tuple[int, ...]
    global_layers: tuple[int, ...]
    patch_feature_count: int


@dataclass(frozen=True)
class PatchMoEWarmStartVerificationReport:
    dcp_key_count: int
    sample_keys: tuple[str, ...]


def _global_ffn_weights(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[int, dict[str, torch.Tensor]]:
    layer_weights: dict[int, dict[str, torch.Tensor]] = {}
    for key, value in state_dict.items():
        match = DENSE_GLOBAL_FFN_RE.match(key)
        if match is None:
            continue
        layer_idx = int(match.group("layer"))
        weight_name = match.group("weight")
        layer_weights.setdefault(layer_idx, {})[weight_name] = value

    if not layer_weights:
        raise ValueError("No dense global Transformer FFN weights found")
    for layer_idx, weights in layer_weights.items():
        if set(weights) != {"w1", "w2", "w3"}:
            raise ValueError(
                f"Global Transformer layer {layer_idx} has incomplete FFN weights: "
                f"{sorted(weights)}"
            )
    return layer_weights


def _router_init_std(
    *, dim: int, layer_idx: int, global_layer_count: int, init_std_factor: str
) -> float:
    factor = {
        "disabled": 1.0,
        "current_depth": math.sqrt(2 * (layer_idx + 1)),
        "global_depth": math.sqrt(2 * (global_layer_count + 1)),
        "dim_ratio": dim / 4096,
    }[init_std_factor]
    return (dim**-0.5) / factor


def _truncated_normal(
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    std: float,
    generator: torch.Generator,
) -> torch.Tensor:
    tensor = torch.empty(shape, dtype=torch.float32)
    nn.init.trunc_normal_(
        tensor,
        mean=0.0,
        std=std,
        a=-3 * std,
        b=3 * std,
        generator=generator,
    )
    return tensor.to(dtype=dtype)


def _scale_dense_ffn_weight(
    weight: torch.Tensor,
    weight_name: str,
    expert_ffn_dim_multiplier: float | None,
) -> torch.Tensor:
    if expert_ffn_dim_multiplier is None:
        return weight
    if weight_name in {"w1", "w3"}:
        hidden_dim = weight.shape[0]
        target_hidden_dim = int(
            math.ceil(hidden_dim * expert_ffn_dim_multiplier / 256) * 256
        )
        return weight[:target_hidden_dim, :].contiguous()
    if weight_name == "w2":
        hidden_dim = weight.shape[1]
        target_hidden_dim = int(
            math.ceil(hidden_dim * expert_ffn_dim_multiplier / 256) * 256
        )
        return weight[:, :target_hidden_dim].contiguous()
    raise ValueError(f"Unknown FFN weight: {weight_name}")


def _paired_partition_ffn_weight(
    weight: torch.Tensor,
    weight_name: str,
    expert_idx: int,
) -> torch.Tensor:
    split_dim = 0 if weight_name in {"w1", "w3"} else 1
    hidden_dim = weight.shape[split_dim]
    if hidden_dim % 2 != 0:
        raise ValueError(
            f"paired_partition requires an even FFN hidden dimension, got {hidden_dim}"
        )
    partition_idx = expert_idx % 2
    start = partition_idx * (hidden_dim // 2)
    expert_weight = weight.narrow(split_dim, start, hidden_dim // 2).contiguous()
    if weight_name == "w2":
        # Top-2 gives paired experts equal 0.5 weights at initialization.
        expert_weight = expert_weight * 2
    return expert_weight


def _paired_router_weight(
    *,
    num_experts: int,
    input_dim: int,
    dtype: torch.dtype,
    std: float,
    generator: torch.Generator,
) -> torch.Tensor:
    pair_weight = _truncated_normal(
        (num_experts // 2, input_dim),
        dtype=dtype,
        std=std,
        generator=generator,
    )
    return pair_weight.repeat_interleave(2, dim=0)


def _pair_router_weight(
    *,
    num_experts: int,
    input_dim: int,
    dtype: torch.dtype,
    std: float,
    generator: torch.Generator,
) -> torch.Tensor:
    return _truncated_normal(
        (num_experts // 2, input_dim),
        dtype=dtype,
        std=std,
        generator=generator,
    )


def _entropy_band_patch_feature_router(
    spec: PatchMoEWarmStartSpec,
    *,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    entropy_feature_idx = spec.patch_features.index("entropy")
    pair_count = spec.num_experts // 2
    # The router uses linear logits. These lines are the upper envelope of
    # -scale * (entropy - center)^2, dropping the common -entropy^2 term.
    # With duplicated lines per pair, Top-2 selects both paired experts while
    # different entropy ranges prefer different pairs.
    centers = torch.linspace(0.5, 3.5, pair_count, dtype=torch.float32)
    slopes = 2.0 * spec.entropy_band_logit_scale * centers
    biases = -spec.entropy_band_logit_scale * centers.square()
    output_rows = (
        pair_count if spec.routing_granularity == "pair" else spec.num_experts
    )
    weight = torch.zeros(output_rows, spec.patch_feature_count, dtype=torch.float32)
    bias = torch.zeros(output_rows, dtype=torch.float32)
    for pair_idx in range(pair_count):
        if spec.routing_granularity == "pair":
            weight[pair_idx, entropy_feature_idx] = slopes[pair_idx]
            bias[pair_idx] = biases[pair_idx]
        else:
            start = 2 * pair_idx
            weight[start : start + 2, entropy_feature_idx] = slopes[pair_idx]
            bias[start : start + 2] = biases[pair_idx]
    return weight.to(dtype=dtype), bias.to(dtype=dtype)



def _length_entropy_anchor_patch_feature_router(
    spec: PatchMoEWarmStartSpec,
    *,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    if spec.entropy_prior_calibration_path is None:
        raise ValueError("Missing length-entropy anchor calibration path")

    with Path(spec.entropy_prior_calibration_path).open() as f:
        calibration = json.load(f)

    pair_count = spec.num_experts // 2
    if int(calibration.get("num_pairs", -1)) != pair_count:
        raise ValueError(
            f"calibration num_pairs must be {pair_count}, got "
            f"{calibration.get('num_pairs')}"
        )

    length_feature_idx = spec.patch_features.index("length")
    entropy_feature_idx = spec.patch_features.index("entropy")

    weight = torch.zeros(pair_count, spec.patch_feature_count, dtype=torch.float32)
    bias = torch.zeros(pair_count, dtype=torch.float32)

    # Linear nearest-anchor logits:
    #   argmax_p scale * <c_p, x> - 0.5 * scale * ||c_p||^2
    # This is equivalent to nearest-center routing in the [length, entropy]
    # feature space up to constants shared by all pairs.
    scale = float(spec.entropy_band_logit_scale)

    seen = set()
    for item in calibration["pairs"]:
        pair_idx = int(item["pair"])
        if pair_idx < 0 or pair_idx >= pair_count:
            raise ValueError(f"invalid pair index in calibration: {pair_idx}")
        seen.add(pair_idx)

        length_center = float(item["center_length_feature"])
        entropy_center = float(item["center_entropy_feature"])

        weight[pair_idx, length_feature_idx] = scale * length_center
        weight[pair_idx, entropy_feature_idx] = scale * entropy_center
        bias[pair_idx] = -0.5 * scale * (
            length_center * length_center + entropy_center * entropy_center
        )

    if seen != set(range(pair_count)):
        raise ValueError(
            f"calibration pairs must cover 0..{pair_count - 1}, got {sorted(seen)}"
        )

    return weight.to(dtype=dtype), bias.to(dtype=dtype)


def _load_entropy_prior_calibration(
    spec: PatchMoEWarmStartSpec,
    *,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if spec.entropy_prior_calibration_path is None:
        raise ValueError("Missing entropy prior calibration path")
    with Path(spec.entropy_prior_calibration_path).open() as f:
        calibration = json.load(f)
    num_pairs = spec.num_experts // 2
    if int(calibration.get("num_pairs", -1)) != num_pairs:
        raise ValueError(
            f"calibration num_pairs must be {num_pairs}, got "
            f"{calibration.get('num_pairs')}"
        )
    if int(calibration.get("num_experts", spec.num_experts)) != spec.num_experts:
        raise ValueError(
            f"calibration num_experts must be {spec.num_experts}, got "
            f"{calibration.get('num_experts')}"
        )
    centers = torch.tensor(calibration["entropy_centers"], dtype=dtype)
    widths = torch.tensor(calibration["entropy_widths"], dtype=dtype)
    static_bias = torch.tensor(calibration["static_pair_bias"], dtype=dtype)
    if centers.numel() != num_pairs or widths.numel() != num_pairs:
        raise ValueError("calibration entropy prior length must match num_pairs")
    if static_bias.numel() != num_pairs:
        raise ValueError("calibration static_pair_bias length must match num_pairs")
    return centers, widths, static_bias


def convert_dense_state_dict_to_patchmoe(
    dense_state_dict: Mapping[str, torch.Tensor],
    spec: PatchMoEWarmStartSpec,
) -> tuple[dict[str, torch.Tensor], PatchMoEWarmStartReport]:
    layer_weights = _global_ffn_weights(dense_state_dict)
    global_layers = tuple(sorted(layer_weights))
    selected_layers = tuple(
        layer_idx
        for layer_idx in global_layers
        if layer_idx % spec.layer_frequency == 0
    )
    selected_layer_set = set(selected_layers)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(spec.router_seed)

    converted: dict[str, torch.Tensor] = {}
    for key, value in dense_state_dict.items():
        match = DENSE_GLOBAL_FFN_RE.match(key)
        if match is None or int(match.group("layer")) not in selected_layer_set:
            converted[key] = value
            continue

        layer_idx = int(match.group("layer"))
        weight_name = match.group("weight")
        prefix = f"global_transformer.layers.{layer_idx}.feed_forward"
        for expert_idx in range(spec.num_experts):
            if spec.expert_init_mode == "paired_partition":
                expert_value = _paired_partition_ffn_weight(
                    value, weight_name, expert_idx
                )
            else:
                expert_value = _scale_dense_ffn_weight(
                    value, weight_name, spec.expert_ffn_dim_multiplier
                )
            converted[f"{prefix}.experts.{expert_idx}.{weight_name}.weight"] = (
                expert_value
            )

    for layer_idx in selected_layers:
        weights = layer_weights[layer_idx]
        dim = weights["w1"].shape[1]
        dtype = weights["w1"].dtype
        prefix = f"global_transformer.layers.{layer_idx}.feed_forward"
        router_std = _router_init_std(
            dim=dim,
            layer_idx=layer_idx,
            global_layer_count=len(global_layers),
            init_std_factor=spec.init_std_factor,
        )
        if spec.routing_granularity == "pair":
            router_weight = _pair_router_weight(
                num_experts=spec.num_experts,
                input_dim=dim,
                dtype=dtype,
                std=router_std,
                generator=generator,
            )
        elif spec.expert_init_mode == "paired_partition":
            router_weight = _paired_router_weight(
                num_experts=spec.num_experts,
                input_dim=dim,
                dtype=dtype,
                std=router_std,
                generator=generator,
            )
        else:
            router_weight = _truncated_normal(
                (spec.num_experts, dim),
                dtype=dtype,
                std=router_std,
                generator=generator,
            )
        converted[f"{prefix}.router.weight"] = router_weight
        if spec.patch_feature_count > 0:
            patch_feature_bias = None
            if spec.patch_feature_init == "entropy_bands":
                patch_feature_router, patch_feature_bias = (
                    _entropy_band_patch_feature_router(spec, dtype=dtype)
                )
            elif spec.patch_feature_init == "length_entropy_anchors":
                patch_feature_router, patch_feature_bias = (
                    _length_entropy_anchor_patch_feature_router(spec, dtype=dtype)
                )
            elif spec.routing_granularity == "pair":
                patch_feature_router = _pair_router_weight(
                    num_experts=spec.num_experts,
                    input_dim=spec.patch_feature_count,
                    dtype=dtype,
                    std=router_std,
                    generator=generator,
                )
            elif (
                spec.expert_init_mode == "paired_partition"
                and spec.pair_patch_feature_router
            ):
                patch_feature_router = _paired_router_weight(
                    num_experts=spec.num_experts,
                    input_dim=spec.patch_feature_count,
                    dtype=dtype,
                    std=router_std,
                    generator=generator,
                )
            else:
                patch_feature_router = _truncated_normal(
                    (spec.num_experts, spec.patch_feature_count),
                    dtype=dtype,
                    std=router_std,
                    generator=generator,
                )
            converted[f"{prefix}.patch_feature_router.weight"] = patch_feature_router
            if spec.patch_feature_bias:
                if patch_feature_bias is None:
                    if spec.routing_granularity == "pair":
                        patch_feature_bias = _pair_router_weight(
                            num_experts=spec.num_experts,
                            input_dim=1,
                            dtype=dtype,
                            std=router_std,
                            generator=generator,
                        ).squeeze(-1)
                    elif (
                        spec.expert_init_mode == "paired_partition"
                        and spec.pair_patch_feature_router
                    ):
                        patch_feature_bias = _paired_router_weight(
                            num_experts=spec.num_experts,
                            input_dim=1,
                            dtype=dtype,
                            std=router_std,
                            generator=generator,
                        ).squeeze(-1)
                    else:
                        patch_feature_bias = _truncated_normal(
                            (spec.num_experts,),
                            dtype=dtype,
                            std=router_std,
                            generator=generator,
                        )
                converted[f"{prefix}.patch_feature_router.bias"] = patch_feature_bias
                converted[f"{prefix}.patch_length_router.bias"] = patch_feature_bias
            # SparseMoEFeedForward keeps this compatibility alias in its state dict.
            converted[f"{prefix}.patch_length_router.weight"] = patch_feature_router

        if spec.entropy_prior_mode == "gaussian_pairs":
            centers, widths, static_bias = _load_entropy_prior_calibration(
                spec, dtype=dtype
            )
            converted[f"{prefix}.entropy_prior_centers"] = centers
            converted[f"{prefix}.entropy_prior_widths"] = widths
            converted[f"{prefix}.entropy_prior_static_bias"] = static_bias
        if spec.pair_bias_mode == "ema_byte_floor":
            num_pairs = spec.num_experts // 2
            converted[f"{prefix}.dynamic_pair_bias"] = torch.zeros(
                num_pairs, dtype=dtype
            )
            converted[f"{prefix}.pair_load_ema"] = torch.zeros(num_pairs, dtype=dtype)
            converted[f"{prefix}.pair_bias_step"] = torch.zeros((), dtype=torch.long)
        if spec.hidden_residual_ramp_end_step > spec.hidden_residual_ramp_start_step:
            converted[f"{prefix}.router_step"] = torch.zeros((), dtype=torch.long)
        if spec.entropy_mlp_hidden_dim > 0:
            converted[f"{prefix}.entropy_router.0.weight"] = _truncated_normal(
                (spec.entropy_mlp_hidden_dim, 1),
                dtype=dtype,
                std=router_std,
                generator=generator,
            )
            converted[f"{prefix}.entropy_router.0.bias"] = torch.zeros(
                spec.entropy_mlp_hidden_dim, dtype=dtype
            )
            converted[f"{prefix}.entropy_router.2.weight"] = _truncated_normal(
                (router_weight.shape[0], spec.entropy_mlp_hidden_dim),
                dtype=dtype,
                std=router_std,
                generator=generator,
            )
            converted[f"{prefix}.entropy_router.2.bias"] = torch.zeros(
                router_weight.shape[0], dtype=dtype
            )

    source_parameter_count = sum(value.numel() for value in dense_state_dict.values())
    source_global_ffn_parameter_count = sum(
        value.numel()
        for weights in layer_weights.values()
        for value in weights.values()
    )
    target_parameter_count = sum(
        value.numel()
        for key, value in converted.items()
        if ".patch_length_router." not in key
    )
    report = PatchMoEWarmStartReport(
        source_parameter_count=source_parameter_count,
        target_parameter_count=target_parameter_count,
        source_global_ffn_parameter_count=source_global_ffn_parameter_count,
        selected_layers=selected_layers,
        global_layers=global_layers,
        patch_feature_count=spec.patch_feature_count,
    )
    return converted, report


def load_consolidated_model_state_dict(path: str | Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError("Expected a consolidated checkpoint with a 'model' state dict")
    model_state_dict = checkpoint["model"]
    if not isinstance(model_state_dict, dict):
        raise ValueError("Checkpoint 'model' entry must be a state dict")
    return model_state_dict


def write_patchmoe_warmstart_dcp(
    output_dir: str | Path,
    converted_state_dict: Mapping[str, torch.Tensor],
    *,
    source_checkpoint: str | Path,
    spec: PatchMoEWarmStartSpec,
    report: PatchMoEWarmStartReport,
    force: bool = False,
) -> Path:
    output_dir = Path(output_dir)
    if output_dir.exists():
        if not force:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. "
                "Pass force=True only when replacing it intentionally."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    dcp.save({"model": dict(converted_state_dict)}, checkpoint_id=output_dir)
    manifest = {
        "format": "patchmoe-warmstart-dcp-v1",
        "source_checkpoint": str(Path(source_checkpoint).resolve()),
        "spec": asdict(spec),
        "report": asdict(report),
    }
    if spec.expert_init_mode == "paired_partition":
        manifest.update(
            {
                "num_experts": spec.num_experts,
                "pair_size": 2,
                "num_pairs": spec.num_experts // 2,
                "top_k": spec.top_k,
                "expert_ffn_dim_multiplier": spec.expert_ffn_dim_multiplier,
                "expert_init": "paired_partition_replicated_per_pair",
                "pair_member_output_scale": 2.0,
                "routing_granularity": spec.routing_granularity,
                "hidden_router_init": (
                    "pair"
                    if spec.routing_granularity == "pair"
                    else "paired"
                    if spec.pair_patch_feature_router
                    else "independent_patch_feature_router"
                ),
            }
        )
    with open(output_dir / MANIFEST_NAME, "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    return output_dir


def load_patchmoe_warmstart_manifest(output_dir: str | Path) -> dict:
    output_dir = Path(output_dir)
    metadata_path = output_dir / ".metadata"
    manifest_path = output_dir / MANIFEST_NAME
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing DCP metadata: {metadata_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing PatchMoE warm-start manifest: {manifest_path}"
        )
    with open(manifest_path) as f:
        manifest = json.load(f)
    if manifest.get("format") != "patchmoe-warmstart-dcp-v1":
        raise ValueError(
            f"Unsupported warm-start manifest format: {manifest.get('format')}"
        )
    return manifest


def _verification_sample_keys(
    converted_state_dict: Mapping[str, torch.Tensor],
) -> tuple[str, ...]:
    keys = set(converted_state_dict)
    samples: list[str] = []

    def append_first_matching(pattern: str):
        matches = sorted(key for key in keys if re.search(pattern, key))
        if matches:
            samples.append(matches[0])

    append_first_matching(r"\.feed_forward\.router\.weight$")
    append_first_matching(r"\.feed_forward\.patch_feature_router\.weight$")
    append_first_matching(r"\.feed_forward\.patch_length_router\.weight$")
    append_first_matching(r"\.feed_forward\.experts\.0\.w1\.weight$")

    expert_w3_keys = sorted(
        key
        for key in keys
        if re.search(r"\.feed_forward\.experts\.\d+\.w3\.weight$", key)
    )
    if expert_w3_keys:
        samples.append(expert_w3_keys[-1])

    dense_ffn_keys = sorted(key for key in keys if DENSE_GLOBAL_FFN_RE.match(key))
    if dense_ffn_keys:
        samples.append(dense_ffn_keys[0])

    local_trunk_keys = [
        key
        for key in keys
        if key.startswith("local_encoder.") and ".feed_forward." not in key
    ]
    preserved_keys = local_trunk_keys or [
        key
        for key in keys
        if ".feed_forward." not in key and "router.weight" not in key
    ]
    if preserved_keys:
        samples.append(
            min(
                preserved_keys,
                key=lambda key: (converted_state_dict[key].numel(), key),
            )
        )
    return tuple(dict.fromkeys(samples))


def verify_patchmoe_warmstart_dcp(
    output_dir: str | Path,
    converted_state_dict: Mapping[str, torch.Tensor],
    *,
    sample_keys: Sequence[str] | None = None,
) -> PatchMoEWarmStartVerificationReport:
    output_dir = Path(output_dir)
    load_patchmoe_warmstart_manifest(output_dir)

    metadata = dcp.FileSystemReader(str(output_dir)).read_metadata()
    actual_dcp_keys = set(metadata.state_dict_metadata)
    expected_dcp_keys = {f"model.{key}" for key in converted_state_dict}
    missing_keys = expected_dcp_keys - actual_dcp_keys
    extra_keys = actual_dcp_keys - expected_dcp_keys
    if missing_keys or extra_keys:
        raise ValueError(
            "DCP key set mismatch: "
            f"missing={sorted(missing_keys)[:5]}, extra={sorted(extra_keys)[:5]}"
        )

    selected_sample_keys = tuple(
        sample_keys or _verification_sample_keys(converted_state_dict)
    )
    if not selected_sample_keys:
        raise ValueError("No DCP verification sample keys selected")
    unknown_sample_keys = set(selected_sample_keys) - set(converted_state_dict)
    if unknown_sample_keys:
        raise ValueError(
            f"Unknown DCP verification sample keys: {sorted(unknown_sample_keys)}"
        )

    loaded = {
        key: torch.empty_like(converted_state_dict[key], device="cpu")
        for key in selected_sample_keys
    }
    dcp.load({"model": loaded}, checkpoint_id=output_dir)
    for key, value in loaded.items():
        if not torch.equal(value, converted_state_dict[key]):
            raise ValueError(f"DCP tensor mismatch: {key}")

    return PatchMoEWarmStartVerificationReport(
        dcp_key_count=len(actual_dcp_keys),
        sample_keys=selected_sample_keys,
    )
