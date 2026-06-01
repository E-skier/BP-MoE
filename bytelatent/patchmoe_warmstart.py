# Copyright (c) Meta Platforms, Inc. and affiliates.

import json
import math
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import torch
import torch.distributed.checkpoint as dcp
from torch import nn

from bytelatent.base_transformer import PATCH_BYTE_TYPE_NAMES

DENSE_GLOBAL_FFN_RE = re.compile(
    r"^global_transformer\.layers\.(?P<layer>\d+)\."
    r"feed_forward\.(?P<weight>w[123])\.weight$"
)
PATCH_FEATURE_CHOICES = ("length", "entropy", "byte_type")
MANIFEST_NAME = "patchmoe_warmstart_manifest.json"


@dataclass(frozen=True)
class PatchMoEWarmStartSpec:
    num_experts: int = 8
    top_k: int = 2
    layer_frequency: int = 4
    patch_features: tuple[str, ...] = ("entropy",)
    router_seed: int = 42
    init_std_factor: str = "current_depth"

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
        if self.init_std_factor not in {
            "disabled",
            "current_depth",
            "global_depth",
            "dim_ratio",
        }:
            raise ValueError(f"Unsupported init_std_factor: {self.init_std_factor}")

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
            converted[f"{prefix}.experts.{expert_idx}.{weight_name}.weight"] = value

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
        converted[f"{prefix}.router.weight"] = _truncated_normal(
            (spec.num_experts, dim),
            dtype=dtype,
            std=router_std,
            generator=generator,
        )
        if spec.patch_feature_count > 0:
            patch_feature_router = _truncated_normal(
                (spec.num_experts, spec.patch_feature_count),
                dtype=dtype,
                std=router_std,
                generator=generator,
            )
            converted[f"{prefix}.patch_feature_router.weight"] = patch_feature_router
            # SparseMoEFeedForward keeps this compatibility alias in its state dict.
            converted[f"{prefix}.patch_length_router.weight"] = patch_feature_router

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
