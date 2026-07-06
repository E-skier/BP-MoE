#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp


def parse_checkpoint(value: str):
    if "=" not in value:
        raise ValueError(
            f"Invalid --checkpoint value: {value}. Expected TAG=/absolute/path"
        )
    tag, path = value.split("=", 1)
    tag = tag.strip()
    path = path.strip()
    if not tag or not path:
        raise ValueError(f"Invalid --checkpoint value: {value}")
    return tag, Path(path)


def init_cpu_dist():
    if not dist.is_available() or dist.is_initialized():
        return False

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29597")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")

    dist.init_process_group(
        backend="gloo",
        rank=0,
        world_size=1,
    )
    return True


def tensor_metadata_items(metadata):
    state_metadata = getattr(metadata, "state_dict_metadata", {})
    output = []

    for key, item in state_metadata.items():
        shape = getattr(item, "size", None)
        properties = getattr(item, "properties", None)
        dtype = getattr(properties, "dtype", None)

        if shape is None or dtype is None:
            continue

        output.append((str(key), item, tuple(shape), dtype))

    return output


def select_router_items(items):
    patterns = [
        "patch_feature_router",
        "patch_feature",
        "router",
    ]

    for pattern in patterns:
        selected = [
            item for item in items
            if pattern in item[0].lower()
        ]
        if selected:
            return pattern, selected

    return "none", []


def safe_quantile(values, q):
    if values.numel() == 0:
        return None
    return float(torch.quantile(values, q).item())


def tensor_stats(tensor: torch.Tensor):
    x = tensor.detach().to(dtype=torch.float32, device="cpu").reshape(-1)

    if x.numel() == 0:
        return {"numel": 0}

    finite = torch.isfinite(x)
    finite_x = x[finite]

    result = {
        "numel": int(x.numel()),
        "finite_fraction": float(finite.float().mean().item()),
        "mean": None,
        "std": None,
        "min": None,
        "max": None,
        "abs_max": None,
        "abs_mean": None,
        "p01": None,
        "p50": None,
        "p99": None,
        "zero_fraction": float((x == 0).float().mean().item()),
    }

    if finite_x.numel() > 0:
        abs_x = finite_x.abs()
        result.update(
            {
                "mean": float(finite_x.mean().item()),
                "std": float(finite_x.std(unbiased=False).item()),
                "min": float(finite_x.min().item()),
                "max": float(finite_x.max().item()),
                "abs_max": float(abs_x.max().item()),
                "abs_mean": float(abs_x.mean().item()),
                "p01": safe_quantile(finite_x, 0.01),
                "p50": safe_quantile(finite_x, 0.50),
                "p99": safe_quantile(finite_x, 0.99),
            }
        )

    if x.numel() <= 256:
        result["raw_values"] = x.tolist()

    return result


def inspect_checkpoint(tag: str, ckpt_dir: Path, out_dir: Path):
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    metadata_file = ckpt_dir / ".metadata"
    if not metadata_file.exists():
        raise FileNotFoundError(
            f"Missing .metadata in checkpoint directory: {ckpt_dir}"
        )

    reader = dcp.FileSystemReader(str(ckpt_dir))
    metadata = reader.read_metadata()

    all_items = tensor_metadata_items(metadata)
    selected_pattern, selected_items = select_router_items(all_items)

    payload = {
        "tag": tag,
        "checkpoint_dir": str(ckpt_dir),
        "selected_pattern": selected_pattern,
        "all_tensor_key_count": len(all_items),
        "selected_tensor_key_count": len(selected_items),
        "all_candidate_keys": [
            key for key, _, _, _ in all_items
            if any(
                p in key.lower()
                for p in ["patch_feature_router", "patch_feature", "router"]
            )
        ],
        "parameters": {},
    }

    if not selected_items:
        print(f"[{tag}] No router-related tensor key found.")
        print(f"[{tag}] Candidate names written to JSON for inspection.")
        out_path = out_dir / f"router_stats_{tag}.json"
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    state = {}
    for key, _, shape, dtype in selected_items:
        state[key] = torch.empty(shape, dtype=dtype, device="cpu")

    planner = dcp.DefaultLoadPlanner(allow_partial_load=True)
    dcp.load(
        state_dict=state,
        checkpoint_id=str(ckpt_dir),
        planner=planner,
    )

    print(f"\n{'=' * 96}")
    print(f"[{tag}] checkpoint: {ckpt_dir}")
    print(f"[{tag}] selected pattern: {selected_pattern}")
    print(f"[{tag}] loaded router tensors: {len(state)}")
    print(f"{'=' * 96}")

    for key in sorted(state):
        tensor = state[key]
        stats = tensor_stats(tensor)

        payload["parameters"][key] = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            **stats,
        }

        print(
            f"{key}\n"
            f"  shape={tuple(tensor.shape)} dtype={tensor.dtype} "
            f"mean={stats.get('mean')} std={stats.get('std')} "
            f"min={stats.get('min')} max={stats.get('max')} "
            f"abs_max={stats.get('abs_max')} "
            f"zero_frac={stats.get('zero_fraction')}"
        )

    out_path = out_dir / f"router_stats_{tag}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n[{tag}] JSON written to: {out_path}")

    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        help="TAG=/absolute/path/to/DCP/checkpoint",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Directory for JSON reports",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = [parse_checkpoint(x) for x in args.checkpoint]

    started_dist = False
    try:
        started_dist = init_cpu_dist()

        for tag, ckpt_dir in checkpoints:
            inspect_checkpoint(tag, ckpt_dir, out_dir)

    finally:
        if started_dist and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
