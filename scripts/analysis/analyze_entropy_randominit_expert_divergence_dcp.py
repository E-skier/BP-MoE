#!/usr/bin/env python3
import argparse
import csv
import json
import math
import re
from pathlib import Path
from collections import defaultdict

import torch
import torch.distributed.checkpoint as dcp


def read_dcp_metadata(ckpt: Path):
    from torch.distributed.checkpoint import FileSystemReader

    reader = FileSystemReader(str(ckpt))
    metadata = reader.read_metadata()
    return metadata


def get_state_metadata(metadata):
    md = getattr(metadata, "state_dict_metadata", None)
    if md is None:
        raise RuntimeError("Cannot find state_dict_metadata in DCP metadata.")
    return md


def get_shape(tensor_md):
    for attr in ("size", "shape"):
        if hasattr(tensor_md, attr):
            x = getattr(tensor_md, attr)
            if x is not None:
                return tuple(x)

    chunks = getattr(tensor_md, "chunks", None)
    if chunks:
        sizes = []
        for c in chunks:
            if hasattr(c, "sizes"):
                sizes = list(c.sizes)
                break
        if sizes:
            return tuple(sizes)

    raise RuntimeError(f"Cannot infer tensor shape from metadata: {tensor_md}")


def get_dtype(tensor_md):
    for attr in ("properties", "tensor_properties"):
        if hasattr(tensor_md, attr):
            props = getattr(tensor_md, attr)
            if props is not None and hasattr(props, "dtype"):
                return props.dtype
    return torch.float32


def is_tensor_md(x):
    name = type(x).__name__.lower()
    return "tensor" in name and "metadata" in name


def dcp_load_one(ckpt: Path, key: str, shape, dtype):
    # Load one tensor by exact flattened DCP key.
    state = {key: torch.empty(shape, dtype=dtype, device="cpu")}

    try:
        dcp.load(state, checkpoint_id=str(ckpt))
    except TypeError:
        from torch.distributed.checkpoint import FileSystemReader
        reader = FileSystemReader(str(ckpt))
        dcp.load_state_dict(state, storage_reader=reader, no_dist=True)

    return state[key]


def layer_id_from_key(k: str):
    m = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", k)
    if m:
        return int(m.group(1))
    m = re.search(r"(?:^|\.)global_layers\.(\d+)(?:\.|$)", k)
    if m:
        return int(m.group(1))
    m = re.search(r"(?:^|\.)blocks\.(\d+)(?:\.|$)", k)
    if m:
        return int(m.group(1))
    return None


def same_pair(a, b):
    return a // 2 == b // 2


def cosine_dist_matrix(x):
    x = x.float().reshape(x.shape[0], -1)
    x = torch.nn.functional.normalize(x, dim=1, eps=1e-12)
    return 1.0 - x @ x.T


def write_inventory(state_md, out: Path):
    inv = out / "dcp_router_expert_key_inventory.txt"
    lines = []

    for key, md in state_md.items():
        if not is_tensor_md(md):
            continue
        low = key.lower()
        if "router" in low or "expert" in low or "moe" in low:
            try:
                shape = get_shape(md)
            except Exception:
                shape = "?"
            dtype = get_dtype(md)
            lines.append(f"{key}\tshape={shape}\tdtype={dtype}")

    inv.write_text("\n".join(lines) + "\n")
    print(f"[inventory] wrote: {inv}")
    print("[inventory] first 120 router/expert/moe keys:")
    for line in lines[:120]:
        print("  " + line)


def find_patch_feature_router_keys(state_md):
    weight_keys = {}
    bias_keys = {}

    for key, md in state_md.items():
        if not is_tensor_md(md):
            continue

        low = key.lower()
        try:
            shape = get_shape(md)
        except Exception:
            continue

        # 目标：entropy patch-feature router，一般 shape 是 [8, 1] / [8]
        if (
            "router" in low
            and "patch" in low
            and key.endswith("weight")
            and len(shape) == 2
            and shape[0] == 8
            and shape[1] >= 1
        ):
            base = key.rsplit(".", 1)[0]
            weight_keys[base] = key

        if (
            "router" in low
            and "patch" in low
            and key.endswith("bias")
            and len(shape) == 1
            and shape[0] == 8
        ):
            base = key.rsplit(".", 1)[0]
            bias_keys[base] = key

    bases = sorted(weight_keys)
    return [(base, weight_keys[base], bias_keys.get(base)) for base in bases]


def analyze_router(ckpt: Path, state_md, router_keys, out: Path):
    router_csv = out / "router_entropy_slopes_biases.csv"
    grid_csv = out / "router_entropy_grid_top2.csv"
    summary_csv = out / "router_summary.csv"
    summary_json = out / "router_summary.json"

    entropy_grid = torch.linspace(0.0, 4.0, 401)
    summary = []

    with router_csv.open("w", newline="") as f_weight, grid_csv.open("w", newline="") as f_grid:
        ww = csv.writer(f_weight)
        wg = csv.writer(f_grid)

        ww.writerow([
            "layer", "base_key", "expert",
            "entropy_slope", "bias", "slope_rank_low_to_high"
        ])

        wg.writerow([
            "layer", "base_key", "entropy",
            "top1", "top2_a", "top2_b", "top2_same_pair",
            "logits_json"
        ])

        for base, w_key, b_key in router_keys:
            w_md = state_md[w_key]
            w = dcp_load_one(ckpt, w_key, get_shape(w_md), get_dtype(w_md)).float()

            if b_key is not None:
                b_md = state_md[b_key]
                b = dcp_load_one(ckpt, b_key, get_shape(b_md), get_dtype(b_md)).float()
            else:
                b = torch.zeros(w.shape[0], dtype=torch.float32)

            slopes = w[:, 0].float()
            bias = b.float()

            order = torch.argsort(slopes)
            ranks = torch.empty_like(order)
            ranks[order] = torch.arange(len(order))

            layer = layer_id_from_key(base)

            for e in range(8):
                ww.writerow([
                    layer, base, e,
                    float(slopes[e]),
                    float(bias[e]),
                    int(ranks[e]),
                ])

            top1_counts = torch.zeros(8)
            top2_counts = torch.zeros(8)
            same_count = 0

            for ent in entropy_grid:
                logits = slopes * ent + bias
                top2 = torch.topk(logits, k=2).indices.tolist()
                top1 = top2[0]

                top1_counts[top1] += 1
                top2_counts[top2[0]] += 1
                top2_counts[top2[1]] += 1
                same_count += int(same_pair(top2[0], top2[1]))

                wg.writerow([
                    layer, base, float(ent),
                    int(top1), int(top2[0]), int(top2[1]),
                    int(same_pair(top2[0], top2[1])),
                    json.dumps([float(x) for x in logits.tolist()]),
                ])

            item = {
                "layer": layer,
                "base_key": base,
                "slope_mean": float(slopes.mean()),
                "slope_std": float(slopes.std(unbiased=False)),
                "slope_min": float(slopes.min()),
                "slope_max": float(slopes.max()),
                "bias_mean": float(bias.mean()),
                "bias_std": float(bias.std(unbiased=False)),
                "bias_min": float(bias.min()),
                "bias_max": float(bias.max()),
                "top2_same_pair_fraction_on_entropy_grid": same_count / len(entropy_grid),
                "top1_fraction_by_expert": [float(x) for x in (top1_counts / len(entropy_grid)).tolist()],
                "top2_fraction_by_expert": [float(x) for x in (top2_counts / (2 * len(entropy_grid))).tolist()],
            }
            summary.append(item)

            print(
                f"[router] layer={str(layer):>2} "
                f"slope_std={item['slope_std']:.4e} "
                f"bias_std={item['bias_std']:.4e} "
                f"same_pair={item['top2_same_pair_fraction_on_entropy_grid']:.3f} "
                f"top1={['%.2f' % x for x in item['top1_fraction_by_expert']]}"
            )

    with summary_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "layer", "base_key",
            "slope_std", "bias_std",
            "top2_same_pair_fraction_on_entropy_grid",
            "top1_fraction_by_expert",
            "top2_fraction_by_expert",
        ])
        for x in summary:
            writer.writerow([
                x["layer"], x["base_key"],
                x["slope_std"], x["bias_std"],
                x["top2_same_pair_fraction_on_entropy_grid"],
                json.dumps(x["top1_fraction_by_expert"]),
                json.dumps(x["top2_fraction_by_expert"]),
            ])

    summary_json.write_text(json.dumps(summary, indent=2))

    print(f"[router] wrote: {router_csv}")
    print(f"[router] wrote: {grid_csv}")
    print(f"[router] wrote: {summary_csv}")
    print(f"[router] wrote: {summary_json}")


def find_stacked_expert_weight_keys(state_md):
    keys = []

    for key, md in state_md.items():
        if not is_tensor_md(md):
            continue

        low = key.lower()
        if "expert" not in low:
            continue
        if "router" in low:
            continue

        try:
            shape = get_shape(md)
        except Exception:
            continue

        # stacked experts: [8, ...]
        if len(shape) >= 2 and shape[0] == 8:
            # 跳过太小的标量/统计张量
            numel = 1
            for s in shape:
                numel *= int(s)
            if numel >= 8 * 1024:
                keys.append((layer_id_from_key(key), key, shape, get_dtype(md), numel))

    keys.sort(key=lambda x: (-1 if x[0] is None else x[0], x[1]))
    return keys


def analyze_ffn(ckpt: Path, state_md, expert_keys, out: Path, max_tensors=None):
    pairwise_csv = out / "ffn_expert_pairwise_cosine_distance.csv"
    summary_csv = out / "ffn_expert_distance_summary.csv"
    summary_json = out / "ffn_expert_distance_summary.json"

    summaries = []

    with pairwise_csv.open("w", newline="") as f_pair:
        wp = csv.writer(f_pair)
        wp.writerow([
            "layer", "tensor_key", "shape",
            "expert_i", "expert_j",
            "same_original_pair",
            "cosine_distance"
        ])

        selected = expert_keys if max_tensors is None else expert_keys[:max_tensors]

        for idx, (layer, key, shape, dtype, numel) in enumerate(selected, 1):
            print(f"[ffn] loading {idx}/{len(selected)} layer={layer} shape={shape} key={key}")

            x = dcp_load_one(ckpt, key, shape, dtype)

            if x.shape[0] != 8:
                continue

            dist = cosine_dist_matrix(x)

            within = []
            cross = []
            all_d = []

            for i in range(8):
                for j in range(i + 1, 8):
                    d = float(dist[i, j])
                    all_d.append(d)

                    if same_pair(i, j):
                        within.append(d)
                        same = 1
                    else:
                        cross.append(d)
                        same = 0

                    wp.writerow([
                        layer, key, json.dumps(list(shape)),
                        i, j, same, d
                    ])

            if not all_d:
                continue

            s = {
                "layer": layer,
                "tensor_key": key,
                "shape": list(shape),
                "mean_pairwise_cosine_distance": sum(all_d) / len(all_d),
                "mean_within_original_pair_distance": sum(within) / len(within) if within else None,
                "mean_cross_pair_distance": sum(cross) / len(cross) if cross else None,
            }

            if s["mean_within_original_pair_distance"] is not None and s["mean_cross_pair_distance"] is not None:
                s["within_minus_cross"] = (
                    s["mean_within_original_pair_distance"] - s["mean_cross_pair_distance"]
                )
            else:
                s["within_minus_cross"] = None

            summaries.append(s)

            print(
                f"[ffn] layer={layer} "
                f"mean={s['mean_pairwise_cosine_distance']:.4e} "
                f"within={s['mean_within_original_pair_distance']:.4e} "
                f"cross={s['mean_cross_pair_distance']:.4e} "
                f"within-cross={s['within_minus_cross']:.4e}"
            )

            del x, dist

    with summary_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "layer", "tensor_key", "shape",
            "mean_pairwise_cosine_distance",
            "mean_within_original_pair_distance",
            "mean_cross_pair_distance",
            "within_minus_cross",
        ])
        for s in summaries:
            w.writerow([
                s["layer"], s["tensor_key"], json.dumps(s["shape"]),
                s["mean_pairwise_cosine_distance"],
                s["mean_within_original_pair_distance"],
                s["mean_cross_pair_distance"],
                s["within_minus_cross"],
            ])

    summary_json.write_text(json.dumps(summaries, indent=2))

    print(f"[ffn] wrote: {pairwise_csv}")
    print(f"[ffn] wrote: {summary_csv}")
    print(f"[ffn] wrote: {summary_json}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-ffn-tensors", type=int, default=None)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    if not args.ckpt.is_dir():
        raise FileNotFoundError(args.ckpt)
    if not (args.ckpt / ".metadata").exists():
        raise FileNotFoundError(args.ckpt / ".metadata")

    print(f"[ckpt] {args.ckpt}")
    print(f"[out]  {args.out}")

    metadata = read_dcp_metadata(args.ckpt)
    state_md = get_state_metadata(metadata)

    print(f"[metadata] entries: {len(state_md):,}")

    write_inventory(state_md, args.out)

    router_keys = find_patch_feature_router_keys(state_md)
    print(f"[router] candidate patch-feature router groups: {len(router_keys)}")

    if router_keys:
        analyze_router(args.ckpt, state_md, router_keys, args.out)
    else:
        print("[router] No patch-feature router keys found.")
        print("[router] Inspect dcp_router_expert_key_inventory.txt and adjust key matching.")

    expert_keys = find_stacked_expert_weight_keys(state_md)
    print(f"[ffn] candidate stacked expert tensors: {len(expert_keys)}")

    key_list_path = args.out / "ffn_candidate_tensor_keys.txt"
    key_list_path.write_text(
        "\n".join(
            f"layer={layer}\tshape={shape}\tdtype={dtype}\tnumel={numel}\tkey={key}"
            for layer, key, shape, dtype, numel in expert_keys
        )
        + "\n"
    )
    print(f"[ffn] wrote candidate list: {key_list_path}")

    if expert_keys:
        analyze_ffn(args.ckpt, state_md, expert_keys, args.out, args.max_ffn_tensors)
    else:
        print("[ffn] No stacked expert tensors found.")

    print("[done]")


if __name__ == "__main__":
    main()
