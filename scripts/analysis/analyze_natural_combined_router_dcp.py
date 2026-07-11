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


def read_metadata(ckpt):
    from torch.distributed.checkpoint import FileSystemReader
    reader = FileSystemReader(str(ckpt))
    return reader.read_metadata().state_dict_metadata


def shape_of(md):
    if hasattr(md, "size"):
        return tuple(md.size)
    if hasattr(md, "shape"):
        return tuple(md.shape)
    chunks = getattr(md, "chunks", None)
    if chunks:
        c = chunks[0]
        if hasattr(c, "sizes"):
            return tuple(c.sizes)
    raise RuntimeError(f"Cannot infer shape: {md}")


def dtype_of(md):
    props = getattr(md, "properties", None)
    if props is not None and hasattr(props, "dtype"):
        return props.dtype
    return torch.float32


def load_one(ckpt, key, shape, dtype):
    state = {key: torch.empty(shape, dtype=dtype, device="cpu")}
    try:
        dcp.load(state, checkpoint_id=str(ckpt))
    except TypeError:
        from torch.distributed.checkpoint import FileSystemReader
        reader = FileSystemReader(str(ckpt))
        dcp.load_state_dict(state, storage_reader=reader, no_dist=True)
    return state[key].float()


def same_pair(a, b):
    return a // 2 == b // 2


def eff_num(ps):
    ps = [p for p in ps if p > 0]
    h = -sum(p * math.log(p) for p in ps)
    return math.exp(h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--entropy-min", type=float, default=0.0)
    ap.add_argument("--entropy-max", type=float, default=4.0)
    ap.add_argument("--entropy-steps", type=int, default=81)
    ap.add_argument("--length-min", type=float, default=1.0)
    ap.add_argument("--length-max", type=float, default=16.0)
    ap.add_argument("--length-steps", type=int, default=16)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    md = read_metadata(args.ckpt)

    pat = re.compile(
        r"^model\.global_transformer\.layers\.(\d+)\.feed_forward\.(patch_feature_router|patch_length_router)\.(weight|bias)$"
    )

    keys = defaultdict(dict)

    for key, meta in md.items():
        m = pat.match(key)
        if not m:
            continue
        layer = int(m.group(1))
        router = m.group(2)
        wb = m.group(3)
        keys[layer][f"{router}.{wb}"] = key

    layers = sorted(keys)
    print(f"layers with natural routers: {len(layers)}")

    entropy_grid = torch.linspace(args.entropy_min, args.entropy_max, args.entropy_steps)
    length_grid = torch.linspace(args.length_min, args.length_max, args.length_steps)

    grid_csv = args.out / "natural_combined_router_2d_grid_top2.csv"
    summary_json = args.out / "natural_combined_router_summary.json"
    summary_csv = args.out / "natural_combined_router_summary.csv"

    summaries = []

    with grid_csv.open("w", newline="") as fg:
        wg = csv.writer(fg)
        wg.writerow([
            "layer", "entropy", "length",
            "top1", "top2_a", "top2_b", "top2_same_pair",
            "logits_json"
        ])

        for layer in layers:
            need = [
                "patch_feature_router.weight",
                "patch_feature_router.bias",
                "patch_length_router.weight",
                "patch_length_router.bias",
            ]
            missing = [x for x in need if x not in keys[layer]]
            if missing:
                print(f"[skip] layer={layer}, missing={missing}")
                continue

            fw_key = keys[layer]["patch_feature_router.weight"]
            fb_key = keys[layer]["patch_feature_router.bias"]
            lw_key = keys[layer]["patch_length_router.weight"]
            lb_key = keys[layer]["patch_length_router.bias"]

            fw = load_one(args.ckpt, fw_key, shape_of(md[fw_key]), dtype_of(md[fw_key]))[:, 0]
            fb = load_one(args.ckpt, fb_key, shape_of(md[fb_key]), dtype_of(md[fb_key]))
            lw = load_one(args.ckpt, lw_key, shape_of(md[lw_key]), dtype_of(md[lw_key]))[:, 0]
            lb = load_one(args.ckpt, lb_key, shape_of(md[lb_key]), dtype_of(md[lb_key]))

            top1_counts = torch.zeros(8)
            top2_counts = torch.zeros(8)
            same_count = 0
            n = 0

            for ent in entropy_grid:
                for length in length_grid:
                    logits = fw * ent + fb + lw * length + lb
                    top2 = torch.topk(logits, k=2).indices.tolist()
                    top1 = top2[0]

                    top1_counts[top1] += 1
                    top2_counts[top2[0]] += 1
                    top2_counts[top2[1]] += 1
                    same = int(same_pair(top2[0], top2[1]))
                    same_count += same
                    n += 1

                    wg.writerow([
                        layer, float(ent), float(length),
                        top1, top2[0], top2[1], same,
                        json.dumps([float(x) for x in logits.tolist()])
                    ])

            top1_frac = (top1_counts / n).tolist()
            top2_frac = (top2_counts / (2 * n)).tolist()

            active_top1 = [i for i, v in enumerate(top1_frac) if v > 0.01]
            active_top2 = [i for i, v in enumerate(top2_frac) if v > 0.01]

            item = {
                "layer": layer,
                "feature_slope_std": float(fw.std(unbiased=False)),
                "length_slope_std": float(lw.std(unbiased=False)),
                "feature_bias_std": float(fb.std(unbiased=False)),
                "length_bias_std": float(lb.std(unbiased=False)),
                "combined_same_pair_fraction": same_count / n,
                "top1_fraction_by_expert": [float(x) for x in top1_frac],
                "top2_fraction_by_expert": [float(x) for x in top2_frac],
                "active_top1_experts": active_top1,
                "active_top2_experts": active_top2,
                "top2_effective_experts": eff_num(top2_frac),
            }
            summaries.append(item)

            print(
                f"[combined] layer={layer:02d} "
                f"feat_std={item['feature_slope_std']:.4e} "
                f"len_std={item['length_slope_std']:.4e} "
                f"same_pair={item['combined_same_pair_fraction']:.3f} "
                f"active_top2={active_top2} "
                f"eff={item['top2_effective_experts']:.2f} "
                f"top2={['%.2f'%v for v in top2_frac]}"
            )

    summary_json.write_text(json.dumps(summaries, indent=2))

    with summary_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "layer",
            "feature_slope_std",
            "length_slope_std",
            "feature_bias_std",
            "length_bias_std",
            "combined_same_pair_fraction",
            "active_top1_count",
            "active_top2_count",
            "top2_effective_experts",
            "top1_fraction_by_expert",
            "top2_fraction_by_expert",
        ])
        for x in summaries:
            w.writerow([
                x["layer"],
                x["feature_slope_std"],
                x["length_slope_std"],
                x["feature_bias_std"],
                x["length_bias_std"],
                x["combined_same_pair_fraction"],
                len(x["active_top1_experts"]),
                len(x["active_top2_experts"]),
                x["top2_effective_experts"],
                json.dumps(x["top1_fraction_by_expert"]),
                json.dumps(x["top2_fraction_by_expert"]),
            ])

    print(f"\nwrote: {grid_csv}")
    print(f"wrote: {summary_json}")
    print(f"wrote: {summary_csv}")


if __name__ == "__main__":
    main()
