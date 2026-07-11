#!/usr/bin/env python3
import argparse
import csv
import json
import math
import re
from pathlib import Path

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
    raise RuntimeError(f"Cannot infer shape from metadata: {md}")


def dtype_of(md):
    props = getattr(md, "properties", None)
    if props is not None and hasattr(props, "dtype"):
        return props.dtype
    return torch.float32


def load_one(ckpt, key, md):
    state = {key: torch.empty(shape_of(md), dtype=dtype_of(md), device="cpu")}
    try:
        dcp.load(state, checkpoint_id=str(ckpt))
    except TypeError:
        from torch.distributed.checkpoint import FileSystemReader
        reader = FileSystemReader(str(ckpt))
        dcp.load_state_dict(state, storage_reader=reader, no_dist=True)
    return state[key].float()


def same_pair(a, b):
    return a // 2 == b // 2


def entropy_eff(ps):
    ps = [float(p) for p in ps if float(p) > 0]
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

    weight_pat = re.compile(
        r"^model\.global_transformer\.layers\.(\d+)\.feed_forward\.patch_feature_router\.weight$"
    )

    layers = []
    for key in md:
        m = weight_pat.match(key)
        if m:
            layers.append(int(m.group(1)))
    layers = sorted(layers)

    print(f"layers with patch_feature_router: {len(layers)}")

    entropy_grid = torch.linspace(args.entropy_min, args.entropy_max, args.entropy_steps)
    length_grid = torch.linspace(args.length_min, args.length_max, args.length_steps)

    grid_csv = args.out / "shared_2d_patch_feature_router_grid_top2.csv"
    summary_csv = args.out / "shared_2d_patch_feature_router_summary.csv"
    summary_json = args.out / "shared_2d_patch_feature_router_summary.json"

    summaries = []

    with grid_csv.open("w", newline="") as fg:
        wg = csv.writer(fg)
        wg.writerow([
            "layer",
            "raw_length",
            "log1p_length",
            "entropy",
            "top1",
            "top2_a",
            "top2_b",
            "top2_same_pair",
            "logits_json",
        ])

        for layer in layers:
            w_key = f"model.global_transformer.layers.{layer}.feed_forward.patch_feature_router.weight"
            b_key = f"model.global_transformer.layers.{layer}.feed_forward.patch_feature_router.bias"

            w = load_one(args.ckpt, w_key, md[w_key])
            b = load_one(args.ckpt, b_key, md[b_key])

            if tuple(w.shape) != (8, 2):
                raise RuntimeError(f"Layer {layer}: expected weight shape (8,2), got {tuple(w.shape)}")
            if tuple(b.shape) != (8,):
                raise RuntimeError(f"Layer {layer}: expected bias shape (8,), got {tuple(b.shape)}")

            # Feature order follows base_transformer.py:
            # [log1p(length), entropy]
            length_axis = w[:, 0]
            entropy_axis = w[:, 1]

            top1_counts = torch.zeros(8)
            top2_counts = torch.zeros(8)
            same_count = 0
            total = 0

            for raw_len in length_grid:
                log_len = torch.log1p(raw_len)
                for ent in entropy_grid:
                    logits = length_axis * log_len + entropy_axis * ent + b
                    top2 = torch.topk(logits, k=2).indices.tolist()
                    top1 = top2[0]
                    same = int(same_pair(top2[0], top2[1]))

                    top1_counts[top1] += 1
                    top2_counts[top2[0]] += 1
                    top2_counts[top2[1]] += 1
                    same_count += same
                    total += 1

                    wg.writerow([
                        layer,
                        float(raw_len),
                        float(log_len),
                        float(ent),
                        top1,
                        top2[0],
                        top2[1],
                        same,
                        json.dumps([float(x) for x in logits.tolist()]),
                    ])

            top1_frac = (top1_counts / total).tolist()
            top2_frac = (top2_counts / (2 * total)).tolist()

            active_top1 = [i for i, v in enumerate(top1_frac) if v > 0.01]
            active_top2 = [i for i, v in enumerate(top2_frac) if v > 0.01]

            length_std = float(length_axis.std(unbiased=False))
            entropy_std = float(entropy_axis.std(unbiased=False))
            bias_std = float(b.std(unbiased=False))

            length_l2 = float(torch.linalg.vector_norm(length_axis))
            entropy_l2 = float(torch.linalg.vector_norm(entropy_axis))
            axis_cos = float(torch.nn.functional.cosine_similarity(
                length_axis.reshape(1, -1),
                entropy_axis.reshape(1, -1),
                dim=1,
            ).item())

            item = {
                "layer": layer,
                "length_axis_std": length_std,
                "entropy_axis_std": entropy_std,
                "bias_std": bias_std,
                "length_axis_l2": length_l2,
                "entropy_axis_l2": entropy_l2,
                "entropy_to_length_l2_ratio": entropy_l2 / max(length_l2, 1e-12),
                "length_entropy_axis_cosine": axis_cos,
                "same_pair_fraction": same_count / total,
                "top1_fraction_by_expert": [float(x) for x in top1_frac],
                "top2_fraction_by_expert": [float(x) for x in top2_frac],
                "active_top1_experts": active_top1,
                "active_top2_experts": active_top2,
                "active_top1_count": len(active_top1),
                "active_top2_count": len(active_top2),
                "top2_effective_experts": entropy_eff(top2_frac),
            }
            summaries.append(item)

            print(
                f"[2d] layer={layer:02d} "
                f"len_std={length_std:.4e} "
                f"ent_std={entropy_std:.4e} "
                f"ratio={item['entropy_to_length_l2_ratio']:.3f} "
                f"axis_cos={axis_cos:+.3f} "
                f"same_pair={item['same_pair_fraction']:.3f} "
                f"active_top2={active_top2} "
                f"eff={item['top2_effective_experts']:.2f} "
                f"top2={['%.2f'%v for v in top2_frac]}"
            )

    summary_json.write_text(json.dumps(summaries, indent=2))

    with summary_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "layer",
            "length_axis_std",
            "entropy_axis_std",
            "bias_std",
            "length_axis_l2",
            "entropy_axis_l2",
            "entropy_to_length_l2_ratio",
            "length_entropy_axis_cosine",
            "same_pair_fraction",
            "active_top1_count",
            "active_top2_count",
            "top2_effective_experts",
            "active_top1_experts",
            "active_top2_experts",
            "top1_fraction_by_expert",
            "top2_fraction_by_expert",
        ])
        for x in summaries:
            writer.writerow([
                x["layer"],
                x["length_axis_std"],
                x["entropy_axis_std"],
                x["bias_std"],
                x["length_axis_l2"],
                x["entropy_axis_l2"],
                x["entropy_to_length_l2_ratio"],
                x["length_entropy_axis_cosine"],
                x["same_pair_fraction"],
                x["active_top1_count"],
                x["active_top2_count"],
                x["top2_effective_experts"],
                json.dumps(x["active_top1_experts"]),
                json.dumps(x["active_top2_experts"]),
                json.dumps(x["top1_fraction_by_expert"]),
                json.dumps(x["top2_fraction_by_expert"]),
            ])

    print(f"\nwrote: {grid_csv}")
    print(f"wrote: {summary_csv}")
    print(f"wrote: {summary_json}")


if __name__ == "__main__":
    main()
