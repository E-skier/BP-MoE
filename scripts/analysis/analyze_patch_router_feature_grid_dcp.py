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


def build_feature_vector(feature_names, raw_len, entropy):
    values = []
    for name in feature_names:
        if name == "length":
            values.append(float(torch.log1p(torch.tensor(float(raw_len))).item()))
        elif name == "entropy":
            values.append(float(entropy))
        else:
            raise ValueError(f"Unsupported feature name: {name}")
    return torch.tensor(values, dtype=torch.float32)


def infer_feature_names(weight_shape, mode):
    in_dim = int(weight_shape[1])
    if mode == "natural_2d":
        if in_dim != 2:
            raise ValueError(f"natural_2d expects in_dim=2, got {in_dim}")
        return ["length", "entropy"]
    if mode == "entropy_only":
        if in_dim != 1:
            raise ValueError(f"entropy_only expects in_dim=1, got {in_dim}")
        return ["entropy"]
    if mode == "length_only":
        if in_dim != 1:
            raise ValueError(f"length_only expects in_dim=1, got {in_dim}")
        return ["length"]
    if mode == "auto":
        if in_dim == 2:
            return ["length", "entropy"]
        if in_dim == 1:
            return ["entropy"]
        raise ValueError(f"Cannot infer feature names for in_dim={in_dim}")
    raise ValueError(f"Unknown mode: {mode}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--tag", required=True)
    ap.add_argument(
        "--mode",
        default="auto",
        choices=["auto", "natural_2d", "entropy_only", "length_only"],
    )
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

    if not layers:
        print("[ERROR] No patch_feature_router found.")
        print("[HINT] This checkpoint is probably hidden-only, or uses no patch-feature router.")
        print("[HINT] It cannot be analyzed by static [length, entropy] feature grid.")
        raise SystemExit(2)

    print(f"[info] tag={args.tag}")
    print(f"[info] ckpt={args.ckpt}")
    print(f"[info] layers with patch_feature_router: {len(layers)}")

    entropy_grid = torch.linspace(args.entropy_min, args.entropy_max, args.entropy_steps)
    length_grid = torch.linspace(args.length_min, args.length_max, args.length_steps)

    grid_csv = args.out / f"{args.tag}_feature_grid_top2.csv"
    summary_csv = args.out / f"{args.tag}_feature_grid_summary.csv"
    summary_json = args.out / f"{args.tag}_feature_grid_summary.json"

    summaries = []

    with grid_csv.open("w", newline="") as fg:
        wg = csv.writer(fg)
        wg.writerow([
            "tag",
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
            b = load_one(args.ckpt, b_key, md[b_key]) if b_key in md else torch.zeros(w.shape[0])

            if w.shape[0] != 8:
                raise RuntimeError(f"Layer {layer}: expected 8 experts, got {tuple(w.shape)}")
            if b.shape[0] != 8:
                raise RuntimeError(f"Layer {layer}: expected bias shape (8,), got {tuple(b.shape)}")

            feature_names = infer_feature_names(tuple(w.shape), args.mode)
            if layer == layers[0]:
                print(f"[info] inferred feature_names={feature_names} from weight_shape={tuple(w.shape)}")

            top1_counts = torch.zeros(8)
            top2_counts = torch.zeros(8)
            same_count = 0
            total = 0

            for raw_len in length_grid:
                log_len = torch.log1p(raw_len)
                for ent in entropy_grid:
                    feat = build_feature_vector(feature_names, raw_len, ent)
                    logits = w @ feat + b

                    top2 = torch.topk(logits, k=2).indices.tolist()
                    top1 = top2[0]
                    same = int(same_pair(top2[0], top2[1]))

                    top1_counts[top1] += 1
                    top2_counts[top2[0]] += 1
                    top2_counts[top2[1]] += 1
                    same_count += same
                    total += 1

                    wg.writerow([
                        args.tag,
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

            item = {
                "tag": args.tag,
                "layer": layer,
                "feature_names": feature_names,
                "weight_shape": list(w.shape),
                "same_pair_fraction": same_count / total,
                "active_top1_experts": active_top1,
                "active_top2_experts": active_top2,
                "active_top1_count": len(active_top1),
                "active_top2_count": len(active_top2),
                "top1_fraction_by_expert": [float(x) for x in top1_frac],
                "top2_fraction_by_expert": [float(x) for x in top2_frac],
                "top2_effective_experts": entropy_eff(top2_frac),
                "feature_axis_std": [float(w[:, i].std(unbiased=False)) for i in range(w.shape[1])],
                "feature_axis_l2": [float(torch.linalg.vector_norm(w[:, i])) for i in range(w.shape[1])],
                "bias_std": float(b.std(unbiased=False)),
            }

            if w.shape[1] == 2:
                length_axis = w[:, 0]
                entropy_axis = w[:, 1]
                item["length_axis_std"] = float(length_axis.std(unbiased=False))
                item["entropy_axis_std"] = float(entropy_axis.std(unbiased=False))
                item["entropy_to_length_l2_ratio"] = float(
                    torch.linalg.vector_norm(entropy_axis) / max(float(torch.linalg.vector_norm(length_axis)), 1e-12)
                )
                item["length_entropy_axis_cosine"] = float(
                    torch.nn.functional.cosine_similarity(
                        length_axis.reshape(1, -1),
                        entropy_axis.reshape(1, -1),
                        dim=1,
                    ).item()
                )
            else:
                item["length_axis_std"] = None
                item["entropy_axis_std"] = float(w[:, 0].std(unbiased=False)) if feature_names[0] == "entropy" else None
                item["entropy_to_length_l2_ratio"] = None
                item["length_entropy_axis_cosine"] = None

            summaries.append(item)

            print(
                f"[grid] tag={args.tag} layer={layer:02d} "
                f"same_pair={item['same_pair_fraction']:.3f} "
                f"active_top2={active_top2} "
                f"active_top2_count={item['active_top2_count']} "
                f"eff={item['top2_effective_experts']:.2f} "
                f"top2={['%.2f'%v for v in top2_frac]}"
            )

    summary_json.write_text(json.dumps(summaries, indent=2))

    with summary_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "tag",
            "layer",
            "feature_names",
            "weight_shape",
            "same_pair_fraction",
            "active_top1_count",
            "active_top2_count",
            "top2_effective_experts",
            "active_top1_experts",
            "active_top2_experts",
            "top1_fraction_by_expert",
            "top2_fraction_by_expert",
            "feature_axis_std",
            "feature_axis_l2",
            "bias_std",
            "length_axis_std",
            "entropy_axis_std",
            "entropy_to_length_l2_ratio",
            "length_entropy_axis_cosine",
        ])
        for x in summaries:
            writer.writerow([
                x["tag"],
                x["layer"],
                json.dumps(x["feature_names"]),
                json.dumps(x["weight_shape"]),
                x["same_pair_fraction"],
                x["active_top1_count"],
                x["active_top2_count"],
                x["top2_effective_experts"],
                json.dumps(x["active_top1_experts"]),
                json.dumps(x["active_top2_experts"]),
                json.dumps(x["top1_fraction_by_expert"]),
                json.dumps(x["top2_fraction_by_expert"]),
                json.dumps(x["feature_axis_std"]),
                json.dumps(x["feature_axis_l2"]),
                x["bias_std"],
                x["length_axis_std"],
                x["entropy_axis_std"],
                x["entropy_to_length_l2_ratio"],
                x["length_entropy_axis_cosine"],
            ])

    n = len(summaries)
    active2 = [x["active_top2_count"] for x in summaries]
    effs = [x["top2_effective_experts"] for x in summaries]
    same = [x["same_pair_fraction"] for x in summaries]

    print("\nAggregate:")
    print(f"  tag: {args.tag}")
    print(f"  layers: {n}")
    print(f"  mean same_pair_frac: {sum(same)/n:.4f}")
    print(f"  mean active_top2_experts: {sum(active2)/n:.4f}")
    print(f"  mean top2 effective experts: {sum(effs)/n:.4f}")
    print(f"  layers with <=2 active top2 experts: {sum(c <= 2 for c in active2)} / {n}")
    print(f"  layers with >=4 active top2 experts: {sum(c >= 4 for c in active2)} / {n}")

    ratios = [x["entropy_to_length_l2_ratio"] for x in summaries if x["entropy_to_length_l2_ratio"] is not None]
    cosines = [x["length_entropy_axis_cosine"] for x in summaries if x["length_entropy_axis_cosine"] is not None]
    if ratios:
        print(f"  mean entropy_to_length_l2_ratio: {sum(ratios)/len(ratios):.4f}")
    if cosines:
        print(f"  mean length_entropy_axis_cosine: {sum(cosines)/len(cosines):.4f}")

    print(f"\nwrote: {grid_csv}")
    print(f"wrote: {summary_csv}")
    print(f"wrote: {summary_json}")


if __name__ == "__main__":
    main()
