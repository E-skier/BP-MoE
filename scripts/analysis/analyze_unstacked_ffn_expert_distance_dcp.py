#!/usr/bin/env python3
import argparse
import csv
import json
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
    return state[key]


def cos_dist_matrix(tensors):
    x = torch.stack([t.float().reshape(-1) for t in tensors], dim=0)
    x = torch.nn.functional.normalize(x, dim=1, eps=1e-12)
    return 1.0 - x @ x.T


def same_pair(i, j):
    return i // 2 == j // 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    md = read_metadata(args.ckpt)

    pat = re.compile(
        r"^model\.global_transformer\.layers\.(\d+)\.feed_forward\.experts\.(\d+)\.(w[123])\.weight$"
    )

    groups = defaultdict(dict)

    for key, meta in md.items():
        m = pat.match(key)
        if not m:
            continue
        layer = int(m.group(1))
        expert = int(m.group(2))
        mat = m.group(3)
        groups[(layer, mat)][expert] = key

    print(f"groups found: {len(groups)}")

    summary_csv = args.out / "ffn_unstacked_expert_distance_summary.csv"
    pair_csv = args.out / "ffn_unstacked_expert_pairwise_distance.csv"

    summaries = []

    with pair_csv.open("w", newline="") as fpair:
        wp = csv.writer(fpair)
        wp.writerow([
            "layer", "matrix", "expert_i", "expert_j",
            "same_original_pair", "cosine_distance"
        ])

        for (layer, mat), expert_to_key in sorted(groups.items()):
            experts = sorted(expert_to_key)

            if len(experts) != 8:
                print(f"[skip] layer={layer} {mat}: expected 8 experts, got {experts}")
                continue

            tensors = []
            print(f"[load] layer={layer} {mat}")

            for e in experts:
                key = expert_to_key[e]
                meta = md[key]
                t = load_one(args.ckpt, key, shape_of(meta), dtype_of(meta))
                tensors.append(t)

            dist = cos_dist_matrix(tensors)

            within = []
            cross = []
            all_d = []

            for i in range(8):
                for j in range(i + 1, 8):
                    d = float(dist[i, j])
                    all_d.append(d)
                    if same_pair(i, j):
                        within.append(d)
                        sp = 1
                    else:
                        cross.append(d)
                        sp = 0

                    wp.writerow([layer, mat, i, j, sp, d])

            mean_all = sum(all_d) / len(all_d)
            mean_within = sum(within) / len(within)
            mean_cross = sum(cross) / len(cross)

            item = {
                "layer": layer,
                "matrix": mat,
                "mean_pairwise_cosine_distance": mean_all,
                "mean_within_original_pair_distance": mean_within,
                "mean_cross_pair_distance": mean_cross,
                "within_minus_cross": mean_within - mean_cross,
            }
            summaries.append(item)

            print(
                f"[summary] layer={layer:02d} {mat} "
                f"mean={mean_all:.4e} "
                f"within={mean_within:.4e} "
                f"cross={mean_cross:.4e} "
                f"within-cross={mean_within-mean_cross:+.4e}"
            )

            del tensors, dist

    with summary_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "layer", "matrix",
            "mean_pairwise_cosine_distance",
            "mean_within_original_pair_distance",
            "mean_cross_pair_distance",
            "within_minus_cross",
        ])
        for x in summaries:
            w.writerow([
                x["layer"], x["matrix"],
                x["mean_pairwise_cosine_distance"],
                x["mean_within_original_pair_distance"],
                x["mean_cross_pair_distance"],
                x["within_minus_cross"],
            ])

    json_path = args.out / "ffn_unstacked_expert_distance_summary.json"
    json_path.write_text(json.dumps(summaries, indent=2))

    print(f"\nwrote: {summary_csv}")
    print(f"wrote: {pair_csv}")
    print(f"wrote: {json_path}")


if __name__ == "__main__":
    main()
