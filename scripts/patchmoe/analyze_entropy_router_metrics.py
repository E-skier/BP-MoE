#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path


def _pearson(xs, ys):
    n = len(xs)
    if n == 0:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def _last_window(path, window):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows[-window:]


def _mean(rows, key):
    vals = [row[key] for row in rows if key in row]
    if not vals:
        return float("nan")
    return sum(vals) / len(vals)


def _expert_vector(row, suffix, num_experts=8):
    return [row.get(f"moe/expert_{i}_{suffix}") for i in range(num_experts)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics_jsonl")
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument("--num-experts", type=int, default=8)
    args = parser.parse_args()

    path = Path(args.metrics_jsonl)
    rows = _last_window(path, args.window)
    if not rows:
        raise SystemExit(f"No metrics rows found: {path}")

    final = rows[-1]
    expert_ids = list(range(args.num_experts))
    final_entropy = _expert_vector(final, "patch_entropy_mean_mean", args.num_experts)
    final_length = _expert_vector(final, "patch_length_mean_mean", args.num_experts)
    final_load = _expert_vector(final, "load_fraction_mean", args.num_experts)
    final_unit = _expert_vector(
        final, "unit_assignment_fraction_mean", args.num_experts
    )

    high_rows = [
        _expert_vector(
            row, "entropy_bucket_high_assignment_fraction_mean", args.num_experts
        )
        for row in rows
    ]
    medium_rows = [
        _expert_vector(
            row, "entropy_bucket_medium_assignment_fraction_mean", args.num_experts
        )
        for row in rows
    ]
    low_rows = [
        _expert_vector(
            row, "entropy_bucket_low_assignment_fraction_mean", args.num_experts
        )
        for row in rows
    ]

    def average_vector(vectors):
        return [
            sum(vec[i] for vec in vectors if vec[i] is not None)
            / max(sum(1 for vec in vectors if vec[i] is not None), 1)
            for i in range(args.num_experts)
        ]

    high = average_vector(high_rows)
    medium = average_vector(medium_rows)
    low = average_vector(low_rows)

    summary = {
        "path": str(path),
        "rows_used": len(rows),
        "first_step": rows[0].get("global_step"),
        "last_step": final.get("global_step"),
        "router_entropy_mean": _mean(rows, "moe/router_entropy_mean"),
        "router_entropy_final": final.get("moe/router_entropy_mean"),
        "ln_num_experts": math.log(args.num_experts),
        "active_experts_mean": _mean(rows, "moe/active_experts_mean"),
        "load_imbalance_mean": _mean(rows, "moe/load_imbalance_mean"),
        "final_expert_load": final_load,
        "final_expert_unit_assignment": final_unit,
        "final_expert_patch_entropy_mean": final_entropy,
        "final_expert_patch_length_mean": final_length,
        "corr_expert_id_vs_final_entropy_mean": _pearson(expert_ids, final_entropy),
        "corr_final_load_vs_final_entropy_mean": _pearson(final_load, final_entropy),
        "corr_final_length_vs_final_entropy_mean": _pearson(
            final_length, final_entropy
        ),
        "window_entropy_bucket_high_assignment": high,
        "window_entropy_bucket_medium_assignment": medium,
        "window_entropy_bucket_low_assignment": low,
        "corr_expert_id_vs_window_high_bucket": _pearson(expert_ids, high),
        "corr_expert_id_vs_window_medium_bucket": _pearson(expert_ids, medium),
        "high_minus_medium_assignment": [h - m for h, m in zip(high, medium)],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
