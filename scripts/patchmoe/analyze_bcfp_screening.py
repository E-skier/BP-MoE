#!/usr/bin/env python3
"""Quick screening analysis for BCFP experiments.

Reads metrics.jsonl from a training run and prints key load-balance
and quality indicators for the last N records.

Usage:
  python scripts/patchmoe/analyze_bcfp_screening.py \
    blt1b_warmstart/bcfp_experiments/bcfp_B_calibrated/metrics.jsonl

Pass --tail N to control how many records are aggregated (default 100).
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _safe_float(record: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = record.get(key, default)
    return float(value) if value is not None else default


def load_metrics(path: Path, tail: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records[-tail:] if len(records) > tail else records


def analyze(records: list[dict[str, Any]]) -> dict[str, float]:
    if not records:
        raise ValueError("No metric records to analyze")

    keys_of_interest = [
        # Quality
        "bpb_avg",
        "loss_avg",
        # Load balance (pair-level)
        "pair_load_cv_mean",
        "pair_max_byte_fraction_mean",
        "pair_min_byte_fraction_mean",
        "pair_dead_count_mean",
        # Load balance (expert-level)
        "load_imbalance_mean",
        "max_load_fraction_mean",
        "min_load_fraction_mean",
        "balance_loss_mean",
        # Router behavior
        "router_entropy_mean",
        "pair_router_entropy_mean",
        "routing_granularity_pair_mean",
        "hidden_state_routing_mean",
        "hidden_residual_scale_mean",
        "side_feature_count_mean",
        # Specialization
        "entropy_selected_pair_corr_mean",
        "top2_same_pair_fraction_mean",
        # System
        "wps_mean",
    ]

    result: dict[str, float] = {}
    for key in keys_of_interest:
        values = [_safe_float(rec, key) for rec in records if key in rec]
        if values:
            result[key] = _mean(values)

    # Pair-level per-pair byte fractions
    pair_keys = sorted(
        {k for rec in records for k in rec if k.startswith("pair_") and k.endswith("_byte_fraction_mean")}
    )
    for pk in pair_keys:
        values = [_safe_float(rec, pk) for rec in records if pk in rec]
        if values:
            result[pk] = _mean(values)

    # Expert-level per-expert load fractions
    expert_keys = sorted(
        {k for rec in records for k in rec if k.startswith("expert_") and k.endswith("_load_fraction_mean")}
    )
    for ek in expert_keys:
        values = [_safe_float(rec, ek) for rec in records if ek in rec]
        if values:
            result[ek] = _mean(values)

    # Step range
    steps = [_safe_float(rec, "global_step") for rec in records if "global_step" in rec]
    if steps:
        result["step_min"] = min(steps)
        result["step_max"] = max(steps)

    return result


def print_report(result: dict[str, float]) -> None:
    def value(key: str) -> float:
        return result.get(key, float("nan"))

    print()
    print("=" * 72)
    print("  BCFP Screening Report")
    print("=" * 72)
    if "step_min" in result:
        print(f"  Step range:  {result['step_min']:.0f} → {result['step_max']:.0f}")

    print()
    print("  ── Quality ──")
    print(f"  BPB:                              {value('bpb_avg'):.4f}")
    print(f"  Loss:                             {value('loss_avg'):.4f}")

    print()
    print("  ── Pair-level Load Balance ──")
    pair_cv = value("pair_load_cv_mean")
    print(f"  Pair load CV:                     {pair_cv:.4f}  {'✓' if pair_cv < 0.3 else '✗ COLLAPSE' if pair_cv > 0.5 else '⚠'}")
    print(f"  Pair max byte fraction:           {value('pair_max_byte_fraction_mean'):.4f}")
    print(f"  Pair min byte fraction:           {value('pair_min_byte_fraction_mean'):.4f}")
    dead = value("pair_dead_count_mean")
    print(f"  Dead pair count (<0.5%):           {dead:.1f}  {'✓' if dead < 1 else '✗ DEAD PAIRS'}")

    print()
    print("  ── Per-Pair Byte Shares ──")
    for pk in sorted(k for k in result if k.startswith("pair_") and k.endswith("_byte_fraction_mean")):
        pair_id = pk.split("_")[1]
        share = result[pk]
        bar = "█" * int(share * 40) if share > 0 else ""
        status = "✓" if 0.15 <= share <= 0.35 else "✗" if share < 0.05 else "⚠"
        print(f"  Pair {pair_id}:  {share:.4f}  {bar} {status}")

    print()
    print("  ── Expert-level Load ──")
    print(f"  Load imbalance:                   {value('load_imbalance_mean'):.4f}")
    print(f"  Max load fraction:                {value('max_load_fraction_mean'):.4f}")
    print(f"  Min load fraction:                {value('min_load_fraction_mean'):.4f}")
    print(f"  Balance loss:                     {value('balance_loss_mean'):.6f}")

    for ek in sorted(k for k in result if k.startswith("expert_") and k.endswith("_load_fraction_mean")):
        expert_id = ek.split("_")[1]
        share = result[ek]
        bar = "█" * int(share * 80) if share > 0 else ""
        print(f"  Expert {expert_id}: {share:.4f}  {bar}")

    print()
    print("  ── Router Behavior ──")
    print(f"  Router entropy:                   {value('router_entropy_mean'):.4f}")
    print(f"  Pair router entropy:              {value('pair_router_entropy_mean'):.4f}")
    print(f"  Top-2 same-pair fraction:         {value('top2_same_pair_fraction_mean'):.4f}")
    print(f"  Entropy-selected pair corr:       {value('entropy_selected_pair_corr_mean'):.4f}")

    print()
    print("  ── System ──")
    print(f"  WPS (words per second):          {value('wps_mean'):.1f}")
    print()

    # Decision
    print("  ── Pass/Fail ──")
    checks = []
    if value("bpb_avg") < 1.0:
        checks.append(("✓", "BPB < 1.0"))
    else:
        checks.append(("✗", f"BPB = {value('bpb_avg'):.4f} ≥ 1.0"))
    if pair_cv < 0.3:
        checks.append(("✓", "Pair load CV < 0.3 (balanced)"))
    elif pair_cv < 0.5:
        checks.append(("⚠", f"Pair load CV = {pair_cv:.4f} (moderate imbalance)"))
    else:
        checks.append(("✗", f"Pair load CV = {pair_cv:.4f} (severe imbalance, consider D)"))
    if dead < 1:
        checks.append(("✓", "No dead pairs"))
    else:
        checks.append(("✗", f"{dead:.0f} dead pairs (consider D)"))
    if value("pair_min_byte_fraction_mean") > 0.05:
        checks.append(("✓", "All pairs above 5% floor"))
    else:
        checks.append(("✗", "Some pair below 5% byte share (consider D)"))

    for symbol, msg in checks:
        print(f"  {symbol}  {msg}")

    all_pass = all(s == "✓" for s, _ in checks)
    print()
    if all_pass:
        print("  ▶ RECOMMENDATION: Proceed to 50k training.")
    else:
        print("  ▶ RECOMMENDATION: Fix issues before scaling. Consider D config.")
    print("=" * 72)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Screening analysis for BCFP experiments")
    parser.add_argument("metrics_jsonl", type=Path, help="Path to metrics.jsonl")
    parser.add_argument("--tail", type=int, default=100, help="Number of last records to aggregate")
    args = parser.parse_args()

    if not args.metrics_jsonl.exists():
        print(f"ERROR: {args.metrics_jsonl} not found", file=sys.stderr)
        sys.exit(1)

    records = load_metrics(args.metrics_jsonl, args.tail)
    result = analyze(records)
    print_report(result)


if __name__ == "__main__":
    main()
