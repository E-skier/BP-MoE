#!/usr/bin/env python3
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PAIR_BYTE_KEYS = [f"moe/pair_{idx}_byte_fraction_mean" for idx in range(4)]


@dataclass(frozen=True)
class ScreeningThresholds:
    min_pair_byte_fraction: float = 0.01
    max_pair_byte_fraction: float = 0.70
    max_bpb_relative_regression: float = 0.015
    expected_steps: int | None = None
    require_checkpoint: bool = False


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics file: {path}")
    rows = []
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path}:{line_no}: {exc}") from exc
    if not rows:
        raise ValueError(f"No metrics rows found in {path}")
    return rows


def mean_metric(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [row.get(key) for row in rows if row.get(key) is not None]
    if not values:
        return None
    return float(sum(float(value) for value in values) / len(values))


def last_metric(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    return None if value is None else float(value)


def checkpoint_present(run_dir: Path) -> bool:
    checkpoint_dir = run_dir / "checkpoints"
    if not checkpoint_dir.exists():
        return False
    return any(path.is_dir() for path in checkpoint_dir.iterdir())


def evaluate_thresholds(
    summary: dict[str, Any], thresholds: ScreeningThresholds
) -> dict[str, Any]:
    reasons: list[str] = []
    if thresholds.expected_steps is not None and (
        summary["last_step"] < thresholds.expected_steps
    ):
        reasons.append(
            f"last_step {summary['last_step']} is below expected {thresholds.expected_steps}"
        )
    if thresholds.require_checkpoint and not summary["checkpoint_present"]:
        reasons.append("checkpoint is required but was not found")
    if summary["oom"]:
        reasons.append("memory/num_ooms is non-zero")
    if summary["pair_dead_count"] > 0:
        reasons.append(f"pair_dead_count is {summary['pair_dead_count']}")
    if summary["pair_min_byte_fraction"] < thresholds.min_pair_byte_fraction:
        reasons.append(
            "pair_min_byte_fraction "
            f"{summary['pair_min_byte_fraction']:.4f} is below "
            f"{thresholds.min_pair_byte_fraction:.4f}"
        )
    if summary["pair_max_byte_fraction"] > thresholds.max_pair_byte_fraction:
        reasons.append(
            "pair_max_byte_fraction "
            f"{summary['pair_max_byte_fraction']:.4f} is above "
            f"{thresholds.max_pair_byte_fraction:.4f}"
        )
    return {"passed": not reasons, "reasons": reasons}


def analyze_run(
    run_dir: str | Path,
    *,
    label: str | None = None,
    tail_records: int = 100,
    thresholds: ScreeningThresholds | None = None,
) -> dict[str, Any]:
    run_path = Path(run_dir)
    rows = read_jsonl(run_path / "metrics.jsonl")
    tail = rows[-tail_records:]
    last = rows[-1]
    pair_bytes = [float(last[key]) for key in PAIR_BYTE_KEYS if key in last]
    if len(pair_bytes) != len(PAIR_BYTE_KEYS):
        raise ValueError(
            f"Expected {len(PAIR_BYTE_KEYS)} pair byte metrics in {run_path}"
        )

    summary = {
        "label": label or run_path.name,
        "run_dir": str(run_path),
        "metrics_records": len(rows),
        "last_step": int(last["global_step"]),
        "tail_records": len(tail),
        "tail_bpb_mean": mean_metric(tail, "bpb/interval_across_gpus"),
        "last_bpb": last_metric(last, "bpb/interval_across_gpus"),
        "tail_loss_mean": mean_metric(tail, "loss/interval_across_gpu"),
        "pair_byte_fraction": pair_bytes,
        "pair_load_cv": last_metric(last, "moe/pair_load_cv_mean"),
        "pair_min_byte_fraction": last_metric(last, "moe/pair_min_byte_fraction_mean"),
        "pair_max_byte_fraction": last_metric(last, "moe/pair_max_byte_fraction_mean"),
        "pair_dead_count": last_metric(last, "moe/pair_dead_count_mean") or 0.0,
        "top2_same_pair_fraction": last_metric(
            last, "moe/top2_same_pair_fraction_mean"
        ),
        "entropy_selected_pair_corr": last_metric(
            last, "moe/entropy_selected_pair_corr_mean"
        ),
        "entropy_expected_pair_corr": last_metric(
            last, "moe/entropy_expected_pair_corr_mean"
        ),
        "wps_tail_mean": mean_metric(tail, "speed/wps"),
        "iter_time_tail_mean": mean_metric(tail, "speed/curr_iter_time"),
        "gpu_max_active_gib": last_metric(last, "memory/max_active_gib"),
        "gpu_max_reserved_gib": last_metric(last, "memory/max_reserved_gib"),
        "ep_alltoall_bytes_tail_mean": mean_metric(
            tail, "moe/ep_all_to_all_bytes_mean"
        ),
        "oom": bool(last_metric(last, "memory/num_ooms") or 0.0),
        "checkpoint_present": checkpoint_present(run_path),
    }
    summary["pass_fail"] = evaluate_thresholds(
        summary, thresholds or ScreeningThresholds()
    )
    return summary


def compare_to_baseline(
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    thresholds: ScreeningThresholds,
) -> dict[str, Any]:
    baseline_bpb = baseline.get("tail_bpb_mean")
    candidate_bpb = candidate.get("tail_bpb_mean")
    if baseline_bpb is None or candidate_bpb is None:
        return candidate
    relative_delta = round((candidate_bpb - baseline_bpb) / baseline_bpb, 6)
    candidate["bpb_relative_delta_vs_baseline"] = relative_delta
    if relative_delta > thresholds.max_bpb_relative_regression:
        candidate["pass_fail"]["passed"] = False
        candidate["pass_fail"]["reasons"].append(
            "tail_bpb_mean regressed by "
            f"{relative_delta * 100:.2f}% vs baseline, limit is "
            f"{thresholds.max_bpb_relative_regression * 100:.2f}%"
        )
    return candidate


def parse_run(value: str) -> tuple[str, Path]:
    if "=" in value:
        label, path = value.split("=", 1)
        if not label:
            raise argparse.ArgumentTypeError("run label cannot be empty")
        return label, Path(path)
    path = Path(value)
    return path.name, path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize PatchMoE CPT metrics and screening pass/fail status."
    )
    parser.add_argument(
        "--run",
        action="append",
        type=parse_run,
        required=True,
        help="Run directory, optionally LABEL=PATH. May be provided multiple times.",
    )
    parser.add_argument("--baseline-label", default=None)
    parser.add_argument("--tail-records", type=int, default=100)
    parser.add_argument("--expected-steps", type=int, default=None)
    parser.add_argument("--require-checkpoint", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    thresholds = ScreeningThresholds(
        expected_steps=args.expected_steps,
        require_checkpoint=args.require_checkpoint,
    )
    summaries = [
        analyze_run(
            path,
            label=label,
            tail_records=args.tail_records,
            thresholds=thresholds,
        )
        for label, path in args.run
    ]
    baseline = None
    if args.baseline_label is not None:
        matches = [
            summary for summary in summaries if summary["label"] == args.baseline_label
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one baseline labelled {args.baseline_label!r}, "
                f"found {len(matches)}"
            )
        baseline = matches[0]
        summaries = [
            summary
            if summary is baseline
            else compare_to_baseline(summary, baseline, thresholds)
            for summary in summaries
        ]

    payload = {"runs": summaries}
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

