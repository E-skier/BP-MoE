import json
from pathlib import Path
import tempfile

from scripts.analyze_patchmoe_cpt import (
    ScreeningThresholds,
    analyze_run,
    compare_to_baseline,
)


def write_metrics(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def metric_row(step: int, bpb: float, pair_bytes: list[float]) -> dict:
    return {
        "global_step": step,
        "bpb/interval_across_gpus": bpb,
        "memory/num_ooms": 0,
        "memory/max_active_gib": 25.5,
        "memory/max_reserved_gib": 36.0,
        "moe/top2_same_pair_fraction_mean": 1.0,
        "moe/entropy_selected_pair_corr_mean": 0.8,
        "moe/pair_load_cv_mean": 0.5,
        "moe/pair_min_byte_fraction_mean": min(pair_bytes),
        "moe/pair_max_byte_fraction_mean": max(pair_bytes),
        "moe/pair_dead_count_mean": 0.0,
        **{
            f"moe/pair_{idx}_byte_fraction_mean": value
            for idx, value in enumerate(pair_bytes)
        },
    }


def test_analyze_run_summarizes_tail_metrics_and_passes_screening_thresholds():
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir) / "B"
        write_metrics(
            run_dir / "metrics.jsonl",
            [
                metric_row(50, 0.9, [0.10, 0.20, 0.30, 0.40]),
                metric_row(100, 0.8, [0.12, 0.22, 0.28, 0.38]),
                metric_row(150, 0.7, [0.14, 0.24, 0.26, 0.36]),
            ],
        )

        summary = analyze_run(run_dir, label="B", tail_records=2)

    assert summary["label"] == "B"
    assert summary["last_step"] == 150
    assert summary["tail_bpb_mean"] == 0.75
    assert summary["pair_byte_fraction"] == [0.14, 0.24, 0.26, 0.36]
    assert summary["pass_fail"]["passed"] is True
    assert summary["pass_fail"]["reasons"] == []


def test_compare_to_baseline_fails_when_bpb_regresses_too_much():
    thresholds = ScreeningThresholds(max_bpb_relative_regression=0.015)
    baseline = {"tail_bpb_mean": 0.8}
    candidate = {
        "tail_bpb_mean": 0.83,
        "pass_fail": {"passed": True, "reasons": []},
    }

    compared = compare_to_baseline(candidate, baseline, thresholds)

    assert compared["bpb_relative_delta_vs_baseline"] == 0.0375
    assert compared["pass_fail"]["passed"] is False
    assert compared["pass_fail"]["reasons"] == [
        "tail_bpb_mean regressed by 3.75% vs baseline, limit is 1.50%"
    ]

