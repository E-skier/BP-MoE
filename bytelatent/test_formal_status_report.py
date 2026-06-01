import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts/patchmoe/formal_status_report.py"
)
spec = importlib.util.spec_from_file_location("formal_status_report", SCRIPT_PATH)
formal_status_report = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(formal_status_report)


def write_metrics(path: Path, *, step: int, bpb: float = 1.25) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "global_step": step,
                "bpb/interval_across_gpus": bpb,
                "speed/wps": 1234.0,
                "moe/load_imbalance_mean": 1.5,
                "moe/side_feature_count_mean": 7.0,
                "moe/congestion_weight_mean": 0.5,
                "moe/router_z_loss_weight_mean": 0.001,
            }
        )
        + "\n"
    )


def write_validation(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "a.arrow": {"n_bytes": 2, "loss_sum": 2.0},
                "b.arrow": {"n_bytes": 4, "loss_sum": 6.0},
            }
        )
    )


def test_status_from_artifacts_reports_done_with_train_and_heldout_metrics(tmp_path):
    run_name = "byte_entropy_type_w005_seed779_200000step"
    run_dir = tmp_path / "runs" / run_name
    eval_root = tmp_path / "eval"
    write_metrics(run_dir / "metrics.jsonl", step=200000, bpb=1.125)
    (run_dir / "checkpoints" / "0000200000").mkdir(parents=True)
    write_validation(eval_root / run_name / "0000200000" / "validation.json")

    row = formal_status_report.status_from_artifacts(
        pipeline="200k",
        variant="byte_entropy_type_w005",
        seed="779",
        run_name=run_name,
        run_dir=run_dir,
        eval_root=eval_root,
        final_step=200000,
    )

    assert row["status"] == "done"
    assert row["last_step"] == 200000
    assert row["progress_fraction"] == 1.0
    assert row["final_train_bpb"] == 1.125
    assert row["moe_side_feature_count"] == 7.0
    assert row["heldout_bpb"] == pytest.approx(8.0 / math.log(2) / 6.0)
    assert row["heldout_n_bytes"] == 6.0


def test_status_from_artifacts_reports_partial_for_running_metrics_only(tmp_path):
    run_name = "dense_compute_matched_seed778_200000step"
    run_dir = tmp_path / "dense" / run_name
    eval_root = tmp_path / "eval"
    write_metrics(run_dir / "metrics.jsonl", step=43790, bpb=1.3671875)

    row = formal_status_report.status_from_artifacts(
        pipeline="200k_dense_compute",
        variant="dense_compute_matched",
        seed="778",
        run_name=run_name,
        run_dir=run_dir,
        eval_root=eval_root,
        final_step=200000,
    )

    assert row["status"] == "partial"
    assert row["last_step"] == 43790
    assert row["progress_fraction"] == pytest.approx(43790 / 200000)
    assert row["has_metrics"] is True
    assert row["has_final_checkpoint"] is False
    assert row["has_eval"] is False


def test_collect_rows_includes_external_dense_compute_controls(tmp_path):
    run_root = tmp_path / "matched"
    eval_root = tmp_path / "matched_eval"
    external_run_root = tmp_path / "dense_external"
    external_eval_root = tmp_path / "dense_external_eval"

    run_name = "dense_seed779_200000step"
    write_metrics(run_root / run_name / "metrics.jsonl", step=200000)
    (run_root / run_name / "checkpoints" / "0000200000").mkdir(parents=True)
    write_validation(eval_root / run_name / "0000200000" / "validation.json")

    external_name = "dense_compute_matched_seed779_200000step"
    write_metrics(external_run_root / external_name / "metrics.jsonl", step=100)

    args = SimpleNamespace(
        stage1_variants=[],
        stage1_run_root=tmp_path / "stage1",
        stage1_eval_root=tmp_path / "stage1_eval",
        stage1_steps=100000,
        seeds=["779"],
        matched_200k_variants=["dense"],
        matched_200k_run_root=run_root,
        matched_200k_eval_root=eval_root,
        matched_200k_steps=200000,
        external_dense_compute=[
            f"779:{external_run_root}:{external_eval_root}:200000",
        ],
    )

    rows = formal_status_report.collect_rows(args)

    assert [(row["pipeline"], row["variant"], row["status"]) for row in rows] == [
        ("200k", "dense", "done"),
        ("200k_dense_compute", "dense_compute_matched", "partial"),
    ]


def test_write_outputs_emits_csv_and_json(tmp_path):
    rows = [
        {
            "pipeline": "stage1",
            "variant": "dense",
            "seed": "",
            "run_name": "stage1_dense_matched",
            "status": "todo",
        }
    ]

    formal_status_report.write_outputs(rows, tmp_path)

    assert (
        (tmp_path / "formal_status.csv")
        .read_text()
        .startswith("pipeline,variant,seed,run_name,status")
    )
    assert json.loads((tmp_path / "formal_status.json").read_text()) == rows
