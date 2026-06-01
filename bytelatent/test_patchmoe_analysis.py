import json
import math

import pandas as pd
import pytest

from bytelatent.plotting.patchmoe_phase2_ablation import (
    merge_summary_with_heldout,
    normalize_run_label,
    read_validation_root,
)


def test_read_validation_root_aggregates_sources_by_loss_sum(tmp_path):
    eval_dir = tmp_path / "eval" / "byte_entropy_seed779_200000step" / "0000200000"
    eval_dir.mkdir(parents=True)
    (eval_dir / "validation.json").write_text(
        json.dumps(
            {
                "source_a.arrow": {"n_bytes": 2, "loss_sum": 2.0, "bpb": 1.0},
                "source_b.arrow": {"n_bytes": 4, "loss_sum": 6.0, "bpb": 2.0},
            }
        )
    )

    by_source, summary = read_validation_root(tmp_path / "eval")

    assert len(by_source) == 2
    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["label"] == "byte_entropy_seed779_200000step"
    assert row["normalized_variant"] == "byte_entropy"
    assert row["heldout_n_bytes"] == 6.0
    assert row["heldout_loss_sum"] == 8.0
    assert row["heldout_bpb"] == pytest.approx(8.0 / math.log(2) / 6.0)


def test_merge_summary_with_heldout_uses_exact_run_label_without_seed_cross_product():
    summary = pd.DataFrame(
        {
            "variant": [
                "byte_entropy_seed778_200000step",
                "byte_entropy_seed779_200000step",
            ],
            "normalized_variant": ["byte_entropy", "byte_entropy"],
            "final_bpb": [1.2, 1.1],
        }
    )
    validation_summary = pd.DataFrame(
        {
            "label": [
                "byte_entropy_seed778_200000step",
                "byte_entropy_seed779_200000step",
            ],
            "normalized_variant": ["byte_entropy", "byte_entropy"],
            "global_step": [200000, 200000],
            "heldout_bpb": [1.05, 1.03],
        }
    )

    merged = merge_summary_with_heldout(summary, validation_summary)

    assert len(merged) == 2
    assert merged["variant"].tolist() == summary["variant"].tolist()
    assert merged["heldout_bpb"].tolist() == [1.05, 1.03]


def test_merge_summary_with_heldout_keeps_latest_eval_step_for_label():
    summary = pd.DataFrame(
        {
            "variant": ["stage1_byte_type_w005_matched"],
            "normalized_variant": ["byte_type_w005"],
        }
    )
    validation_summary = pd.DataFrame(
        {
            "label": [
                "stage1_byte_type_w005_matched",
                "stage1_byte_type_w005_matched",
            ],
            "normalized_variant": ["byte_type_w005", "byte_type_w005"],
            "global_step": [50000, 100000],
            "heldout_bpb": [1.2, 1.1],
        }
    )

    merged = merge_summary_with_heldout(summary, validation_summary)

    assert len(merged) == 1
    assert merged.iloc[0]["global_step"] == 100000
    assert merged.iloc[0]["heldout_bpb"] == 1.1


def test_normalize_run_label_handles_stage1_and_seed_suffixes():
    assert normalize_run_label("stage1_byte_type_w005_matched") == "byte_type_w005"
    assert (
        normalize_run_label("byte_entropy_type_w005_seed779_200000step")
        == "byte_entropy_type_w005"
    )
