import csv
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "patchmoe" / "summarize_opencompass_results.py"


def load_summary_module():
    spec = importlib.util.spec_from_file_location("opencompass_summary", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_summary(path, rows):
    path.parent.mkdir(parents=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "version",
                "metric",
                "mode",
                "entropy_only",
                "hidden_only",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def test_build_wide_rows_uses_opencompass_summary_columns(tmp_path):
    module = load_summary_module()
    summary = tmp_path / "latest" / "summary" / "summary.csv"
    write_summary(
        summary,
        [
            {
                "dataset": "mmlu",
                "version": "aaa",
                "metric": "accuracy",
                "mode": "ppl",
                "entropy_only": "37.5",
                "hidden_only": "32.0",
            },
            {
                "dataset": "SuperGLUE_BoolQ",
                "version": "bbb",
                "metric": "accuracy",
                "mode": "ppl",
                "entropy_only": "62.0",
                "hidden_only": "58.0",
            },
        ],
    )

    rows = module.read_all([tmp_path])
    wide = module.build_wide_rows(rows, module.canonical_dataset_specs(None))

    entropy_row = next(row for row in wide if row["Method"] == "entropy_only")
    assert entropy_row["MMLU"] == "37.5"
    assert entropy_row["BoolQ"] == "62"
    assert entropy_row["Average"] == "49.75"


def test_mmlu_subject_rows_are_averaged_when_group_row_is_missing(tmp_path):
    module = load_summary_module()
    summary = tmp_path / "summary" / "summary.csv"
    write_summary(
        summary,
        [
            {
                "dataset": "lukaemon_mmlu_abstract_algebra",
                "version": "aaa",
                "metric": "accuracy",
                "mode": "ppl",
                "entropy_only": "20",
                "hidden_only": "-",
            },
            {
                "dataset": "lukaemon_mmlu_anatomy",
                "version": "aaa",
                "metric": "accuracy",
                "mode": "ppl",
                "entropy_only": "40",
                "hidden_only": "-",
            },
        ],
    )

    rows = module.read_all([tmp_path])
    wide = module.build_wide_rows(rows, module.canonical_dataset_specs(None))

    entropy_row = next(row for row in wide if row["Method"] == "entropy_only")
    assert entropy_row["MMLU"] == "30"


def test_cli_writes_long_and_wide_csv(tmp_path):
    summary = tmp_path / "oc" / "latest" / "summary" / "summary.csv"
    write_summary(
        summary,
        [
            {
                "dataset": "hellaswag",
                "version": "ccc",
                "metric": "accuracy",
                "mode": "ppl",
                "entropy_only": "45.25",
                "hidden_only": "41.0",
            }
        ],
    )
    output_dir = tmp_path / "out"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(tmp_path / "oc"),
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "opencompass_long.csv" in result.stdout
    assert (output_dir / "opencompass_long.csv").exists()
    assert (output_dir / "opencompass_paper_table.csv").exists()
    assert "HellaSwag" in (output_dir / "opencompass_paper_table.csv").read_text()


def test_missing_opencompass_columns_raise_clear_error(tmp_path):
    module = load_summary_module()
    bad = tmp_path / "bad.csv"
    bad.write_text("name,score\nx,1\n")

    with pytest.raises(ValueError, match="missing columns"):
        module.read_opencompass_summary(bad)
