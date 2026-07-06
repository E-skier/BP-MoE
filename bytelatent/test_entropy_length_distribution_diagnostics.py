import math
from pathlib import Path
import tempfile

import pyarrow as pa

from scripts.diagnostics.analyze_entropy_length_distribution import (
    classify_entropy_bucket,
    classify_length_bucket,
    summarize_entropy_length_distribution,
)
from scripts.diagnostics.build_entropy_length_balanced_source import (
    dominant_entropy_length_bucket,
    entropy_length_cell_byte_counts,
    select_records_per_bucket,
    select_records_by_top_cell_fraction,
    _write_arrow,
)


def test_bucket_boundaries_match_training_metrics():
    assert classify_length_bucket(4) == "short"
    assert classify_length_bucket(5) == "medium"
    assert classify_length_bucket(8) == "medium"
    assert classify_length_bucket(9) == "long"

    assert classify_entropy_bucket(1.0) == "low"
    assert classify_entropy_bucket(1.01) == "medium"
    assert classify_entropy_bucket(2.0) == "medium"
    assert classify_entropy_bucket(2.01) == "high"


def test_summarize_entropy_length_distribution_tracks_byte_mass():
    report = summarize_entropy_length_distribution(
        patch_lengths=[2, 6, 10, 10],
        patch_entropies=[0.5, 1.5, 2.5, 0.5],
    )

    assert report["total_patches"] == 4
    assert report["total_bytes"] == 28
    assert report["mean_patch_length"] == 7.0

    cells = {cell["bucket"]: cell for cell in report["cells"]}
    assert cells["low/short"]["patch_count"] == 1
    assert cells["low/short"]["byte_count"] == 2
    assert math.isclose(cells["low/short"]["byte_fraction"], 2 / 28)

    assert cells["medium/medium"]["patch_count"] == 1
    assert cells["medium/medium"]["byte_count"] == 6
    assert math.isclose(cells["medium/medium"]["byte_fraction"], 6 / 28)

    assert cells["high/long"]["patch_count"] == 1
    assert cells["high/long"]["byte_count"] == 10
    assert math.isclose(cells["high/long"]["byte_fraction"], 10 / 28)

    assert cells["low/long"]["patch_count"] == 1
    assert cells["low/long"]["byte_count"] == 10
    assert math.isclose(cells["low/long"]["byte_fraction"], 10 / 28)

    length = {row["bucket"]: row for row in report["length_marginals"]}
    assert length["long"]["patch_count"] == 2
    assert length["long"]["byte_count"] == 20
    assert math.isclose(length["long"]["byte_fraction"], 20 / 28)

    entropy = {row["bucket"]: row for row in report["entropy_marginals"]}
    assert entropy["low"]["patch_count"] == 2
    assert entropy["low"]["byte_count"] == 12
    assert math.isclose(entropy["low"]["byte_fraction"], 12 / 28)


def test_dominant_entropy_length_bucket_uses_byte_mass_not_patch_count():
    bucket = dominant_entropy_length_bucket(
        patch_lengths=[2, 2, 12],
        patch_entropies=[2.5, 2.5, 0.5],
    )

    assert bucket == "low/long"


def test_entropy_length_cell_byte_counts_tracks_cross_cells():
    counts = entropy_length_cell_byte_counts(
        patch_lengths=[2, 6, 10],
        patch_entropies=[2.5, 1.5, 0.5],
    )

    assert counts == {
        "high/short": 2.0,
        "medium/medium": 6.0,
        "low/long": 10.0,
    }


def test_select_records_per_bucket_balances_nonempty_buckets():
    records = [
        {"sample_id": "low-0", "dominant_bucket": "low/long"},
        {"sample_id": "low-1", "dominant_bucket": "low/long"},
        {"sample_id": "med-0", "dominant_bucket": "medium/short"},
        {"sample_id": "med-1", "dominant_bucket": "medium/short"},
        {"sample_id": "high-0", "dominant_bucket": "high/short"},
    ]

    selected, metadata = select_records_per_bucket(records, records_per_bucket=1)

    assert [row["sample_id"] for row in selected] == ["low-0", "med-0", "high-0"]
    assert metadata["available_by_bucket"] == {
        "low/long": 2,
        "medium/short": 2,
        "high/short": 1,
    }
    assert metadata["selected_by_bucket"] == {
        "low/long": 1,
        "medium/short": 1,
        "high/short": 1,
    }


def test_select_records_by_top_cell_fraction_allows_targeted_duplicates():
    records = [
        {
            "sample_id": "mixed",
            "dominant_bucket": "low/long",
            "total_bytes": 10.0,
            "cell_byte_counts": {"low/long": 8.0, "high/short": 2.0},
        },
        {
            "sample_id": "high-rich",
            "dominant_bucket": "low/medium",
            "total_bytes": 10.0,
            "cell_byte_counts": {"low/medium": 6.0, "high/short": 4.0},
        },
    ]

    selected, metadata = select_records_by_top_cell_fraction(
        records,
        records_per_bucket=1,
    )

    high_rows = [
        row for row in selected if row["selection_bucket"] == "high/short"
    ]
    assert [row["source_sample_id"] for row in high_rows] == ["high-rich"]
    assert high_rows[0]["sample_id"] == "high-rich::selected::high_short::000000"
    assert metadata["available_by_bucket"]["high/short"] == 2
    assert metadata["selected_by_bucket"]["high/short"] == 1


def test_write_arrow_accepts_python_float_entropies():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "sample.arrow"
        _write_arrow(
            path,
            [
                {
                    "sample_id": "a",
                    "text": "abc",
                    "entropies": [0.25, 1.5, 2.25],
                }
            ],
        )

        with pa.memory_map(str(path), "r") as source:
            reader = pa.ipc.open_file(source)
            batch = reader.get_batch(0)

    assert batch.column("sample_id").to_pylist() == ["a"]
    assert batch.column("text").to_pylist() == ["abc"]
    assert batch.column("entropies").to_pylist() == [[0.25, 1.5, 2.25]]
