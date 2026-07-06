import json
from pathlib import Path
import tempfile

import pyarrow as pa
import torch

from scripts.calibrate_patchmoe_router import (
    calibrate_entropy_prior,
    collect_entropy_patch_statistics_from_arrow,
    write_calibration,
)


def test_calibrate_entropy_prior_uses_valid_byte_weighted_quantiles():
    report = calibrate_entropy_prior(
        patch_lengths=torch.tensor([10, 10, 10, 10, 0]),
        patch_entropies=torch.tensor([0.0, 1.0, 2.0, 3.0, 100.0]),
        num_pairs=4,
        prior_scale=0.5,
        bias_steps=25,
    )

    assert report["num_pairs"] == 4
    assert report["num_experts"] == 8
    assert report["pair_size"] == 2
    assert report["entropy_centers"] == [0.0, 1.0, 2.0, 3.0]
    assert report["calibration_num_valid_patches"] == 4
    assert report["calibration_total_bytes"] == 40


def test_calibrate_entropy_prior_writes_json_with_balanced_soft_byte_share():
    patch_entropies = torch.tensor(
        [0.0, 0.05, 1.0, 1.05, 2.0, 2.05, 3.0, 3.05]
    )
    patch_lengths = torch.tensor([10, 10, 10, 10, 10, 10, 10, 10])
    report = calibrate_entropy_prior(
        patch_lengths=patch_lengths,
        patch_entropies=patch_entropies,
        num_pairs=4,
        prior_scale=0.5,
        bias_steps=100,
        bias_lr=0.5,
    )

    assert max(abs(share - 0.25) for share in report["calibrated_byte_share"]) <= 0.05
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "router_calibration.json"
        write_calibration(report, output_path)
        reloaded = json.loads(output_path.read_text())

    assert reloaded["version"] == 1
    assert reloaded["static_pair_bias"] == report["static_pair_bias"]


def test_collect_entropy_patch_statistics_from_arrow_uses_entropy_patcher():
    with tempfile.TemporaryDirectory() as tmpdir:
        arrow_path = Path(tmpdir) / "entropy.arrow"
        table = pa.table(
            {
                "sample_id": ["row0", "empty", "row1"],
                "text": ["abcd", "", "efgh"],
                "entropies": [
                    [0.1, 1.5, 0.2, 1.6],
                    [],
                    [0.3, 0.4, 2.0, 0.6],
                ],
            }
        )
        with pa.OSFile(str(arrow_path), "wb") as sink:
            with pa.ipc.new_file(sink, table.schema) as writer:
                writer.write_table(table)

        lengths, entropies, metadata = collect_entropy_patch_statistics_from_arrow(
            [arrow_path],
            threshold=1.0,
            include_next_token=False,
        )

    assert lengths.tolist() == [1.0, 1.0, 2.0, 1.0, 2.0, 1.0]
    assert torch.allclose(
        entropies,
        torch.tensor([0.1, 1.5, 0.9, 0.3, 1.2, 0.6], dtype=torch.float32),
    )
    assert metadata["num_rows_seen"] == 3
    assert metadata["num_rows_used"] == 2
    assert metadata["num_valid_patches"] == 6
    assert metadata["total_bytes"] == 8
