#!/usr/bin/env python3
import argparse
import glob
import json
from collections import OrderedDict, defaultdict
from pathlib import Path
import sys
from typing import Iterable, Sequence

import numpy as np
import pyarrow as pa

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.calibrate_patchmoe_router import (  # noqa: E402
    _entropy_patch_lengths,
    _iter_arrow_record_batches,
    _mean_patch_entropies,
)
from scripts.diagnostics.analyze_entropy_length_distribution import (  # noqa: E402
    ENTROPY_BUCKETS,
    LENGTH_BUCKETS,
    classify_entropy_bucket,
    classify_length_bucket,
)


def dominant_entropy_length_bucket(
    patch_lengths: Sequence[float],
    patch_entropies: Sequence[float],
) -> str:
    byte_counts = entropy_length_cell_byte_counts(patch_lengths, patch_entropies)
    if not byte_counts:
        raise ValueError("cannot classify a record with no positive-length patches")
    return max(byte_counts.items(), key=lambda item: item[1])[0]


def entropy_length_cell_byte_counts(
    patch_lengths: Sequence[float],
    patch_entropies: Sequence[float],
) -> dict[str, float]:
    if len(patch_lengths) != len(patch_entropies):
        raise ValueError("patch_lengths and patch_entropies must have the same length")

    byte_counts: dict[str, float] = defaultdict(float)
    for length, entropy in zip(patch_lengths, patch_entropies):
        length = float(length)
        if length <= 0:
            continue
        bucket = (
            f"{classify_entropy_bucket(float(entropy))}/"
            f"{classify_length_bucket(length)}"
        )
        byte_counts[bucket] += length
    return dict(byte_counts)


def bucket_order() -> list[str]:
    return [
        f"{entropy_bucket}/{length_bucket}"
        for entropy_bucket in ENTROPY_BUCKETS
        for length_bucket in LENGTH_BUCKETS
    ]


def select_records_per_bucket(
    records: Sequence[dict],
    *,
    records_per_bucket: int,
) -> tuple[list[dict], dict]:
    if records_per_bucket <= 0:
        raise ValueError("records_per_bucket must be positive")

    grouped: dict[str, list[dict]] = OrderedDict((bucket, []) for bucket in bucket_order())
    for record in records:
        bucket = record["dominant_bucket"]
        grouped.setdefault(bucket, []).append(record)

    selected: list[dict] = []
    available_by_bucket: dict[str, int] = {}
    selected_by_bucket: dict[str, int] = {}
    for bucket, bucket_records in grouped.items():
        if not bucket_records:
            continue
        available_by_bucket[bucket] = len(bucket_records)
        chosen = bucket_records[:records_per_bucket]
        selected.extend(chosen)
        selected_by_bucket[bucket] = len(chosen)

    return selected, {
        "available_by_bucket": available_by_bucket,
        "selected_by_bucket": selected_by_bucket,
        "records_per_bucket": records_per_bucket,
    }


def select_records_by_top_cell_fraction(
    records: Sequence[dict],
    *,
    records_per_bucket: int,
) -> tuple[list[dict], dict]:
    if records_per_bucket <= 0:
        raise ValueError("records_per_bucket must be positive")

    selected: list[dict] = []
    available_by_bucket: dict[str, int] = {}
    selected_by_bucket: dict[str, int] = {}
    for bucket in bucket_order():
        scored = []
        for idx, record in enumerate(records):
            total_bytes = float(record.get("total_bytes", 0.0))
            if total_bytes <= 0:
                continue
            cell_bytes = float(record.get("cell_byte_counts", {}).get(bucket, 0.0))
            if cell_bytes <= 0:
                continue
            scored.append((cell_bytes / total_bytes, idx, record))
        if not scored:
            continue
        scored.sort(key=lambda item: (-item[0], item[1]))
        available_by_bucket[bucket] = len(scored)
        chosen = scored[:records_per_bucket]
        selected_by_bucket[bucket] = len(chosen)
        safe_bucket = bucket.replace("/", "_")
        for rank, (_, _, record) in enumerate(chosen):
            copied = dict(record)
            copied["source_sample_id"] = record["sample_id"]
            copied["sample_id"] = (
                f"{record['sample_id']}::selected::{safe_bucket}::{rank:06d}"
            )
            copied["selection_bucket"] = bucket
            selected.append(copied)

    return selected, {
        "available_by_bucket": available_by_bucket,
        "selected_by_bucket": selected_by_bucket,
        "records_per_bucket": records_per_bucket,
    }


def _dedupe_paths(paths: Iterable[str | Path]) -> list[Path]:
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        candidate = Path(path)
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(candidate)
    return deduped


def _load_arrow_paths(args: argparse.Namespace) -> list[Path]:
    paths = [Path(path) for path in args.arrow_file or []]
    if args.arrow_glob:
        paths.extend(Path(path) for path in sorted(glob.glob(args.arrow_glob)))
    paths = _dedupe_paths(paths)
    if not paths:
        raise ValueError("no Arrow files matched the requested input")
    return paths


def collect_candidate_records(
    arrow_paths: Sequence[Path],
    *,
    max_rows_per_file: int,
    patch_size: float,
    threshold: float,
    threshold_add: float | None,
    monotonicity: bool,
) -> tuple[list[dict], dict]:
    records: list[dict] = []
    metadata = {
        "num_arrow_files_seen": 0,
        "num_rows_seen": 0,
        "num_rows_used": 0,
        "num_rows_skipped": 0,
        "max_rows_per_file": max_rows_per_file,
    }
    for arrow_path in arrow_paths:
        metadata["num_arrow_files_seen"] += 1
        file_rows = 0
        for batch in _iter_arrow_record_batches(arrow_path):
            columns = batch.to_pydict()
            for sample_id, text, entropies in zip(
                columns["sample_id"],
                columns["text"],
                columns["entropies"],
            ):
                if file_rows >= max_rows_per_file:
                    break
                metadata["num_rows_seen"] += 1
                file_rows += 1
                if not entropies:
                    metadata["num_rows_skipped"] += 1
                    continue

                token_entropies = [float(value) for value in entropies]
                patch_lengths = _entropy_patch_lengths(
                    token_entropies,
                    patch_size=patch_size,
                    threshold=threshold,
                    threshold_add=threshold_add,
                    monotonicity=monotonicity,
                    include_next_token=False,
                )
                patch_entropies = _mean_patch_entropies(
                    token_entropies,
                    patch_lengths,
                )
                cell_byte_counts = entropy_length_cell_byte_counts(
                    patch_lengths,
                    patch_entropies,
                )
                try:
                    dominant_bucket = dominant_entropy_length_bucket(
                        patch_lengths,
                        patch_entropies,
                    )
                except ValueError:
                    metadata["num_rows_skipped"] += 1
                    continue

                records.append(
                    {
                        "sample_id": str(sample_id),
                        "text": text,
                        "entropies": token_entropies,
                        "dominant_bucket": dominant_bucket,
                        "cell_byte_counts": cell_byte_counts,
                        "total_bytes": float(sum(cell_byte_counts.values())),
                    }
                )
                metadata["num_rows_used"] += 1
            if file_rows >= max_rows_per_file:
                break
    return records, metadata


def _write_jsonl_placeholder(path: Path, source_name: str, chunk_idx: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "sample_id": f"{source_name}.placeholder.{chunk_idx:05d}",
        "text": "placeholder; training reads the preprocessed Arrow shard",
    }
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")


def _write_arrow(path: Path, records: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = pa.schema(
        [
            pa.field("sample_id", pa.string(), nullable=False),
            pa.field("text", pa.string(), nullable=False),
            pa.field("entropies", pa.list_(pa.float16()), nullable=False),
        ]
    )
    table = pa.Table.from_arrays(
        [
            pa.array([record["sample_id"] for record in records], type=pa.string()),
            pa.array([record["text"] for record in records], type=pa.string()),
            pa.array(
                [
                    [np.float16(value) for value in record["entropies"]]
                    for record in records
                ],
                type=pa.list_(pa.float16()),
            ),
        ],
        schema=schema,
    )
    with pa.OSFile(str(path), "wb") as sink:
        with pa.ipc.new_file(sink, schema) as writer:
            writer.write_table(table)
    path.with_suffix(path.suffix + ".complete").write_text("done\n")


def write_balanced_source(
    records: Sequence[dict],
    *,
    data_root: Path,
    preprocess_root: Path,
    entropy_model_name: str,
    source_name: str,
    num_chunks: int,
) -> dict:
    if num_chunks <= 0:
        raise ValueError("num_chunks must be positive")
    if not records:
        raise ValueError("cannot write an empty balanced source")

    source_dir = data_root / source_name
    preprocessed_dir = preprocess_root / source_name / entropy_model_name
    chunks = [[] for _ in range(num_chunks)]
    for idx, record in enumerate(records):
        chunks[idx % num_chunks].append(record)

    output_chunks = []
    for chunk_idx, chunk_records in enumerate(chunks):
        jsonl_path = source_dir / f"{source_name}.chunk.{chunk_idx:05d}.jsonl"
        arrow_path = (
            preprocessed_dir
            / f"{source_name}.chunk.{chunk_idx:05d}.jsonl.shard_00.arrow"
        )
        _write_jsonl_placeholder(jsonl_path, source_name, chunk_idx)
        _write_arrow(arrow_path, chunk_records)
        output_chunks.append(
            {
                "chunk_idx": chunk_idx,
                "jsonl_path": str(jsonl_path),
                "arrow_path": str(arrow_path),
                "num_rows": len(chunk_records),
            }
        )

    return {
        "source_name": source_name,
        "source_dir": str(source_dir),
        "preprocessed_dir": str(preprocessed_dir),
        "entropy_model_name": entropy_model_name,
        "num_chunks": num_chunks,
        "num_records": len(records),
        "chunks": output_chunks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a row-level entropy x length balanced Arrow training source."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--arrow-glob")
    input_group.add_argument("--arrow-file", action="append")
    parser.add_argument("--source-name", required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--preprocess-root",
        type=Path,
        default=Path("data/entropy_preprocessed_stage1"),
    )
    parser.add_argument("--entropy-model-name", default="transformer_100m")
    parser.add_argument("--max-rows-per-file", type=int, default=2000)
    parser.add_argument("--records-per-bucket", type=int, default=1000)
    parser.add_argument(
        "--selection-mode",
        choices=("top-cell-fraction", "dominant"),
        default="top-cell-fraction",
    )
    parser.add_argument("--num-chunks", type=int, default=6)
    parser.add_argument("--patch-size", type=float, default=4.5)
    parser.add_argument("--threshold", type=float, default=1.335442066192627)
    parser.add_argument("--threshold-add", type=float)
    parser.add_argument("--monotonicity", action="store_true")
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    arrow_paths = _load_arrow_paths(args)
    records, collect_metadata = collect_candidate_records(
        arrow_paths,
        max_rows_per_file=args.max_rows_per_file,
        patch_size=args.patch_size,
        threshold=args.threshold,
        threshold_add=args.threshold_add,
        monotonicity=args.monotonicity,
    )
    if args.selection_mode == "dominant":
        selected, select_metadata = select_records_per_bucket(
            records,
            records_per_bucket=args.records_per_bucket,
        )
    else:
        selected, select_metadata = select_records_by_top_cell_fraction(
            records,
            records_per_bucket=args.records_per_bucket,
        )
    select_metadata["selection_mode"] = args.selection_mode
    source_metadata = write_balanced_source(
        selected,
        data_root=args.data_root,
        preprocess_root=args.preprocess_root,
        entropy_model_name=args.entropy_model_name,
        source_name=args.source_name,
        num_chunks=args.num_chunks,
    )
    manifest = {
        "input": {
            "arrow_files": [str(path) for path in arrow_paths],
            "patch_size": args.patch_size,
            "threshold": args.threshold,
            "threshold_add": args.threshold_add,
            "monotonicity": args.monotonicity,
        },
        "collection": collect_metadata,
        "selection": select_metadata,
        "source": source_metadata,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
