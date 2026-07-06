#!/usr/bin/env python3
import argparse
import glob
import json
from pathlib import Path
import sys
from typing import Iterable, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.calibrate_patchmoe_router import (  # noqa: E402
    collect_entropy_patch_statistics_from_arrow,
)

LENGTH_BUCKETS = ("short", "medium", "long")
ENTROPY_BUCKETS = ("low", "medium", "high")


def classify_length_bucket(length: float) -> str:
    if length <= 4:
        return "short"
    if length <= 8:
        return "medium"
    return "long"


def classify_entropy_bucket(entropy: float) -> str:
    if entropy <= 1.0:
        return "low"
    if entropy <= 2.0:
        return "medium"
    return "high"


def _new_stats(bucket: str) -> dict:
    return {
        "bucket": bucket,
        "patch_count": 0,
        "patch_fraction": 0.0,
        "byte_count": 0.0,
        "byte_fraction": 0.0,
        "length_sum": 0.0,
        "entropy_sum": 0.0,
        "byte_entropy_sum": 0.0,
        "mean_patch_length": 0.0,
        "mean_patch_entropy": 0.0,
        "byte_weighted_patch_entropy": 0.0,
    }


def _finalize_stats(stats: dict, total_patches: int, total_bytes: float) -> dict:
    patch_count = int(stats["patch_count"])
    byte_count = float(stats["byte_count"])
    if total_patches > 0:
        stats["patch_fraction"] = patch_count / total_patches
    if total_bytes > 0:
        stats["byte_fraction"] = byte_count / total_bytes
    if patch_count > 0:
        stats["mean_patch_length"] = stats["length_sum"] / patch_count
        stats["mean_patch_entropy"] = stats["entropy_sum"] / patch_count
    if byte_count > 0:
        stats["byte_weighted_patch_entropy"] = (
            stats["byte_entropy_sum"] / byte_count
        )
    stats["byte_count"] = int(byte_count) if byte_count.is_integer() else byte_count
    return stats


def summarize_entropy_length_distribution(
    patch_lengths: Sequence[float],
    patch_entropies: Sequence[float],
) -> dict:
    if len(patch_lengths) != len(patch_entropies):
        raise ValueError("patch_lengths and patch_entropies must have the same length")

    cells = {
        f"{entropy_bucket}/{length_bucket}": _new_stats(
            f"{entropy_bucket}/{length_bucket}"
        )
        for entropy_bucket in ENTROPY_BUCKETS
        for length_bucket in LENGTH_BUCKETS
    }
    length_marginals = {bucket: _new_stats(bucket) for bucket in LENGTH_BUCKETS}
    entropy_marginals = {bucket: _new_stats(bucket) for bucket in ENTROPY_BUCKETS}

    valid_rows = [
        (float(length), float(entropy))
        for length, entropy in zip(patch_lengths, patch_entropies)
        if float(length) > 0
    ]
    total_patches = len(valid_rows)
    total_bytes = sum(length for length, _ in valid_rows)
    entropy_sum = sum(entropy for _, entropy in valid_rows)
    byte_entropy_sum = sum(length * entropy for length, entropy in valid_rows)

    for length, entropy in valid_rows:
        length_bucket = classify_length_bucket(length)
        entropy_bucket = classify_entropy_bucket(entropy)
        for stats in (
            cells[f"{entropy_bucket}/{length_bucket}"],
            length_marginals[length_bucket],
            entropy_marginals[entropy_bucket],
        ):
            stats["patch_count"] += 1
            stats["byte_count"] += length
            stats["length_sum"] += length
            stats["entropy_sum"] += entropy
            stats["byte_entropy_sum"] += length * entropy

    ordered_cells = [
        _finalize_stats(cells[f"{entropy_bucket}/{length_bucket}"], total_patches, total_bytes)
        for entropy_bucket in ENTROPY_BUCKETS
        for length_bucket in LENGTH_BUCKETS
    ]
    ordered_length = [
        _finalize_stats(length_marginals[bucket], total_patches, total_bytes)
        for bucket in LENGTH_BUCKETS
    ]
    ordered_entropy = [
        _finalize_stats(entropy_marginals[bucket], total_patches, total_bytes)
        for bucket in ENTROPY_BUCKETS
    ]

    return {
        "total_patches": total_patches,
        "total_bytes": int(total_bytes) if float(total_bytes).is_integer() else total_bytes,
        "mean_patch_length": total_bytes / total_patches if total_patches else 0.0,
        "mean_patch_entropy": entropy_sum / total_patches if total_patches else 0.0,
        "byte_weighted_patch_entropy": (
            byte_entropy_sum / total_bytes if total_bytes else 0.0
        ),
        "cells": ordered_cells,
        "length_marginals": ordered_length,
        "entropy_marginals": ordered_entropy,
        "dominant_cells_by_byte_fraction": sorted(
            [cell for cell in ordered_cells if cell["patch_count"] > 0],
            key=lambda cell: cell["byte_fraction"],
            reverse=True,
        ),
    }


def _fmt_fraction(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def _markdown_table(rows: Iterable[dict], *, title: str) -> list[str]:
    out = [
        f"### {title}",
        "",
        "| bucket | patches | patch % | bytes | byte % | mean len | mean entropy | byte-w entropy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        out.append(
            "| {bucket} | {patch_count} | {patch_fraction} | {byte_count} | "
            "{byte_fraction} | {mean_patch_length:.3f} | {mean_patch_entropy:.3f} | "
            "{byte_weighted_patch_entropy:.3f} |".format(
                bucket=row["bucket"],
                patch_count=row["patch_count"],
                patch_fraction=_fmt_fraction(row["patch_fraction"]),
                byte_count=row["byte_count"],
                byte_fraction=_fmt_fraction(row["byte_fraction"]),
                mean_patch_length=row["mean_patch_length"],
                mean_patch_entropy=row["mean_patch_entropy"],
                byte_weighted_patch_entropy=row["byte_weighted_patch_entropy"],
            )
        )
    out.append("")
    return out


def format_markdown_report(report: dict) -> str:
    lines = [
        "# Entropy-Length Data Distribution Diagnostic",
        "",
        "## Summary",
        "",
        f"- total patches: {report['total_patches']}",
        f"- total bytes: {report['total_bytes']}",
        f"- mean patch length: {report['mean_patch_length']:.4f}",
        f"- mean patch entropy: {report['mean_patch_entropy']:.4f}",
        f"- byte-weighted patch entropy: {report['byte_weighted_patch_entropy']:.4f}",
        "",
    ]
    metadata = report.get("collection_metadata")
    if metadata:
        lines.extend(
            [
                "## Collection Metadata",
                "",
                f"- arrow files seen: {metadata.get('num_arrow_files_seen')}",
                f"- rows seen: {metadata.get('num_rows_seen')}",
                f"- rows used: {metadata.get('num_rows_used')}",
                "",
            ]
        )
    lines.extend(_markdown_table(report["cells"], title="Entropy x Length Cells"))
    lines.extend(_markdown_table(report["length_marginals"], title="Length Marginals"))
    lines.extend(_markdown_table(report["entropy_marginals"], title="Entropy Marginals"))
    lines.extend(
        _markdown_table(
            report["dominant_cells_by_byte_fraction"][:5],
            title="Top Cells by Byte Fraction",
        )
    )
    return "\n".join(lines).rstrip() + "\n"


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


def _collect_patch_statistics(args: argparse.Namespace, arrow_paths: list[Path]):
    if args.max_rows_per_file is None:
        return collect_entropy_patch_statistics_from_arrow(
            arrow_paths,
            patch_size=args.patch_size,
            threshold=args.threshold,
            threshold_add=args.threshold_add,
            monotonicity=args.monotonicity,
            include_next_token=False,
            max_rows=args.max_rows,
            max_patches=args.max_patches,
            max_bytes=args.max_bytes,
        )

    if args.max_rows is not None:
        raise ValueError("--max-rows cannot be combined with --max-rows-per-file")
    if args.max_patches is not None:
        raise ValueError("--max-patches cannot be combined with --max-rows-per-file")
    if args.max_bytes is not None:
        raise ValueError("--max-bytes cannot be combined with --max-rows-per-file")

    all_lengths = []
    all_entropies = []
    metadata = {
        "num_arrow_files_seen": 0,
        "num_rows_seen": 0,
        "num_rows_used": 0,
        "num_valid_patches": 0,
        "total_bytes": 0,
        "max_rows_per_file": args.max_rows_per_file,
    }
    for arrow_path in arrow_paths:
        lengths, entropies, file_metadata = collect_entropy_patch_statistics_from_arrow(
            [arrow_path],
            patch_size=args.patch_size,
            threshold=args.threshold,
            threshold_add=args.threshold_add,
            monotonicity=args.monotonicity,
            include_next_token=False,
            max_rows=args.max_rows_per_file,
        )
        all_lengths.append(lengths)
        all_entropies.append(entropies)
        for key in (
            "num_arrow_files_seen",
            "num_rows_seen",
            "num_rows_used",
            "num_valid_patches",
            "total_bytes",
        ):
            metadata[key] += int(file_metadata.get(key, 0))

    return torch.cat(all_lengths), torch.cat(all_entropies), metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze byte mass over entropy x patch-length buckets."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--arrow-glob")
    input_group.add_argument("--arrow-file", action="append")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--patch-size", type=float, default=4.5)
    parser.add_argument("--threshold", type=float, default=1.335442066192627)
    parser.add_argument("--threshold-add", type=float)
    parser.add_argument("--monotonicity", action="store_true")
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--max-rows-per-file", type=int)
    parser.add_argument("--max-patches", type=int)
    parser.add_argument("--max-bytes", type=int)
    args = parser.parse_args()

    arrow_paths = _load_arrow_paths(args)
    patch_lengths, patch_entropies, metadata = _collect_patch_statistics(
        args, arrow_paths
    )
    report = summarize_entropy_length_distribution(
        patch_lengths.tolist(),
        patch_entropies.tolist(),
    )
    report["collection_metadata"] = metadata
    report["input"] = {
        "arrow_files": [str(path) for path in arrow_paths],
        "patch_size": args.patch_size,
        "threshold": args.threshold,
        "threshold_add": args.threshold_add,
        "monotonicity": args.monotonicity,
        "max_rows": args.max_rows,
        "max_rows_per_file": args.max_rows_per_file,
        "max_patches": args.max_patches,
        "max_bytes": args.max_bytes,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    markdown = format_markdown_report(report)
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(markdown)
    print(markdown)


if __name__ == "__main__":
    main()
