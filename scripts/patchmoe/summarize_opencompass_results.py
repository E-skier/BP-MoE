#!/usr/bin/env python3
"""Summarize OpenCompass CSV outputs for PatchMoE paper tables."""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


METADATA_COLUMNS = ("dataset", "version", "metric", "mode")
DEFAULT_PAPER_DATASETS = (
    ("MMLU", ("mmlu", "lukaemon_mmlu")),
    ("BoolQ", ("boolq", "superglue_boolq", "SuperGLUE_BoolQ")),
    ("OpenBookQA", ("openbookqa",)),
    ("ARC-E", ("ARC_e", "ARC-e", "arc_easy", "ARC-Easy")),
    ("ARC-C", ("ARC_c", "ARC-c", "arc_challenge", "ARC-Challenge")),
    ("HellaSwag", ("hellaswag",)),
    ("PIQA", ("piqa",)),
)


@dataclass(frozen=True)
class ScoreRow:
    method: str
    dataset: str
    version: str
    metric: str
    mode: str
    score: float | None
    source_file: str
    source_mtime: float


def normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def parse_score(value: str | None) -> float | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped or stripped == "-":
        return None
    if stripped.endswith("%"):
        stripped = stripped[:-1].strip()
    try:
        parsed = float(stripped)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def discover_summary_csvs(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() == ".csv" else []
    if not path.exists():
        return []

    candidates: list[Path] = []
    direct = path / "summary"
    if direct.is_dir():
        candidates.extend(direct.glob("*.csv"))
    latest = path / "latest" / "summary"
    if latest.is_dir():
        candidates.extend(latest.glob("*.csv"))
    candidates.extend(path.glob("*/summary/*.csv"))
    candidates.extend(path.rglob("summary/*.csv"))

    unique: dict[str, Path] = {}
    for candidate in candidates:
        try:
            key = str(candidate.resolve())
        except OSError:
            key = str(candidate)
        unique[key] = candidate
    return sorted(unique.values(), key=lambda item: str(item))


def read_opencompass_summary(path: Path) -> list[ScoreRow]:
    rows: list[ScoreRow] = []
    mtime = path.stat().st_mtime
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return rows
        fieldnames = [item.strip() if item else "" for item in reader.fieldnames]
        lower_fields = {item.lower(): item for item in fieldnames}
        missing = [item for item in METADATA_COLUMNS if item not in lower_fields]
        if missing:
            raise ValueError(
                f"{path} does not look like an OpenCompass summary CSV; "
                f"missing columns: {', '.join(missing)}"
            )

        metadata_fields = {lower_fields[item] for item in METADATA_COLUMNS}
        model_fields = [item for item in fieldnames if item not in metadata_fields]
        for raw in reader:
            dataset = raw.get(lower_fields["dataset"], "").strip()
            if not dataset:
                continue
            version = raw.get(lower_fields["version"], "").strip()
            metric = raw.get(lower_fields["metric"], "").strip()
            mode = raw.get(lower_fields["mode"], "").strip()
            for method in model_fields:
                score = parse_score(raw.get(method))
                rows.append(
                    ScoreRow(
                        method=method.strip(),
                        dataset=dataset,
                        version=version,
                        metric=metric,
                        mode=mode,
                        score=score,
                        source_file=str(path),
                        source_mtime=mtime,
                    )
                )
    return rows


def read_all(paths: Iterable[Path]) -> list[ScoreRow]:
    rows: list[ScoreRow] = []
    seen: set[str] = set()
    for root in paths:
        for summary_csv in discover_summary_csvs(root):
            key = str(summary_csv.resolve())
            if key in seen:
                continue
            seen.add(key)
            rows.extend(read_opencompass_summary(summary_csv))
    return rows


def canonical_dataset_specs(spec: str | None) -> list[tuple[str, tuple[str, ...]]]:
    if not spec:
        return list(DEFAULT_PAPER_DATASETS)
    datasets: list[tuple[str, tuple[str, ...]]] = []
    for item in spec.split(";"):
        if not item.strip():
            continue
        if ":" not in item:
            raise ValueError(
                "paper dataset spec must use 'Column:alias1,alias2;...' format"
            )
        label, aliases = item.split(":", 1)
        alias_values = tuple(
            alias.strip() for alias in aliases.split(",") if alias.strip()
        )
        if not label.strip() or not alias_values:
            raise ValueError(f"invalid paper dataset spec item: {item!r}")
        datasets.append((label.strip(), alias_values))
    return datasets


def choose_latest_rows(rows: Iterable[ScoreRow]) -> list[ScoreRow]:
    latest: dict[tuple[str, str, str, str], ScoreRow] = {}
    for row in rows:
        key = (row.method, row.dataset, row.metric, row.mode)
        existing = latest.get(key)
        if existing is None or row.source_mtime >= existing.source_mtime:
            latest[key] = row
    return list(latest.values())


def select_score(
    rows: list[ScoreRow],
    method: str,
    aliases: tuple[str, ...],
    label: str,
) -> float | None:
    alias_keys = {normalize_key(alias) for alias in aliases}
    method_rows = [
        row for row in rows if row.method == method and row.score is not None
    ]

    exact = [
        row
        for row in method_rows
        if normalize_key(row.dataset) in alias_keys and row.metric.lower() == "accuracy"
    ]
    if exact:
        return exact[-1].score

    exact = [row for row in method_rows if normalize_key(row.dataset) in alias_keys]
    if exact:
        return exact[-1].score

    if label == "MMLU":
        subject_rows = [
            row
            for row in method_rows
            if normalize_key(row.dataset).startswith("lukaemonmmlu")
            and row.metric.lower() == "accuracy"
            and row.score is not None
        ]
        if subject_rows:
            return sum(
                row.score for row in subject_rows if row.score is not None
            ) / len(subject_rows)
    return None


def build_wide_rows(
    rows: list[ScoreRow],
    paper_datasets: list[tuple[str, tuple[str, ...]]],
) -> list[dict[str, str]]:
    latest_rows = choose_latest_rows(rows)
    methods = sorted({row.method for row in latest_rows if row.method})
    output: list[dict[str, str]] = []
    for method in methods:
        result: dict[str, str] = {"Method": method}
        scores: list[float] = []
        for label, aliases in paper_datasets:
            score = select_score(latest_rows, method, aliases, label)
            if score is None:
                result[label] = ""
                continue
            result[label] = f"{score:.4g}"
            scores.append(score)
        result["Average"] = f"{sum(scores) / len(scores):.4g}" if scores else ""
        output.append(result)
    return output


def write_long_csv(rows: list[ScoreRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "dataset",
                "version",
                "metric",
                "mode",
                "score",
                "source_file",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "method": row.method,
                    "dataset": row.dataset,
                    "version": row.version,
                    "metric": row.metric,
                    "mode": row.mode,
                    "score": "" if row.score is None else f"{row.score:.12g}",
                    "source_file": row.source_file,
                }
            )


def write_wide_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        fieldnames = list(rows[0].keys())
    else:
        fieldnames = [
            "Method",
            *(label for label, _ in DEFAULT_PAPER_DATASETS),
            "Average",
        ]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert OpenCompass summary CSV files into PatchMoE paper tables."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="OpenCompass work dirs, timestamp dirs, or summary CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/opencompass/summary"),
        help="Directory for opencompass_long.csv and opencompass_paper_table.csv.",
    )
    parser.add_argument(
        "--paper-datasets",
        default=None,
        help="Override paper columns: 'Name:alias1,alias2;Name2:alias3'.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_all(args.paths)
    if not rows:
        print("No OpenCompass summary CSV rows found.", file=sys.stderr)
        return 1

    paper_datasets = canonical_dataset_specs(args.paper_datasets)
    wide_rows = build_wide_rows(rows, paper_datasets)
    long_csv = args.output_dir / "opencompass_long.csv"
    wide_csv = args.output_dir / "opencompass_paper_table.csv"
    write_long_csv(rows, long_csv)
    write_wide_csv(wide_rows, wide_csv)
    print(f"Wrote: {long_csv}")
    print(f"Wrote: {wide_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
