#!/usr/bin/env python3
"""Prepare local files required by OpenCompass datasets without built-in URLs."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from datasets import load_dataset


DEFAULT_CACHE = Path.home() / ".cache" / "opencompass"
PIQA_TRAIN_DEV_URL = (
    "https://storage.googleapis.com/ai2-mosaic/public/physicaliqa/"
    "physicaliqa-train-dev.zip"
)


def convert_openbookqa_example(example: dict) -> dict:
    choices = example["choices"]
    return {
        "id": example.get("id"),
        "question": {
            "stem": example["question_stem"],
            "choices": [
                {"label": label, "text": text}
                for label, text in zip(choices["label"], choices["text"])
            ],
        },
        "answerKey": example["answerKey"],
        **({"fact1": example["fact1"]} if "fact1" in example else {}),
    }


def write_jsonl(records, output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def prepare_openbookqa(cache_dir: Path) -> None:
    root = cache_dir / "data" / "openbookqa"
    outputs = [
        ("main", root / "Main" / "test.jsonl"),
        ("additional", root / "Additional" / "test_complete.jsonl"),
    ]
    for dataset_name, output_path in outputs:
        dataset = load_dataset("allenai/openbookqa", dataset_name, split="test")
        count = write_jsonl(
            (convert_openbookqa_example(example) for example in dataset), output_path
        )
        print(f"Wrote {count} OpenBookQA {dataset_name} examples: {output_path}")


def prepare_boolq(output_root: Path) -> None:
    dataset = load_dataset("super_glue", "boolq", split="validation")
    output_path = output_root / "opencompass" / "boolq"

    def records():
        for example in dataset:
            yield {
                "question": example["question"],
                "passage": example["passage"],
                "label": "true" if int(example["label"]) == 1 else "false",
            }

    count = write_jsonl(records(), output_path)
    print(f"Wrote {count} BoolQ validation examples: {output_path}")


def prepare_piqa(cache_dir: Path) -> None:
    root = cache_dir / "data" / "piqa"
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        archive_path = tmp_dir_path / "physicaliqa-train-dev.zip"
        urllib.request.urlretrieve(PIQA_TRAIN_DEV_URL, archive_path)
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(tmp_dir_path)
        source = tmp_dir_path / "physicaliqa-train-dev"
        files = ["train.jsonl", "train-labels.lst", "dev.jsonl", "dev-labels.lst"]
        for name in files:
            shutil.copyfile(source / name, root / name)
            print(f"Wrote PIQA {name}: {root / name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE,
        help="OpenCompass cache root. Matches COMPASS_DATA_CACHE when set.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["openbookqa", "piqa", "boolq"],
        choices=["openbookqa", "piqa", "boolq"],
        help="Extra local datasets to prepare.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected = set(args.datasets)
    if "openbookqa" in selected:
        prepare_openbookqa(args.cache_dir)
    if "piqa" in selected:
        prepare_piqa(args.cache_dir)
    if "boolq" in selected:
        prepare_boolq(Path.cwd())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
