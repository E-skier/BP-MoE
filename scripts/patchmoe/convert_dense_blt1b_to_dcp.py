#!/usr/bin/env python
"""Convert a consolidated BLT checkpoint into DCP format for training init."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="consolidated.pth path")
    parser.add_argument("--output-dir", required=True, help="output DCP directory")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    source = Path(args.source)
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        if not args.force:
            raise FileExistsError(output_dir)
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    checkpoint = torch.load(
        source,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError("Expected consolidated checkpoint with a 'model' entry")

    dcp.save({"model": checkpoint["model"]}, checkpoint_id=output_dir)
    manifest = {
        "format": "dense-blt-dcp-v1",
        "source_checkpoint": str(source.resolve()),
    }
    with (output_dir / "dense_blt_dcp_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
