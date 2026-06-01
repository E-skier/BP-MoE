#!/usr/bin/env -S uv run python

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from bytelatent.patchmoe_warmstart import (
    PatchMoEWarmStartSpec,
    convert_dense_state_dict_to_patchmoe,
    load_consolidated_model_state_dict,
    load_patchmoe_warmstart_manifest,
    verify_patchmoe_warmstart_dcp,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify a BLT-1B PatchMoE warm-start DCP against the released dense checkpoint."
    )
    parser.add_argument(
        "--source",
        default="hf-weights/blt_1b/consolidated.pth",
        help="Released BLT-1B consolidated checkpoint.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="hf-weights/blt_1b_patchmoe_entropy_all_layers_dcp",
        help="PatchMoE DCP warm-start directory.",
    )
    return parser.parse_args()


def json_normalize(value):
    return json.loads(json.dumps(value))


def main():
    args = parse_args()
    source = Path(args.source)
    checkpoint_dir = Path(args.checkpoint_dir)
    manifest = load_patchmoe_warmstart_manifest(checkpoint_dir)
    if Path(manifest["source_checkpoint"]).resolve() != source.resolve():
        raise ValueError(
            "Warm-start manifest source checkpoint does not match --source"
        )

    manifest_spec = manifest["spec"]
    spec = PatchMoEWarmStartSpec(
        **{
            **manifest_spec,
            "patch_features": tuple(manifest_spec["patch_features"]),
        }
    )
    dense_state_dict = load_consolidated_model_state_dict(source)
    converted_state_dict, report = convert_dense_state_dict_to_patchmoe(
        dense_state_dict, spec
    )
    if json_normalize(asdict(spec)) != manifest_spec:
        raise ValueError("Warm-start manifest spec does not match reconstructed spec")
    if json_normalize(asdict(report)) != manifest["report"]:
        raise ValueError(
            "Warm-start manifest report does not match reconstructed report"
        )

    verification = verify_patchmoe_warmstart_dcp(checkpoint_dir, converted_state_dict)
    print(
        json.dumps(
            {
                "checkpoint_dir": str(checkpoint_dir.resolve()),
                "verification": asdict(verification),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
