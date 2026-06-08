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
    parser.add_argument("--expect-num-experts", type=int)
    parser.add_argument("--expect-top-k", type=int)
    parser.add_argument("--expect-layer-frequency", type=int)
    parser.add_argument(
        "--expect-patch-features",
        help="Comma-separated expected router side features, for example: entropy",
    )
    parser.add_argument("--expect-expert-ffn-dim-multiplier", type=float)
    parser.add_argument(
        "--expect-expert-init-mode",
        choices=("replicated_prefix", "paired_partition"),
    )
    parser.add_argument(
        "--expect-global-layer-count",
        type=int,
        help="Expected number of BLT global Transformer layers in the source checkpoint.",
    )
    parser.add_argument(
        "--require-all-global-layers",
        action="store_true",
        help="Require every discovered global FFN layer to be replaced by PatchMoE.",
    )
    return parser.parse_args()


def json_normalize(value):
    return json.loads(json.dumps(value))


def _assert_equal(name, actual, expected):
    if expected is not None and actual != expected:
        raise ValueError(f"{name} mismatch: expected {expected!r}, got {actual!r}")


def _assert_manifest_matches_expectations(args, manifest):
    spec = manifest["spec"]
    report = manifest["report"]

    _assert_equal("num_experts", spec["num_experts"], args.expect_num_experts)
    _assert_equal("top_k", spec["top_k"], args.expect_top_k)
    _assert_equal(
        "layer_frequency", spec["layer_frequency"], args.expect_layer_frequency
    )
    _assert_equal(
        "expert_init_mode",
        spec["expert_init_mode"],
        args.expect_expert_init_mode,
    )
    if args.expect_patch_features is not None:
        expected_features = [
            feature
            for feature in args.expect_patch_features.split(",")
            if feature
        ]
        _assert_equal("patch_features", spec["patch_features"], expected_features)
    if args.expect_expert_ffn_dim_multiplier is not None:
        actual = spec["expert_ffn_dim_multiplier"]
        if actual is None or abs(actual - args.expect_expert_ffn_dim_multiplier) > 1e-9:
            raise ValueError(
                "expert_ffn_dim_multiplier mismatch: "
                f"expected {args.expect_expert_ffn_dim_multiplier!r}, got {actual!r}"
            )
    if args.expect_global_layer_count is not None:
        _assert_equal(
            "global layer count",
            len(report["global_layers"]),
            args.expect_global_layer_count,
        )
    if args.require_all_global_layers:
        if report["selected_layers"] != report["global_layers"]:
            raise ValueError(
                "Warm-start is not all-layer PatchMoE: "
                f"selected={report['selected_layers']}, "
                f"global={report['global_layers']}"
            )


def main():
    args = parse_args()
    source = Path(args.source)
    checkpoint_dir = Path(args.checkpoint_dir)
    manifest = load_patchmoe_warmstart_manifest(checkpoint_dir)
    if Path(manifest["source_checkpoint"]).resolve() != source.resolve():
        raise ValueError(
            "Warm-start manifest source checkpoint does not match --source"
        )
    _assert_manifest_matches_expectations(args, manifest)

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
