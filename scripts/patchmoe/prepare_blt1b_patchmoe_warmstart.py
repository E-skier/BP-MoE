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
    write_patchmoe_warmstart_dcp,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert the released dense BLT-1B checkpoint into a PatchMoE DCP warm-start."
    )
    parser.add_argument(
        "--source",
        default="hf-weights/blt_1b/consolidated.pth",
        help="Released BLT-1B consolidated checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        default="hf-weights/blt_1b_patchmoe_entropy_all_layers_dcp",
        help="Output DCP directory consumed by checkpoint.init_ckpt_path.",
    )
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--layer-frequency", type=int, default=1)
    parser.add_argument(
        "--patch-features",
        default="entropy",
        help="Comma-separated router side features: length,entropy,byte_type.",
    )
    parser.add_argument("--router-seed", type=int, default=42)
    parser.add_argument("--init-std-factor", default="current_depth")
    parser.add_argument("--expert-ffn-dim-multiplier", type=float, default=None)
    parser.add_argument(
        "--patch-feature-bias",
        action="store_true",
        help="Add an affine bias to the patch-feature router.",
    )
    parser.add_argument(
        "--expert-init-mode",
        default="replicated_prefix",
        choices=("replicated_prefix", "paired_partition"),
    )
    parser.add_argument(
        "--patch-feature-init",
        default="random",
        choices=("random", "entropy_bands"),
        help="Initialize patch-feature router randomly or with entropy-ordered expert-pair bands.",
    )
    parser.add_argument(
        "--entropy-band-logit-scale",
        type=float,
        default=1.0,
        help="Logit scale for --patch-feature-init entropy_bands.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect the mapping and parameter counts without writing the DCP checkpoint.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace output-dir if it already exists.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    patch_features = tuple(
        feature for feature in args.patch_features.split(",") if feature
    )
    spec = PatchMoEWarmStartSpec(
        num_experts=args.num_experts,
        top_k=args.top_k,
        layer_frequency=args.layer_frequency,
        patch_features=patch_features,
        router_seed=args.router_seed,
        init_std_factor=args.init_std_factor,
        expert_ffn_dim_multiplier=args.expert_ffn_dim_multiplier,
        expert_init_mode=args.expert_init_mode,
        patch_feature_bias=args.patch_feature_bias,
        patch_feature_init=args.patch_feature_init,
        entropy_band_logit_scale=args.entropy_band_logit_scale,
    )
    source = Path(args.source)
    if not source.is_file():
        raise FileNotFoundError(f"Missing source BLT checkpoint: {source}")

    print(f"Loading released BLT checkpoint with mmap: {source}")
    dense_state_dict = load_consolidated_model_state_dict(source)
    converted_state_dict, report = convert_dense_state_dict_to_patchmoe(
        dense_state_dict, spec
    )
    print(json.dumps({"spec": asdict(spec), "report": asdict(report)}, indent=2))
    print(
        "Target bf16 parameter storage: "
        f"{report.target_parameter_count * 2 / 1024**3:.2f} GiB"
    )

    if args.dry_run:
        print("Dry-run complete; no checkpoint written.")
        return

    output_dir = write_patchmoe_warmstart_dcp(
        args.output_dir,
        converted_state_dict,
        source_checkpoint=source,
        spec=spec,
        report=report,
        force=args.force,
    )
    print(f"Wrote PatchMoE warm-start DCP: {output_dir}")
    print(f"Training override: checkpoint.init_ckpt_path={output_dir}")


if __name__ == "__main__":
    main()
