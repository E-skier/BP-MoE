#!/usr/bin/env python
"""Safe training entrypoint for BP-MoE/PatchMoE smoke runs.

This wrapper keeps the original `bytelatent.train` training loop, but replaces
the FSDP grouping-plan builder with a version that only lists modules that the
current BLT configuration actually creates.

Why this exists:
  The upstream `build_fsdp_grouping_plan()` assumes optional modules such as
  local cross-attention layers and hash embeddings always exist. Tiny/smoke
  configs often disable or shrink those modules, so the original plan may ask
  FSDP to wrap a missing/None module and fail before the first training step.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _ensure_repo_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def _as_list_or_empty(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [x for x in value.split(",") if x]
    return list(value)


def safe_build_fsdp_grouping_plan(model_args):
    """Build a BLT-aware FSDP wrapping plan.

    The plan must match modules that exist for the selected config. BLT has many
    optional pieces, so we gate each optional path by its config flag.
    """
    from bytelatent.transformer import LMTransformerArgs

    group_plan = []

    if isinstance(model_args, LMTransformerArgs):
        group_plan.append(("tok_embeddings", False))
        for i in range(model_args.n_layers):
            group_plan.append((f"layers.{i}", False))
        group_plan.append(("output", True))
        return group_plan

    for i in range(model_args.n_layers_local_encoder):
        group_plan.append((f"local_encoder.layers.{i}", False))

    if getattr(model_args, "cross_attn_encoder", False):
        n_cross = (
            model_args.n_layers_local_encoder
            if getattr(model_args, "cross_attn_all_layers_encoder", False)
            else 1
        )
        for i in range(n_cross):
            group_plan.append((f"local_encoder.cross_attn_layers.{i}", False))

    for i in range(model_args.n_layers_local_decoder):
        group_plan.append((f"local_decoder.layers.{i}", False))

    if getattr(model_args, "cross_attn_decoder", False):
        n_cross = (
            model_args.n_layers_local_decoder
            if getattr(model_args, "cross_attn_all_layers_decoder", False)
            else 1
        )
        for i in range(n_cross):
            group_plan.append((f"local_decoder.cross_attn_layers.{i}", False))

    for i in range(model_args.n_layers_global):
        group_plan.append((f"global_transformer.layers.{i}", False))

    hash_sizes = _as_list_or_empty(getattr(model_args, "encoder_hash_byte_group_size", None))
    n_hash_fns = int(getattr(model_args, "encoder_hash_byte_group_nb_functions", 0) or 0)
    for i in range(len(hash_sizes) * n_hash_fns):
        group_plan.append((f"encoder_hash_tok_embedding.{i}", False))

    return group_plan


def main() -> None:
    _ensure_repo_on_path()

    import bytelatent.train as train_mod

    train_mod.build_fsdp_grouping_plan = safe_build_fsdp_grouping_plan

    print("[safe_train] using BLT-aware safe FSDP grouping plan", flush=True)
    print(f"[safe_train] cwd={os.getcwd()}", flush=True)
    print(f"[safe_train] argv={' '.join(sys.argv[1:])}", flush=True)

    train_mod.main()


if __name__ == "__main__":
    main()
