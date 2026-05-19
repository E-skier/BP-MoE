#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export TINY_ROOT="$PWD/data_tiny"
export BLT_ALLOW_MISSING_FLEX_ATTENTION=1
export BLT_SUPPRESS_ATTN_ERROR=1

mkdir -p "$TINY_ROOT/dclm_baseline_1.0"

python - <<'PY'
import json
import os

path = os.path.join(
    os.environ["TINY_ROOT"], "dclm_baseline_1.0", "tiny.chunk.00.jsonl"
)
with open(path, "w") as f:
    for i in range(2000):
        text = "PatchMoE byte level smoke test. " * 20
        f.write(json.dumps({"id": str(i), "text": text}) + "\n")
PY

torchrun --standalone --nproc-per-node=1 -m bytelatent.train \
  config=apps/main/configs/patchmoe_stage0.yaml \
  dump_dir="$PWD/runs/patchmoe_tiny" \
  steps=5 max_steps=5 \
  data.root_dir="$TINY_ROOT" \
  data.file_format=json \
  data.preprocess_dir=null \
  data.load_async=false \
  data.batch_size=1 \
  data.seq_len=64 \
  data.max_encoder_seq_length=256 \
  data.tokenizer_args.init_kwargs.bpe_tokenizer_path=/tmp/unused.tokenizer.model \
  model.attn_impl=sdpa \
  model.cross_attn_encoder=false \
  model.cross_attn_decoder=false \
  model.cross_attn_use_flex_attention=false \
  model.dim=64 \
  model.dim_token=64 \
  model.dim_global=64 \
  model.dim_local_encoder=64 \
  model.dim_local_decoder=64 \
  model.max_encoder_seq_length=256 \
  model.max_seqlen=64 \
  model.max_length=64 \
  model.n_heads=4 \
  model.n_heads_global=4 \
  model.n_heads_local_encoder=4 \
  model.n_heads_local_decoder=4 \
  model.n_layers_global=2 \
  model.n_layers_local_encoder=1 \
  model.n_layers_local_decoder=1 \
  model.cross_attn_k=1 \
  model.cross_attn_nheads=1 \
  model.cross_attn_window_encoder=64 \
  model.cross_attn_window_decoder=64 \
  model.local_attention_window_len=64 \
  model.moe_num_experts=2 \
  model.moe_top_k=1 \
  distributed.fsdp_type=no_shard \
  distributed.dp_shard=1 \
  distributed.dp_replicate=1 \
  distributed.matmul_allow_tf32=true \
  checkpoint.dump.every=-1 \
  checkpoint.eval.every=-1 \
  logging.freq=1
