#!/usr/bin/env bash
set -euo pipefail

# One-GPU PatchMoE tiny smoke launcher.
#
# Goal:
#   Verify that the BP-MoE training loop, CUDA environment, model construction,
#   forward/backward, optimizer step, metrics logging, and MoE metrics work on
#   a single GPU before scaling to multi-GPU distributed training.
#
# Usage:
#   cd /data1/pengfeigao/BP-MoE
#   GPU_ID=3 bash scripts/patchmoe/run_tiny_patchmoe_safe.sh
#
# Useful overrides:
#   STEPS=20 GPU_ID=3 RUN_DIR=$PWD/runs/my_tiny bash scripts/patchmoe/run_tiny_patchmoe_safe.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

GPU_ID="${GPU_ID:-0}"
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
fi

export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export TINY_ROOT="${TINY_ROOT:-$ROOT_DIR/data_tiny}"
export RUN_DIR="${RUN_DIR:-$ROOT_DIR/runs/patchmoe_tiny_safe}"
export BLT_ALLOW_MISSING_FLEX_ATTENTION="${BLT_ALLOW_MISSING_FLEX_ATTENTION:-1}"
export BLT_SUPPRESS_ATTN_ERROR="${BLT_SUPPRESS_ATTN_ERROR:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

PYTHON_BIN="${PYTHON_BIN:-python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
STEPS="${STEPS:-5}"
SEQ_LEN="${SEQ_LEN:-64}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_ENCODER_SEQ_LENGTH="${MAX_ENCODER_SEQ_LENGTH:-256}"

echo "==> BP-MoE tiny smoke"
echo "    ROOT_DIR=$ROOT_DIR"
echo "    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "    RUN_DIR=$RUN_DIR"
echo "    STEPS=$STEPS"

echo "==> Checking Python/PyTorch/CUDA"
"$PYTHON_BIN" - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("visible gpu count:", torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"visible cuda:{i}: {p.name}, {p.total_memory / 1024**3:.1f} GiB")
PY

echo "==> Creating tiny JSONL dataset"
mkdir -p "$TINY_ROOT/dclm_baseline_1.0"
"$PYTHON_BIN" - <<'PY'
import json
import os

path = os.path.join(
    os.environ["TINY_ROOT"], "dclm_baseline_1.0", "tiny.chunk.00.jsonl"
)
with open(path, "w", encoding="utf-8") as f:
    for i in range(2000):
        text = "PatchMoE byte level smoke test. " * 20
        f.write(json.dumps({"id": str(i), "text": text}, ensure_ascii=False) + "\n")
print(path)
PY

mkdir -p "$RUN_DIR"

echo "==> Launching one-process torchrun training"
"$TORCHRUN_BIN" --standalone --nproc-per-node=1 scripts/patchmoe/safe_train.py \
  config=apps/main/configs/patchmoe_stage0.yaml \
  dump_dir="$RUN_DIR" \
  name=patchmoe_tiny_safe \
  steps="$STEPS" max_steps="$STEPS" \
  data.root_dir="$TINY_ROOT" \
  data.sources='{dclm_baseline_1.0: 1.0}' \
  data.file_format=json \
  data.preprocess_dir=null \
  data.load_async=false \
  data.batch_size="$BATCH_SIZE" \
  data.seq_len="$SEQ_LEN" \
  data.max_encoder_seq_length="$MAX_ENCODER_SEQ_LENGTH" \
  data.tokenizer_args.name=blt \
  data.tokenizer_args.init_kwargs.bpe_tokenizer_path=/tmp/unused.tokenizer.model \
  model.attn_impl=sdpa \
  model.cross_attn_encoder=true \
  model.cross_attn_decoder=true \
  model.cross_attn_use_flex_attention=false \
  model.dim=64 \
  model.dim_token=64 \
  model.dim_global=64 \
  model.dim_local_encoder=64 \
  model.dim_local_decoder=64 \
  model.max_encoder_seq_length="$MAX_ENCODER_SEQ_LENGTH" \
  model.max_seqlen="$SEQ_LEN" \
  model.max_length="$SEQ_LEN" \
  model.n_heads=4 \
  model.n_heads_global=4 \
  model.n_heads_local_encoder=4 \
  model.n_heads_local_decoder=4 \
  model.n_layers_global=2 \
  model.n_layers_local_encoder=1 \
  model.n_layers_local_decoder=1 \
  model.cross_attn_k=1 \
  model.cross_attn_nheads=1 \
  model.cross_attn_window_encoder="$SEQ_LEN" \
  model.cross_attn_window_decoder="$SEQ_LEN" \
  model.local_attention_window_len="$SEQ_LEN" \
  model.moe_num_experts=2 \
  model.moe_top_k=1 \
  distributed.fsdp_type=no_shard \
  distributed.dp_shard=1 \
  distributed.dp_replicate=1 \
  distributed.tp_size=1 \
  distributed.matmul_allow_tf32=true \
  checkpoint.dump.every=-1 \
  checkpoint.eval.every=-1 \
  eval_on_gpus=1 \
  logging.freq=1 \
  env.NCCL_DEBUG="$NCCL_DEBUG"

echo "==> Training finished"
echo "==> Last metrics:"
tail -n 5 "$RUN_DIR/metrics.jsonl" || true