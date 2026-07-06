#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/data1/pengfeigao/BP-MoE"
RUN_DIR="$ROOT_DIR/experiments/bcfp_cpt/D_topcell_balanced_10k_6gpu_nofused_clip0_nockpt"
CKPT_DIR="$RUN_DIR/checkpoints/0000010000"
EVAL_DIR="$RUN_DIR/heldout_eval_expanded_100k_8000batches_gpu2"
HELDOUT_ARROW_DIR="$ROOT_DIR/data/entropy_preprocessed_stage1_heldout_expanded/fineweb_edu_10bt_heldout_expanded_100000/transformer_100m"

cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export BLT_ALLOW_MISSING_FLEX_ATTENTION=1
export BLT_SUPPRESS_ATTN_ERROR=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_SERVICE_FORCE_INTEL=GNU
export NCCL_DEBUG=WARN
export NCCL_IB_TIMEOUT=22
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1

exec env CUDA_VISIBLE_DEVICES=2 \
  "$ROOT_DIR/.venv/bin/python" -m bytelatent.eval \
  "ckpt_dir=$CKPT_DIR" \
  "dump_dir=$EVAL_DIR" \
  "metric_log_dir=$EVAL_DIR" \
  "global_step=10000" \
  "consolidate_if_needed=true" \
  "run_ppl=true" \
  "run_tasks=false" \
  "validation.use_val_from_train_src=false" \
  "validation.root_dir=$HELDOUT_ARROW_DIR" \
  "validation.sources=[fineweb_edu_10bt_heldout_expanded_100000.chunk.00002.first_100000.jsonl.shard_00.arrow,fineweb_edu_10bt_heldout_expanded_100000.chunk.00003.first_100000.jsonl.shard_00.arrow]" \
  "validation.batch_size=1" \
  "validation.max_n_batches=8000"
