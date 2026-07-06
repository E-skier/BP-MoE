#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/data1/pengfeigao/BP-MoE"
RUN_DIR="$ROOT_DIR/experiments/bcfp_cpt/C_byte_aux_001_screenlike_5k_6gpu_nofused_clip0_ckpt100k"
CKPT_DIR="$RUN_DIR/checkpoints/0000005000"
EVAL_DIR="$RUN_DIR/heldout_eval_expanded_100k_8000batches_0to5_retry2"
HELDOUT_ARROW_DIR="$ROOT_DIR/data/entropy_preprocessed_stage1_heldout_expanded/fineweb_edu_10bt_heldout_expanded_100000/transformer_100m"

cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export BLT_ALLOW_MISSING_FLEX_ATTENTION=1
export BLT_SUPPRESS_ATTN_ERROR=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_SERVICE_FORCE_INTEL=GNU
export UV_CACHE_DIR="$ROOT_DIR/.uv-cache"
export UV_LINK_MODE=copy
export BLT_EVAL_SPAWN_METHOD=fork

exec env CUDA_VISIBLE_DEVICES=0 \
  /home/pengfeigao/.local/bin/uv run python -m bytelatent.eval \
  "ckpt_dir=$CKPT_DIR" \
  "dump_dir=$EVAL_DIR" \
  "metric_log_dir=$EVAL_DIR" \
  "global_step=5000" \
  "consolidate_if_needed=true" \
  "run_ppl=true" \
  "run_tasks=false" \
  "validation.use_val_from_train_src=false" \
  "validation.root_dir=$HELDOUT_ARROW_DIR" \
  "validation.sources=[fineweb_edu_10bt_heldout_expanded_100000.chunk.00002.first_100000.jsonl.shard_00.arrow,fineweb_edu_10bt_heldout_expanded_100000.chunk.00003.first_100000.jsonl.shard_00.arrow]" \
  "validation.batch_size=1" \
  "validation.max_n_batches=8000"
