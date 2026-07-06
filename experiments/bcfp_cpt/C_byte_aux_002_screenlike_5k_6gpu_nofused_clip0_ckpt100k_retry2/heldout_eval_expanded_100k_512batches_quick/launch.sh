#!/usr/bin/env bash
set -euo pipefail

ROOT="/data1/pengfeigao/BP-MoE"
RUN_DIR="$ROOT/experiments/bcfp_cpt/C_byte_aux_002_screenlike_5k_6gpu_nofused_clip0_ckpt100k_retry2"
DUMP_DIR="$RUN_DIR/heldout_eval_expanded_100k_512batches_quick"
CKPT_DIR="$RUN_DIR/checkpoints/0000005000"
HELDOUT_DIR="$ROOT/data/entropy_preprocessed_stage1_heldout_expanded/fineweb_edu_10bt_heldout_expanded_100000/transformer_100m"

cd "$ROOT"

CUDA_VISIBLE_DEVICES=5 \
LOCAL_RANK=0 \
RANK=0 \
WORLD_SIZE=1 \
MASTER_ADDR=127.0.0.1 \
MASTER_PORT=29919 \
uv run python -m bytelatent.eval \
  "ckpt_dir=$CKPT_DIR" \
  "dump_dir=$DUMP_DIR" \
  "metric_log_dir=$DUMP_DIR" \
  "global_step=5000" \
  "consolidate_if_needed=true" \
  "run_ppl=true" \
  "run_tasks=false" \
  "validation.use_val_from_train_src=false" \
  "validation.root_dir=$HELDOUT_DIR" \
  "validation.sources=[fineweb_edu_10bt_heldout_expanded_100000.chunk.00002.first_100000.jsonl.shard_00.arrow,fineweb_edu_10bt_heldout_expanded_100000.chunk.00003.first_100000.jsonl.shard_00.arrow]" \
  "validation.batch_size=1" \
  "validation.max_n_batches=512"
