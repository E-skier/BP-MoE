#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/data1/pengfeigao/BP-MoE"
RUN_DIR="$ROOT_DIR/experiments/bcfp_cpt/D_mix90_bal10_screenlike_5k_4gpu_2to5_nofused_clip0_ckptfinal"
EVAL_DIR="$RUN_DIR/heldout_eval_expanded_100k_8000batches_gpu2"
CKPT_DIR="$RUN_DIR/checkpoints/0000005000"

mkdir -p "$RUN_DIR" "$EVAL_DIR"

{
  date
  echo "pipeline: start train"
} | tee "$RUN_DIR/pipeline.log"

"$RUN_DIR/launch_train.sh" > "$RUN_DIR/train.log" 2>&1

{
  date
  echo "pipeline: train done; start heldout eval"
} | tee -a "$RUN_DIR/pipeline.log"

"$EVAL_DIR/launch.sh" > "$EVAL_DIR/eval.log" 2>&1

{
  date
  echo "pipeline: heldout eval done; remove consolidated temp checkpoint"
} | tee -a "$RUN_DIR/pipeline.log"

rm -rf "$CKPT_DIR/consolidated"

{
  date
  echo "pipeline: complete"
} | tee -a "$RUN_DIR/pipeline.log"
