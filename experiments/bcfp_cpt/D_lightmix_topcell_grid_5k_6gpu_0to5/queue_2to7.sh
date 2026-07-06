#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/data1/pengfeigao/BP-MoE"
GRID_DIR="$ROOT_DIR/experiments/bcfp_cpt/D_lightmix_topcell_grid_5k_6gpu_0to5"
LAUNCH_ONE="$GRID_DIR/launch_one.sh"
QUEUE_LOG="$GRID_DIR/queue_2to7.log"

cd "$ROOT_DIR"

export TRAIN_CUDA_VISIBLE_DEVICES="2,3,4,5,6,7"
export EVAL_CUDA_VISIBLE_DEVICES="2"
export NPROC_PER_NODE="6"
export DP_SHARD="6"

{
  date
  echo "queue: start light-mix topcell grid on GPUs $TRAIN_CUDA_VISIBLE_DEVICES"
  echo "queue: eval GPU will be $EVAL_CUDA_VISIBLE_DEVICES"
} | tee -a "$QUEUE_LOG"

"$LAUNCH_ONE" "D_mix90_topcell10_screenlike_5k_6gpu_2to7_nofused_clip0_ckptfinal" "0.9" "0.1" 2>&1 | tee -a "$QUEUE_LOG"
"$LAUNCH_ONE" "D_mix80_topcell20_screenlike_5k_6gpu_2to7_nofused_clip0_ckptfinal" "0.8" "0.2" 2>&1 | tee -a "$QUEUE_LOG"

{
  date
  echo "queue: complete light-mix topcell grid on GPUs $TRAIN_CUDA_VISIBLE_DEVICES"
} | tee -a "$QUEUE_LOG"
