#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/data1/pengfeigao/BP-MoE"
GRID_DIR="$ROOT_DIR/experiments/bcfp_cpt/D_lightmix_topcell_grid_5k_6gpu_0to5"
LAUNCH_ONE="$GRID_DIR/launch_one.sh"
QUEUE_LOG="$GRID_DIR/queue.log"

cd "$ROOT_DIR"

{
  date
  echo "queue: start light-mix topcell grid"
  echo "queue: CUDA_VISIBLE_DEVICES will be 0,1,2,3,4,5 for training and 0 for eval"
} | tee -a "$QUEUE_LOG"

"$LAUNCH_ONE" "D_mix90_topcell10_screenlike_5k_6gpu_0to5_nofused_clip0_ckptfinal" "0.9" "0.1" 2>&1 | tee -a "$QUEUE_LOG"
"$LAUNCH_ONE" "D_mix80_topcell20_screenlike_5k_6gpu_0to5_nofused_clip0_ckptfinal" "0.8" "0.2" 2>&1 | tee -a "$QUEUE_LOG"

{
  date
  echo "queue: complete light-mix topcell grid"
} | tee -a "$QUEUE_LOG"

