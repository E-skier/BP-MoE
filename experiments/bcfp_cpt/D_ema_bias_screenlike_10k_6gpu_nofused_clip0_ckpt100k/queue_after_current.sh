#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/data1/pengfeigao/BP-MoE"
RUN_DIR="$ROOT_DIR/experiments/bcfp_cpt/D_ema_bias_screenlike_10k_6gpu_nofused_clip0_ckpt100k"
C001_EVAL_DIR="$ROOT_DIR/experiments/bcfp_cpt/C_byte_aux_001_screenlike_10k_6gpu_nofused_clip0_ckpt100k/heldout_eval_expanded_100k_8000batches_0to5"
C001_CONSOLIDATED="$ROOT_DIR/experiments/bcfp_cpt/C_byte_aux_001_screenlike_10k_6gpu_nofused_clip0_ckpt100k/checkpoints/0000010000/consolidated"

cd "$ROOT_DIR"

echo "$(date) queue: waiting for active C001 eval and D 4GPU sessions"
while tmux has-session -t c001_10k_eval_full_gpu2 2>/dev/null || tmux has-session -t d_ema_10k_4gpu_0134 2>/dev/null; do
  sleep 300
done

if [[ -f "$C001_EVAL_DIR/validation.json" && -d "$C001_CONSOLIDATED" ]]; then
  echo "$(date) queue: removing completed C001 consolidated checkpoint"
  rm -rf "$C001_CONSOLIDATED"
else
  echo "$(date) queue: C001 validation.json missing or consolidated already absent; not deleting checkpoint"
fi

echo "$(date) queue: waiting for GPUs 0-5 to be idle"
while true; do
  mapfile -t used_mib < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  busy=0
  for idx in 0 1 2 3 4 5; do
    if (( ${used_mib[$idx]} > 1024 )); then
      busy=1
      break
    fi
  done
  if (( busy == 0 )); then
    break
  fi
  sleep 300
done

echo "$(date) queue: launching D 6GPU 10k"
exec "$RUN_DIR/launch.sh"
