#!/usr/bin/env bash
set -euo pipefail

# Safely prepare or launch one formal PatchMoE variant.
#
# Default behavior is non-mutating: run the target pipeline preflight and print
# the exact command. Set MODE=run to launch it in the background after a GPU
# memory guard passes.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PIPELINE="${PIPELINE:-200k}"
VARIANT="${VARIANT:-${1:-}}"
MODE="${MODE:-print}"
GPU="${GPU:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
if [[ -z "${GRAD_ACC_STEPS+x}" ]]; then
  if (( NPROC_PER_NODE == 1 )); then
    GRAD_ACC_STEPS=2
  else
    GRAD_ACC_STEPS=1
  fi
fi
LOAD_ASYNC="${LOAD_ASYNC:-false}"
LOG_FREQ="${LOG_FREQ:-10}"
MAX_GPU_USED_MB="${MAX_GPU_USED_MB:-20000}"
MAX_GPU_UTIL_PCT="${MAX_GPU_UTIL_PCT:-20}"
FORCE_GPU="${FORCE_GPU:-0}"

if [[ -z "$VARIANT" ]]; then
  echo "usage: VARIANT=name [PIPELINE=200k|stage1] [MODE=print|run] [GPU=0] $0" >&2
  echo "       or: $0 VARIANT" >&2
  exit 2
fi
if [[ "$MODE" != "print" && "$MODE" != "run" ]]; then
  echo "MODE must be print or run" >&2
  exit 2
fi

quote_cmd() {
  printf '%q ' "$@"
  printf '\n'
}

check_gpu() {
  if [[ "$FORCE_GPU" == "1" ]]; then
    echo "Skipping GPU memory guard because FORCE_GPU=1"
    return 0
  fi
  local query used util
  query="$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -i "$GPU" | head -n 1)"
  IFS=',' read -r used util <<<"$query"
  used="${used//[[:space:]]/}"
  util="${util//[[:space:]]/}"
  if [[ -z "$used" || -z "$util" ]]; then
    echo "Unable to read GPU $GPU memory/utilization" >&2
    return 1
  fi
  if (( used > MAX_GPU_USED_MB )); then
    echo "GPU $GPU currently uses ${used}MiB, above MAX_GPU_USED_MB=${MAX_GPU_USED_MB}; refusing to launch." >&2
    echo "Use a freer GPU, wait, or set FORCE_GPU=1 only if you have verified the allocation is safe." >&2
    return 1
  fi
  if (( util > MAX_GPU_UTIL_PCT )); then
    echo "GPU $GPU utilization is ${util}%, above MAX_GPU_UTIL_PCT=${MAX_GPU_UTIL_PCT}; refusing to launch." >&2
    echo "Use a freer GPU, wait, or set FORCE_GPU=1 only if you have verified the allocation is safe." >&2
    return 1
  fi
  echo "OK   GPU $GPU guard: ${used}MiB <= ${MAX_GPU_USED_MB}MiB and ${util}% <= ${MAX_GPU_UTIL_PCT}%"
}

case "$PIPELINE" in
  200k)
    SCRIPT="scripts/patchmoe/run_byte_entropy_200k_matched_controls.sh"
    SEED="${SEED:-779}"
    OUT_ROOT="${OUT_ROOT:-$ROOT_DIR/runs/byte_entropy_200k_matched_controls}"
    EVAL_ROOT="${EVAL_ROOT:-$ROOT_DIR/runs/byte_entropy_200k_matched_controls_heldout_eval}"
    ANALYSIS_ROOT="${ANALYSIS_ROOT:-$ROOT_DIR/runs/byte_entropy_200k_matched_controls_analysis}"
    env_args=(
      "CUDA_VISIBLE_DEVICES=$GPU"
      "EVAL_CUDA_VISIBLE_DEVICES=$GPU"
      "NPROC_PER_NODE=$NPROC_PER_NODE"
      "GRAD_ACC_STEPS=$GRAD_ACC_STEPS"
      "LOAD_ASYNC=$LOAD_ASYNC"
      "LOG_FREQ=$LOG_FREQ"
      "SEEDS=$SEED"
      "VARIANTS=$VARIANT"
      "OUT_ROOT=$OUT_ROOT"
      "EVAL_ROOT=$EVAL_ROOT"
      "ANALYSIS_ROOT=$ANALYSIS_ROOT"
    )
    log_path="$OUT_ROOT/pipeline_${VARIANT}_seed${SEED}_gpu${GPU}.log"
    ;;
  stage1)
    SCRIPT="scripts/patchmoe/run_stage1_matched_eval_pipeline.sh"
    OUT_ROOT="${OUT_ROOT:-$ROOT_DIR/runs/stage1_matched_budget}"
    EVAL_ROOT="${EVAL_ROOT:-$ROOT_DIR/runs/stage1_heldout_eval}"
    ANALYSIS_ROOT="${ANALYSIS_ROOT:-$ROOT_DIR/runs/stage1_matched_budget_analysis}"
    env_args=(
      "CUDA_VISIBLE_DEVICES=$GPU"
      "EVAL_CUDA_VISIBLE_DEVICES=$GPU"
      "NPROC_PER_NODE=$NPROC_PER_NODE"
      "GRAD_ACC_STEPS=$GRAD_ACC_STEPS"
      "LOAD_ASYNC=$LOAD_ASYNC"
      "LOG_FREQ=$LOG_FREQ"
      "VARIANTS=$VARIANT"
      "OUT_ROOT=$OUT_ROOT"
      "EVAL_ROOT=$EVAL_ROOT"
      "ANALYSIS_ROOT=$ANALYSIS_ROOT"
    )
    log_path="$OUT_ROOT/pipeline_${VARIANT}_gpu${GPU}.log"
    ;;
  *)
    echo "PIPELINE must be 200k or stage1" >&2
    exit 2
    ;;
esac

# Validate the variant before running the heavier preflight.
env "${env_args[@]}" "$SCRIPT" --print-overrides "$VARIANT" >/dev/null

echo "Running preflight for PIPELINE=$PIPELINE VARIANT=$VARIANT"
env "${env_args[@]}" "$SCRIPT" --preflight

cmd=(env "${env_args[@]}" "$SCRIPT")
echo "Prepared command:"
quote_cmd "${cmd[@]}"
echo "Log path if MODE=run: $log_path"

if [[ "$MODE" == "print" ]]; then
  exit 0
fi

check_gpu
mkdir -p "$OUT_ROOT"
nohup "${cmd[@]}" >"$log_path" 2>&1 &
pid=$!
echo "Launched PIPELINE=$PIPELINE VARIANT=$VARIANT on GPU=$GPU pid=$pid"
echo "Log: $log_path"
