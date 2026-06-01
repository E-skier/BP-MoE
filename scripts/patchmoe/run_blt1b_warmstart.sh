#!/usr/bin/env bash
set -euo pipefail

# Prepare or launch a guarded BLT-1B PatchMoE warm-start smoke run.
#
# Default MODE=print performs preflight and prints the exact command. Use
# MODE=run only after the selected GPUs are free, or MODE=wait to poll the GPU
# guard and launch once resources are available. This launcher intentionally
# starts with a short 10-step continued-pretraining run before longer budgets.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

UV_BIN="${UV_BIN:-/home/ubuntu/.local/bin/uv}"
CONFIG="${CONFIG:-apps/main/configs/patchmoe_blt1b_warmstart.yaml}"
SOURCE_CKPT="${SOURCE_CKPT:-$ROOT_DIR/hf-weights/blt_1b/consolidated.pth}"
INIT_CKPT_DIR="${INIT_CKPT_DIR:-$ROOT_DIR/hf-weights/blt_1b_patchmoe_entropy_all_layers_dcp}"
OUT_ROOT="${OUT_ROOT:-$ROOT_DIR/runs/blt1b_warmstart}"
RUN_NAME="${RUN_NAME:-patchmoe_blt1b_warmstart_entropy_all_layers_smoke}"
RUN_DIR="${RUN_DIR:-$OUT_ROOT/$RUN_NAME}"
LAUNCH_LOCK_DIR="$OUT_ROOT/.${RUN_NAME}.launch.lock"

MODE="${MODE:-print}"
GPUS="${GPUS:-0,1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
STEPS="${STEPS:-1000}"
MAX_STEPS="${MAX_STEPS:-10}"
BATCH_SIZE="${BATCH_SIZE:-1}"
SEQ_LEN="${SEQ_LEN:-256}"
MAX_ENCODER_SEQ_LENGTH="${MAX_ENCODER_SEQ_LENGTH:-1536}"
GRAD_ACC_STEPS="${GRAD_ACC_STEPS:-1}"
LOAD_ASYNC="${LOAD_ASYNC:-false}"
LOG_FREQ="${LOG_FREQ:-1}"
ENABLE_INTRA_NODE_COMM="${ENABLE_INTRA_NODE_COMM:-1}"
MAX_GPU_USED_MB="${MAX_GPU_USED_MB:-20000}"
MAX_GPU_UTIL_PCT="${MAX_GPU_UTIL_PCT:-20}"
FORCE_GPU="${FORCE_GPU:-0}"
GPU_WAIT_INTERVAL_SECONDS="${GPU_WAIT_INTERVAL_SECONDS:-60}"

DATA_ROOT="${DATA_ROOT:-$ROOT_DIR/data}"
SOURCE="${SOURCE:-fineweb_edu_10bt}"
PREPROCESS_DIR="${PREPROCESS_DIR:-$ROOT_DIR/data/entropy_preprocessed_stage1}"
ENTROPY_MODEL_NAME="${ENTROPY_MODEL_NAME:-transformer_100m}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/tmp/unused.tokenizer.model}"

export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export BLT_ALLOW_MISSING_FLEX_ATTENTION="${BLT_ALLOW_MISSING_FLEX_ATTENTION:-1}"
export BLT_SUPPRESS_ATTN_ERROR="${BLT_SUPPRESS_ATTN_ERROR:-1}"

if [[ "$MODE" != "print" && "$MODE" != "run" && "$MODE" != "wait" ]]; then
  echo "MODE must be print, run, or wait" >&2
  exit 2
fi

quote_cmd() {
  printf '%q ' "$@"
  printf '\n'
}

check_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "FAIL missing file: $path" >&2
    return 1
  fi
  echo "OK   file: $path"
}

check_gpu() {
  if [[ "$FORCE_GPU" == "1" ]]; then
    echo "Skipping GPU memory guard because FORCE_GPU=1"
    return 0
  fi

  local gpu query used util
  IFS=',' read -ra gpu_ids <<<"$GPUS"
  for gpu in "${gpu_ids[@]}"; do
    query="$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -i "$gpu" | head -n 1)"
    IFS=',' read -r used util <<<"$query"
    used="${used//[[:space:]]/}"
    util="${util//[[:space:]]/}"
    if (( used > MAX_GPU_USED_MB )); then
      echo "GPU $gpu uses ${used}MiB, above MAX_GPU_USED_MB=${MAX_GPU_USED_MB}; refusing to launch." >&2
      return 1
    fi
    if (( util > MAX_GPU_UTIL_PCT )); then
      echo "GPU $gpu utilization is ${util}%, above MAX_GPU_UTIL_PCT=${MAX_GPU_UTIL_PCT}; refusing to launch." >&2
      return 1
    fi
    echo "OK   GPU $gpu guard: ${used}MiB and ${util}% utilization"
  done
}

preflight() {
  local preprocessed_source_dir="$PREPROCESS_DIR/$SOURCE/$ENTROPY_MODEL_NAME"
  check_file "$CONFIG"
  check_file "$SOURCE_CKPT"
  check_file "$INIT_CKPT_DIR/.metadata"
  check_file "$INIT_CKPT_DIR/patchmoe_warmstart_manifest.json"
  if [[ ! -d "$preprocessed_source_dir" ]]; then
    echo "FAIL missing entropy-preprocessed data: $preprocessed_source_dir" >&2
    return 1
  fi
  if ! find "$preprocessed_source_dir" -name '*.arrow.complete' -print -quit | grep -q .; then
    echo "FAIL no completed entropy arrow shards in: $preprocessed_source_dir" >&2
    return 1
  fi
  echo "OK   entropy-preprocessed data: $preprocessed_source_dir"
  "$UV_BIN" run python scripts/patchmoe/verify_blt1b_patchmoe_warmstart.py \
    --source "$SOURCE_CKPT" \
    --checkpoint-dir "$INIT_CKPT_DIR"
}

wait_for_gpu() {
  if (( GPU_WAIT_INTERVAL_SECONDS <= 0 )); then
    echo "GPU_WAIT_INTERVAL_SECONDS must be positive" >&2
    return 2
  fi
  echo "Waiting for GPU guard every ${GPU_WAIT_INTERVAL_SECONDS}s before one-time launch."
  until check_gpu; do
    echo "GPU guard still busy at $(date --iso-8601=seconds); waiting."
    sleep "$GPU_WAIT_INTERVAL_SECONDS"
  done
}

cleanup_launch_lock() {
  rm -f "$LAUNCH_LOCK_DIR/owner.pid"
  rmdir "$LAUNCH_LOCK_DIR" 2>/dev/null || true
}

acquire_launch_lock() {
  local owner_pid=""
  mkdir -p "$OUT_ROOT"
  if ! mkdir "$LAUNCH_LOCK_DIR" 2>/dev/null; then
    if [[ -f "$LAUNCH_LOCK_DIR/owner.pid" ]]; then
      owner_pid="$(<"$LAUNCH_LOCK_DIR/owner.pid")"
    fi
    if [[ "$owner_pid" =~ ^[0-9]+$ ]] && kill -0 "$owner_pid" 2>/dev/null; then
      echo "Launch lock is held by live pid=$owner_pid: $LAUNCH_LOCK_DIR" >&2
      echo "Another BLT-1B warm-start launcher is already waiting or launching." >&2
      return 1
    fi
    echo "Removing stale launch lock: $LAUNCH_LOCK_DIR"
    rm -f "$LAUNCH_LOCK_DIR/owner.pid"
    if ! rmdir "$LAUNCH_LOCK_DIR" 2>/dev/null || ! mkdir "$LAUNCH_LOCK_DIR"; then
      echo "Failed to recover launch lock: $LAUNCH_LOCK_DIR" >&2
      return 1
    fi
  fi
  printf '%s\n' "$$" >"$LAUNCH_LOCK_DIR/owner.pid"
  trap cleanup_launch_lock EXIT
  if [[ -e "$RUN_DIR/pipeline.log" ]]; then
    echo "Refusing to overwrite existing smoke log: $RUN_DIR/pipeline.log" >&2
    return 1
  fi
}

cmd=(
  env
  "CUDA_VISIBLE_DEVICES=$GPUS"
  "$UV_BIN" run torchrun --standalone "--nproc-per-node=$NPROC_PER_NODE"
  -m bytelatent.train
  "config=$CONFIG"
  "dump_dir=$RUN_DIR"
  "name=$RUN_NAME"
  "steps=$STEPS"
  "max_steps=$MAX_STEPS"
  "grad_acc_steps=$GRAD_ACC_STEPS"
  "data.root_dir=$DATA_ROOT"
  "data.sources={$SOURCE: 1.0}"
  "data.batch_size=$BATCH_SIZE"
  "data.seq_len=$SEQ_LEN"
  "data.max_encoder_seq_length=$MAX_ENCODER_SEQ_LENGTH"
  "data.load_async=$LOAD_ASYNC"
  "logging.freq=$LOG_FREQ"
  "data.preprocess_dir=$PREPROCESS_DIR"
  "data.entropy_model_name=$ENTROPY_MODEL_NAME"
  "data.tokenizer_args.init_kwargs.bpe_tokenizer_path=$TOKENIZER_PATH"
  "model.max_seqlen=$SEQ_LEN"
  "model.max_length=$SEQ_LEN"
  "model.max_encoder_seq_length=$MAX_ENCODER_SEQ_LENGTH"
  "checkpoint.path=$RUN_DIR/checkpoints"
  "checkpoint.init_ckpt_path=$INIT_CKPT_DIR"
  "distributed.dp_shard=$NPROC_PER_NODE"
  "distributed.dp_replicate=1"
  "eval_on_gpus=$NPROC_PER_NODE"
  "env.ENABLE_INTRA_NODE_COMM=\"$ENABLE_INTRA_NODE_COMM\""
  "env.NCCL_DEBUG=WARN"
)

preflight
echo "Prepared BLT-1B PatchMoE warm-start command:"
quote_cmd "${cmd[@]}"
echo "Log path if MODE=run or MODE=wait: $RUN_DIR/pipeline.log"

if [[ "$MODE" == "print" ]]; then
  exit 0
fi

acquire_launch_lock
if [[ "$MODE" == "wait" ]]; then
  wait_for_gpu
else
  check_gpu
fi
mkdir -p "$RUN_DIR"
nohup "${cmd[@]}" >"$RUN_DIR/pipeline.log" 2>&1 &
pid=$!
printf '%s\n' "$pid" >"$RUN_DIR/train.pid"
echo "Launched BLT-1B PatchMoE warm-start pid=$pid"
echo "Log: $RUN_DIR/pipeline.log"
