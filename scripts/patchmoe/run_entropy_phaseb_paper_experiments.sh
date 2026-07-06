#!/usr/bin/env bash
set -euo pipefail

# Paper-critical Entropy PhaseB experiment launcher.
#
# This script intentionally does not change model code. It coordinates:
#   1. held-out eval for the existing entropy_bands_phaseB_100k 50k checkpoint
#   2. hidden-only 50k rerun with robust checkpoint retention
#   3. entropy PhaseB balance-loss controls at w=0.01 and w=0.05
#   4. read-only markdown summary tables for quality and system cost
#
# Default ACTION=preflight is read-only. Use MODE=run for train/eval launch.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ -z "${UV_BIN:-}" ]]; then
  if command -v uv >/dev/null 2>&1; then
    UV_BIN="$(command -v uv)"
  else
    UV_BIN="/home/pengfeigao/.local/bin/uv"
  fi
fi
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-$ROOT_DIR/.venv/bin/torchrun}"

ACTION="${ACTION:-preflight}"
MODE="${MODE:-print}"

CONFIG="${CONFIG:-apps/main/configs/patchmoe_blt1b_warmstart.yaml}"
OUT_ROOT="${OUT_ROOT:-$ROOT_DIR/blt1b_warmstart/paper_phaseb_experiments}"
EVAL_ROOT="${EVAL_ROOT:-$ROOT_DIR/runs/entropy_phaseb_paper_heldout_eval}"
SUMMARY_OUT="${SUMMARY_OUT:-$ROOT_DIR/runs/entropy_phaseb_paper_tables.md}"

DATA_ROOT="${DATA_ROOT:-$ROOT_DIR/data}"
SOURCE="${SOURCE:-fineweb_edu_10bt}"
PREPROCESS_DIR="${PREPROCESS_DIR:-$ROOT_DIR/data/entropy_preprocessed_stage1}"
ENTROPY_MODEL_NAME="${ENTROPY_MODEL_NAME:-transformer_100m}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/tmp/unused.tokenizer.model}"

PHASEA_INIT_CKPT="${PHASEA_INIT_CKPT:-$ROOT_DIR/blt1b_warmstart/entropy_bands_phaseA_1k/entropy_bands_phaseA_1000step/checkpoints/0000001000}"
HIDDEN_INIT_CKPT="${HIDDEN_INIT_CKPT:-$ROOT_DIR/blt1b_warmstart/initial_weights/blt_1b_patchmoe_hidden_only_active_matched_dcp}"
PHASEB_TARGET_CKPT="${PHASEB_TARGET_CKPT:-$ROOT_DIR/blt1b_warmstart/entropy_bands_phaseB_100k/checkpoints/0000050000}"
PHASEB_TARGET_RUN_DIR="${PHASEB_TARGET_RUN_DIR:-$ROOT_DIR/blt1b_warmstart/entropy_bands_phaseB_100k}"

HELDOUT_LINES="${HELDOUT_LINES:-50000}"
HELDOUT_SOURCE_NAME="${HELDOUT_SOURCE_NAME:-${SOURCE}_heldout}"
HELDOUT_RAW_DIR="${HELDOUT_RAW_DIR:-$DATA_ROOT/stage1_heldout/$HELDOUT_SOURCE_NAME}"
HELDOUT_RAW_FILE="${HELDOUT_RAW_FILE:-$HELDOUT_RAW_DIR/$HELDOUT_SOURCE_NAME.chunk.00002.first_${HELDOUT_LINES}.jsonl}"
HELDOUT_SOURCE_FILE="${HELDOUT_SOURCE_FILE:-$DATA_ROOT/$SOURCE/$SOURCE.chunk.00002.jsonl}"
HELDOUT_PREPROCESS_ROOT="${HELDOUT_PREPROCESS_ROOT:-$ROOT_DIR/data/entropy_preprocessed_stage1_heldout}"
HELDOUT_ARROW_DIR="$HELDOUT_PREPROCESS_ROOT/$HELDOUT_SOURCE_NAME/$ENTROPY_MODEL_NAME"
HELDOUT_ARROW_NAME="$(basename "$HELDOUT_RAW_FILE").shard_00.arrow"
HELDOUT_ARROW_FILE="$HELDOUT_ARROW_DIR/$HELDOUT_ARROW_NAME"

TARGETS="${TARGETS:-phaseb_balance_w001 phaseb_balance_w005 hidden_only_w000_50k}"
VARIANT="${VARIANT:-}"
GPUS="${GPUS:-1,2}"
GPU="${GPU:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
EP_SIZE="${EP_SIZE:-$NPROC_PER_NODE}"
STEPS="${STEPS:-50000}"
MAX_STEPS="${MAX_STEPS:-$STEPS}"
BATCH_SIZE="${BATCH_SIZE:-1}"
SEQ_LEN="${SEQ_LEN:-256}"
MAX_ENCODER_SEQ_LENGTH="${MAX_ENCODER_SEQ_LENGTH:-1536}"
GRAD_ACC_STEPS="${GRAD_ACC_STEPS:-1}"
LOAD_ASYNC="${LOAD_ASYNC:-false}"
LOG_FREQ="${LOG_FREQ:-10}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-5000}"
CHECKPOINT_KEEP="${CHECKPOINT_KEEP:-3}"
ENABLE_INTRA_NODE_COMM="${ENABLE_INTRA_NODE_COMM:-1}"

EVAL_CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES:-$GPU}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
# Empty or "full" means full held-out shard. Set EVAL_MAX_BATCHES=2000 for a fast smoke eval.
EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-}"
EVAL_LABEL="${EVAL_LABEL:-entropy_bands_phaseB_100k}"
EVAL_STEP="${EVAL_STEP:-0000050000}"

MAX_GPU_USED_MB="${MAX_GPU_USED_MB:-20000}"
MAX_GPU_UTIL_PCT="${MAX_GPU_UTIL_PCT:-20}"
FORCE_GPU="${FORCE_GPU:-0}"
GPU_WAIT_INTERVAL_SECONDS="${GPU_WAIT_INTERVAL_SECONDS:-60}"

export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export BLT_ALLOW_MISSING_FLEX_ATTENTION="${BLT_ALLOW_MISSING_FLEX_ATTENTION:-1}"
export BLT_SUPPRESS_ATTN_ERROR="${BLT_SUPPRESS_ATTN_ERROR:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

quote_cmd() {
  printf '%q ' "$@"
  printf '\n'
}

usage() {
  cat <<'EOF'
Usage:
  ACTION=preflight scripts/patchmoe/run_entropy_phaseb_paper_experiments.sh
  ACTION=list scripts/patchmoe/run_entropy_phaseb_paper_experiments.sh
  ACTION=train VARIANT=phaseb_balance_w001 MODE=run GPUS=1,2 scripts/patchmoe/run_entropy_phaseb_paper_experiments.sh
  ACTION=train-all MODE=run GPUS=1,2 scripts/patchmoe/run_entropy_phaseb_paper_experiments.sh
  ACTION=eval-phaseb50k MODE=run GPU=1 scripts/patchmoe/run_entropy_phaseb_paper_experiments.sh
  ACTION=summarize scripts/patchmoe/run_entropy_phaseb_paper_experiments.sh

Actions:
  preflight       Read-only artifact and GPU/data check.
  list            Print supported variants and actions.
  train           Print or launch one VARIANT.
  train-all       Print or launch all TARGETS sequentially in this shell/background job.
  eval-phaseb50k  Print or launch held-out eval for entropy_bands_phaseB_100k/checkpoints/0000050000.
  eval            Print or launch held-out eval for CKPT_DIR/RUN_DIR/EVAL_LABEL/EVAL_STEP.
  summarize       Generate quality vs system-cost markdown tables from existing artifacts.

Variants:
  phaseb_repro_w000      Entropy PhaseB baseline, balance loss 0.00.
  phaseb_balance_w001    Entropy PhaseB control, balance loss 0.01.
  phaseb_balance_w005    Entropy PhaseB control, balance loss 0.05.
  hidden_only_w000_50k   Hidden-only 50k rerun, balance loss 0.00, checkpoint every 5k.
EOF
}

check_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "FAIL missing file: $path" >&2
    return 1
  fi
  echo "OK   file: $path"
}

check_dir() {
  local path="$1"
  if [[ ! -d "$path" ]]; then
    echo "FAIL missing directory: $path" >&2
    return 1
  fi
  echo "OK   directory: $path"
}

check_gpu_list() {
  if [[ "$FORCE_GPU" == "1" ]]; then
    echo "Skipping GPU guard because FORCE_GPU=1"
    return 0
  fi

  local gpu query used util
  IFS=',' read -ra gpu_ids <<<"$1"
  for gpu in "${gpu_ids[@]}"; do
    query="$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -i "$gpu" | head -n 1)"
    IFS=',' read -r used util <<<"$query"
    used="${used//[[:space:]]/}"
    util="${util//[[:space:]]/}"
    if [[ -z "$used" || -z "$util" ]]; then
      echo "FAIL unable to read GPU $gpu state" >&2
      return 1
    fi
    if (( used > MAX_GPU_USED_MB )); then
      echo "FAIL GPU $gpu uses ${used}MiB, above MAX_GPU_USED_MB=$MAX_GPU_USED_MB" >&2
      return 1
    fi
    if (( util > MAX_GPU_UTIL_PCT )); then
      echo "FAIL GPU $gpu utilization ${util}%, above MAX_GPU_UTIL_PCT=$MAX_GPU_UTIL_PCT" >&2
      return 1
    fi
    echo "OK   GPU $gpu guard: ${used}MiB, ${util}%"
  done
}

wait_for_gpu_list() {
  local gpu_list="$1"
  if (( GPU_WAIT_INTERVAL_SECONDS <= 0 )); then
    echo "GPU_WAIT_INTERVAL_SECONDS must be positive" >&2
    return 2
  fi
  until check_gpu_list "$gpu_list"; do
    echo "GPU guard busy at $(date --iso-8601=seconds); waiting ${GPU_WAIT_INTERVAL_SECONDS}s."
    sleep "$GPU_WAIT_INTERVAL_SECONDS"
  done
}

prepare_heldout() {
  if [[ ! -f "$HELDOUT_SOURCE_FILE" ]]; then
    echo "Missing held-out source file: $HELDOUT_SOURCE_FILE" >&2
    return 1
  fi
  mkdir -p "$HELDOUT_RAW_DIR"
  if [[ ! -f "$HELDOUT_RAW_FILE" ]]; then
    echo "Creating held-out subset: $HELDOUT_RAW_FILE"
    tmp_file="$HELDOUT_RAW_FILE.tmp"
    head -n "$HELDOUT_LINES" "$HELDOUT_SOURCE_FILE" >"$tmp_file"
    mv "$tmp_file" "$HELDOUT_RAW_FILE"
  fi
  if [[ -f "$HELDOUT_ARROW_FILE" ]]; then
    if [[ ! -f "$HELDOUT_ARROW_FILE.complete" ]]; then
      echo "Using held-out arrow without .complete marker: $HELDOUT_ARROW_FILE"
    fi
    return 0
  fi
  if [[ ! -f "$HELDOUT_ARROW_FILE.complete" ]]; then
    echo "Preprocessing held-out shard: $HELDOUT_ARROW_FILE"
    SOURCE_DIR="$HELDOUT_RAW_DIR" \
    SOURCE_NAME="$HELDOUT_SOURCE_NAME" \
    OUTPUT_ROOT="$HELDOUT_PREPROCESS_ROOT" \
    ENTROPY_MODEL_NAME="$ENTROPY_MODEL_NAME" \
    NPROC=1 \
    MAX_FILES=1 \
      scripts/patchmoe/preprocess_fineweb_entropy_stage1.sh
  fi
}

variant_run_name() {
  case "$1" in
    phaseb_repro_w000) printf 'entropy_bands_phaseB_repro_w000_50k' ;;
    phaseb_balance_w001) printf 'entropy_bands_phaseB_balance001_50k' ;;
    phaseb_balance_w005) printf 'entropy_bands_phaseB_balance005_50k' ;;
    hidden_only_w000_50k) printf 'hidden_only_blt1b_50000step_savefix_balance0' ;;
    *) echo "Unknown variant: $1" >&2; return 1 ;;
  esac
}

variant_init_ckpt() {
  case "$1" in
    phaseb_repro_w000|phaseb_balance_w001|phaseb_balance_w005) printf '%s' "$PHASEA_INIT_CKPT" ;;
    hidden_only_w000_50k) printf '%s' "$HIDDEN_INIT_CKPT" ;;
    *) echo "Unknown variant: $1" >&2; return 1 ;;
  esac
}

variant_overrides() {
  case "$1" in
    phaseb_repro_w000)
      printf '%s\n' \
        model.moe_num_experts=8 \
        model.moe_top_k=2 \
        model.moe_ffn_dim_multiplier=0.5 \
        model.moe_balance_loss_weight=0.0 \
        model.moe_router_jitter=0.0 \
        model.moe_router_congestion_weight=0.0 \
        model.moe_router_z_loss_weight=0.0 \
        model.moe_router_use_hidden_state=false \
        model.moe_router_use_patch_length=false \
        model.moe_router_use_patch_entropy=true \
        model.moe_router_use_patch_byte_features=false \
        model.moe_balance_cost=byte
      ;;
    phaseb_balance_w001)
      printf '%s\n' \
        model.moe_num_experts=8 \
        model.moe_top_k=2 \
        model.moe_ffn_dim_multiplier=0.5 \
        model.moe_balance_loss_weight=0.01 \
        model.moe_router_jitter=0.0 \
        model.moe_router_congestion_weight=0.0 \
        model.moe_router_z_loss_weight=0.0 \
        model.moe_router_use_hidden_state=false \
        model.moe_router_use_patch_length=false \
        model.moe_router_use_patch_entropy=true \
        model.moe_router_use_patch_byte_features=false \
        model.moe_balance_cost=byte
      ;;
    phaseb_balance_w005)
      printf '%s\n' \
        model.moe_num_experts=8 \
        model.moe_top_k=2 \
        model.moe_ffn_dim_multiplier=0.5 \
        model.moe_balance_loss_weight=0.05 \
        model.moe_router_jitter=0.0 \
        model.moe_router_congestion_weight=0.0 \
        model.moe_router_z_loss_weight=0.0 \
        model.moe_router_use_hidden_state=false \
        model.moe_router_use_patch_length=false \
        model.moe_router_use_patch_entropy=true \
        model.moe_router_use_patch_byte_features=false \
        model.moe_balance_cost=byte
      ;;
    hidden_only_w000_50k)
      printf '%s\n' \
        model.moe_num_experts=8 \
        model.moe_top_k=2 \
        model.moe_ffn_dim_multiplier=0.5 \
        model.moe_balance_loss_weight=0.0 \
        model.moe_router_jitter=0.0 \
        model.moe_router_congestion_weight=0.0 \
        model.moe_router_z_loss_weight=0.0 \
        model.moe_router_use_hidden_state=true \
        model.moe_router_use_patch_length=false \
        model.moe_router_use_patch_entropy=false \
        model.moe_router_use_patch_byte_features=false \
        model.moe_balance_cost=byte
      ;;
    *) echo "Unknown variant: $1" >&2; return 1 ;;
  esac
}

train_cmd_for_variant() {
  local variant="$1"
  local run_name run_dir init_ckpt
  run_name="$(variant_run_name "$variant")"
  run_dir="$OUT_ROOT/$run_name"
  init_ckpt="$(variant_init_ckpt "$variant")"
  mapfile -t overrides < <(variant_overrides "$variant")

  cmd=(
    env
    "CUDA_VISIBLE_DEVICES=$GPUS"
    "PYTHONPATH=$ROOT_DIR:${PYTHONPATH:-}"
    "BLT_ALLOW_MISSING_FLEX_ATTENTION=$BLT_ALLOW_MISSING_FLEX_ATTENTION"
    "BLT_SUPPRESS_ATTN_ERROR=$BLT_SUPPRESS_ATTN_ERROR"
    "NCCL_P2P_DISABLE=$NCCL_P2P_DISABLE"
    "NCCL_IB_DISABLE=$NCCL_IB_DISABLE"
    "$TORCHRUN_BIN" --standalone "--nproc-per-node=$NPROC_PER_NODE"
    -m bytelatent.train
    "config=$CONFIG"
    "dump_dir=$run_dir"
    "name=$run_name"
    "seed=42"
    "steps=$STEPS"
    "max_steps=$MAX_STEPS"
    "grad_acc_steps=$GRAD_ACC_STEPS"
    "data.root_dir=$DATA_ROOT"
    "data.sources={$SOURCE: 1.0}"
    "data.batch_size=$BATCH_SIZE"
    "data.seq_len=$SEQ_LEN"
    "data.max_encoder_seq_length=$MAX_ENCODER_SEQ_LENGTH"
    "data.load_async=$LOAD_ASYNC"
    "data.preprocess_dir=$PREPROCESS_DIR"
    "data.entropy_model_name=$ENTROPY_MODEL_NAME"
    "data.tokenizer_args.init_kwargs.bpe_tokenizer_path=$TOKENIZER_PATH"
    "model.attn_impl=sdpa"
    "model.max_seqlen=$SEQ_LEN"
    "model.max_length=$SEQ_LEN"
    "model.max_encoder_seq_length=$MAX_ENCODER_SEQ_LENGTH"
    "model.moe_ep_size=$EP_SIZE"
    "checkpoint.path=$run_dir/checkpoints"
    "checkpoint.init_ckpt_path=$init_ckpt"
    "checkpoint.dump.every=$CHECKPOINT_EVERY"
    "checkpoint.dump.keep=$CHECKPOINT_KEEP"
    "checkpoint.eval.every=$CHECKPOINT_EVERY"
    "checkpoint.eval.keep=$CHECKPOINT_KEEP"
    "distributed.fsdp_type=full_shard"
    "distributed.dp_shard=$NPROC_PER_NODE"
    "distributed.dp_replicate=1"
    "distributed.tp_size=1"
    "distributed.selective_activation_checkpointing=false"
    "distributed.compile=false"
    "distributed.model_dtype=bf16"
    "optim.lr=5e-6"
    "optim.warmup=0"
    "optim.scheduler=constant"
    "optim.lr_min_ratio=1.0"
    "optim.clip=0.0"
    "optim.fused=false"
    "logging.freq=$LOG_FREQ"
    "logging.wandb=null"
    "eval_on_gpus=$NPROC_PER_NODE"
    "env.ENABLE_INTRA_NODE_COMM=\"$ENABLE_INTRA_NODE_COMM\""
    "env.NCCL_DEBUG=WARN"
    "${overrides[@]}"
  )
}

eval_cmd() {
  local ckpt_dir="$1"
  local run_dir="$2"
  local eval_label="$3"
  local eval_step="$4"
  local eval_dir="$EVAL_ROOT/$eval_label/$eval_step"
  local global_step="$eval_step"
  if [[ "$global_step" =~ ^[0-9]+$ ]]; then
    global_step=$((10#$global_step))
  fi
  cmd=(
    env
    "CUDA_VISIBLE_DEVICES=$EVAL_CUDA_VISIBLE_DEVICES"
    "PYTHONPATH=$ROOT_DIR:${PYTHONPATH:-}"
    "BLT_ALLOW_MISSING_FLEX_ATTENTION=$BLT_ALLOW_MISSING_FLEX_ATTENTION"
    "BLT_SUPPRESS_ATTN_ERROR=$BLT_SUPPRESS_ATTN_ERROR"
    "NCCL_P2P_DISABLE=$NCCL_P2P_DISABLE"
    "NCCL_IB_DISABLE=$NCCL_IB_DISABLE"
    "$PYTHON_BIN" -m bytelatent.eval
    "ckpt_dir=$ckpt_dir"
    "dump_dir=$eval_dir"
    "metric_log_dir=$run_dir"
    "global_step=$global_step"
    "consolidate_if_needed=true"
    "run_ppl=true"
    "run_tasks=false"
    "validation.use_val_from_train_src=false"
    "validation.root_dir=$HELDOUT_ARROW_DIR"
    "validation.sources=[$HELDOUT_ARROW_NAME]"
    "validation.batch_size=$EVAL_BATCH_SIZE"
  )
  if [[ -n "$EVAL_MAX_BATCHES" && "$EVAL_MAX_BATCHES" != "full" ]]; then
    cmd+=("validation.max_n_batches=$EVAL_MAX_BATCHES")
  fi
}

preflight() {
  local failures=0
  echo "Entropy PhaseB paper experiment preflight"
  echo "ACTION=$ACTION MODE=$MODE"
  echo "OUT_ROOT=$OUT_ROOT"
  echo "EVAL_ROOT=$EVAL_ROOT"
  check_file "$CONFIG" || failures=1
  check_file "$PYTHON_BIN" || failures=1
  check_file "$TORCHRUN_BIN" || failures=1
  check_dir "$PREPROCESS_DIR/$SOURCE/$ENTROPY_MODEL_NAME" || failures=1
  check_file "$PHASEB_TARGET_CKPT/params.json" || failures=1
  check_file "$PHASEB_TARGET_CKPT/.metadata" || failures=1
  check_dir "$PHASEA_INIT_CKPT" || failures=1
  check_file "$HIDDEN_INIT_CKPT/patchmoe_warmstart_manifest.json" || failures=1
  check_file "$HELDOUT_SOURCE_FILE" || failures=1
  if [[ -f "$HELDOUT_ARROW_FILE.complete" ]]; then
    echo "OK   held-out arrow: $HELDOUT_ARROW_FILE"
  elif [[ -f "$HELDOUT_ARROW_FILE" ]]; then
    echo "OK   held-out arrow exists without .complete marker: $HELDOUT_ARROW_FILE"
  else
    echo "TODO held-out arrow will be prepared: $HELDOUT_ARROW_FILE"
  fi
  echo "Training targets: $TARGETS"
  for target in $TARGETS; do
    local run_name run_dir ckpt_dir
    run_name="$(variant_run_name "$target")" || failures=1
    run_dir="$OUT_ROOT/$run_name"
    ckpt_dir="$run_dir/checkpoints/$(printf '%010d' "$STEPS")"
    if [[ -d "$ckpt_dir" ]]; then
      echo "DONE checkpoint target=$target: $ckpt_dir"
    elif [[ -f "$run_dir/metrics.jsonl" ]]; then
      echo "PARTIAL target=$target: $run_dir/metrics.jsonl"
    else
      echo "TODO target=$target: $run_dir"
    fi
  done
  check_gpu_list "$GPUS" || true
  check_gpu_list "$EVAL_CUDA_VISIBLE_DEVICES" || true
  return "$failures"
}

launch_or_print() {
  local log_path="$1"
  local pid_path="$2"
  if [[ "$MODE" != "print" && "$MODE" != "run" && "$MODE" != "wait" ]]; then
    echo "MODE must be print, run, or wait" >&2
    return 2
  fi
  echo "Prepared command:"
  quote_cmd "${cmd[@]}"
  echo "Log path if launched: $log_path"
  if [[ "$MODE" == "print" ]]; then
    return 0
  fi
  mkdir -p "$(dirname "$log_path")"
  if [[ "$MODE" == "wait" ]]; then
    wait_for_gpu_list "$GPUS"
  else
    check_gpu_list "$GPUS"
  fi
  nohup "${cmd[@]}" >"$log_path" 2>&1 &
  local pid=$!
  printf '%s\n' "$pid" >"$pid_path"
  echo "Launched pid=$pid"
  echo "PID file: $pid_path"
  echo "Log: $log_path"
}

run_train_variant() {
  local variant="$1"
  local run_name run_dir final_ckpt
  run_name="$(variant_run_name "$variant")"
  run_dir="$OUT_ROOT/$run_name"
  final_ckpt="$run_dir/checkpoints/$(printf '%010d' "$STEPS")"
  if [[ -d "$final_ckpt" && "${FORCE_TRAIN:-0}" != "1" ]]; then
    echo "Skipping completed checkpoint: $final_ckpt"
    return 0
  fi
  train_cmd_for_variant "$variant"
  launch_or_print "$run_dir/train.log" "$run_dir/train.pid"
}

run_train_all() {
  local script="$ROOT_DIR/scripts/patchmoe/run_entropy_phaseb_paper_experiments.sh"
  cmd=(env)
  for name in UV_BIN PYTHON_BIN TORCHRUN_BIN CONFIG OUT_ROOT EVAL_ROOT DATA_ROOT SOURCE PREPROCESS_DIR ENTROPY_MODEL_NAME TOKENIZER_PATH PHASEA_INIT_CKPT HIDDEN_INIT_CKPT GPUS NPROC_PER_NODE EP_SIZE STEPS MAX_STEPS BATCH_SIZE SEQ_LEN MAX_ENCODER_SEQ_LENGTH GRAD_ACC_STEPS LOAD_ASYNC LOG_FREQ CHECKPOINT_EVERY CHECKPOINT_KEEP ENABLE_INTRA_NODE_COMM MAX_GPU_USED_MB MAX_GPU_UTIL_PCT FORCE_GPU; do
    cmd+=("$name=${!name}")
  done
  cmd+=("$script" "__train_all_inner")
  launch_or_print "$OUT_ROOT/train_all.log" "$OUT_ROOT/train_all.pid"
}

run_eval_action() {
  local ckpt_dir="${CKPT_DIR:-$PHASEB_TARGET_CKPT}"
  local run_dir="${RUN_DIR:-$PHASEB_TARGET_RUN_DIR}"
  local eval_label="${LABEL:-$EVAL_LABEL}"
  local eval_step="${STEP:-$EVAL_STEP}"
  if [[ "$MODE" == "print" ]]; then
    if [[ -f "$HELDOUT_ARROW_FILE" ]]; then
      echo "Using held-out arrow: $HELDOUT_ARROW_FILE"
    else
      echo "Held-out arrow is missing and will be prepared when MODE=run or MODE=wait: $HELDOUT_ARROW_FILE"
    fi
  else
    prepare_heldout
  fi
  if [[ ! -d "$ckpt_dir" ]]; then
    echo "Missing checkpoint directory: $ckpt_dir" >&2
    return 1
  fi
  eval_cmd "$ckpt_dir" "$run_dir" "$eval_label" "$eval_step"
  local eval_dir="$EVAL_ROOT/$eval_label/$eval_step"
  local old_gpus="$GPUS"
  GPUS="$EVAL_CUDA_VISIBLE_DEVICES"
  launch_or_print "$eval_dir/eval.log" "$eval_dir/eval.pid"
  GPUS="$old_gpus"
}

summarize() {
  mkdir -p "$(dirname "$SUMMARY_OUT")"
  SUMMARY_OUT="$SUMMARY_OUT" OUT_ROOT="$OUT_ROOT" EVAL_ROOT="$EVAL_ROOT" ROOT_DIR="$ROOT_DIR" STEPS="$STEPS" "$PYTHON_BIN" - <<'PY'
import json
import math
import os
from pathlib import Path

root = Path(os.environ["ROOT_DIR"])
out_root = Path(os.environ["OUT_ROOT"])
eval_root = Path(os.environ["EVAL_ROOT"])
summary_out = Path(os.environ["SUMMARY_OUT"])
steps = int(os.environ["STEPS"])
final_step_dir = f"{steps:010d}"

new_phaseb_eval = eval_root / "entropy_bands_phaseB_100k/0000050000/validation.json"
old_phaseb_eval = root / "runs/stage1_heldout_eval/entropy_bands_phaseB_100k/0000050000/validation.json"
known_runs = [
    (
        "entropy_bands_phaseB_100k_existing",
        root / "blt1b_warmstart/entropy_bands_phaseB_100k",
        new_phaseb_eval if new_phaseb_eval.exists() else old_phaseb_eval,
        "0000050000",
    ),
]
if out_root.exists():
    for run_dir in sorted(p for p in out_root.iterdir() if p.is_dir()):
        known_runs.append((run_dir.name, run_dir, eval_root / run_dir.name / final_step_dir / "validation.json", final_step_dir))

def read_last_jsonl(path: Path):
    if not path.exists():
        return None
    last = ""
    with path.open() as f:
        for line in f:
            if line.strip():
                last = line
    return json.loads(last) if last else None

def read_validation(path: Path):
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    total_bytes = 0.0
    total_loss = 0.0
    source_count = 0
    for metrics in data.values():
        n_bytes = float(metrics.get("n_bytes") or 0.0)
        loss_sum = metrics.get("loss_sum")
        if loss_sum is None and metrics.get("loss_mean") is not None:
            loss_sum = float(metrics["loss_mean"]) * n_bytes
        total_bytes += n_bytes
        total_loss += float(loss_sum or 0.0)
        source_count += 1
    if total_bytes <= 0:
        return {"source_count": source_count, "heldout_n_bytes": total_bytes}
    loss_mean = total_loss / total_bytes
    return {
        "source_count": source_count,
        "heldout_n_bytes": total_bytes,
        "heldout_loss_mean": loss_mean,
        "heldout_ppl": math.exp(loss_mean),
        "heldout_bpb": total_loss / math.log(2) / total_bytes,
    }

def fmt(x, nd=6):
    if x is None:
        return ""
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)

quality_rows = []
system_rows = []
for label, run_dir, eval_path, step_dir in known_runs:
    metrics = read_last_jsonl(run_dir / "metrics.jsonl") or {}
    validation = read_validation(eval_path) or {}
    ckpt_dir = run_dir / "checkpoints" / step_dir
    quality_rows.append([
        label,
        str(metrics.get("global_step", "")),
        "yes" if ckpt_dir.is_dir() else "no",
        fmt(metrics.get("bpb/interval_across_gpus")),
        fmt(metrics.get("loss/interval_across_gpu")),
        fmt(validation.get("heldout_bpb")),
        fmt(validation.get("heldout_loss_mean")),
        fmt(validation.get("heldout_n_bytes"), 0),
        str(eval_path),
    ])
    system_rows.append([
        label,
        str(metrics.get("global_step", "")),
        fmt(metrics.get("speed/wps"), 2),
        fmt(metrics.get("speed/curr_iter_time"), 4),
        fmt(metrics.get("speed/FLOPS"), 2),
        fmt(metrics.get("memory/max_active_gib"), 3),
        fmt(metrics.get("memory/max_reserved_gib"), 3),
        fmt(metrics.get("memory/num_alloc_retries"), 0),
        fmt(metrics.get("moe/ep_all_to_all_bytes_mean"), 0),
        fmt(metrics.get("moe/active_experts_mean"), 3),
        fmt(metrics.get("moe/max_load_fraction_mean"), 6),
        fmt(metrics.get("moe/min_load_fraction_mean"), 6),
        fmt(metrics.get("moe/expert_6_load_fraction_mean"), 6),
        fmt(metrics.get("moe/expert_7_load_fraction_mean"), 6),
        fmt(metrics.get("moe/load_imbalance_mean"), 6),
    ])

def table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)

text = []
text.append("# Entropy PhaseB Paper Experiment Tables")
text.append("")
text.append("Generated by `scripts/patchmoe/run_entropy_phaseb_paper_experiments.sh ACTION=summarize`.")
text.append("")
text.append("## Quality Table")
text.append("")
text.append(table(
    ["run", "last_step", "checkpoint", "train_bpb", "train_loss", "heldout_bpb", "heldout_loss", "heldout_bytes", "eval_path"],
    quality_rows,
))
text.append("")
text.append("## System Cost Table")
text.append("")
text.append(table(
    ["run", "last_step", "wps", "iter_s", "FLOPS", "max_active_gib", "max_reserved_gib", "alloc_retries", "all_to_all_bytes", "active_experts", "max_load", "min_load", "expert6_load", "expert7_load", "load_imbalance"],
    system_rows,
))
text.append("")
summary_out.write_text("\n".join(text) + "\n")
print(f"Wrote {summary_out}")
PY
}

if [[ "${1:-}" == "__train_all_inner" ]]; then
  prepare_heldout
  for target in $TARGETS; do
    ACTION=train MODE=run VARIANT="$target" "$ROOT_DIR/scripts/patchmoe/run_entropy_phaseb_paper_experiments.sh"
  done
  exit 0
fi

case "$ACTION" in
  list)
    usage
    ;;
  preflight)
    preflight
    ;;
  train)
    if [[ -z "$VARIANT" ]]; then
      echo "ACTION=train requires VARIANT" >&2
      usage >&2
      exit 2
    fi
    preflight
    run_train_variant "$VARIANT"
    ;;
  train-all)
    preflight
    run_train_all
    ;;
  eval-phaseb50k)
    CKPT_DIR="$PHASEB_TARGET_CKPT" RUN_DIR="$PHASEB_TARGET_RUN_DIR" LABEL="$EVAL_LABEL" STEP="$EVAL_STEP" run_eval_action
    ;;
  eval)
    run_eval_action
    ;;
  summarize)
    summarize
    ;;
  *)
    echo "Unknown ACTION: $ACTION" >&2
    usage >&2
    exit 2
    ;;
esac
