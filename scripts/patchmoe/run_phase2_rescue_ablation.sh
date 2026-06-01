#!/usr/bin/env bash
set -euo pipefail

# Run the Phase-2 rescue ablation after the first Stage-1 matched-budget result.
#
# Goal:
#   1. Keep the same 100k-step/small held-out eval protocol.
#   2. Move the main line to byte-count balancing.
#   3. Re-test router side features under byte balancing.
#   4. Re-test entropy_byte balancing only at weaker loss weights.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

UV_BIN="${UV_BIN:-/home/ubuntu/.local/bin/uv}"
CONFIG="${CONFIG:-apps/main/configs/patchmoe_stage1_candidate.yaml}"
OUT_ROOT="${OUT_ROOT:-$ROOT_DIR/runs/phase2_rescue_ablation}"
DATA_ROOT="${DATA_ROOT:-$ROOT_DIR/data}"
SOURCE="${SOURCE:-fineweb_edu_10bt}"
PREPROCESS_DIR="${PREPROCESS_DIR:-$ROOT_DIR/data/entropy_preprocessed_stage1}"
ENTROPY_MODEL_NAME="${ENTROPY_MODEL_NAME:-transformer_100m}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/tmp/unused.tokenizer.model}"

STEPS="${STEPS:-100000}"
MAX_STEPS="${MAX_STEPS:-$STEPS}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
BATCH_SIZE="${BATCH_SIZE:-1}"
SEQ_LEN="${SEQ_LEN:-2048}"
GRAD_ACC_STEPS="${GRAD_ACC_STEPS:-1}"
LOAD_ASYNC="${LOAD_ASYNC:-false}"
LOG_FREQ="${LOG_FREQ:-10}"
ENABLE_INTRA_NODE_COMM="${ENABLE_INTRA_NODE_COMM:-1}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-10000}"
CHECKPOINT_KEEP="${CHECKPOINT_KEEP:-1}"
REMOVE_EVAL_CONSOLIDATED="${REMOVE_EVAL_CONSOLIDATED:-1}"

HELDOUT_SOURCE_FILE="${HELDOUT_SOURCE_FILE:-$DATA_ROOT/$SOURCE/$SOURCE.chunk.00002.jsonl}"
HELDOUT_LINES="${HELDOUT_LINES:-50000}"
HELDOUT_SOURCE_NAME="${HELDOUT_SOURCE_NAME:-${SOURCE}_heldout}"
HELDOUT_RAW_DIR="${HELDOUT_RAW_DIR:-$DATA_ROOT/stage1_heldout/$HELDOUT_SOURCE_NAME}"
HELDOUT_RAW_FILE="${HELDOUT_RAW_FILE:-$HELDOUT_RAW_DIR/$HELDOUT_SOURCE_NAME.chunk.00002.first_${HELDOUT_LINES}.jsonl}"
HELDOUT_PREPROCESS_ROOT="${HELDOUT_PREPROCESS_ROOT:-$ROOT_DIR/data/entropy_preprocessed_stage1_heldout}"
HELDOUT_ARROW_DIR="$HELDOUT_PREPROCESS_ROOT/$HELDOUT_SOURCE_NAME/$ENTROPY_MODEL_NAME"
HELDOUT_ARROW_NAME="$(basename "$HELDOUT_RAW_FILE").shard_00.arrow"
HELDOUT_ARROW_FILE="$HELDOUT_ARROW_DIR/$HELDOUT_ARROW_NAME"

EVAL_ROOT="${EVAL_ROOT:-$ROOT_DIR/runs/phase2_rescue_heldout_eval}"
EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-2000}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
EVAL_CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES:-0}"

VARIANTS="${VARIANTS:-byte_hidden_only_w005 byte_length_w005 byte_entropy_w005 byte_entropy_length_w005 entropy_byte_entropy_length_w0005 entropy_byte_entropy_length_w001 entropy_byte_entropy_length_w002}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"
FORCE_EVAL="${FORCE_EVAL:-0}"

# Reuse previous matched-budget runs when they are exactly equivalent. This
# keeps the rescue sweep focused on missing variants instead of re-spending GPU
# days on already completed controls.
REUSE_STAGE1_MATCHED="${REUSE_STAGE1_MATCHED:-1}"
DENSE_REF_RUN_DIR="${DENSE_REF_RUN_DIR:-$ROOT_DIR/runs/stage1_matched_budget/stage1_dense_matched}"
BYTE_ENTROPY_LENGTH_REF_RUN_DIR="${BYTE_ENTROPY_LENGTH_REF_RUN_DIR:-$ROOT_DIR/runs/stage1_matched_budget/stage1_no_entropy_byte_balance_matched}"

export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export BLT_ALLOW_MISSING_FLEX_ATTENTION="${BLT_ALLOW_MISSING_FLEX_ATTENTION:-1}"
export BLT_SUPPRESS_ATTN_ERROR="${BLT_SUPPRESS_ATTN_ERROR:-1}"

final_step_dir="$(printf "%010d" "$STEPS")"
preprocessed_source_dir="$PREPROCESS_DIR/$SOURCE/$ENTROPY_MODEL_NAME"

if [[ ! -d "$preprocessed_source_dir" ]]; then
  echo "Missing training entropy-preprocessed data: $preprocessed_source_dir" >&2
  exit 1
fi
if ! find "$preprocessed_source_dir" -name '*.arrow.complete' -print -quit | grep -q .; then
  echo "No completed training entropy arrow shards found in: $preprocessed_source_dir" >&2
  exit 1
fi
if [[ ! -f "$HELDOUT_SOURCE_FILE" ]]; then
  echo "Missing held-out source file: $HELDOUT_SOURCE_FILE" >&2
  exit 1
fi

variant_overrides() {
  local variant="$1"
  case "$variant" in
    byte_hidden_only_w005)
      printf '%s\n' \
        model.moe_num_experts=8 \
        model.moe_top_k=2 \
        model.moe_balance_loss_weight=0.05 \
        model.moe_router_congestion_weight=0.0 \
        model.moe_router_z_loss_weight=0.0 \
        model.moe_router_use_patch_length=false \
        model.moe_router_use_patch_entropy=false \
        model.moe_router_use_patch_byte_features=false \
        model.moe_balance_cost=byte
      ;;
    byte_length_w005)
      printf '%s\n' \
        model.moe_num_experts=8 \
        model.moe_top_k=2 \
        model.moe_balance_loss_weight=0.05 \
        model.moe_router_congestion_weight=0.0 \
        model.moe_router_z_loss_weight=0.0 \
        model.moe_router_use_patch_length=true \
        model.moe_router_use_patch_entropy=false \
        model.moe_router_use_patch_byte_features=false \
        model.moe_balance_cost=byte
      ;;
    byte_entropy_w005)
      printf '%s\n' \
        model.moe_num_experts=8 \
        model.moe_top_k=2 \
        model.moe_balance_loss_weight=0.05 \
        model.moe_router_congestion_weight=0.0 \
        model.moe_router_z_loss_weight=0.0 \
        model.moe_router_use_patch_length=false \
        model.moe_router_use_patch_entropy=true \
        model.moe_router_use_patch_byte_features=false \
        model.moe_balance_cost=byte
      ;;
    byte_entropy_length_w005)
      printf '%s\n' \
        model.moe_num_experts=8 \
        model.moe_top_k=2 \
        model.moe_balance_loss_weight=0.05 \
        model.moe_router_congestion_weight=0.0 \
        model.moe_router_z_loss_weight=0.0 \
        model.moe_router_use_patch_length=true \
        model.moe_router_use_patch_entropy=true \
        model.moe_router_use_patch_byte_features=false \
        model.moe_balance_cost=byte
      ;;
    entropy_byte_entropy_length_w0005)
      printf '%s\n' \
        model.moe_num_experts=8 \
        model.moe_top_k=2 \
        model.moe_balance_loss_weight=0.005 \
        model.moe_router_congestion_weight=0.0 \
        model.moe_router_z_loss_weight=0.0 \
        model.moe_router_use_patch_length=true \
        model.moe_router_use_patch_entropy=true \
        model.moe_router_use_patch_byte_features=false \
        model.moe_balance_cost=entropy_byte
      ;;
    entropy_byte_entropy_length_w001)
      printf '%s\n' \
        model.moe_num_experts=8 \
        model.moe_top_k=2 \
        model.moe_balance_loss_weight=0.01 \
        model.moe_router_congestion_weight=0.0 \
        model.moe_router_z_loss_weight=0.0 \
        model.moe_router_use_patch_length=true \
        model.moe_router_use_patch_entropy=true \
        model.moe_router_use_patch_byte_features=false \
        model.moe_balance_cost=entropy_byte
      ;;
    entropy_byte_entropy_length_w002)
      printf '%s\n' \
        model.moe_num_experts=8 \
        model.moe_top_k=2 \
        model.moe_balance_loss_weight=0.02 \
        model.moe_router_congestion_weight=0.0 \
        model.moe_router_z_loss_weight=0.0 \
        model.moe_router_use_patch_length=true \
        model.moe_router_use_patch_entropy=true \
        model.moe_router_use_patch_byte_features=false \
        model.moe_balance_cost=entropy_byte
      ;;
    *)
      echo "Unknown Phase-2 rescue variant: $variant" >&2
      echo "Known variants: byte_hidden_only_w005 byte_length_w005 byte_entropy_w005 byte_entropy_length_w005 entropy_byte_entropy_length_w0005 entropy_byte_entropy_length_w001 entropy_byte_entropy_length_w002" >&2
      return 1
      ;;
  esac
}

prepare_heldout() {
  mkdir -p "$HELDOUT_RAW_DIR"
  if [[ ! -f "$HELDOUT_RAW_FILE" ]]; then
    echo "Creating held-out JSONL subset: $HELDOUT_RAW_FILE"
    tmp_file="$HELDOUT_RAW_FILE.tmp"
    head -n "$HELDOUT_LINES" "$HELDOUT_SOURCE_FILE" >"$tmp_file"
    mv "$tmp_file" "$HELDOUT_RAW_FILE"
  else
    echo "Using existing held-out JSONL subset: $HELDOUT_RAW_FILE"
  fi

  if [[ ! -f "$HELDOUT_ARROW_FILE.complete" ]]; then
    echo "Preprocessing held-out entropy shard: $HELDOUT_ARROW_FILE"
    SOURCE_DIR="$HELDOUT_RAW_DIR" \
    SOURCE_NAME="$HELDOUT_SOURCE_NAME" \
    OUTPUT_ROOT="$HELDOUT_PREPROCESS_ROOT" \
    ENTROPY_MODEL_NAME="$ENTROPY_MODEL_NAME" \
    NPROC=1 \
    MAX_FILES=1 \
      scripts/patchmoe/preprocess_fineweb_entropy_stage1.sh
  else
    echo "Using existing held-out entropy shard: $HELDOUT_ARROW_FILE"
  fi
}

run_eval() {
  local label="$1"
  local run_dir="$2"
  local ckpt_dir="$3"
  local step="$4"
  local eval_dir="$EVAL_ROOT/$label/$step"
  local global_step="$step"
  if [[ "$global_step" =~ ^[0-9]+$ ]]; then
    global_step=$((10#$global_step))
  fi

  if [[ ! -d "$ckpt_dir" ]]; then
    echo "Missing checkpoint for eval: $ckpt_dir" >&2
    return 1
  fi
  if [[ "$FORCE_EVAL" != "1" && -f "$eval_dir/validation.json" ]]; then
    echo "Skipping completed eval: $eval_dir/validation.json"
    return 0
  fi

  mkdir -p "$eval_dir"
  echo "Evaluating $label checkpoint $step on held-out shard"
  CUDA_VISIBLE_DEVICES="$EVAL_CUDA_VISIBLE_DEVICES" "$UV_BIN" run python -m bytelatent.eval \
    "ckpt_dir=$ckpt_dir" \
    "dump_dir=$eval_dir" \
    "metric_log_dir=$run_dir" \
    "global_step=$global_step" \
    "consolidate_if_needed=true" \
    "run_ppl=true" \
    "run_tasks=false" \
    "validation.use_val_from_train_src=false" \
    "validation.root_dir=$HELDOUT_ARROW_DIR" \
    "validation.sources=[$HELDOUT_ARROW_NAME]" \
    "validation.batch_size=$EVAL_BATCH_SIZE" \
    "validation.max_n_batches=$EVAL_MAX_BATCHES"

  if [[ "$REMOVE_EVAL_CONSOLIDATED" == "1" ]]; then
    rm -rf "$ckpt_dir/consolidated"
  fi
}

train_variant() {
  local variant="$1"
  local run_name="phase2_rescue_${variant}"
  local run_dir="$OUT_ROOT/$run_name"
  local ckpt_dir="$run_dir/checkpoints/$final_step_dir"

  if [[ "$variant" == "byte_entropy_length_w005" && "$REUSE_STAGE1_MATCHED" == "1" ]]; then
    local ref_ckpt_dir="$BYTE_ENTROPY_LENGTH_REF_RUN_DIR/checkpoints/$final_step_dir"
    if [[ -d "$ref_ckpt_dir" ]]; then
      echo "Reusing completed byte+entropy+length reference: $BYTE_ENTROPY_LENGTH_REF_RUN_DIR"
      run_eval "$run_name" "$BYTE_ENTROPY_LENGTH_REF_RUN_DIR" "$ref_ckpt_dir" "$final_step_dir"
      return 0
    fi
  fi

  mkdir -p "$run_dir"
  if [[ "$FORCE_TRAIN" != "1" && -d "$ckpt_dir" ]]; then
    echo "Skipping completed training run: $run_name"
    run_eval "$run_name" "$run_dir" "$ckpt_dir" "$final_step_dir"
    return 0
  fi

  mapfile -t overrides < <(variant_overrides "$variant")
  echo "Training Phase-2 rescue variant: $variant"
  "$UV_BIN" run torchrun --standalone --nproc-per-node="$NPROC_PER_NODE" \
    -m bytelatent.train \
    "config=$CONFIG" \
    "dump_dir=$run_dir" \
    "name=$run_name" \
    "steps=$STEPS" \
    "max_steps=$MAX_STEPS" \
    "grad_acc_steps=$GRAD_ACC_STEPS" \
    "data.root_dir=$DATA_ROOT" \
    "data.sources={$SOURCE: 1.0}" \
    "data.batch_size=$BATCH_SIZE" \
    "data.seq_len=$SEQ_LEN" \
    "data.load_async=$LOAD_ASYNC" \
    "logging.freq=$LOG_FREQ" \
    "data.preprocess_dir=$PREPROCESS_DIR" \
    "data.entropy_model_name=$ENTROPY_MODEL_NAME" \
    "data.tokenizer_args.init_kwargs.bpe_tokenizer_path=$TOKENIZER_PATH" \
    "checkpoint.path=$run_dir/checkpoints" \
    "checkpoint.dump.every=$CHECKPOINT_EVERY" \
    "checkpoint.dump.keep=$CHECKPOINT_KEEP" \
    "checkpoint.eval.every=$CHECKPOINT_EVERY" \
    "checkpoint.eval.keep=$CHECKPOINT_KEEP" \
    "distributed.dp_shard=$NPROC_PER_NODE" \
    "distributed.dp_replicate=1" \
    "eval_on_gpus=$NPROC_PER_NODE" \
    "env.ENABLE_INTRA_NODE_COMM=\"$ENABLE_INTRA_NODE_COMM\"" \
    "env.NCCL_DEBUG=WARN" \
    "${overrides[@]}"

  run_eval "$run_name" "$run_dir" "$ckpt_dir" "$final_step_dir"
}

prepare_heldout

if [[ "$REUSE_STAGE1_MATCHED" == "1" ]]; then
  dense_ckpt_dir="$DENSE_REF_RUN_DIR/checkpoints/$final_step_dir"
  if [[ -d "$dense_ckpt_dir" ]]; then
    run_eval "phase2_rescue_dense_ref" "$DENSE_REF_RUN_DIR" "$dense_ckpt_dir" "$final_step_dir"
  else
    echo "Dense reference checkpoint not found, skipping: $dense_ckpt_dir"
  fi
fi

for variant in $VARIANTS; do
  train_variant "$variant"
done

echo "Phase-2 rescue ablation pipeline complete."
echo "Eval root: $EVAL_ROOT"
echo "Run root:  $OUT_ROOT"
