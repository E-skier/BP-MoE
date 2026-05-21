#!/usr/bin/env bash
set -euo pipefail

# Run the Stage-1 held-out evaluation and matched-step/token comparison set.
#
# The candidate long run used fineweb_edu_10bt chunks 00000 and 00001. This
# script builds a fixed held-out subset from chunk 00002, preprocesses entropy
# arrows for it, evaluates the existing candidate checkpoint, then trains and
# evaluates matched-budget controls.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

UV_BIN="${UV_BIN:-/home/ubuntu/.local/bin/uv}"
CONFIG="${CONFIG:-apps/main/configs/patchmoe_stage1_candidate.yaml}"
OUT_ROOT="${OUT_ROOT:-$ROOT_DIR/runs/stage1_matched_budget}"
DATA_ROOT="${DATA_ROOT:-$ROOT_DIR/data}"
SOURCE="${SOURCE:-fineweb_edu_10bt}"
PREPROCESS_DIR="${PREPROCESS_DIR:-$ROOT_DIR/data/entropy_preprocessed_stage1}"
ENTROPY_MODEL_NAME="${ENTROPY_MODEL_NAME:-transformer_100m}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/tmp/unused.tokenizer.model}"

CANDIDATE_RUN_DIR="${CANDIDATE_RUN_DIR:-$ROOT_DIR/runs/stage1_candidate/patchmoe_stage1_8e_top2_entropy_byte_full}"
CANDIDATE_NAME="${CANDIDATE_NAME:-candidate_entropy_router_entropy_byte}"

STEPS="${STEPS:-100000}"
MAX_STEPS="${MAX_STEPS:-$STEPS}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
BATCH_SIZE="${BATCH_SIZE:-1}"
SEQ_LEN="${SEQ_LEN:-2048}"
GRAD_ACC_STEPS="${GRAD_ACC_STEPS:-1}"
LOAD_ASYNC="${LOAD_ASYNC:-false}"
LOG_FREQ="${LOG_FREQ:-10}"
ENABLE_INTRA_NODE_COMM="${ENABLE_INTRA_NODE_COMM:-1}"

HELDOUT_SOURCE_FILE="${HELDOUT_SOURCE_FILE:-$DATA_ROOT/$SOURCE/$SOURCE.chunk.00002.jsonl}"
HELDOUT_LINES="${HELDOUT_LINES:-50000}"
HELDOUT_SOURCE_NAME="${HELDOUT_SOURCE_NAME:-${SOURCE}_heldout}"
HELDOUT_RAW_DIR="${HELDOUT_RAW_DIR:-$DATA_ROOT/stage1_heldout/$HELDOUT_SOURCE_NAME}"
HELDOUT_RAW_FILE="${HELDOUT_RAW_FILE:-$HELDOUT_RAW_DIR/$HELDOUT_SOURCE_NAME.chunk.00002.first_${HELDOUT_LINES}.jsonl}"
HELDOUT_PREPROCESS_ROOT="${HELDOUT_PREPROCESS_ROOT:-$ROOT_DIR/data/entropy_preprocessed_stage1_heldout}"
HELDOUT_ARROW_DIR="$HELDOUT_PREPROCESS_ROOT/$HELDOUT_SOURCE_NAME/$ENTROPY_MODEL_NAME"
HELDOUT_ARROW_NAME="$(basename "$HELDOUT_RAW_FILE").shard_00.arrow"
HELDOUT_ARROW_FILE="$HELDOUT_ARROW_DIR/$HELDOUT_ARROW_NAME"

EVAL_ROOT="${EVAL_ROOT:-$ROOT_DIR/runs/stage1_heldout_eval}"
EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-2000}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
EVAL_CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES:-0}"

VARIANTS="${VARIANTS-dense no_entropy_router no_entropy_byte_balance}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"
FORCE_EVAL="${FORCE_EVAL:-0}"

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
    dense)
      printf '%s\n' \
        model.moe_num_experts=0 \
        model.moe_top_k=1 \
        model.moe_balance_loss_weight=0.0 \
        model.moe_router_jitter=0.0 \
        model.moe_router_use_patch_length=false \
        model.moe_router_use_patch_entropy=false \
        model.moe_balance_cost=patch
      ;;
    no_entropy_router)
      printf '%s\n' \
        model.moe_num_experts=8 \
        model.moe_top_k=2 \
        model.moe_balance_loss_weight=0.05 \
        model.moe_router_use_patch_length=true \
        model.moe_router_use_patch_entropy=false \
        model.moe_balance_cost=entropy_byte
      ;;
    no_entropy_byte_balance)
      printf '%s\n' \
        model.moe_num_experts=8 \
        model.moe_top_k=2 \
        model.moe_balance_loss_weight=0.05 \
        model.moe_router_use_patch_length=true \
        model.moe_router_use_patch_entropy=true \
        model.moe_balance_cost=byte
      ;;
    *)
      echo "Unknown Stage-1 matched-budget variant: $variant" >&2
      echo "Known variants: dense no_entropy_router no_entropy_byte_balance" >&2
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
}

train_variant() {
  local variant="$1"
  local run_name="stage1_${variant}_matched"
  local run_dir="$OUT_ROOT/$run_name"
  local ckpt_dir="$run_dir/checkpoints/$final_step_dir"

  mkdir -p "$run_dir"
  if [[ "$FORCE_TRAIN" != "1" && -d "$ckpt_dir" ]]; then
    echo "Skipping completed training run: $run_name"
    run_eval "$run_name" "$run_dir" "$ckpt_dir" "$final_step_dir"
    return 0
  fi

  mapfile -t overrides < <(variant_overrides "$variant")
  echo "Training matched-budget variant: $variant"
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
    "distributed.dp_shard=$NPROC_PER_NODE" \
    "distributed.dp_replicate=1" \
    "eval_on_gpus=$NPROC_PER_NODE" \
    "env.ENABLE_INTRA_NODE_COMM=\"$ENABLE_INTRA_NODE_COMM\"" \
    "env.NCCL_DEBUG=WARN" \
    "${overrides[@]}"

  run_eval "$run_name" "$run_dir" "$ckpt_dir" "$final_step_dir"
}

prepare_heldout

candidate_ckpt="$CANDIDATE_RUN_DIR/checkpoints/$final_step_dir"
run_eval "$CANDIDATE_NAME" "$CANDIDATE_RUN_DIR" "$candidate_ckpt" "$final_step_dir"

for variant in $VARIANTS; do
  train_variant "$variant"
done

echo "Stage-1 matched-budget pipeline complete."
echo "Eval root: $EVAL_ROOT"
echo "Run root:  $OUT_ROOT"
