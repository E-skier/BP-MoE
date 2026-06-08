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
EP_SIZE="${EP_SIZE:-$NPROC_PER_NODE}"
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
ANALYSIS_ROOT="${ANALYSIS_ROOT:-$ROOT_DIR/runs/stage1_matched_budget_analysis}"
EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-2000}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
EVAL_CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES:-0}"

VARIANTS="${VARIANTS:-dense byte_hidden_only_w005 byte_entropy_w005 byte_type_w005 byte_entropy_type_w005}"
KNOWN_VARIANTS="dense byte_hidden_only_w005 byte_entropy_w005 byte_type_w005 byte_entropy_type_w005 byte_entropy_type_congestion_w05 byte_entropy_type_congestion_zloss_w0001 byte_entropy_assignment_w005 byte_entropy_assignment_congestion_zloss_w005 byte_entropy_type_assignment_congestion_zloss_w005 byte_entropy_length_type_w005 no_entropy_router no_entropy_byte_balance"
FORCE_TRAIN="${FORCE_TRAIN:-0}"
FORCE_EVAL="${FORCE_EVAL:-0}"

if (( EP_SIZE <= 0 || NPROC_PER_NODE % EP_SIZE != 0 )); then
  echo "EP_SIZE must be positive and divide NPROC_PER_NODE" >&2
  exit 2
fi
if (( 8 % EP_SIZE != 0 )); then
  echo "EP_SIZE must divide the configured 8 experts" >&2
  exit 2
fi

export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export BLT_ALLOW_MISSING_FLEX_ATTENTION="${BLT_ALLOW_MISSING_FLEX_ATTENTION:-1}"
export BLT_SUPPRESS_ATTN_ERROR="${BLT_SUPPRESS_ATTN_ERROR:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

final_step_dir="$(printf "%010d" "$STEPS")"
preprocessed_source_dir="$PREPROCESS_DIR/$SOURCE/$ENTROPY_MODEL_NAME"

if [[ "${1:-}" == "--list" ]]; then
  printf "%s\n" $KNOWN_VARIANTS
  exit 0
fi

variant_overrides() {
  local variant="$1"
  case "$variant" in
    dense)
      printf "%s\n" \
        model.moe_num_experts=0 \
        model.moe_top_k=1 \
        model.moe_balance_loss_weight=0.0 \
        model.moe_router_congestion_weight=0.0 \
        model.moe_router_z_loss_weight=0.0 \
        model.moe_router_jitter=0.0 \
        model.moe_router_use_patch_length=false \
        model.moe_router_use_patch_entropy=false \
        model.moe_router_use_patch_byte_features=false \
        model.moe_balance_cost=patch
      ;;
    byte_hidden_only_w005)
      printf "%s\n" \
        model.moe_num_experts=8 \
        model.moe_top_k=2 \
        model.moe_balance_loss_weight=0.05 \
        model.moe_router_congestion_weight=0.0 \
        model.moe_router_z_loss_weight=0.0 \
        model.moe_router_jitter=0.01 \
        model.moe_router_use_patch_length=false \
        model.moe_router_use_patch_entropy=false \
        model.moe_router_use_patch_byte_features=false \
        model.moe_balance_cost=byte
      ;;
    byte_entropy_w005)
      printf "%s\n" \
        model.moe_num_experts=8 \
        model.moe_top_k=2 \
        model.moe_balance_loss_weight=0.05 \
        model.moe_router_congestion_weight=0.0 \
        model.moe_router_z_loss_weight=0.0 \
        model.moe_router_jitter=0.01 \
        model.moe_router_use_patch_length=false \
        model.moe_router_use_patch_entropy=true \
        model.moe_router_use_patch_byte_features=false \
        model.moe_balance_cost=byte
      ;;
    byte_type_w005)
      printf "%s\n" \
        model.moe_num_experts=8 \
        model.moe_top_k=2 \
        model.moe_balance_loss_weight=0.05 \
        model.moe_router_congestion_weight=0.0 \
        model.moe_router_z_loss_weight=0.0 \
        model.moe_router_jitter=0.01 \
        model.moe_router_use_patch_length=false \
        model.moe_router_use_patch_entropy=false \
        model.moe_router_use_patch_byte_features=true \
        model.moe_balance_cost=byte
      ;;
    byte_entropy_type_w005)
      printf "%s\n" \
        model.moe_num_experts=8 \
        model.moe_top_k=2 \
        model.moe_balance_loss_weight=0.05 \
        model.moe_router_congestion_weight=0.0 \
        model.moe_router_z_loss_weight=0.0 \
        model.moe_router_jitter=0.01 \
        model.moe_router_use_patch_length=false \
        model.moe_router_use_patch_entropy=true \
        model.moe_router_use_patch_byte_features=true \
        model.moe_balance_cost=byte
      ;;
    byte_entropy_type_congestion_w05)
      printf "%s\n" \
        model.moe_num_experts=8 \
        model.moe_top_k=2 \
        model.moe_balance_loss_weight=0.05 \
        model.moe_router_congestion_weight=0.5 \
        model.moe_router_z_loss_weight=0.0 \
        model.moe_router_jitter=0.01 \
        model.moe_router_use_patch_length=false \
        model.moe_router_use_patch_entropy=true \
        model.moe_router_use_patch_byte_features=true \
        model.moe_balance_cost=byte
      ;;
    byte_entropy_type_congestion_zloss_w0001)
      printf "%s\n" \
        model.moe_num_experts=8 \
        model.moe_top_k=2 \
        model.moe_balance_loss_weight=0.05 \
        model.moe_router_congestion_weight=0.5 \
        model.moe_router_z_loss_weight=0.001 \
        model.moe_router_jitter=0.01 \
        model.moe_router_use_patch_length=false \
        model.moe_router_use_patch_entropy=true \
        model.moe_router_use_patch_byte_features=true \
        model.moe_balance_cost=byte
      ;;
    byte_entropy_assignment_w005)
      printf "%s\n" \
        model.moe_num_experts=8 \
        model.moe_top_k=2 \
        model.moe_balance_loss_weight=0.05 \
        model.moe_assignment_balance_loss_weight=0.05 \
        model.moe_router_congestion_weight=0.0 \
        model.moe_router_z_loss_weight=0.0 \
        model.moe_router_jitter=0.01 \
        model.moe_router_use_patch_length=false \
        model.moe_router_use_patch_entropy=true \
        model.moe_router_use_patch_byte_features=false \
        model.moe_balance_cost=byte
      ;;
    byte_entropy_assignment_congestion_zloss_w005)
      printf "%s\n" \
        model.moe_num_experts=8 \
        model.moe_top_k=2 \
        model.moe_balance_loss_weight=0.05 \
        model.moe_assignment_balance_loss_weight=0.05 \
        model.moe_router_congestion_weight=0.5 \
        model.moe_router_z_loss_weight=0.001 \
        model.moe_router_jitter=0.01 \
        model.moe_router_use_patch_length=false \
        model.moe_router_use_patch_entropy=true \
        model.moe_router_use_patch_byte_features=false \
        model.moe_balance_cost=byte
      ;;
    byte_entropy_type_assignment_congestion_zloss_w005)
      printf "%s\n" \
        model.moe_num_experts=8 \
        model.moe_top_k=2 \
        model.moe_balance_loss_weight=0.05 \
        model.moe_assignment_balance_loss_weight=0.05 \
        model.moe_router_congestion_weight=0.5 \
        model.moe_router_z_loss_weight=0.001 \
        model.moe_router_jitter=0.01 \
        model.moe_router_use_patch_length=false \
        model.moe_router_use_patch_entropy=true \
        model.moe_router_use_patch_byte_features=true \
        model.moe_balance_cost=byte
      ;;
    byte_entropy_length_type_w005)
      printf "%s\n" \
        model.moe_num_experts=8 \
        model.moe_top_k=2 \
        model.moe_balance_loss_weight=0.05 \
        model.moe_router_congestion_weight=0.0 \
        model.moe_router_z_loss_weight=0.0 \
        model.moe_router_jitter=0.01 \
        model.moe_router_use_patch_length=true \
        model.moe_router_use_patch_entropy=true \
        model.moe_router_use_patch_byte_features=true \
        model.moe_balance_cost=byte
      ;;
    no_entropy_router)
      printf "%s\n" \
        model.moe_num_experts=8 \
        model.moe_top_k=2 \
        model.moe_balance_loss_weight=0.05 \
        model.moe_router_congestion_weight=0.0 \
        model.moe_router_z_loss_weight=0.0 \
        model.moe_router_jitter=0.01 \
        model.moe_router_use_patch_length=true \
        model.moe_router_use_patch_entropy=false \
        model.moe_router_use_patch_byte_features=false \
        model.moe_balance_cost=entropy_byte
      ;;
    no_entropy_byte_balance)
      printf "%s\n" \
        model.moe_num_experts=8 \
        model.moe_top_k=2 \
        model.moe_balance_loss_weight=0.05 \
        model.moe_router_congestion_weight=0.0 \
        model.moe_router_z_loss_weight=0.0 \
        model.moe_router_jitter=0.01 \
        model.moe_router_use_patch_length=true \
        model.moe_router_use_patch_entropy=true \
        model.moe_router_use_patch_byte_features=false \
        model.moe_balance_cost=byte
      ;;
    *)
      echo "Unknown Stage-1 matched-budget variant: $variant" >&2
      echo "Known variants: $KNOWN_VARIANTS" >&2
      return 1
      ;;
  esac
}

if [[ "${1:-}" == "--print-overrides" ]]; then
  if [[ "${2:-}" == "" || "${3:-}" != "" ]]; then
    echo "usage: $0 --print-overrides VARIANT" >&2
    exit 2
  fi
  variant_overrides "$2"
  exit 0
fi

preflight() {
  local failures=0
  local shard_count=0
  local candidate_ckpt="$CANDIDATE_RUN_DIR/checkpoints/$final_step_dir"

  echo "Stage-1 matched-budget preflight"
  echo "Config:        $CONFIG"
  echo "Train data:    $preprocessed_source_dir"
  echo "Held-out src:  $HELDOUT_SOURCE_FILE"
  echo "Candidate ckpt:$candidate_ckpt"
  echo "Variants:      $VARIANTS"
  echo "Run root:      $OUT_ROOT"
  echo "Eval root:     $EVAL_ROOT"
  echo "Analysis root: $ANALYSIS_ROOT"

  if [[ -f "$CONFIG" ]]; then
    echo "OK   config exists"
  else
    echo "FAIL missing config: $CONFIG" >&2
    failures=1
  fi

  if [[ -d "$preprocessed_source_dir" ]]; then
    shard_count="$(find "$preprocessed_source_dir" -name "*.arrow.complete" -print | wc -l | tr -d ' ')"
    if [[ "$shard_count" != "0" ]]; then
      echo "OK   training entropy shards complete: $shard_count"
    else
      echo "FAIL no completed training entropy shards in: $preprocessed_source_dir" >&2
      failures=1
    fi
  else
    echo "FAIL missing training entropy-preprocessed directory: $preprocessed_source_dir" >&2
    failures=1
  fi

  if [[ -f "$HELDOUT_SOURCE_FILE" ]]; then
    echo "OK   held-out source exists"
  else
    echo "FAIL missing held-out source file: $HELDOUT_SOURCE_FILE" >&2
    failures=1
  fi

  if [[ -d "$candidate_ckpt" ]]; then
    echo "OK   candidate checkpoint exists"
  else
    echo "FAIL missing candidate checkpoint: $candidate_ckpt" >&2
    failures=1
  fi

  for variant in $VARIANTS; do
    if ! variant_overrides "$variant" >/dev/null; then
      failures=1
      continue
    fi
    local run_name="stage1_${variant}_matched"
    local run_dir="$OUT_ROOT/$run_name"
    local ckpt_dir="$run_dir/checkpoints/$final_step_dir"
    local eval_file="$EVAL_ROOT/$run_name/$final_step_dir/validation.json"
    local metrics_file="$run_dir/metrics.jsonl"

    if [[ -d "$ckpt_dir" ]]; then
      echo "DONE checkpoint $variant: $ckpt_dir"
    elif [[ -f "$metrics_file" ]]; then
      local last_step
      last_step="$(tail -n 1 "$metrics_file" | sed -n 's/.*"global_step": *\([0-9][0-9]*\).*/\1/p')"
      echo "PARTIAL train $variant: metrics present, last_step=${last_step:-unknown}"
    else
      echo "TODO train $variant: $run_dir"
    fi

    if [[ -f "$eval_file" ]]; then
      echo "DONE eval $variant: $eval_file"
    else
      echo "TODO eval $variant: $eval_file"
    fi
  done

  return "$failures"
}

if [[ "${1:-}" == "--preflight" ]]; then
  preflight
  exit $?
fi

if [[ "${1:-}" != "" ]]; then
  echo "usage: $0 [--list|--preflight|--print-overrides VARIANT]" >&2
  exit 2
fi

if [[ ! -d "$preprocessed_source_dir" ]]; then
  echo "Missing training entropy-preprocessed data: $preprocessed_source_dir" >&2
  exit 1
fi
if ! find "$preprocessed_source_dir" -name "*.arrow.complete" -print -quit | grep -q .; then
  echo "No completed training entropy arrow shards found in: $preprocessed_source_dir" >&2
  exit 1
fi
if [[ ! -f "$HELDOUT_SOURCE_FILE" ]]; then
  echo "Missing held-out source file: $HELDOUT_SOURCE_FILE" >&2
  exit 1
fi

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

analyze_runs() {
  mkdir -p "$ANALYSIS_ROOT"
  "$UV_BIN" run python -m bytelatent.plotting.patchmoe_phase2_ablation \
    "$OUT_ROOT" \
    "$ANALYSIS_ROOT" \
    --eval-root "$EVAL_ROOT"
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
    "model.moe_ep_size=$EP_SIZE" \
    "checkpoint.path=$run_dir/checkpoints" \
    "distributed.dp_shard=$NPROC_PER_NODE" \
    "distributed.dp_replicate=1" \
    "eval_on_gpus=$NPROC_PER_NODE" \
    "optim.fused=false" \
    "optim.clip=0" \
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

analyze_runs

echo "Stage-1 matched-budget pipeline complete."
echo "Eval root:     $EVAL_ROOT"
echo "Run root:      $OUT_ROOT"
echo "Analysis root: $ANALYSIS_ROOT"
