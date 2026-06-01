#!/usr/bin/env bash
set -euo pipefail

# Run 200k-step matched controls for the current best PatchMoE setting:
#   byte-cost balancing + entropy-only side-feature routing.
#
# Defaults keep completed references and add the decisive byte-type controls:
#   dense                   no MoE
#   byte_hidden_only_w005   MoE + byte balance, no side features
#   byte_entropy_w005       MoE + byte balance, entropy side feature
#   byte_type_w005          MoE + byte balance, byte-pattern side features
#   byte_entropy_type_w005  MoE + byte balance, entropy + byte-pattern features
#
# The held-out eval protocol matches run_byte_entropy_repro_scale.sh.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

UV_BIN="${UV_BIN:-/home/ubuntu/.local/bin/uv}"
CONFIG="${CONFIG:-apps/main/configs/patchmoe_stage1_candidate.yaml}"
OUT_ROOT="${OUT_ROOT:-$ROOT_DIR/runs/byte_entropy_200k_matched_controls}"
EVAL_ROOT="${EVAL_ROOT:-$ROOT_DIR/runs/byte_entropy_200k_matched_controls_heldout_eval}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-$ROOT_DIR/runs/byte_entropy_200k_matched_controls_analysis}"

DATA_ROOT="${DATA_ROOT:-$ROOT_DIR/data}"
SOURCE="${SOURCE:-fineweb_edu_10bt}"
PREPROCESS_DIR="${PREPROCESS_DIR:-$ROOT_DIR/data/entropy_preprocessed_stage1}"
ENTROPY_MODEL_NAME="${ENTROPY_MODEL_NAME:-transformer_100m}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/tmp/unused.tokenizer.model}"

SEEDS="${SEEDS:-779}"
VARIANTS="${VARIANTS:-dense byte_hidden_only_w005 byte_entropy_w005 byte_type_w005 byte_entropy_type_w005}"
KNOWN_VARIANTS="dense dense_compute_matched byte_hidden_only_w005 byte_length_w005 byte_entropy_w005 byte_type_w005 byte_entropy_type_w005 byte_entropy_type_congestion_w05 byte_entropy_type_congestion_zloss_w0001 byte_entropy_length_w005 byte_entropy_length_type_w005 entropy_byte_entropy_length_w0005 entropy_byte_entropy_length_w001"

STEPS="${STEPS:-200000}"
MAX_STEPS="${MAX_STEPS:-$STEPS}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
BATCH_SIZE="${BATCH_SIZE:-1}"
SEQ_LEN="${SEQ_LEN:-2048}"
GRAD_ACC_STEPS="${GRAD_ACC_STEPS:-1}"
LOAD_ASYNC="${LOAD_ASYNC:-false}"
LOG_FREQ="${LOG_FREQ:-10}"
ENABLE_INTRA_NODE_COMM="${ENABLE_INTRA_NODE_COMM:-1}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-20000}"
CHECKPOINT_KEEP="${CHECKPOINT_KEEP:-1}"
REMOVE_EVAL_CONSOLIDATED="${REMOVE_EVAL_CONSOLIDATED:-1}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"
FORCE_EVAL="${FORCE_EVAL:-0}"

HELDOUT_CHUNKS="${HELDOUT_CHUNKS:-00002 00003}"
HELDOUT_LINES_PER_CHUNK="${HELDOUT_LINES_PER_CHUNK:-100000}"
HELDOUT_SOURCE_NAME="${HELDOUT_SOURCE_NAME:-${SOURCE}_heldout_expanded_${HELDOUT_LINES_PER_CHUNK}}"
HELDOUT_RAW_DIR="${HELDOUT_RAW_DIR:-$DATA_ROOT/stage1_heldout/$HELDOUT_SOURCE_NAME}"
HELDOUT_PREPROCESS_ROOT="${HELDOUT_PREPROCESS_ROOT:-$ROOT_DIR/data/entropy_preprocessed_stage1_heldout_expanded}"
HELDOUT_ARROW_DIR="$HELDOUT_PREPROCESS_ROOT/$HELDOUT_SOURCE_NAME/$ENTROPY_MODEL_NAME"
HELDOUT_PREPROCESS_NPROC="${HELDOUT_PREPROCESS_NPROC:-2}"

EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-8000}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
EVAL_CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES:-0}"

export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export BLT_ALLOW_MISSING_FLEX_ATTENTION="${BLT_ALLOW_MISSING_FLEX_ATTENTION:-1}"
export BLT_SUPPRESS_ATTN_ERROR="${BLT_SUPPRESS_ATTN_ERROR:-1}"

if [[ "${1:-}" == "--list" ]]; then
  printf "%s\n" $KNOWN_VARIANTS
  exit 0
fi

final_step_dir="$(printf "%010d" "$STEPS")"
preprocessed_source_dir="$PREPROCESS_DIR/$SOURCE/$ENTROPY_MODEL_NAME"

declare -a HELDOUT_ARROW_NAMES=()

variant_overrides() {
  local variant="$1"
  case "$variant" in
    dense)
      printf '%s\n' \
        model.moe_num_experts=0 \
        model.moe_top_k=1 \
        model.moe_balance_loss_weight=0.0 \
        model.moe_router_jitter=0.0 \
        model.moe_router_congestion_weight=0.0 \
        model.moe_router_z_loss_weight=0.0 \
        model.moe_router_use_patch_length=false \
        model.moe_router_use_patch_entropy=false \
        model.moe_router_use_patch_byte_features=false \
        model.moe_balance_cost=patch
      ;;
    dense_compute_matched)
      printf '%s\n' \
        model.moe_num_experts=0 \
        model.moe_top_k=1 \
        model.moe_balance_loss_weight=0.0 \
        model.moe_router_jitter=0.0 \
        model.moe_router_congestion_weight=0.0 \
        model.moe_router_z_loss_weight=0.0 \
        model.moe_router_use_patch_length=false \
        model.moe_router_use_patch_entropy=false \
        model.moe_router_use_patch_byte_features=false \
        model.moe_balance_cost=patch \
        model.ffn_dim_multiplier_global=2.0
      ;;
    byte_hidden_only_w005)
      printf '%s\n' \
        model.moe_num_experts=8 \
        model.moe_top_k=2 \
        model.moe_balance_loss_weight=0.05 \
        model.moe_router_jitter=0.01 \
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
        model.moe_router_jitter=0.01 \
        model.moe_router_congestion_weight=0.0 \
        model.moe_router_z_loss_weight=0.0 \
        model.moe_router_use_patch_length=true \
        model.moe_router_use_patch_entropy=false \
        model.moe_router_use_patch_byte_features=false \
        model.moe_balance_cost=byte
      ;;
    byte_entropy_length_w005)
      printf '%s\n' \
        model.moe_num_experts=8 \
        model.moe_top_k=2 \
        model.moe_balance_loss_weight=0.05 \
        model.moe_router_jitter=0.01 \
        model.moe_router_congestion_weight=0.0 \
        model.moe_router_z_loss_weight=0.0 \
        model.moe_router_use_patch_length=true \
        model.moe_router_use_patch_entropy=true \
        model.moe_router_use_patch_byte_features=false \
        model.moe_balance_cost=byte
      ;;
    byte_entropy_w005)
      printf "%s\n" \
        model.moe_num_experts=8 \
        model.moe_top_k=2 \
        model.moe_balance_loss_weight=0.05 \
        model.moe_router_jitter=0.01 \
        model.moe_router_congestion_weight=0.0 \
        model.moe_router_z_loss_weight=0.0 \
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
        model.moe_router_jitter=0.01 \
        model.moe_router_congestion_weight=0.0 \
        model.moe_router_z_loss_weight=0.0 \
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
        model.moe_router_jitter=0.01 \
        model.moe_router_congestion_weight=0.0 \
        model.moe_router_z_loss_weight=0.0 \
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
        model.moe_router_jitter=0.01 \
        model.moe_router_congestion_weight=0.5 \
        model.moe_router_z_loss_weight=0.0 \
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
        model.moe_router_jitter=0.01 \
        model.moe_router_congestion_weight=0.5 \
        model.moe_router_z_loss_weight=0.001 \
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
        model.moe_router_jitter=0.01 \
        model.moe_router_congestion_weight=0.0 \
        model.moe_router_z_loss_weight=0.0 \
        model.moe_router_use_patch_length=true \
        model.moe_router_use_patch_entropy=true \
        model.moe_router_use_patch_byte_features=true \
        model.moe_balance_cost=byte
      ;;
    entropy_byte_entropy_length_w0005)
      printf '%s\n' \
        model.moe_num_experts=8 \
        model.moe_top_k=2 \
        model.moe_balance_loss_weight=0.005 \
        model.moe_router_jitter=0.01 \
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
        model.moe_router_jitter=0.01 \
        model.moe_router_congestion_weight=0.0 \
        model.moe_router_z_loss_weight=0.0 \
        model.moe_router_use_patch_length=true \
        model.moe_router_use_patch_entropy=true \
        model.moe_router_use_patch_byte_features=false \
        model.moe_balance_cost=entropy_byte
      ;;
    *)
      echo "Unknown 200k matched-control variant: $variant" >&2
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

  echo "200k matched-control preflight"
  echo "Config:        $CONFIG"
  echo "Train data:    $preprocessed_source_dir"
  echo "Held-out src:  $DATA_ROOT/$SOURCE chunks: $HELDOUT_CHUNKS"
  echo "Seeds:         $SEEDS"
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

  for chunk in $HELDOUT_CHUNKS; do
    local source_file="$DATA_ROOT/$SOURCE/$SOURCE.chunk.$chunk.jsonl"
    if [[ -f "$source_file" ]]; then
      echo "OK   held-out source chunk exists: $source_file"
    else
      echo "FAIL missing held-out source chunk: $source_file" >&2
      failures=1
    fi
  done

  for seed in $SEEDS; do
    for variant in $VARIANTS; do
      if ! variant_overrides "$variant" >/dev/null; then
        failures=1
        continue
      fi
      local run_name="${variant}_seed${seed}_${STEPS}step"
      local run_dir="$OUT_ROOT/$run_name"
      local ckpt_dir="$run_dir/checkpoints/$final_step_dir"
      local eval_file="$EVAL_ROOT/$run_name/$final_step_dir/validation.json"
      local metrics_file="$run_dir/metrics.jsonl"

      if [[ -d "$ckpt_dir" ]]; then
        echo "DONE checkpoint seed=$seed variant=$variant: $ckpt_dir"
      elif [[ -f "$metrics_file" ]]; then
        local last_step
        last_step="$(tail -n 1 "$metrics_file" | sed -n 's/.*"global_step": *\([0-9][0-9]*\).*/\1/p')"
        echo "PARTIAL train seed=$seed variant=$variant: last_step=${last_step:-unknown}"
      else
        echo "TODO train seed=$seed variant=$variant: $run_dir"
      fi

      if [[ -f "$eval_file" ]]; then
        echo "DONE eval seed=$seed variant=$variant: $eval_file"
      else
        echo "TODO eval seed=$seed variant=$variant: $eval_file"
      fi
    done
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

prepare_heldout() {
  mkdir -p "$HELDOUT_RAW_DIR"
  HELDOUT_ARROW_NAMES=()

  for chunk in $HELDOUT_CHUNKS; do
    local source_file="$DATA_ROOT/$SOURCE/$SOURCE.chunk.$chunk.jsonl"
    local raw_file="$HELDOUT_RAW_DIR/$HELDOUT_SOURCE_NAME.chunk.$chunk.first_${HELDOUT_LINES_PER_CHUNK}.jsonl"

    if [[ ! -f "$source_file" ]]; then
      echo "Missing held-out source chunk: $source_file" >&2
      exit 1
    fi

    if [[ ! -f "$raw_file" ]]; then
      echo "Creating held-out subset: $raw_file"
      local tmp_file="$raw_file.tmp"
      head -n "$HELDOUT_LINES_PER_CHUNK" "$source_file" >"$tmp_file"
      mv "$tmp_file" "$raw_file"
    else
      echo "Using existing held-out subset: $raw_file"
    fi

    HELDOUT_ARROW_NAMES+=("$(basename "$raw_file").shard_00.arrow")
  done

  echo "Preparing expanded held-out entropy shards under: $HELDOUT_ARROW_DIR"
  SOURCE_DIR="$HELDOUT_RAW_DIR" \
  SOURCE_NAME="$HELDOUT_SOURCE_NAME" \
  OUTPUT_ROOT="$HELDOUT_PREPROCESS_ROOT" \
  ENTROPY_MODEL_NAME="$ENTROPY_MODEL_NAME" \
  NPROC="$HELDOUT_PREPROCESS_NPROC" \
  MAX_FILES=all \
    scripts/patchmoe/preprocess_fineweb_entropy_stage1.sh
}

validation_sources_arg() {
  local result="["
  local first=1
  for name in "${HELDOUT_ARROW_NAMES[@]}"; do
    if [[ "$first" -eq 0 ]]; then
      result+=","
    fi
    result+="$name"
    first=0
  done
  result+="]"
  printf '%s' "$result"
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
  echo "Evaluating $label checkpoint $step on expanded held-out shards"
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
    "validation.sources=$(validation_sources_arg)" \
    "validation.batch_size=$EVAL_BATCH_SIZE" \
    "validation.max_n_batches=$EVAL_MAX_BATCHES"

  if [[ "$REMOVE_EVAL_CONSOLIDATED" == "1" ]]; then
    rm -rf "$ckpt_dir/consolidated"
  fi
}

analyze_runs() {
  mkdir -p "$ANALYSIS_ROOT"
  "$UV_BIN" run python -m bytelatent.plotting.patchmoe_phase2_ablation \
    "$OUT_ROOT" \
    "$ANALYSIS_ROOT" \
    --eval-root "$EVAL_ROOT"
}

train_variant_seed() {
  local variant="$1"
  local seed="$2"
  local run_name="${variant}_seed${seed}_${STEPS}step"
  local run_dir="$OUT_ROOT/$run_name"
  local ckpt_dir="$run_dir/checkpoints/$final_step_dir"

  mkdir -p "$run_dir"
  if [[ "$FORCE_TRAIN" != "1" && -d "$ckpt_dir" ]]; then
    echo "Skipping completed training run: $run_name"
    run_eval "$run_name" "$run_dir" "$ckpt_dir" "$final_step_dir"
    return 0
  fi

  mapfile -t overrides < <(variant_overrides "$variant")
  echo "Training 200k matched control: $run_name"
  "$UV_BIN" run torchrun --standalone --nproc-per-node="$NPROC_PER_NODE" \
    -m bytelatent.train \
    "config=$CONFIG" \
    "dump_dir=$run_dir" \
    "name=$run_name" \
    "steps=$STEPS" \
    "max_steps=$MAX_STEPS" \
    "seed=$seed" \
    "model.seed=$seed" \
    "data.seed=$seed" \
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
  analyze_runs
}

prepare_heldout

for seed in $SEEDS; do
  for variant in $VARIANTS; do
    train_variant_seed "$variant" "$seed"
  done
done

analyze_runs

echo "200k matched-control pipeline complete."
echo "Run root:      $OUT_ROOT"
echo "Eval root:     $EVAL_ROOT"
echo "Analysis root: $ANALYSIS_ROOT"
