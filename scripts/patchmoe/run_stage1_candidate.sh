#!/usr/bin/env bash
set -euo pipefail

# Launch the Stage-1 PatchMoE candidate selected from the Phase-2 ablation.
# This expects entropy-preprocessed data under:
#   $PREPROCESS_DIR/$SOURCE/$ENTROPY_MODEL_NAME

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

UV_BIN="${UV_BIN:-/home/ubuntu/.local/bin/uv}"
CONFIG="${CONFIG:-apps/main/configs/patchmoe_stage1_candidate.yaml}"
OUT_ROOT="${OUT_ROOT:-$ROOT_DIR/runs/stage1_candidate}"
RUN_NAME="${RUN_NAME:-patchmoe_stage1_8e_top2_entropy_byte}"
RUN_DIR="${RUN_DIR:-$OUT_ROOT/$RUN_NAME}"

NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
STEPS="${STEPS:-100000}"
MAX_STEPS="${MAX_STEPS:-$STEPS}"
BATCH_SIZE="${BATCH_SIZE:-1}"
SEQ_LEN="${SEQ_LEN:-2048}"
GRAD_ACC_STEPS="${GRAD_ACC_STEPS:-1}"
LOAD_ASYNC="${LOAD_ASYNC:-false}"
LOG_FREQ="${LOG_FREQ:-10}"
ENABLE_INTRA_NODE_COMM="${ENABLE_INTRA_NODE_COMM:-1}"

DATA_ROOT="${DATA_ROOT:-$ROOT_DIR/data}"
SOURCE="${SOURCE:-fineweb_edu_10bt}"
PREPROCESS_DIR="${PREPROCESS_DIR:-$ROOT_DIR/data/entropy_preprocessed_stage1}"
ENTROPY_MODEL_NAME="${ENTROPY_MODEL_NAME:-transformer_100m}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/tmp/unused.tokenizer.model}"

export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export BLT_ALLOW_MISSING_FLEX_ATTENTION="${BLT_ALLOW_MISSING_FLEX_ATTENTION:-1}"
export BLT_SUPPRESS_ATTN_ERROR="${BLT_SUPPRESS_ATTN_ERROR:-1}"

preprocessed_source_dir="$PREPROCESS_DIR/$SOURCE/$ENTROPY_MODEL_NAME"
if [[ ! -d "$preprocessed_source_dir" ]]; then
  echo "Missing entropy-preprocessed Stage-1 data:" >&2
  echo "  $preprocessed_source_dir" >&2
  echo "Prepare entropy arrow shards before launching Stage-1." >&2
  exit 1
fi

if ! find "$preprocessed_source_dir" -name '*.arrow.complete' -print -quit | grep -q .; then
  echo "No completed entropy arrow shards found in:" >&2
  echo "  $preprocessed_source_dir" >&2
  exit 1
fi

mkdir -p "$RUN_DIR"

"$UV_BIN" run torchrun --standalone --nproc-per-node="$NPROC_PER_NODE" \
  -m bytelatent.train \
  "config=$CONFIG" \
  "dump_dir=$RUN_DIR" \
  "name=$RUN_NAME" \
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
  "checkpoint.path=$RUN_DIR/checkpoints" \
  "distributed.dp_shard=$NPROC_PER_NODE" \
  "distributed.dp_replicate=1" \
  "eval_on_gpus=$NPROC_PER_NODE" \
  "env.ENABLE_INTRA_NODE_COMM=\"$ENABLE_INTRA_NODE_COMM\"" \
  "env.NCCL_DEBUG=WARN"
