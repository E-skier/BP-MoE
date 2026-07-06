#!/usr/bin/env bash
set -euo pipefail

# 6-GPU BLT-1B dense continued-pretraining launcher for this workspace.
# Default MODE=print only runs preflight checks and prints the command.
# Use MODE=run to launch training.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

UV_BIN="${UV_BIN:-}"
if [[ -z "$UV_BIN" ]]; then
  UV_BIN="$(command -v uv || true)"
fi
if [[ -z "$UV_BIN" && -x "$HOME/.local/bin/uv" ]]; then
  UV_BIN="$HOME/.local/bin/uv"
fi
if [[ -z "$UV_BIN" ]]; then
  echo "FAIL uv not found. Install uv or pass UV_BIN=/path/to/uv." >&2
  exit 2
fi

MODE="${MODE:-print}"
if [[ "$MODE" != "print" && "$MODE" != "run" ]]; then
  echo "MODE must be print or run" >&2
  exit 2
fi

CONFIG="${CONFIG:-apps/main/configs/patchmoe_blt1b_warmstart.yaml}"
GPUS="${GPUS:-0,1,2,3,4,5}"
NPROC_PER_NODE="${NPROC_PER_NODE:-6}"

RUN_NAME="${RUN_NAME:-dense_blt1b_50000step_active_matched_6gpu}"
OUT_ROOT="${OUT_ROOT:-$ROOT_DIR/runs/blt1b_warmstart/active_matched_50k}"
RUN_DIR="${RUN_DIR:-$OUT_ROOT/$RUN_NAME}"

DATA_ROOT="${DATA_ROOT:-$ROOT_DIR/data}"
SOURCE="${SOURCE:-fineweb_edu_10bt}"
PREPROCESS_DIR="${PREPROCESS_DIR:-$ROOT_DIR/data/entropy_preprocessed_stage1}"
ENTROPY_MODEL_NAME="${ENTROPY_MODEL_NAME:-transformer_100m}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/tmp/unused.tokenizer.model}"

# This must be a torch.distributed.checkpoint directory, not consolidated.pth.
INIT_CKPT_PATH="${INIT_CKPT_PATH:-$ROOT_DIR/hf-weights/blt_1b_dense_official_dcp}"

STEPS="${STEPS:-50000}"
MAX_STEPS="${MAX_STEPS:-50000}"
GRAD_ACC_STEPS="${GRAD_ACC_STEPS:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
SEQ_LEN="${SEQ_LEN:-256}"
MAX_ENCODER_SEQ_LENGTH="${MAX_ENCODER_SEQ_LENGTH:-1536}"
LR="${LR:-0.00005}"
WARMUP="${WARMUP:-100}"
LOG_FREQ="${LOG_FREQ:-10}"
CKPT_EVERY="${CKPT_EVERY:-1000}"
CKPT_KEEP="${CKPT_KEEP:-1}"

export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export BLT_ALLOW_MISSING_FLEX_ATTENTION="${BLT_ALLOW_MISSING_FLEX_ATTENTION:-1}"
export BLT_SUPPRESS_ATTN_ERROR="${BLT_SUPPRESS_ATTN_ERROR:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export ENABLE_INTRA_NODE_COMM="${ENABLE_INTRA_NODE_COMM:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

quote_cmd() {
  printf '%q ' "$@"
  printf '\n'
}

preflight() {
  local preprocessed_source_dir="$PREPROCESS_DIR/$SOURCE/$ENTROPY_MODEL_NAME"

  [[ -f "$CONFIG" ]] || { echo "FAIL missing config: $CONFIG" >&2; return 1; }
  [[ -d "$DATA_ROOT/$SOURCE" ]] || { echo "FAIL missing data source dir: $DATA_ROOT/$SOURCE" >&2; return 1; }
  [[ -d "$preprocessed_source_dir" ]] || { echo "FAIL missing entropy preprocess dir: $preprocessed_source_dir" >&2; return 1; }
  find "$preprocessed_source_dir" -name '*.arrow' -print -quit | grep -q . || {
    echo "FAIL no entropy .arrow shards under: $preprocessed_source_dir" >&2
    return 1
  }
  [[ -d "$INIT_CKPT_PATH" ]] || {
    echo "FAIL missing init DCP directory: $INIT_CKPT_PATH" >&2
    echo "      Set INIT_CKPT_PATH=/path/to/blt_1b_dense_official_dcp before running." >&2
    return 1
  }
  [[ -f "$INIT_CKPT_PATH/.metadata" ]] || {
    echo "FAIL init checkpoint is not DCP format: $INIT_CKPT_PATH/.metadata not found" >&2
    echo "      This training path expects torch.distributed.checkpoint format, not hf-weights/blt_1b/consolidated.pth." >&2
    return 1
  }

  echo "OK config: $CONFIG"
  echo "OK data: $DATA_ROOT/$SOURCE"
  echo "OK entropy preprocess: $preprocessed_source_dir"
  echo "OK init DCP: $INIT_CKPT_PATH"
}

cmd=(
  env
  "CUDA_VISIBLE_DEVICES=$GPUS"
  "$UV_BIN" run torchrun --standalone "--nproc-per-node=$NPROC_PER_NODE"
  -m bytelatent.train
  "config=$CONFIG"
  "steps=$STEPS"
  "max_steps=$MAX_STEPS"
  "grad_acc_steps=$GRAD_ACC_STEPS"
  "data.root_dir=$DATA_ROOT"
  "data.sources={$SOURCE: 1.0}"
  "data.batch_size=$BATCH_SIZE"
  "data.seq_len=$SEQ_LEN"
  "data.max_encoder_seq_length=$MAX_ENCODER_SEQ_LENGTH"
  "data.load_async=false"
  "data.preprocess_dir=$PREPROCESS_DIR"
  "data.entropy_model_name=$ENTROPY_MODEL_NAME"
  "data.tokenizer_args.init_kwargs.bpe_tokenizer_path=$TOKENIZER_PATH"
  "model.attn_impl=sdpa"
  "model.max_seqlen=$SEQ_LEN"
  "model.max_length=$SEQ_LEN"
  "model.max_encoder_seq_length=$MAX_ENCODER_SEQ_LENGTH"
  "checkpoint.dump.every=$CKPT_EVERY"
  "checkpoint.dump.keep=$CKPT_KEEP"
  "checkpoint.eval.every=$CKPT_EVERY"
  "checkpoint.eval.keep=$CKPT_KEEP"
  "distributed.fsdp_type=full_shard"
  "distributed.dp_shard=$NPROC_PER_NODE"
  "distributed.dp_replicate=1"
  "distributed.tp_size=1"
  "distributed.selective_activation_checkpointing=false"
  "distributed.compile=false"
  "distributed.model_dtype=bf16"
  "optim.lr=$LR"
  "optim.warmup=$WARMUP"
  "optim.lr_min_ratio=0.1"
  "optim.clip=0.0"
  "optim.fused=false"
  "logging.freq=$LOG_FREQ"
  "logging.wandb=null"
  "eval_on_gpus=$NPROC_PER_NODE"
  "dump_dir=$RUN_DIR"
  "name=$RUN_NAME"
  "checkpoint.path=$RUN_DIR/checkpoints"
  "checkpoint.init_ckpt_path=$INIT_CKPT_PATH"
  "model.moe_num_experts=0"
  "model.moe_top_k=1"
  "model.moe_ep_size=1"
  "model.moe_balance_loss_weight=0.0"
  "model.moe_router_jitter=0.0"
  "model.moe_router_congestion_weight=0.0"
  "model.moe_router_z_loss_weight=0.0"
  "model.moe_router_use_patch_length=false"
  "model.moe_router_use_patch_entropy=false"
  "model.moe_router_use_patch_byte_features=false"
  "model.moe_balance_cost=patch"
)

preflight
echo "Prepared 6-GPU training command:"
quote_cmd "${cmd[@]}"
echo "Run dir: $RUN_DIR"

if [[ "$MODE" == "print" ]]; then
  exit 0
fi

mkdir -p "$RUN_DIR"
quote_cmd "${cmd[@]}" >"$RUN_DIR/launch_command.txt"
"${cmd[@]}" 2>&1 | tee "$RUN_DIR/train.log"
