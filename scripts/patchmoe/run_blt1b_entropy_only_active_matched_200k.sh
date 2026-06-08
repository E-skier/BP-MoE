#!/usr/bin/env bash
set -euo pipefail

# BLT-1B all-layer PatchMoE entropy-only continued-pretraining.
#
# This launcher targets the active-matched BLT-1B PatchMoE variant:
# - 25/25 global Transformer FFNs replaced by PatchMoE (moe_layer_frequency=1)
# - 8 experts, Top-2
# - entropy-only router side feature
# - per-expert FFN multiplier 0.5 for active-FLOP matching
#
# If the warm-start DCP is missing, PREPARE_INIT=auto creates it from the
# released dense BLT-1B checkpoint before training.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export BLT_ALLOW_MISSING_FLEX_ATTENTION="${BLT_ALLOW_MISSING_FLEX_ATTENTION:-1}"
export BLT_SUPPRESS_ATTN_ERROR="${BLT_SUPPRESS_ATTN_ERROR:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export ENABLE_INTRA_NODE_COMM="${ENABLE_INTRA_NODE_COMM:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

CONDA_ENV="${CONDA_ENV:-bpmoe}"
GPUS="${GPUS:-0,1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
EP_SIZE="${EP_SIZE:-2}"
CONFIG="${CONFIG:-apps/main/configs/patchmoe_blt1b_warmstart_data_abs.yaml}"
SOURCE_CKPT="${SOURCE_CKPT:-/data/zydata/BP-MoE/hf-weights/blt_1b/consolidated.pth}"
ENTROPY_INIT="${ENTROPY_INIT:-/data2/BP-MoE-runs/blt1b_warmstart/blt_1b_patchmoe_entropy_only_active_matched_dcp}"
OUT_ROOT="${OUT_ROOT:-/data2/BP-MoE-runs/blt1b_warmstart/active_matched_200k}"
DATA_ROOT="${DATA_ROOT:-/data/zydata}"
PREPROCESS_DIR="${PREPROCESS_DIR:-/data/zydata/entropy_preprocessed_stage1}"
SOURCE="${SOURCE:-fineweb_edu_10bt}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/tmp/unused.tokenizer.model}"
ENTROPY_MODEL_NAME="${ENTROPY_MODEL_NAME:-transformer_100m}"

STEPS="${STEPS:-200000}"
CKPT_EVERY="${CKPT_EVERY:-20000}"
CKPT_KEEP="${CKPT_KEEP:-1}"
SEQ_LEN="${SEQ_LEN:-256}"
MAX_ENCODER_SEQ_LENGTH="${MAX_ENCODER_SEQ_LENGTH:-1536}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACC_STEPS="${GRAD_ACC_STEPS:-1}"
LOAD_ASYNC="${LOAD_ASYNC:-false}"
LOG_FREQ="${LOG_FREQ:-10}"
LR="${LR:-0.00005}"
WARMUP="${WARMUP:-100}"
OPTIM_CLIP="${OPTIM_CLIP:-0.0}"

MOE_NUM_EXPERTS="${MOE_NUM_EXPERTS:-8}"
MOE_TOP_K="${MOE_TOP_K:-2}"
MOE_LAYER_FREQUENCY="${MOE_LAYER_FREQUENCY:-1}"
MOE_FFN_DIM_MULTIPLIER="${MOE_FFN_DIM_MULTIPLIER:-0.5}"
MOE_BALANCE_LOSS_WEIGHT="${MOE_BALANCE_LOSS_WEIGHT:-0.05}"
MOE_ROUTER_JITTER="${MOE_ROUTER_JITTER:-0.01}"
EXPERT_INIT_MODE="${EXPERT_INIT_MODE:-replicated_prefix}"
EXPECTED_GLOBAL_LAYERS="${EXPECTED_GLOBAL_LAYERS:-25}"

PREPARE_INIT="${PREPARE_INIT:-auto}"
PREPARE_FORCE="${PREPARE_FORCE:-0}"
VERIFY_INIT="${VERIFY_INIT:-1}"

NAME="${NAME:-entropy_only_blt1b_200000step_active_matched}"
RUN_DIR="${RUN_DIR:-$OUT_ROOT/$NAME}"
LOG_PATH="${LOG_PATH:-$RUN_DIR/train.log}"

if (( EP_SIZE <= 0 || NPROC_PER_NODE % EP_SIZE != 0 )); then
  echo "EP_SIZE must be positive and divide NPROC_PER_NODE" >&2
  exit 2
fi
if (( MOE_NUM_EXPERTS % EP_SIZE != 0 )); then
  echo "EP_SIZE must divide MOE_NUM_EXPERTS" >&2
  exit 2
fi
if [[ "$MOE_LAYER_FREQUENCY" != "1" ]]; then
  echo "This full all-layer launcher requires MOE_LAYER_FREQUENCY=1" >&2
  exit 2
fi
if [[ "$MOE_TOP_K" != "2" || "$MOE_NUM_EXPERTS" != "8" ]]; then
  echo "This active-matched entropy-only launcher expects 8 experts and Top-2" >&2
  exit 2
fi

python_cmd=(conda run --no-capture-output -n "$CONDA_ENV" python)
train_cmd=(conda run --no-capture-output -n "$CONDA_ENV" torchrun --standalone "--nproc-per-node=$NPROC_PER_NODE")

check_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "FAIL missing file: $path" >&2
    return 1
  fi
  echo "OK   file: $path"
}

check_entropy_data() {
  local preprocessed_source_dir="$PREPROCESS_DIR/$SOURCE/$ENTROPY_MODEL_NAME"
  if [[ ! -d "$preprocessed_source_dir" ]]; then
    echo "FAIL missing entropy-preprocessed data: $preprocessed_source_dir" >&2
    return 1
  fi
  if ! find "$preprocessed_source_dir" -name '*.arrow.complete' -print -quit | grep -q .; then
    echo "FAIL no completed entropy arrow shards in: $preprocessed_source_dir" >&2
    return 1
  fi
  echo "OK   entropy-preprocessed data: $preprocessed_source_dir"
}

prepare_init_if_needed() {
  if [[ -f "$ENTROPY_INIT/.metadata" && -f "$ENTROPY_INIT/patchmoe_warmstart_manifest.json" ]]; then
    echo "OK   warm-start DCP exists: $ENTROPY_INIT"
    return 0
  fi
  if [[ "$PREPARE_INIT" != "1" && "$PREPARE_INIT" != "auto" ]]; then
    echo "FAIL missing warm-start DCP: $ENTROPY_INIT" >&2
    echo "Set PREPARE_INIT=1 or create it with scripts/patchmoe/prepare_blt1b_patchmoe_warmstart.py" >&2
    return 1
  fi
  check_file "$SOURCE_CKPT"
  local force_arg=()
  if [[ "$PREPARE_FORCE" == "1" ]]; then
    force_arg=(--force)
  fi
  echo "Preparing entropy-only active-matched BLT-1B PatchMoE warm-start DCP: $ENTROPY_INIT"
  "${python_cmd[@]}" scripts/patchmoe/prepare_blt1b_patchmoe_warmstart.py \
    --source "$SOURCE_CKPT" \
    --output-dir "$ENTROPY_INIT" \
    --num-experts "$MOE_NUM_EXPERTS" \
    --top-k "$MOE_TOP_K" \
    --layer-frequency "$MOE_LAYER_FREQUENCY" \
    --patch-features entropy \
    --expert-ffn-dim-multiplier "$MOE_FFN_DIM_MULTIPLIER" \
    --expert-init-mode "$EXPERT_INIT_MODE" \
    "${force_arg[@]}"
}

verify_init() {
  if [[ "$VERIFY_INIT" != "1" ]]; then
    echo "Skipping warm-start DCP verification because VERIFY_INIT=$VERIFY_INIT"
    return 0
  fi
  echo "Verifying entropy-only active-matched all-layer warm-start DCP."
  "${python_cmd[@]}" scripts/patchmoe/verify_blt1b_patchmoe_warmstart.py \
    --source "$SOURCE_CKPT" \
    --checkpoint-dir "$ENTROPY_INIT" \
    --expect-num-experts "$MOE_NUM_EXPERTS" \
    --expect-top-k "$MOE_TOP_K" \
    --expect-layer-frequency "$MOE_LAYER_FREQUENCY" \
    --expect-patch-features entropy \
    --expect-expert-ffn-dim-multiplier "$MOE_FFN_DIM_MULTIPLIER" \
    --expect-expert-init-mode "$EXPERT_INIT_MODE" \
    --expect-global-layer-count "$EXPECTED_GLOBAL_LAYERS" \
    --require-all-global-layers
}

check_file "$CONFIG"
check_entropy_data
prepare_init_if_needed
verify_init
mkdir -p "$RUN_DIR"

echo "Starting BLT-1B all-layer PatchMoE entropy-only active-matched training: $NAME"
echo "Run dir: $RUN_DIR"
echo "Log: $LOG_PATH"

CUDA_VISIBLE_DEVICES="$GPUS" "${train_cmd[@]}" -m bytelatent.train \
  "config=$CONFIG" \
  "dump_dir=$RUN_DIR" \
  "name=$NAME" \
  "steps=$STEPS" \
  "max_steps=$STEPS" \
  "grad_acc_steps=$GRAD_ACC_STEPS" \
  "data.root_dir=$DATA_ROOT" \
  "data.sources={$SOURCE: 1.0}" \
  "data.batch_size=$BATCH_SIZE" \
  "data.seq_len=$SEQ_LEN" \
  "data.max_encoder_seq_length=$MAX_ENCODER_SEQ_LENGTH" \
  "data.load_async=$LOAD_ASYNC" \
  "data.preprocess_dir=$PREPROCESS_DIR" \
  "data.entropy_model_name=$ENTROPY_MODEL_NAME" \
  "data.tokenizer_args.init_kwargs.bpe_tokenizer_path=$TOKENIZER_PATH" \
  "model.attn_impl=sdpa" \
  "model.max_seqlen=$SEQ_LEN" \
  "model.max_length=$SEQ_LEN" \
  "model.max_encoder_seq_length=$MAX_ENCODER_SEQ_LENGTH" \
  "model.moe_num_experts=$MOE_NUM_EXPERTS" \
  "model.moe_top_k=$MOE_TOP_K" \
  "model.moe_layer_frequency=$MOE_LAYER_FREQUENCY" \
  "model.moe_ep_size=$EP_SIZE" \
  "model.moe_ffn_dim_multiplier=$MOE_FFN_DIM_MULTIPLIER" \
  "model.moe_balance_loss_weight=$MOE_BALANCE_LOSS_WEIGHT" \
  "model.moe_assignment_balance_loss_weight=0.0" \
  "model.moe_router_jitter=$MOE_ROUTER_JITTER" \
  "model.moe_router_congestion_weight=0.0" \
  "model.moe_router_z_loss_weight=0.0" \
  "model.moe_router_use_patch_length=false" \
  "model.moe_router_use_patch_entropy=true" \
  "model.moe_router_use_patch_byte_features=false" \
  "model.moe_balance_cost=byte" \
  "checkpoint.path=$RUN_DIR/checkpoints" \
  "checkpoint.init_ckpt_path=$ENTROPY_INIT" \
  "checkpoint.dump.every=$CKPT_EVERY" \
  "checkpoint.dump.keep=$CKPT_KEEP" \
  "checkpoint.eval.every=$CKPT_EVERY" \
  "checkpoint.eval.keep=$CKPT_KEEP" \
  "distributed.fsdp_type=full_shard" \
  "distributed.dp_shard=$NPROC_PER_NODE" \
  "distributed.dp_replicate=1" \
  "distributed.tp_size=1" \
  "distributed.selective_activation_checkpointing=false" \
  "distributed.compile=false" \
  "distributed.model_dtype=bf16" \
  "optim.lr=$LR" \
  "optim.warmup=$WARMUP" \
  "optim.lr_min_ratio=0.1" \
  "optim.clip=$OPTIM_CLIP" \
  "optim.fused=false" \
  "logging.freq=$LOG_FREQ" \
  "logging.wandb=null" \
  "eval_on_gpus=$NPROC_PER_NODE" \
  2>&1 | tee "$LOG_PATH"
