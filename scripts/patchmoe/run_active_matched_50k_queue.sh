#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export BLT_ALLOW_MISSING_FLEX_ATTENTION=1
export BLT_SUPPRESS_ATTN_ERROR=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export ENABLE_INTRA_NODE_COMM=1
export NCCL_DEBUG=WARN

GPUS="${GPUS:-0,1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
CONFIG="${CONFIG:-apps/main/configs/patchmoe_blt1b_warmstart_data_abs.yaml}"
OUT_ROOT="${OUT_ROOT:-/data2/BP-MoE-runs/blt1b_warmstart/active_matched_50k}"
DATA_ROOT="${DATA_ROOT:-/data/zydata}"
PREPROCESS_DIR="${PREPROCESS_DIR:-/data/zydata/entropy_preprocessed_stage1}"
SOURCE="${SOURCE:-fineweb_edu_10bt}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/tmp/unused.tokenizer.model}"
ENTROPY_MODEL_NAME="${ENTROPY_MODEL_NAME:-transformer_100m}"
STEPS="${STEPS:-50000}"
CKPT_EVERY="${CKPT_EVERY:-1000}"

DENSE_INIT="${DENSE_INIT:-/data2/BP-MoE-runs/blt1b_warmstart/blt_1b_dense_official_dcp}"
HIDDEN_INIT="${HIDDEN_INIT:-/data2/BP-MoE-runs/blt1b_warmstart/blt_1b_patchmoe_hidden_only_active_matched_dcp}"
ENTROPY_INIT="${ENTROPY_INIT:-/data2/BP-MoE-runs/blt1b_warmstart/blt_1b_patchmoe_entropy_only_active_matched_dcp}"

common_args=(
  "config=$CONFIG"
  "steps=$STEPS"
  "max_steps=$STEPS"
  "grad_acc_steps=1"
  "data.root_dir=$DATA_ROOT"
  "data.sources={$SOURCE: 1.0}"
  "data.batch_size=1"
  "data.seq_len=256"
  "data.max_encoder_seq_length=1536"
  "data.load_async=false"
  "data.preprocess_dir=$PREPROCESS_DIR"
  "data.entropy_model_name=$ENTROPY_MODEL_NAME"
  "data.tokenizer_args.init_kwargs.bpe_tokenizer_path=$TOKENIZER_PATH"
  "model.attn_impl=sdpa"
  "model.max_seqlen=256"
  "model.max_length=256"
  "model.max_encoder_seq_length=1536"
  "checkpoint.dump.every=$CKPT_EVERY"
  "checkpoint.dump.keep=1"
  "checkpoint.eval.every=$CKPT_EVERY"
  "checkpoint.eval.keep=1"
  "distributed.fsdp_type=full_shard"
  "distributed.dp_shard=$NPROC_PER_NODE"
  "distributed.dp_replicate=1"
  "distributed.tp_size=1"
  "distributed.selective_activation_checkpointing=false"
  "distributed.compile=false"
  "distributed.model_dtype=bf16"
  "optim.lr=0.00005"
  "optim.warmup=100"
  "optim.lr_min_ratio=0.1"
  "optim.clip=0.0"
  "optim.fused=false"
  "logging.freq=10"
  "logging.wandb=null"
  "eval_on_gpus=$NPROC_PER_NODE"
)

run_variant() {
  local name="$1"
  local init_ckpt="$2"
  shift 2
  local run_dir="$OUT_ROOT/$name"
  mkdir -p "$run_dir"
  echo "[$(date --iso-8601=seconds)] starting $name"
  CUDA_VISIBLE_DEVICES="$GPUS" conda run --no-capture-output -n bpmoe \
    torchrun --standalone "--nproc-per-node=$NPROC_PER_NODE" -m bytelatent.train \
    "${common_args[@]}" \
    "dump_dir=$run_dir" \
    "name=$name" \
    "checkpoint.path=$run_dir/checkpoints" \
    "checkpoint.init_ckpt_path=$init_ckpt" \
    "$@" \
    2>&1 | tee "$run_dir/pipeline.log"
  echo "[$(date --iso-8601=seconds)] finished $name"
}

run_variant dense_blt1b_50000step_active_matched "$DENSE_INIT" \
  "model.moe_num_experts=0" \
  "model.moe_top_k=1" \
  "model.moe_ep_size=1" \
  "model.moe_balance_loss_weight=0.0" \
  "model.moe_router_jitter=0.0" \
  "model.moe_router_congestion_weight=0.0" \
  "model.moe_router_z_loss_weight=0.0" \
  "model.moe_router_use_patch_length=false" \
  "model.moe_router_use_patch_entropy=false" \
  "model.moe_router_use_patch_byte_features=false" \
  "model.moe_balance_cost=patch"

run_variant hidden_only_blt1b_50000step_active_matched "$HIDDEN_INIT" \
  "model.moe_num_experts=8" \
  "model.moe_top_k=2" \
  "model.moe_ep_size=2" \
  "model.moe_ffn_dim_multiplier=0.5" \
  "model.moe_balance_loss_weight=0.05" \
  "model.moe_router_jitter=0.01" \
  "model.moe_router_congestion_weight=0.0" \
  "model.moe_router_z_loss_weight=0.0" \
  "model.moe_router_use_patch_length=false" \
  "model.moe_router_use_patch_entropy=false" \
  "model.moe_router_use_patch_byte_features=false" \
  "model.moe_balance_cost=byte"

run_variant entropy_only_blt1b_50000step_active_matched "$ENTROPY_INIT" \
  "model.moe_num_experts=8" \
  "model.moe_top_k=2" \
  "model.moe_ep_size=2" \
  "model.moe_ffn_dim_multiplier=0.5" \
  "model.moe_balance_loss_weight=0.05" \
  "model.moe_router_jitter=0.01" \
  "model.moe_router_congestion_weight=0.0" \
  "model.moe_router_z_loss_weight=0.0" \
  "model.moe_router_use_patch_length=false" \
  "model.moe_router_use_patch_entropy=true" \
  "model.moe_router_use_patch_byte_features=false" \
  "model.moe_balance_cost=byte"
