#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/data1/pengfeigao/BP-MoE
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

export BLT_ALLOW_MISSING_FLEX_ATTENTION=1
export BLT_SUPPRESS_ATTN_ERROR=1
export ENABLE_INTRA_NODE_COMM=1

# Use NCCL safely on this server.
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
export WANDB_MODE=disabled

CONFIG=apps/main/configs/patchmoe_blt1b_warmstart_data_abs.yaml

SUBSET_ROOT=${PROJECT_ROOT}/data/subsets/fineweb_edu_10bt_c00000_c00005
PREPROCESS_ROOT=${PROJECT_ROOT}/data/entropy_preprocessed_stage1

RUN_ROOT=${PROJECT_ROOT}/blt1b_warmstart/natural_granularity_AC1_A5k_B15k_6gpu_c00000_c00005

# Existing complete checkpoint:
# Phase A 5k + Phase B 4k = total 9k.
INIT_CKPT=${RUN_ROOT}/phaseB_to_total20k/checkpoints/0000004000

# New resume directory.
# Do not write back into the old phaseB_to_total20k directory.
RESUME_DIR=${RUN_ROOT}/phaseB_resume_from4000_to_total20k
RESUME_CKPT_ROOT=${RESUME_DIR}/checkpoints

# Remaining PhaseB steps:
# Original PhaseB target = 15k.
# Already completed PhaseB = 4k.
# Remaining = 11k.
REMAINING_STEPS=11000

echo "============================================================"
echo "Resume natural_granularity_AC1"
echo "Input checkpoint:"
echo "  ${INIT_CKPT}"
echo "Output directory:"
echo "  ${RESUME_DIR}"
echo "Remaining steps:"
echo "  ${REMAINING_STEPS}"
echo "Expected final:"
echo "  Phase A 5k + Phase B 15k = total 20k"
echo "Checkpoint policy:"
echo "  checkpoint.dump.keep=1"
echo "  checkpoint.eval.keep=1"
echo "============================================================"

if [[ ! -d "${INIT_CKPT}" ]]; then
  echo "ERROR: INIT_CKPT does not exist:"
  echo "  ${INIT_CKPT}"
  exit 1
fi

if [[ ! -f "${INIT_CKPT}/complete.json" && ! -f "${INIT_CKPT}/.metadata" ]]; then
  echo "ERROR: INIT_CKPT does not look complete:"
  echo "  ${INIT_CKPT}"
  exit 1
fi

if [[ -e "${RESUME_DIR}" ]]; then
  echo "ERROR: RESUME_DIR already exists:"
  echo "  ${RESUME_DIR}"
  echo "This script will not delete or overwrite anything."
  exit 1
fi

echo
echo "Disk space before training:"
df -h /data1

COMMON_ARGS=(
  "config=${CONFIG}"
  "seed=42"
  "grad_acc_steps=1"

  "data.root_dir=${SUBSET_ROOT}"
  "data.sources={fineweb_edu_10bt: 1.0}"
  "data.batch_size=1"
  "data.seq_len=256"
  "data.max_encoder_seq_length=1536"
  "data.load_async=true"
  "data.async_persist_type=approximate"
  "data.prefetch_size=200"
  "data.preprocess_dir=${PREPROCESS_ROOT}"
  "data.entropy_model_name=transformer_100m"
  "data.arrow_batch_size=20"
  "data.buffer_size=512"
  "data.tokenizer_args.init_kwargs.bpe_tokenizer_path=/tmp/unused.tokenizer.model"

  "model.attn_impl=sdpa"
  "model.max_seqlen=256"
  "model.max_length=256"
  "model.max_encoder_seq_length=1536"

  "model.moe_num_experts=8"
  "model.moe_top_k=2"
  "model.moe_ep_size=2"
  "model.moe_layer_frequency=1"
  "model.moe_ffn_dim_multiplier=0.5"

  "model.moe_router_use_hidden_state=false"
  "model.moe_router_patch_feature_bias=true"
  "model.moe_router_normalize_patch_entropy=false"
  "model.moe_router_use_patch_entropy=true"
  "model.moe_router_use_patch_length=true"
  "model.moe_router_use_patch_byte_features=false"

  "model.moe_balance_loss_weight=0.02"
  "model.moe_balance_cost=byte"
  "model.moe_router_jitter=0.0"
  "model.moe_router_congestion_weight=0.0"
  "model.moe_router_z_loss_weight=0.0"

  "distributed.fsdp_type=full_shard"
  "distributed.dp_shard=6"
  "distributed.dp_replicate=1"
  "distributed.tp_size=1"
  "distributed.selective_activation_checkpointing=false"
  "distributed.compile=false"
  "distributed.model_dtype=bf16"
  "distributed.matmul_allow_tf32=true"

  "optim.clip=0.0"
  "optim.fused=false"
  "optim.lr=0.000002"
  "optim.scheduler=constant"
  "optim.warmup=0"
  "optim.lr_min_ratio=1.0"

  "logging.freq=10"
  "logging.wandb=null"
  "eval_on_gpus=6"
)

.venv/bin/python -m torch.distributed.run \
  --standalone \
  --nproc-per-node=6 \
  -m bytelatent.train \
  "${COMMON_ARGS[@]}" \
  "dump_dir=${RESUME_DIR}" \
  "name=natural_granularity_AC1_resume_from_B4k_to_total20k_6gpu_c00000_c00005" \
  "steps=${REMAINING_STEPS}" \
  "max_steps=${REMAINING_STEPS}" \
  "checkpoint.path=${RESUME_CKPT_ROOT}" \
  "checkpoint.init_ckpt_path=${INIT_CKPT}" \
  "checkpoint.dump.every=1000" \
  "checkpoint.dump.keep=1" \
  "checkpoint.eval.every=1000" \
  "checkpoint.eval.keep=1"

echo
echo "Resume training finished successfully."
echo "Expected final local checkpoint:"
echo "  ${RESUME_CKPT_ROOT}/0000011000"
echo
echo "Disk space after training:"
df -h /data1
