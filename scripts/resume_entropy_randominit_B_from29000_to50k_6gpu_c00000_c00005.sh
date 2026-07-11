#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/data1/pengfeigao/BP-MoE
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

export BLT_ALLOW_MISSING_FLEX_ATTENTION=1
export BLT_SUPPRESS_ATTN_ERROR=1
export ENABLE_INTRA_NODE_COMM=1
unset NCCL_P2P_DISABLE || true

export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
export WANDB_MODE=disabled

CONFIG=apps/main/configs/patchmoe_blt1b_warmstart_data_abs.yaml

SUBSET_ROOT=${PROJECT_ROOT}/data/subsets/fineweb_edu_10bt_c00000_c00005
PREPROCESS_ROOT=${PROJECT_ROOT}/data/entropy_preprocessed_stage1

RESUME_CKPT=${PROJECT_ROOT}/blt1b_warmstart/entropy_randominit_A1k_B50k_6gpu_c00000_c00005/phaseB_to_total50k/checkpoints/0000029000

RESUME_DIR=${PROJECT_ROOT}/blt1b_warmstart/entropy_randominit_A1k_B50k_6gpu_c00000_c00005/phaseB_resume_from29000_to_total50k
RESUME_CKPT_ROOT=${RESUME_DIR}/checkpoints

if [[ ! -d "${RESUME_CKPT}" || ! -f "${RESUME_CKPT}/.metadata" ]]; then
  echo "ERROR: resume checkpoint is invalid:"
  echo "  ${RESUME_CKPT}"
  exit 1
fi

echo "============================================================"
echo "Resume entropy-randominit Phase B"
echo "Resume checkpoint:"
echo "  ${RESUME_CKPT}"
echo "Already done:"
echo "  Phase A 1,000 + Phase B 29,000 = 30,000 updates"
echo "Remaining:"
echo "  20,000 updates"
echo "Expected final total:"
echo "  50,000 updates"
echo "============================================================"

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
  "model.moe_router_use_patch_length=false"
  "model.moe_router_use_patch_byte_features=false"

  "model.moe_balance_loss_weight=0.0"
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
  "name=entropy_randominit_phaseB_resume_from29000_to_total50k_6gpu_c00000_c00005" \
  "steps=20000" \
  "max_steps=20000" \
  "checkpoint.path=${RESUME_CKPT_ROOT}" \
  "checkpoint.init_ckpt_path=${RESUME_CKPT}" \
  "checkpoint.dump.every=1000" \
  "checkpoint.dump.keep=1" \
  "checkpoint.eval.every=1000000000" \
  "checkpoint.eval.keep=1" \
  "optim.lr=0.000005" \
  "optim.scheduler=constant" \
  "optim.warmup=0" \
  "optim.lr_min_ratio=1.0"

echo
echo "Resume finished successfully."
echo "Final resumed checkpoint root:"
echo "  ${RESUME_CKPT_ROOT}"
