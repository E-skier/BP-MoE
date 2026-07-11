#!/usr/bin/env bash
set -euo pipefail

# This script is sourced by wrapper scripts.
# Required variables from wrapper:
#   RUN_NAME
#   INIT_CKPT_CANDIDATES
#   MODEL_OVERRIDES

PROJECT_ROOT=/data1/pengfeigao/BP-MoE
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

export BLT_ALLOW_MISSING_FLEX_ATTENTION=1
export BLT_SUPPRESS_ATTN_ERROR=1
export ENABLE_INTRA_NODE_COMM=1

export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
export WANDB_MODE=disabled

CONFIG=apps/main/configs/patchmoe_blt1b_warmstart_data_abs.yaml

SUBSET_ROOT=${PROJECT_ROOT}/data/subsets/fineweb_edu_10bt_c00000_c00005
PREPROCESS_ROOT=${PROJECT_ROOT}/data/entropy_preprocessed_stage1

RUN_ROOT=${PROJECT_ROOT}/blt1b_warmstart/${RUN_NAME}
PHASEA_DIR=${RUN_ROOT}/phaseA_5k
PHASEB_DIR=${RUN_ROOT}/phaseB_15k_from_A5k

PHASEA_CKPT_ROOT=${PHASEA_DIR}/checkpoints
PHASEB_CKPT_ROOT=${PHASEB_DIR}/checkpoints

PHASEA_FINAL=${PHASEA_CKPT_ROOT}/0000005000
PHASEB_FINAL=${PHASEB_CKPT_ROOT}/0000015000

PHASEA_STEPS=5000
PHASEB_STEPS=15000

DELETE_PHASEA_AFTER_PHASEB=${DELETE_PHASEA_AFTER_PHASEB:-1}

find_init_ckpt() {
  if [[ -n "${INIT_CKPT:-}" ]]; then
    if [[ -d "${INIT_CKPT}" ]]; then
      echo "${INIT_CKPT}"
      return 0
    fi
    echo "ERROR: user-provided INIT_CKPT does not exist: ${INIT_CKPT}" >&2
    return 1
  fi

  for p in "${INIT_CKPT_CANDIDATES[@]}"; do
    if [[ -d "${p}" ]]; then
      echo "${p}"
      return 0
    fi
  done

  echo "ERROR: Cannot find init checkpoint for ${RUN_NAME}." >&2
  echo "Tried:" >&2
  for p in "${INIT_CKPT_CANDIDATES[@]}"; do
    echo "  ${p}" >&2
  done
  echo >&2
  echo "Useful search commands:" >&2
  echo "  find ${PROJECT_ROOT} -type d -path '*initial_weights*' | sort" >&2
  echo "  find ${PROJECT_ROOT} -type d | grep -Ei 'entropy|hidden|random|patchmoe' | sort | head -100" >&2
  echo >&2
  echo "Then rerun with:" >&2
  echo "  INIT_CKPT=/path/to/init bash <wrapper_script>" >&2
  return 1
}

INIT_CKPT_RESOLVED=$(find_init_ckpt)

echo "============================================================"
echo "Router ablation run"
echo "RUN_NAME: ${RUN_NAME}"
echo "INIT_CKPT: ${INIT_CKPT_RESOLVED}"
echo "PHASEA_DIR: ${PHASEA_DIR}"
echo "PHASEB_DIR: ${PHASEB_DIR}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "Checkpoint policy:"
echo "  checkpoint.dump.keep=1"
echo "  checkpoint.eval.keep=1"
echo "  DELETE_PHASEA_AFTER_PHASEB=${DELETE_PHASEA_AFTER_PHASEB}"
echo "============================================================"

echo
echo "Disk before training:"
df -h /data1

if [[ -e "${RUN_ROOT}" ]]; then
  echo
  echo "ERROR: RUN_ROOT already exists:"
  echo "  ${RUN_ROOT}"
  echo "This script will not overwrite existing runs."
  exit 1
fi

mkdir -p "${RUN_ROOT}"

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

  "logging.freq=10"
  "logging.wandb=null"
  "eval_on_gpus=6"
)

echo
echo "============================================================"
echo "Phase A: 5k"
echo "============================================================"

.venv/bin/python -m torch.distributed.run \
  --standalone \
  --nproc-per-node=6 \
  -m bytelatent.train \
  "${COMMON_ARGS[@]}" \
  "${MODEL_OVERRIDES[@]}" \
  "dump_dir=${PHASEA_DIR}" \
  "name=${RUN_NAME}_phaseA_5k" \
  "steps=${PHASEA_STEPS}" \
  "max_steps=${PHASEA_STEPS}" \
  "checkpoint.path=${PHASEA_CKPT_ROOT}" \
  "checkpoint.init_ckpt_path=${INIT_CKPT_RESOLVED}" \
  "checkpoint.dump.every=1000" \
  "checkpoint.dump.keep=1" \
  "checkpoint.eval.every=1000" \
  "checkpoint.eval.keep=1" \
  "optim.clip=0.0" \
  "optim.fused=false" \
  "optim.lr=0.00005" \
  "optim.scheduler=cosine" \
  "optim.warmup=100"

if [[ ! -d "${PHASEA_FINAL}" ]]; then
  echo "ERROR: Phase A final checkpoint missing:"
  echo "  ${PHASEA_FINAL}"
  exit 1
fi

echo
echo "============================================================"
echo "Phase B: 15k from Phase A 5k"
echo "============================================================"

.venv/bin/python -m torch.distributed.run \
  --standalone \
  --nproc-per-node=6 \
  -m bytelatent.train \
  "${COMMON_ARGS[@]}" \
  "${MODEL_OVERRIDES[@]}" \
  "dump_dir=${PHASEB_DIR}" \
  "name=${RUN_NAME}_phaseB_15k_from_A5k" \
  "steps=${PHASEB_STEPS}" \
  "max_steps=${PHASEB_STEPS}" \
  "checkpoint.path=${PHASEB_CKPT_ROOT}" \
  "checkpoint.init_ckpt_path=${PHASEA_FINAL}" \
  "checkpoint.dump.every=1000" \
  "checkpoint.dump.keep=1" \
  "checkpoint.eval.every=1000" \
  "checkpoint.eval.keep=1" \
  "optim.clip=0.0" \
  "optim.fused=false" \
  "optim.lr=0.000002" \
  "optim.scheduler=constant" \
  "optim.warmup=0" \
  "optim.lr_min_ratio=1.0"

if [[ ! -d "${PHASEB_FINAL}" ]]; then
  echo "ERROR: Phase B final checkpoint missing:"
  echo "  ${PHASEB_FINAL}"
  exit 1
fi

if [[ "${DELETE_PHASEA_AFTER_PHASEB}" == "1" ]]; then
  echo
  echo "Deleting Phase A checkpoint to save disk:"
  echo "  ${PHASEA_CKPT_ROOT}"
  rm -rf "${PHASEA_CKPT_ROOT}"
fi

echo
echo "Finished ${RUN_NAME}"
echo "Final checkpoint:"
echo "  ${PHASEB_FINAL}"
echo
echo "Disk after training:"
df -h /data1
