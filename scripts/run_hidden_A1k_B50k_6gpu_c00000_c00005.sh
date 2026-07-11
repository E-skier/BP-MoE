#!/usr/bin/env bash
set -euo pipefail

cd /data1/pengfeigao/BP-MoE

# ============================================================
# Hidden-only PatchMoE baseline:
#   Phase A:  1,000 steps, lr=5e-5, cosine schedule
#   Phase B: 49,000 steps, lr=5e-6, constant schedule
# Total: 50,000 optimizer steps
# GPUs: 0--5
# Data: FineWeb-Edu 10B chunks 00000--00005
# ============================================================

PROJECT_ROOT=/data1/pengfeigao/BP-MoE
PYTHON=${PROJECT_ROOT}/.venv/bin/python

RAW_DATA_DIR=${PROJECT_ROOT}/data/fineweb_edu_10bt
PREPROCESS_ROOT=${PROJECT_ROOT}/data/entropy_preprocessed_stage1
ARROW_DIR=${PREPROCESS_ROOT}/fineweb_edu_10bt/transformer_100m

# 仅保留 chunk 00000--00005 的数据视图
SUBSET_ROOT=${PROJECT_ROOT}/data/subsets/fineweb_edu_10bt_c00000_c00005
SUBSET_DATA_DIR=${SUBSET_ROOT}/fineweb_edu_10bt

# Hidden-only warmstart DCP
WARMSTART_DIR=${PROJECT_ROOT}/blt1b_warmstart/initial_weights/blt_1b_patchmoe_hidden_only_active_matched_dcp

# 独立输出目录，避免和 entropy-bands 实验混淆
RUN_ROOT=${PROJECT_ROOT}/blt1b_warmstart/hidden_A1k_B50k_6gpu_c00000_c00005

PHASE_A_DIR=${RUN_ROOT}/phaseA_1k
PHASE_A_CKPT_ROOT=${PHASE_A_DIR}/checkpoints

PHASE_B_DIR=${RUN_ROOT}/phaseB_to_total50k
PHASE_B_CKPT_ROOT=${PHASE_B_DIR}/checkpoints

CHUNKS=(
  00000
  00001
  00002
  00003
  00004
  00005
)

# ------------------------------------------------------------
# Preflight checks
# ------------------------------------------------------------

if [[ ! -x "${PYTHON}" ]]; then
  echo "ERROR: Python environment not found:"
  echo "  ${PYTHON}"
  exit 1
fi

if [[ ! -d "${WARMSTART_DIR}" ]]; then
  echo "ERROR: Hidden-only warmstart DCP not found:"
  echo "  ${WARMSTART_DIR}"
  exit 1
fi

# 重建仅有六个 chunk 的 raw JSONL 软链接目录
rm -rf "${SUBSET_ROOT}"
mkdir -p "${SUBSET_DATA_DIR}"
mkdir -p "${PHASE_A_DIR}"
mkdir -p "${PHASE_B_DIR}"

for CHUNK_ID in "${CHUNKS[@]}"; do
  RAW_FILE=${RAW_DATA_DIR}/fineweb_edu_10bt.chunk.${CHUNK_ID}.jsonl
  ARROW_FILE=${ARROW_DIR}/fineweb_edu_10bt.chunk.${CHUNK_ID}.jsonl.shard_00.arrow
  COMPLETE_FILE=${ARROW_FILE}.complete

  if [[ ! -f "${RAW_FILE}" ]]; then
    echo "ERROR: Raw JSONL chunk not found:"
    echo "  ${RAW_FILE}"
    exit 1
  fi

  if [[ ! -f "${ARROW_FILE}" ]]; then
    echo "ERROR: Preprocessed Arrow file not found:"
    echo "  ${ARROW_FILE}"
    exit 1
  fi

  if [[ ! -f "${COMPLETE_FILE}" ]]; then
    echo "ERROR: Arrow completion marker not found:"
    echo "  ${COMPLETE_FILE}"
    exit 1
  fi

  ln -s "${RAW_FILE}" "${SUBSET_DATA_DIR}/$(basename "${RAW_FILE}")"
done

echo "============================================================"
echo "Experiment: Hidden-only PatchMoE"
echo "Dataset:    FineWeb-Edu 10B"
echo "Chunks:     00000--00005"
echo "GPUs:       0,1,2,3,4,5"
echo "Phase A:    1,000 steps"
echo "Phase B:    +49,000 steps"
echo "Total:      50,000 optimizer steps"
echo "============================================================"

echo
echo "Visible raw chunks:"
ls -lh "${SUBSET_DATA_DIR}"
echo

# ------------------------------------------------------------
# Shared training arguments
# ------------------------------------------------------------

COMMON_ARGS=(
  config=apps/main/configs/patchmoe_blt1b_warmstart_data_abs.yaml
  seed=42
  grad_acc_steps=1

  "data.root_dir=${SUBSET_ROOT}"
  'data.sources={fineweb_edu_10bt: 1.0}'
  data.batch_size=1
  data.seq_len=256
  data.max_encoder_seq_length=1536
  data.load_async=true
  data.prefetch_size=200
  data.buffer_size=512
  "data.preprocess_dir=${PREPROCESS_ROOT}"
  data.entropy_model_name=transformer_100m
  data.tokenizer_args.init_kwargs.bpe_tokenizer_path=/tmp/unused.tokenizer.model

  model.attn_impl=sdpa
  model.max_seqlen=256
  model.max_length=256
  model.max_encoder_seq_length=1536

  model.moe_num_experts=8
  model.moe_top_k=2
  model.moe_ep_size=2
  model.moe_layer_frequency=1
  model.moe_ffn_dim_multiplier=0.5

  # ---------------- Hidden-only router ----------------
  model.moe_router_use_hidden_state=true
  model.moe_router_patch_feature_bias=false
  model.moe_router_normalize_patch_entropy=false
  model.moe_router_use_patch_entropy=false
  model.moe_router_use_patch_length=false
  model.moe_router_use_patch_byte_features=false

  model.moe_balance_loss_weight=0.0
  model.moe_router_jitter=0.0
  model.moe_router_congestion_weight=0.0
  model.moe_router_z_loss_weight=0.0

  distributed.fsdp_type=full_shard
  distributed.dp_shard=6
  distributed.dp_replicate=1
  distributed.tp_size=1
  distributed.selective_activation_checkpointing=false
  distributed.compile=false
  distributed.model_dtype=bf16
  distributed.matmul_allow_tf32=true

  optim.clip=0.0
  optim.fused=false

  logging.freq=10
  logging.wandb=null
  env.NCCL_DEBUG=WARN
  eval_on_gpus=6
)

run_torchrun() {
  unset NCCL_P2P_DISABLE
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
  PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH:-} \
  BLT_ALLOW_MISSING_FLEX_ATTENTION=1 \
  BLT_SUPPRESS_ATTN_ERROR=1 \
  NCCL_IB_DISABLE=1 \
  ENABLE_INTRA_NODE_COMM=1 \
  NCCL_DEBUG=WARN \
  WANDB_MODE=disabled \
  "${PYTHON}" -m torch.distributed.run \
    --standalone \
    --nproc-per-node=6 \
    -m bytelatent.train \
    "$@"
}

# ============================================================
# Phase A: 1,000 steps
# ============================================================

echo
echo "===================== Phase A: 1k ========================="

run_torchrun \
  "${COMMON_ARGS[@]}" \
  "dump_dir=${PHASE_A_DIR}" \
  name=hidden_phaseA_1k_6gpu_c00000_c00005 \
  steps=1000 \
  max_steps=1000 \
  "checkpoint.path=${PHASE_A_CKPT_ROOT}" \
  "checkpoint.init_ckpt_path=${WARMSTART_DIR}" \
  checkpoint.dump.every=1000 \
  checkpoint.dump.keep=1 \
  checkpoint.eval.every=1000000000 \
  checkpoint.eval.keep=1 \
  optim.lr=0.00005 \
  optim.scheduler=cosine \
  optim.warmup=100 \
  optim.lr_min_ratio=0.1

# 定位 Phase A 最新 DCP checkpoint
PHASE_A_INIT_CKPT=$(
  find "${PHASE_A_CKPT_ROOT}" \
    -mindepth 1 \
    -maxdepth 1 \
    -type f \
    -name ".metadata" \
    -printf "%h\n" \
  | sort \
  | tail -n 1
)

if [[ -z "${PHASE_A_INIT_CKPT}" || ! -f "${PHASE_A_INIT_CKPT}/.metadata" ]]; then
  echo "ERROR: Phase A finished, but no valid DCP checkpoint was found."
  echo "Expected under:"
  echo "  ${PHASE_A_CKPT_ROOT}"
  exit 1
fi

echo
echo "Phase A checkpoint used for Phase B:"
echo "  ${PHASE_A_INIT_CKPT}"

# ============================================================
# Phase B: 49,000 additional steps
# ============================================================

echo
echo "================= Phase B: +49k to total 50k ==============="

run_torchrun \
  "${COMMON_ARGS[@]}" \
  "dump_dir=${PHASE_B_DIR}" \
  name=hidden_phaseB_total50k_6gpu_c00000_c00005 \
  steps=49000 \
  max_steps=49000 \
  "checkpoint.path=${PHASE_B_CKPT_ROOT}" \
  "checkpoint.init_ckpt_path=${PHASE_A_INIT_CKPT}" \
  checkpoint.dump.every=1000 \
  checkpoint.dump.keep=1 \
  checkpoint.eval.every=1000000000 \
  checkpoint.eval.keep=1 \
  optim.lr=0.000005 \
  optim.scheduler=constant \
  optim.warmup=0 \
  optim.lr_min_ratio=1.0

# ------------------------------------------------------------
# Cleanup Phase A checkpoint after successful Phase B
# ------------------------------------------------------------

echo
echo "Phase B finished successfully."
echo "Removing temporary Phase A checkpoint:"
echo "  ${PHASE_A_CKPT_ROOT}"

rm -rf "${PHASE_A_CKPT_ROOT}"

echo
echo "============================================================"
echo "Completed: hidden-only 50k training."
echo "Latest retained checkpoint:"
echo "  ${PHASE_B_CKPT_ROOT}"
echo "============================================================"
