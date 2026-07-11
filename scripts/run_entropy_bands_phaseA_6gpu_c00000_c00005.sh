#!/usr/bin/env bash
set -euo pipefail

cd /data1/pengfeigao/BP-MoE

PROJECT_ROOT=/data1/pengfeigao/BP-MoE
PYTHON=${PROJECT_ROOT}/.venv/bin/python

# 原始 JSONL 数据所在目录
RAW_DATA_DIR=${PROJECT_ROOT}/data/fineweb_edu_10bt

# entropy preprocessing 后 Arrow 文件所在目录
PREPROCESS_DIR=${PROJECT_ROOT}/data/entropy_preprocessed_stage1
ARROW_DIR=${PREPROCESS_DIR}/fineweb_edu_10bt/transformer_100m

# 仅包含 chunk 00000--00005 的隔离数据根目录。
# 训练 loader 只会扫描这个目录，因此不会误读后续 chunk。
SUBSET_ROOT=${PROJECT_ROOT}/data/subsets/fineweb_edu_10bt_c00000_c00005
SUBSET_DATA_DIR=${SUBSET_ROOT}/fineweb_edu_10bt

# 本次训练输出目录
RUN_DIR=${PROJECT_ROOT}/blt1b_warmstart/entropy_bands_phaseA_1k_6gpu_c00000_c00005
CKPT_DIR=${RUN_DIR}/checkpoints

# 已由 .venv/bin/python 生成的 entropy-bands warmstart DCP
WARMSTART_DIR=${PROJECT_ROOT}/blt1b_warmstart/initial_weights/blt_1b_patchmoe_entropy_bands_paired_dcp

CHUNKS=(
  00000
  00001
  00002
  00003
  00004
  00005
)

if [[ ! -x "${PYTHON}" ]]; then
  echo "ERROR: Python environment not found: ${PYTHON}"
  exit 1
fi

if [[ ! -d "${WARMSTART_DIR}" ]]; then
  echo "ERROR: Warmstart DCP directory not found:"
  echo "  ${WARMSTART_DIR}"
  exit 1
fi

# 重建只含六个 chunk 的 source 目录，严格避免其他 chunk 被扫描。
rm -rf "${SUBSET_ROOT}"
mkdir -p "${SUBSET_DATA_DIR}"
mkdir -p "${RUN_DIR}"

for CHUNK_ID in "${CHUNKS[@]}"; do
  RAW_FILE=${RAW_DATA_DIR}/fineweb_edu_10bt.chunk.${CHUNK_ID}.jsonl
  ARROW_FILE=${ARROW_DIR}/fineweb_edu_10bt.chunk.${CHUNK_ID}.jsonl.shard_00.arrow
  COMPLETE_FILE=${ARROW_FILE}.complete

  if [[ ! -f "${RAW_FILE}" ]]; then
    echo "ERROR: Original JSONL chunk not found:"
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
    echo "The BLT Arrow loader requires <arrow-file>.complete."
    exit 1
  fi

  ln -s "${RAW_FILE}" "${SUBSET_DATA_DIR}/$(basename "${RAW_FILE}")"
done

echo "============================================================"
echo "Training data: FineWeb-Edu 10B Tokens, chunks 00000--00005"
echo "Source subset: ${SUBSET_DATA_DIR}"
echo "Arrow directory: ${ARROW_DIR}"
echo "Visible GPUs: 0,1,2,3,4,5"
echo "============================================================"

ls -lh "${SUBSET_DATA_DIR}"
echo

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH:-} \
BLT_ALLOW_MISSING_FLEX_ATTENTION=1 \
BLT_SUPPRESS_ATTN_ERROR=1 \
NCCL_P2P_DISABLE=0 \
NCCL_IB_DISABLE=1 \
ENABLE_INTRA_NODE_COMM=1 \
NCCL_DEBUG=WARN \
WANDB_MODE=disabled \
"${PYTHON}" -m torch.distributed.run \
  --standalone \
  --nproc-per-node=6 \
  -m bytelatent.train \
  config=apps/main/configs/patchmoe_blt1b_warmstart_data_abs.yaml \
  seed=42 \
  dump_dir=${RUN_DIR} \
  name=entropy_bands_phaseA_1000step_6gpu_c00000_c00005 \
  steps=1000 \
  max_steps=1000 \
  grad_acc_steps=1 \
  data.root_dir=${SUBSET_ROOT} \
  'data.sources={fineweb_edu_10bt: 1.0}' \
  data.batch_size=1 \
  data.seq_len=256 \
  data.max_encoder_seq_length=1536 \
  data.load_async=false \
  data.preprocess_dir=${PREPROCESS_DIR} \
  data.entropy_model_name=transformer_100m \
  data.tokenizer_args.init_kwargs.bpe_tokenizer_path=/tmp/unused.tokenizer.model \
  model.attn_impl=sdpa \
  model.max_seqlen=256 \
  model.max_length=256 \
  model.max_encoder_seq_length=1536 \
  model.moe_num_experts=8 \
  model.moe_top_k=2 \
  model.moe_ep_size=2 \
  model.moe_layer_frequency=1 \
  model.moe_ffn_dim_multiplier=0.5 \
  model.moe_router_use_hidden_state=false \
  model.moe_router_patch_feature_bias=true \
  model.moe_router_normalize_patch_entropy=false \
  model.moe_router_use_patch_entropy=true \
  model.moe_router_use_patch_length=false \
  model.moe_router_use_patch_byte_features=false \
  model.moe_balance_loss_weight=0.0 \
  model.moe_router_jitter=0.0 \
  model.moe_router_congestion_weight=0.0 \
  model.moe_router_z_loss_weight=0.0 \
  checkpoint.path=${CKPT_DIR} \
  checkpoint.init_ckpt_path=${WARMSTART_DIR} \
  checkpoint.dump.every=1000 \
  checkpoint.dump.keep=1 \
  checkpoint.eval.every=100000 \
  checkpoint.eval.keep=1 \
  distributed.fsdp_type=full_shard \
  distributed.dp_shard=6 \
  distributed.dp_replicate=1 \
  distributed.tp_size=1 \
  distributed.selective_activation_checkpointing=false \
  distributed.compile=false \
  distributed.model_dtype=bf16 \
  optim.lr=0.00005 \
  optim.scheduler=cosine \
  optim.warmup=100 \
  optim.lr_min_ratio=0.1 \
  optim.clip=0.0 \
  optim.fused=false \
  logging.freq=10 \
  logging.wandb=null \
  eval_on_gpus=6