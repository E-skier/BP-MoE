#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/data1/pengfeigao/BP-MoE"
RUN_DIR="$ROOT_DIR/experiments/bcfp_cpt/C_byte_aux_001_screenlike_10k_6gpu_nofused_clip0_ckpt100k"

cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export BLT_ALLOW_MISSING_FLEX_ATTENTION=1
export BLT_SUPPRESS_ATTN_ERROR=1
export ENABLE_INTRA_NODE_COMM=1
export NCCL_DEBUG=WARN
export NCCL_IB_TIMEOUT=22
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_SERVICE_FORCE_INTEL=GNU

exec env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
  /home/pengfeigao/.local/bin/uv run torchrun --standalone --nproc-per-node=6 \
  -m bytelatent.train \
  config=apps/main/configs/bcfp_C_byte_aux.yaml \
  dump_dir="$RUN_DIR" \
  name=C_byte_aux_001_screenlike_10k_6gpu_nofused_clip0_ckpt100k \
  steps=10000 \
  max_steps=10000 \
  grad_acc_steps=1 \
  "data.root_dir=$ROOT_DIR/data" \
  "data.sources={fineweb_edu_10bt: 1.0}" \
  "data.batch_size=1" \
  "data.seq_len=256" \
  "data.max_encoder_seq_length=1536" \
  "data.load_async=false" \
  "data.preprocess_dir=$ROOT_DIR/data/entropy_preprocessed_stage1" \
  "data.entropy_model_name=transformer_100m" \
  "data.tokenizer_args.init_kwargs.bpe_tokenizer_path=/tmp/unused.tokenizer.model" \
  "model.max_seqlen=256" \
  "model.max_length=256" \
  "model.max_encoder_seq_length=1536" \
  "model.moe_ep_size=2" \
  "model.moe_balance_loss_weight=0.001" \
  "model.moe_balance_final_weight=0.0002" \
  "model.moe_balance_schedule=linear_decay" \
  "model.moe_balance_start_step=0" \
  "model.moe_balance_peak_step=250" \
  "model.moe_balance_decay_start_step=1000" \
  "model.moe_balance_end_step=5000" \
  "checkpoint.path=$RUN_DIR/checkpoints" \
  "checkpoint.init_ckpt_path=$ROOT_DIR/blt1b_warmstart/initial_weights/bcfp_pair_calibrated_dcp" \
  "checkpoint.dump.every=100000" \
  "checkpoint.dump.keep=0" \
  "checkpoint.eval.every=100000" \
  "checkpoint.eval.keep=0" \
  "distributed.dp_shard=6" \
  "distributed.dp_replicate=1" \
  "optim.clip=0.0" \
  "optim.fused=false" \
  "logging.freq=50" \
  "logging.wandb=null" \
  "eval_on_gpus=6"
