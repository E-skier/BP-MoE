#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/data1/pengfeigao/BP-MoE"
RUN_DIR="$ROOT_DIR/experiments/bcfp_cpt/D_ema_bias_screenlike_10k_4gpu_nofused_clip0_ckpt100k"

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

exec env CUDA_VISIBLE_DEVICES=0,1,3,4 \
  "$ROOT_DIR/.venv/bin/python" -m torch.distributed.run --standalone --nproc-per-node=4 \
  -m bytelatent.train \
  config=apps/main/configs/bcfp_D_ema_bias.yaml \
  dump_dir="$RUN_DIR" \
  name=D_ema_bias_screenlike_10k_4gpu_nofused_clip0_ckpt100k \
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
  "model.moe_balance_loss_weight=0.0" \
  "model.moe_balance_schedule=constant" \
  "model.moe_pair_bias_mode=ema_byte_floor" \
  "model.moe_pair_bias_ema=0.95" \
  "model.moe_pair_bias_update_interval=20" \
  "model.moe_pair_bias_lr=0.05" \
  "model.moe_pair_bias_clip=1.5" \
  "model.moe_pair_min_byte_fraction=0.05" \
  "model.moe_pair_max_byte_fraction=0.55" \
  "model.moe_router_use_hidden_state=true" \
  "model.moe_entropy_prior_hidden_scale=0.0" \
  "model.moe_hidden_residual_ramp_start_step=1000" \
  "model.moe_hidden_residual_ramp_end_step=3000" \
  "model.moe_hidden_residual_final_scale=0.25" \
  "checkpoint.path=$RUN_DIR/checkpoints" \
  "checkpoint.init_ckpt_path=$ROOT_DIR/blt1b_warmstart/initial_weights/bcfp_pair_D_ema_bias_dcp" \
  "checkpoint.dump.every=100000" \
  "checkpoint.dump.keep=0" \
  "checkpoint.eval.every=100000" \
  "checkpoint.eval.keep=0" \
  "distributed.dp_shard=4" \
  "distributed.dp_replicate=1" \
  "optim.clip=0.0" \
  "optim.fused=false" \
  "logging.freq=50" \
  "logging.wandb=null" \
  "eval_on_gpus=4"
