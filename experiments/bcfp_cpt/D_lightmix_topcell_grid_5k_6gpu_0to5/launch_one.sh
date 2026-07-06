#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 RUN_NAME NATURAL_WEIGHT TOPCELL_WEIGHT" >&2
  exit 2
fi

ROOT_DIR="/data1/pengfeigao/BP-MoE"
RUN_NAME="$1"
NATURAL_WEIGHT="$2"
TOPCELL_WEIGHT="$3"
RUN_DIR="$ROOT_DIR/experiments/bcfp_cpt/$RUN_NAME"
CKPT_DIR="$RUN_DIR/checkpoints/0000005000"
EVAL_DIR="$RUN_DIR/heldout_eval_expanded_100k_8000batches_gpu0"
HELDOUT_ARROW_DIR="$ROOT_DIR/data/entropy_preprocessed_stage1_heldout_expanded/fineweb_edu_10bt_heldout_expanded_100000/transformer_100m"
TOPCELL_SOURCE="fineweb_edu_10bt_entropy_length_topcell_2kpf_1kpb"
TRAIN_CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}"
EVAL_CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-6}"
DP_SHARD="${DP_SHARD:-6}"

mkdir -p "$RUN_DIR" "$EVAL_DIR"
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

last_step=0
if [[ -s "$RUN_DIR/metrics.jsonl" ]]; then
  last_step="$(tail -n 1 "$RUN_DIR/metrics.jsonl" | jq -r '.global_step // 0')"
fi

if [[ "$last_step" -lt 5000 ]]; then
  {
    date
    echo "pipeline: start train RUN_NAME=$RUN_NAME natural=$NATURAL_WEIGHT topcell=$TOPCELL_WEIGHT"
    echo "pipeline: train devices=$TRAIN_CUDA_VISIBLE_DEVICES nproc=$NPROC_PER_NODE dp_shard=$DP_SHARD"
  } | tee -a "$RUN_DIR/pipeline.log"

  env CUDA_VISIBLE_DEVICES="$TRAIN_CUDA_VISIBLE_DEVICES" \
    "$ROOT_DIR/.venv/bin/python" -m torch.distributed.run --standalone --nproc-per-node="$NPROC_PER_NODE" \
    -m bytelatent.train \
    config=apps/main/configs/bcfp_D_ema_bias.yaml \
    dump_dir="$RUN_DIR" \
    "name=$RUN_NAME" \
    steps=5000 \
    max_steps=5000 \
    grad_acc_steps=1 \
    "data.root_dir=$ROOT_DIR/data" \
    "data.sources={fineweb_edu_10bt: $NATURAL_WEIGHT, $TOPCELL_SOURCE: $TOPCELL_WEIGHT}" \
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
    "checkpoint.dump.every=5000" \
    "checkpoint.dump.keep=1" \
    "checkpoint.eval.every=100000" \
    "checkpoint.eval.keep=0" \
    "distributed.dp_shard=$DP_SHARD" \
    "distributed.dp_replicate=1" \
    "optim.clip=0.0" \
    "optim.fused=false" \
    "logging.freq=50" \
    "logging.wandb=null" \
    "eval_on_gpus=6" \
    > "$RUN_DIR/train.log" 2>&1

  {
    date
    echo "pipeline: train done"
  } | tee -a "$RUN_DIR/pipeline.log"
else
  echo "pipeline: skip train, last_step=$last_step" | tee -a "$RUN_DIR/pipeline.log"
fi

if [[ ! -d "$CKPT_DIR" ]]; then
  echo "missing checkpoint: $CKPT_DIR" | tee -a "$RUN_DIR/pipeline.log" >&2
  exit 1
fi

if [[ ! -s "$EVAL_DIR/results.json" ]]; then
  {
    date
    echo "pipeline: start fixed natural heldout eval"
    echo "pipeline: eval devices=$EVAL_CUDA_VISIBLE_DEVICES"
  } | tee -a "$RUN_DIR/pipeline.log"

  env CUDA_VISIBLE_DEVICES="$EVAL_CUDA_VISIBLE_DEVICES" \
    "$ROOT_DIR/.venv/bin/python" -m bytelatent.eval \
    "ckpt_dir=$CKPT_DIR" \
    "dump_dir=$EVAL_DIR" \
    "metric_log_dir=$EVAL_DIR" \
    "global_step=5000" \
    "consolidate_if_needed=true" \
    "run_ppl=true" \
    "run_tasks=false" \
    "validation.use_val_from_train_src=false" \
    "validation.root_dir=$HELDOUT_ARROW_DIR" \
    "validation.sources=[fineweb_edu_10bt_heldout_expanded_100000.chunk.00002.first_100000.jsonl.shard_00.arrow,fineweb_edu_10bt_heldout_expanded_100000.chunk.00003.first_100000.jsonl.shard_00.arrow]" \
    "validation.batch_size=1" \
    "validation.max_n_batches=8000" \
    > "$EVAL_DIR/eval.log" 2>&1

  rm -rf "$CKPT_DIR/consolidated"

  {
    date
    echo "pipeline: heldout eval done"
  } | tee -a "$RUN_DIR/pipeline.log"
else
  echo "pipeline: skip eval, results.json exists" | tee -a "$RUN_DIR/pipeline.log"
fi
