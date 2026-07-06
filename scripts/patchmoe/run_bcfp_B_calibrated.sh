#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# BCFP Experiment B: Byte-Calibrated Gaussian Entropy Prior
# ============================================================
#
# 核心配置:
#   - Pair-level routing (4 pairs, 8 experts, top-2 per pair)
#   - Gaussian entropy prior with byte-calibrated centers/widths/bias
#   - No learned patch features, no hidden-state router
#   - No aux balance loss, no congestion/z-loss
#   - Pure prior-driven routing for interpretable, balanced expert assignment
#
# 预期效果:
#   - 4 pairs 各占 ~25% byte share
#   - BPB 接近或优于 entropy PhaseB (0.833)
#   - top2_same_pair_fraction ≈ 1.0 (paired routing 的固有特性)
#   - pair_load_cv < 0.2
#
# 用法:
#   # 1k screening (约 1 小时)
#   STEPS=1000 GPUS=0,1,2,3,4,5,6,7 ./scripts/patchmoe/run_bcfp_B_calibrated.sh
#
#   # 5k checkpoint
#   STEPS=5000 GPUS=0,1,2,3,4,5,6,7 ./scripts/patchmoe/run_bcfp_B_calibrated.sh
#
#   # 50k full run
#   STEPS=50000 GPUS=0,1,2,3,4,5,6,7 ./scripts/patchmoe/run_bcfp_B_calibrated.sh
# ============================================================

ROOT_DIR="/data1/pengfeigao/BP-MoE"
cd "$ROOT_DIR"

# ---- Launcher binaries --------------------------------------------------
TORCHRUN="${TORCHRUN:-$ROOT_DIR/.venv/bin/torchrun}"
if [[ ! -x "$TORCHRUN" ]]; then
  echo "Missing executable torchrun at $TORCHRUN" >&2
  exit 2
fi

# ---- Experiment identity --------------------------------------------------
RUN_NAME="${RUN_NAME:-bcfp_B_calibrated}"
OUT_ROOT="${OUT_ROOT:-$ROOT_DIR/blt1b_warmstart/bcfp_experiments}"
RUN_DIR="${RUN_DIR:-$OUT_ROOT/$RUN_NAME}"
STEPS="${STEPS:-1000}"

# ---- Data paths -----------------------------------------------------------
DATA_ROOT="${DATA_ROOT:-$ROOT_DIR/data}"
PREPROCESS_DIR="${PREPROCESS_DIR:-$ROOT_DIR/data/entropy_preprocessed_stage1}"
SOURCE="${SOURCE:-fineweb_edu_10bt}"
ENTROPY_MODEL_NAME="${ENTROPY_MODEL_NAME:-transformer_100m}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/tmp/unused.tokenizer.model}"

# ---- Model config (B-specific) -------------------------------------------
CONFIG="${CONFIG:-apps/main/configs/bcfp_B_calibrated.yaml}"
INIT_CKPT_PATH="${INIT_CKPT_PATH:-$ROOT_DIR/blt1b_warmstart/initial_weights/bcfp_pair_calibrated_dcp}"
CALIBRATION_PATH="${CALIBRATION_PATH:-$ROOT_DIR/blt1b_warmstart/initial_weights/bcfp_pair_calibrated/router_calibration.json}"

# ---- GPU allocation -------------------------------------------------------
NUM_GPUS="${NUM_GPUS:-8}"
if [[ -z "${GPUS:-}" ]]; then
  if [[ ! "$NUM_GPUS" =~ ^[0-9]+$ ]] || (( NUM_GPUS < 4 || NUM_GPUS > 8 )); then
    echo "NUM_GPUS must be between 4 and 8; got $NUM_GPUS" >&2
    exit 2
  fi
  GPUS="0"
  for (( gpu = 1; gpu < NUM_GPUS; gpu++ )); do
    GPUS+=",$gpu"
  done
fi

IFS=',' read -r -a GPU_IDS <<<"$GPUS"
NPROC_PER_NODE="${NPROC_PER_NODE:-${#GPU_IDS[@]}}"

if [[ ! "$NPROC_PER_NODE" =~ ^[0-9]+$ ]]; then
  echo "NPROC_PER_NODE must be an integer; got $NPROC_PER_NODE" >&2
  exit 2
fi

if (( ${#GPU_IDS[@]} != NPROC_PER_NODE )); then
  echo "GPUS count (${#GPU_IDS[@]}) != NPROC_PER_NODE ($NPROC_PER_NODE)" >&2
  exit 2
fi

# Expert parallelism: 8 experts, divisible by {1,2,4,8}
if (( NPROC_PER_NODE % 2 == 0 && 8 % 2 == 0 )); then
  DEFAULT_EP_SIZE=2
else
  DEFAULT_EP_SIZE=1
fi
EP_SIZE="${EP_SIZE:-$DEFAULT_EP_SIZE}"
if (( 8 % EP_SIZE != 0 || NPROC_PER_NODE % EP_SIZE != 0 )); then
  echo "EP_SIZE=$EP_SIZE must divide both num_experts=8 and NPROC_PER_NODE=$NPROC_PER_NODE" >&2
  exit 2
fi

# ---- Hyperparameters (tuned for BLT-1B PatchMoE) -------------------------
LR="${LR:-5e-5}"
WARMUP="${WARMUP:-100}"
BATCH_SIZE="${BATCH_SIZE:-1}"
SEQ_LEN="${SEQ_LEN:-256}"
MAX_ENCODER_SEQ_LENGTH="${MAX_ENCODER_SEQ_LENGTH:-1536}"
GRAD_ACC_STEPS="${GRAD_ACC_STEPS:-1}"
LOG_FREQ="${LOG_FREQ:-10}"

# Checkpoint frequency: adapt to run length
if (( STEPS <= 1000 )); then
  CKPT_EVERY=250
  EVAL_EVERY=250
elif (( STEPS <= 5000 )); then
  CKPT_EVERY=1000
  EVAL_EVERY=1000
else
  CKPT_EVERY=5000
  EVAL_EVERY=5000
fi

# ---- Pre-flight checks ----------------------------------------------------
check_file() {
  if [[ ! -f "$1" ]]; then
    echo "ERROR: Missing file: $1" >&2
    exit 2
  fi
}

check_dir() {
  if [[ ! -d "$1" ]]; then
    echo "ERROR: Missing directory: $1" >&2
    exit 2
  fi
}

check_file "$CONFIG"
check_file "$CALIBRATION_PATH"
check_dir "$INIT_CKPT_PATH"
check_dir "$DATA_ROOT"
check_dir "$PREPROCESS_DIR/$SOURCE/$ENTROPY_MODEL_NAME"

# Verify each GPU's data shard exists
mapfile -t source_chunks < <(find "$DATA_ROOT/$SOURCE" -maxdepth 1 -type f -name '*.chunk.*.jsonl' | sort | head -n "$NPROC_PER_NODE")
if (( ${#source_chunks[@]} < NPROC_PER_NODE )); then
  echo "ERROR: Need $NPROC_PER_NODE source chunks, found ${#source_chunks[@]}" >&2
  exit 2
fi
for source_chunk in "${source_chunks[@]}"; do
  shard_name="$(basename "$source_chunk").shard_00.arrow"
  shard_path="$PREPROCESS_DIR/$SOURCE/$ENTROPY_MODEL_NAME/$shard_name"
  if [[ ! -f "$shard_path" ]]; then
    echo "ERROR: Missing arrow shard: $shard_path" >&2
    exit 2
  fi
  if [[ ! -f "$shard_path.complete" ]]; then
    echo "WARNING: Missing .complete marker for $shard_name, creating it" >&2
    touch "$shard_path.complete"
  fi
done

# ---- Print configuration summary ------------------------------------------
echo "============================================"
echo " BCFP Experiment B: Calibrated Prior"
echo "============================================"
echo "  RUN_NAME:       $RUN_NAME"
echo "  RUN_DIR:        $RUN_DIR"
echo "  STEPS:          $STEPS"
echo "  GPUS:           $GPUS (${NPROC_PER_NODE} GPUs)"
echo "  EP_SIZE:        $EP_SIZE"
echo "  LR:             $LR"
echo "  BATCH_SIZE:     $BATCH_SIZE"
echo "  SEQ_LEN:        $SEQ_LEN"
echo "  INIT_CKPT:      $INIT_CKPT_PATH"
echo "  CALIBRATION:    $CALIBRATION_PATH"
echo "  CONFIG:         $CONFIG"
echo "============================================"
echo ""

# ---- Launch ---------------------------------------------------------------
mkdir -p "$RUN_DIR"

CUDA_VISIBLE_DEVICES="$GPUS" \
PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}" \
BLT_ALLOW_MISSING_FLEX_ATTENTION=1 \
BLT_SUPPRESS_ATTN_ERROR=1 \
NCCL_P2P_DISABLE=1 \
NCCL_IB_DISABLE=1 \
ENABLE_INTRA_NODE_COMM=1 \
NCCL_DEBUG=WARN \
"$TORCHRUN" --standalone --nproc-per-node="$NPROC_PER_NODE" -m bytelatent.train \
  config="$CONFIG" \
  name="$RUN_NAME" \
  dump_dir="$RUN_DIR" \
  steps="$STEPS" \
  max_steps="$STEPS" \
  grad_acc_steps="$GRAD_ACC_STEPS" \
  data.root_dir="$DATA_ROOT" \
  "data.sources={$SOURCE: 1.0}" \
  data.batch_size="$BATCH_SIZE" \
  data.seq_len="$SEQ_LEN" \
  data.max_encoder_seq_length="$MAX_ENCODER_SEQ_LENGTH" \
  data.load_async=false \
  data.preprocess_dir="$PREPROCESS_DIR" \
  data.entropy_model_name="$ENTROPY_MODEL_NAME" \
  data.tokenizer_args.init_kwargs.bpe_tokenizer_path="$TOKENIZER_PATH" \
  model.attn_impl=sdpa \
  model.max_seqlen="$SEQ_LEN" \
  model.max_length="$SEQ_LEN" \
  model.max_encoder_seq_length="$MAX_ENCODER_SEQ_LENGTH" \
  model.moe_entropy_prior_calibration_path="$CALIBRATION_PATH" \
  checkpoint.path="$RUN_DIR/checkpoints" \
  checkpoint.init_ckpt_path="$INIT_CKPT_PATH" \
  checkpoint.dump.every="$CKPT_EVERY" \
  checkpoint.dump.keep=1 \
  checkpoint.eval.every="$EVAL_EVERY" \
  checkpoint.eval.keep=1 \
  distributed.fsdp_type=full_shard \
  distributed.dp_shard="$NPROC_PER_NODE" \
  distributed.dp_replicate=1 \
  distributed.tp_size=1 \
  distributed.selective_activation_checkpointing=false \
  distributed.compile=false \
  distributed.model_dtype=bf16 \
  model.moe_ep_size="$EP_SIZE" \
  optim.lr="$LR" \
  optim.warmup="$WARMUP" \
  optim.lr_min_ratio=0.1 \
  optim.clip=0.0 \
  optim.fused=false \
  logging.freq="$LOG_FREQ" \
  logging.wandb=null \
  eval_on_gpus="$NPROC_PER_NODE"

echo ""
echo "Training completed. Run directory: $RUN_DIR"
