#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/data1/pengfeigao/BP-MoE"
cd "$ROOT_DIR"

TORCHRUN="${TORCHRUN:-$ROOT_DIR/.venv/bin/torchrun}"
if [[ ! -x "$TORCHRUN" ]]; then
  echo "Missing executable torchrun at $TORCHRUN" >&2
  echo "Create or sync the uv environment first, or override with TORCHRUN=/path/to/torchrun" >&2
  exit 2
fi

CONFIG="${CONFIG:-apps/main/configs/patchmoe_blt1b_warmstart.yaml}"
RUN_NAME="${RUN_NAME:-entropy_only_blt1b_50000step_active_matched}"
OUT_ROOT="${OUT_ROOT:-$ROOT_DIR/runs/blt1b_warmstart/active_matched_50k}"
RUN_DIR="${RUN_DIR:-$OUT_ROOT/$RUN_NAME}"
DATA_ROOT="${DATA_ROOT:-$ROOT_DIR/data}"
PREPROCESS_DIR="${PREPROCESS_DIR:-$ROOT_DIR/data/entropy_preprocessed_stage1}"
SOURCE="${SOURCE:-fineweb_edu_10bt}"
ENTROPY_MODEL_NAME="${ENTROPY_MODEL_NAME:-transformer_100m}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/tmp/unused.tokenizer.model}"
INIT_CKPT_PATH="${INIT_CKPT_PATH:-$ROOT_DIR/hf-weights/blt_1b_patchmoe_entropy_only_active_matched_dcp}"

check_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "Missing file: $path" >&2
    exit 2
  fi
}

check_dir() {
  local path="$1"
  if [[ ! -d "$path" ]]; then
    echo "Missing directory: $path" >&2
    exit 2
  fi
}

if [[ ! -d "$INIT_CKPT_PATH" ]]; then
  echo "Missing warm-start DCP directory: $INIT_CKPT_PATH" >&2
  echo "Generate it with:" >&2
  echo "  $ROOT_DIR/.venv/bin/python scripts/patchmoe/prepare_blt1b_patchmoe_warmstart.py --source hf-weights/blt_1b/consolidated.pth --output-dir $INIT_CKPT_PATH --num-experts 8 --top-k 2 --patch-features entropy --expert-ffn-dim-multiplier 0.5 --force" >&2
  echo "Or point INIT_CKPT_PATH=/path/to/existing_dcp at launch time." >&2
  exit 2
fi

# Defaults to all 8 local A100s. Override either NUM_GPUS=4..8 for the first
# N GPUs, or GPUS=0,2,4,6 to choose explicit device ids.
NUM_GPUS="${NUM_GPUS:-8}"
if [[ -z "${GPUS:-}" ]]; then
  if [[ ! "$NUM_GPUS" =~ ^[0-9]+$ ]] || (( NUM_GPUS < 4 || NUM_GPUS > 8 )); then
    echo "NUM_GPUS must be an integer between 4 and 8; got $NUM_GPUS" >&2
    exit 2
  fi
  GPUS="0"
  for (( gpu = 1; gpu < NUM_GPUS; gpu++ )); do
    GPUS+=",$gpu"
  done
fi

IFS=',' read -r -a GPU_IDS <<<"$GPUS"
NPROC_PER_NODE="${NPROC_PER_NODE:-${#GPU_IDS[@]}}"

if [[ ! "$NPROC_PER_NODE" =~ ^[0-9]+$ ]] || (( NPROC_PER_NODE < 4 || NPROC_PER_NODE > 8 )); then
  echo "NPROC_PER_NODE must be between 4 and 8 for this launcher; got $NPROC_PER_NODE" >&2
  exit 2
fi

if (( ${#GPU_IDS[@]} != NPROC_PER_NODE )); then
  echo "GPUS has ${#GPU_IDS[@]} entries but NPROC_PER_NODE=$NPROC_PER_NODE" >&2
  echo "Use matching values, for example: GPUS=0,1,2,3 NPROC_PER_NODE=4 $0" >&2
  exit 2
fi

if (( NPROC_PER_NODE % 2 == 0 )); then
  DEFAULT_EP_SIZE=2
else
  DEFAULT_EP_SIZE=1
fi
EP_SIZE="${EP_SIZE:-$DEFAULT_EP_SIZE}"

if [[ ! "$EP_SIZE" =~ ^[0-9]+$ ]] || (( EP_SIZE <= 0 || NPROC_PER_NODE % EP_SIZE != 0 )); then
  echo "EP_SIZE must be positive and divide NPROC_PER_NODE; got EP_SIZE=$EP_SIZE NPROC_PER_NODE=$NPROC_PER_NODE" >&2
  exit 2
fi

if (( 8 % EP_SIZE != 0 )); then
  echo "EP_SIZE must divide the configured 8 experts; got EP_SIZE=$EP_SIZE" >&2
  exit 2
fi

check_file "$CONFIG"
check_file "$ROOT_DIR/hf-weights/blt_1b/params.json"
check_dir "$DATA_ROOT"
check_dir "$PREPROCESS_DIR/$SOURCE/$ENTROPY_MODEL_NAME"

mapfile -t source_chunks < <(find "$DATA_ROOT/$SOURCE" -maxdepth 1 -type f -name '*.chunk.*.jsonl' | sort | head -n "$NPROC_PER_NODE")
if (( ${#source_chunks[@]} < NPROC_PER_NODE )); then
  echo "Need at least $NPROC_PER_NODE source chunks for $NPROC_PER_NODE GPU(s), found ${#source_chunks[@]} in $DATA_ROOT/$SOURCE" >&2
  exit 2
fi
for source_chunk in "${source_chunks[@]}"; do
  shard_glob="$PREPROCESS_DIR/$SOURCE/$ENTROPY_MODEL_NAME/$(basename "$source_chunk").shard_*.arrow"
  mapfile -t arrow_shards < <(find "$PREPROCESS_DIR/$SOURCE/$ENTROPY_MODEL_NAME" -maxdepth 1 -type f -name "$(basename "$source_chunk").shard_*.arrow" | sort)
  if (( ${#arrow_shards[@]} == 0 )); then
    echo "Missing entropy-preprocessed arrow shard(s) for: $source_chunk" >&2
    echo "Expected match: $shard_glob" >&2
    echo "Run preprocessing first, for example: NUM_GPUS=$NPROC_PER_NODE NPROC=$NPROC_PER_NODE MAX_FILES=$NPROC_PER_NODE ./scripts/patchmoe/preprocess_fineweb_entropy_stage1.sh" >&2
    exit 2
  fi
  for arrow_shard in "${arrow_shards[@]}"; do
    if [[ ! -f "$arrow_shard.complete" ]]; then
      echo "Missing completion marker: $arrow_shard.complete" >&2
      echo "The data loader refuses to consume entropy shards without this marker." >&2
      echo "If the arrow file was fully written and verified, create the marker with: touch '$arrow_shard.complete'" >&2
      exit 2
    fi
  done
done

echo "Launching entropy_only_blt1b_50000step_active_matched"
echo "  ROOT_DIR=$ROOT_DIR"
echo "  TORCHRUN=$TORCHRUN"
echo "  CONFIG=$CONFIG"
echo "  RUN_DIR=$RUN_DIR"
echo "  DATA_ROOT=$DATA_ROOT"
echo "  PREPROCESS_DIR=$PREPROCESS_DIR"
echo "  INIT_CKPT_PATH=$INIT_CKPT_PATH"
echo "  GPUS=$GPUS"
echo "  NPROC_PER_NODE=$NPROC_PER_NODE"
echo "  EP_SIZE=$EP_SIZE"

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
dump_dir="$RUN_DIR" \
name="$RUN_NAME" \
steps=50000 \
max_steps=50000 \
grad_acc_steps=1 \
data.root_dir="$DATA_ROOT" \
"data.sources={$SOURCE: 1.0}" \
data.batch_size=1 \
data.seq_len=256 \
data.max_encoder_seq_length=1536 \
data.load_async=false \
data.preprocess_dir="$PREPROCESS_DIR" \
data.entropy_model_name="$ENTROPY_MODEL_NAME" \
data.tokenizer_args.init_kwargs.bpe_tokenizer_path="$TOKENIZER_PATH" \
model.attn_impl=sdpa \
model.max_seqlen=256 \
model.max_length=256 \
model.max_encoder_seq_length=1536 \
model.moe_num_experts=8 \
model.moe_top_k=2 \
model.moe_ep_size="$EP_SIZE" \
model.moe_ffn_dim_multiplier=0.5 \
model.moe_balance_loss_weight=0.05 \
model.moe_router_jitter=0.01 \
model.moe_router_congestion_weight=0.0 \
model.moe_router_z_loss_weight=0.0 \
model.moe_router_use_patch_length=false \
model.moe_router_use_patch_entropy=true \
model.moe_router_use_patch_byte_features=false \
model.moe_balance_cost=byte \
checkpoint.path="$RUN_DIR/checkpoints" \
checkpoint.init_ckpt_path="$INIT_CKPT_PATH" \
checkpoint.dump.every=1000 \
checkpoint.dump.keep=1 \
checkpoint.eval.every=1000 \
checkpoint.eval.keep=1 \
distributed.fsdp_type=full_shard \
distributed.dp_shard="$NPROC_PER_NODE" \
distributed.dp_replicate=1 \
distributed.tp_size=1 \
distributed.selective_activation_checkpointing=false \
distributed.compile=false \
distributed.model_dtype=bf16 \
optim.lr=0.00005 \
optim.warmup=100 \
optim.lr_min_ratio=0.1 \
optim.clip=0.0 \
optim.fused=false \
logging.freq=10 \
logging.wandb=null \
eval_on_gpus="$NPROC_PER_NODE"
