#!/usr/bin/env bash
set -euo pipefail

# Phase-2 PatchMoE smoke ablation grid.
# This keeps the model/data scale tiny and varies only the implemented
# routing-feature and load-balancing knobs from the roadmap.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

UV_BIN="${UV_BIN:-/home/ubuntu/.local/bin/uv}"
BASE_CONFIG="${BASE_CONFIG:-apps/main/configs/patchmoe_stage0.yaml}"
OUT_ROOT="${OUT_ROOT:-$ROOT_DIR/runs/phase2_ablation_smoke}"
DATA_ROOT="${DATA_ROOT:-$ROOT_DIR/data}"
PREPROCESS_DIR="${PREPROCESS_DIR:-$ROOT_DIR/data/entropy_preprocessed_smoke}"
SOURCE="${SOURCE:-fineweb_edu_10bt_entropy_smoke}"
ENTROPY_MODEL_NAME="${ENTROPY_MODEL_NAME:-transformer_100m}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/tmp/unused.tokenizer.model}"
STEPS="${STEPS:-1}"
MAX_STEPS="${MAX_STEPS:-$STEPS}"
BATCH_SIZE="${BATCH_SIZE:-1}"
SEQ_LEN="${SEQ_LEN:-64}"
MAX_ENCODER_SEQ_LENGTH="${MAX_ENCODER_SEQ_LENGTH:-256}"
MOE_BALANCE_LOSS_WEIGHT="${MOE_BALANCE_LOSS_WEIGHT:-0.01}"

export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export BLT_ALLOW_MISSING_FLEX_ATTENTION="${BLT_ALLOW_MISSING_FLEX_ATTENTION:-1}"
export BLT_SUPPRESS_ATTN_ERROR="${BLT_SUPPRESS_ATTN_ERROR:-1}"

CORE_VARIANTS=(
  dense_entropy_patches
  hidden_patch_balance
  entropy_patch_balance
  length_patch_balance
  entropy_length_patch_balance
  entropy_length_byte_balance
  entropy_length_entropy_byte_balance
)

EXTRA_VARIANTS=(
  entropy_entropy_byte_balance
  length_byte_balance
  top2_entropy_length_entropy_byte
  experts4_top2_entropy_length_entropy_byte
  byte_type_patch_balance
  entropy_length_byte_type_entropy_byte_balance
  entropy_length_byte_type_entropy_byte_congestion_w05
  entropy_length_byte_type_entropy_byte_congestion_zloss_w0001
)

usage() {
  cat <<USAGE
Usage: $0 [--list] [--print-overrides variant] [--all] [variant ...]

Runs Phase-2 PatchMoE smoke ablations on the entropy-preprocessed shard.

Environment overrides:
  STEPS=$STEPS
  OUT_ROOT=$OUT_ROOT
  PREPROCESS_DIR=$PREPROCESS_DIR
  SOURCE=$SOURCE

Default variants:
$(printf '  %s\n' "${CORE_VARIANTS[@]}")

Extra variants with --all:
$(printf '  %s\n' "${EXTRA_VARIANTS[@]}")
USAGE
}

list_variants() {
  printf '%s\n' "${CORE_VARIANTS[@]}" "${EXTRA_VARIANTS[@]}"
}

variant_overrides() {
  local variant="$1"
  case "$variant" in
    dense_entropy_patches)
      printf '%s\n' \
        model.moe_num_experts=0 \
        model.moe_top_k=1 \
        model.moe_balance_loss_weight="$MOE_BALANCE_LOSS_WEIGHT"0.0 \
        model.moe_router_use_patch_length=false \
        model.moe_router_use_patch_entropy=false \
        model.moe_balance_cost=patch
      ;;
    hidden_patch_balance)
      printf '%s\n' \
        model.moe_num_experts=2 \
        model.moe_top_k=1 \
        model.moe_balance_loss_weight="$MOE_BALANCE_LOSS_WEIGHT" \
        model.moe_router_use_patch_length=false \
        model.moe_router_use_patch_entropy=false \
        model.moe_balance_cost=patch
      ;;
    entropy_patch_balance)
      printf '%s\n' \
        model.moe_num_experts=2 \
        model.moe_top_k=1 \
        model.moe_balance_loss_weight="$MOE_BALANCE_LOSS_WEIGHT" \
        model.moe_router_use_patch_length=false \
        model.moe_router_use_patch_entropy=true \
        model.moe_balance_cost=patch
      ;;
    length_patch_balance)
      printf '%s\n' \
        model.moe_num_experts=2 \
        model.moe_top_k=1 \
        model.moe_balance_loss_weight="$MOE_BALANCE_LOSS_WEIGHT" \
        model.moe_router_use_patch_length=true \
        model.moe_router_use_patch_entropy=false \
        model.moe_balance_cost=patch
      ;;
    entropy_length_patch_balance)
      printf '%s\n' \
        model.moe_num_experts=2 \
        model.moe_top_k=1 \
        model.moe_balance_loss_weight="$MOE_BALANCE_LOSS_WEIGHT" \
        model.moe_router_use_patch_length=true \
        model.moe_router_use_patch_entropy=true \
        model.moe_balance_cost=patch
      ;;
    entropy_length_byte_balance)
      printf '%s\n' \
        model.moe_num_experts=2 \
        model.moe_top_k=1 \
        model.moe_balance_loss_weight="$MOE_BALANCE_LOSS_WEIGHT" \
        model.moe_router_use_patch_length=true \
        model.moe_router_use_patch_entropy=true \
        model.moe_balance_cost=byte
      ;;
    entropy_length_entropy_byte_balance)
      printf '%s\n' \
        model.moe_num_experts=2 \
        model.moe_top_k=1 \
        model.moe_balance_loss_weight="$MOE_BALANCE_LOSS_WEIGHT" \
        model.moe_router_use_patch_length=true \
        model.moe_router_use_patch_entropy=true \
        model.moe_balance_cost=entropy_byte
      ;;
    byte_type_patch_balance)
      printf '%s\n' \
        model.moe_num_experts=2 \
        model.moe_top_k=1 \
        model.moe_balance_loss_weight="$MOE_BALANCE_LOSS_WEIGHT" \
        model.moe_router_use_patch_length=false \
        model.moe_router_use_patch_entropy=false \
        model.moe_router_use_patch_byte_features=true \
        model.moe_balance_cost=patch
      ;;
    entropy_length_byte_type_entropy_byte_balance)
      printf '%s\n' \
        model.moe_num_experts=2 \
        model.moe_top_k=1 \
        model.moe_balance_loss_weight="$MOE_BALANCE_LOSS_WEIGHT" \
        model.moe_router_use_patch_length=true \
        model.moe_router_use_patch_entropy=true \
        model.moe_router_use_patch_byte_features=true \
        model.moe_balance_cost=entropy_byte
      ;;
    entropy_length_byte_type_entropy_byte_congestion_w05)
      printf '%s\n' \
        model.moe_num_experts=2 \
        model.moe_top_k=1 \
        model.moe_balance_loss_weight="$MOE_BALANCE_LOSS_WEIGHT" \
        model.moe_router_congestion_weight=0.5 \
        model.moe_router_use_patch_length=true \
        model.moe_router_use_patch_entropy=true \
        model.moe_router_use_patch_byte_features=true \
        model.moe_balance_cost=entropy_byte
      ;;
    entropy_length_byte_type_entropy_byte_congestion_zloss_w0001)
      printf '%s\n' \
        model.moe_num_experts=2 \
        model.moe_top_k=1 \
        model.moe_balance_loss_weight="$MOE_BALANCE_LOSS_WEIGHT" \
        model.moe_router_congestion_weight=0.5 \
        model.moe_router_z_loss_weight=0.001 \
        model.moe_router_use_patch_length=true \
        model.moe_router_use_patch_entropy=true \
        model.moe_router_use_patch_byte_features=true \
        model.moe_balance_cost=entropy_byte
      ;;
    entropy_entropy_byte_balance)
      printf '%s\n' \
        model.moe_num_experts=2 \
        model.moe_top_k=1 \
        model.moe_balance_loss_weight="$MOE_BALANCE_LOSS_WEIGHT" \
        model.moe_router_use_patch_length=false \
        model.moe_router_use_patch_entropy=true \
        model.moe_balance_cost=entropy_byte
      ;;
    length_byte_balance)
      printf '%s\n' \
        model.moe_num_experts=2 \
        model.moe_top_k=1 \
        model.moe_balance_loss_weight="$MOE_BALANCE_LOSS_WEIGHT" \
        model.moe_router_use_patch_length=true \
        model.moe_router_use_patch_entropy=false \
        model.moe_balance_cost=byte
      ;;
    top2_entropy_length_entropy_byte)
      printf '%s\n' \
        model.moe_num_experts=2 \
        model.moe_top_k=2 \
        model.moe_balance_loss_weight="$MOE_BALANCE_LOSS_WEIGHT" \
        model.moe_router_use_patch_length=true \
        model.moe_router_use_patch_entropy=true \
        model.moe_balance_cost=entropy_byte
      ;;
    experts4_top2_entropy_length_entropy_byte)
      printf '%s\n' \
        model.moe_num_experts=4 \
        model.moe_top_k=2 \
        model.moe_balance_loss_weight="$MOE_BALANCE_LOSS_WEIGHT" \
        model.moe_router_use_patch_length=true \
        model.moe_router_use_patch_entropy=true \
        model.moe_balance_cost=entropy_byte
      ;;
    *)
      echo "Unknown Phase-2 ablation variant: $variant" >&2
      echo "Known variants:" >&2
      list_variants >&2
      return 1
      ;;
  esac
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "${1:-}" == "--list" ]]; then
  list_variants
  exit 0
fi

if [[ "${1:-}" == "--print-overrides" ]]; then
  if [[ "${2:-}" == "" || "${3:-}" != "" ]]; then
    echo "usage: $0 --print-overrides VARIANT" >&2
    exit 2
  fi
  variant_overrides "$2"
  exit 0
fi

declare -a variants
if [[ "${1:-}" == "--all" ]]; then
  variants=("${CORE_VARIANTS[@]}" "${EXTRA_VARIANTS[@]}")
  shift
elif [[ "$#" -gt 0 ]]; then
  variants=("$@")
else
  variants=("${CORE_VARIANTS[@]}")
fi

if [[ ! -d "$PREPROCESS_DIR/$SOURCE/$ENTROPY_MODEL_NAME" ]]; then
  echo "Missing entropy-preprocessed shard directory:" >&2
  echo "  $PREPROCESS_DIR/$SOURCE/$ENTROPY_MODEL_NAME" >&2
  echo "Create it first with bytelatent.preprocess.preprocess_entropies." >&2
  exit 1
fi

mkdir -p "$OUT_ROOT"

common_overrides=(
  "config=$BASE_CONFIG"
  "steps=$STEPS"
  "max_steps=$MAX_STEPS"
  "seed=777"
  "grad_acc_steps=1"
  "data.root_dir=$DATA_ROOT"
  "data.sources={$SOURCE: 1.0}"
  "data.file_format=arrow"
  "data.preprocess_dir=$PREPROCESS_DIR"
  "data.load_async=false"
  "data.batch_size=$BATCH_SIZE"
  "data.seq_len=$SEQ_LEN"
  "data.max_encoder_seq_length=$MAX_ENCODER_SEQ_LENGTH"
  "data.entropy_model_name=$ENTROPY_MODEL_NAME"
  "data.tokenizer_args.init_kwargs.bpe_tokenizer_path=$TOKENIZER_PATH"
  "data.patcher_args.patching_mode=entropy"
  "data.patcher_args.patch_size=6.0"
  "model.patching_mode=entropy"
  "model.moe_router_use_patch_byte_features=false"
  "model.moe_router_congestion_weight=0.0"
  "model.moe_router_z_loss_weight=0.0"
  "model.attn_impl=sdpa"
  "model.cross_attn_encoder=true"
  "model.cross_attn_decoder=true"
  "model.cross_attn_use_flex_attention=false"
  "model.dim=64"
  "model.dim_token=64"
  "model.dim_global=64"
  "model.dim_local_encoder=64"
  "model.dim_local_decoder=64"
  "model.max_encoder_seq_length=$MAX_ENCODER_SEQ_LENGTH"
  "model.max_seqlen=$SEQ_LEN"
  "model.max_length=$SEQ_LEN"
  "model.n_heads=4"
  "model.n_heads_global=4"
  "model.n_heads_local_encoder=4"
  "model.n_heads_local_decoder=4"
  "model.n_layers_global=2"
  "model.n_layers_local_encoder=1"
  "model.n_layers_local_decoder=1"
  "model.cross_attn_k=1"
  "model.cross_attn_nheads=1"
  "model.cross_attn_window_encoder=$SEQ_LEN"
  "model.cross_attn_window_decoder=$SEQ_LEN"
  "model.local_attention_window_len=$SEQ_LEN"
  "distributed.fsdp_type=no_shard"
  "distributed.dp_shard=1"
  "distributed.dp_replicate=1"
  "distributed.matmul_allow_tf32=true"
  "checkpoint.dump.every=-1"
  "checkpoint.eval.every=-1"
  "logging.freq=1"
)

for variant in "${variants[@]}"; do
  run_dir="$OUT_ROOT/$variant"
  mapfile -t per_variant_overrides < <(variant_overrides "$variant")

  echo "==> Running Phase-2 ablation: $variant"
  "$UV_BIN" run python -m bytelatent.train \
    "${common_overrides[@]}" \
    "dump_dir=$run_dir" \
    "name=phase2_ablation_$variant" \
    "checkpoint.path=$run_dir/checkpoints" \
    "${per_variant_overrides[@]}"
done
