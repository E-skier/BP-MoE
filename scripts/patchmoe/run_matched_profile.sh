#!/usr/bin/env bash
set -euo pipefail

# Profile the active-compute comparison variants on one GPU with an identical
# batch shape. xformers writes the measured FLOPs summary from the PyTorch trace.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

UV_BIN="${UV_BIN:-/home/ubuntu/.local/bin/uv}"
CONFIG="${CONFIG:-apps/main/configs/patchmoe_stage1_candidate.yaml}"
OUT_ROOT="${OUT_ROOT:-$ROOT_DIR/runs/patchmoe_matched_profile}"

DATA_ROOT="${DATA_ROOT:-$ROOT_DIR/data}"
SOURCE="${SOURCE:-fineweb_edu_10bt}"
PREPROCESS_DIR="${PREPROCESS_DIR:-$ROOT_DIR/data/entropy_preprocessed_stage1}"
ENTROPY_MODEL_NAME="${ENTROPY_MODEL_NAME:-transformer_100m}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/tmp/unused.tokenizer.model}"

VARIANTS="${VARIANTS:-dense_compute_matched byte_hidden_only_w005 byte_entropy_w005 byte_entropy_type_w005}"
KNOWN_VARIANTS="dense_compute_matched byte_hidden_only_w005 byte_entropy_w005 byte_entropy_type_w005 byte_entropy_type_congestion_w05 byte_entropy_type_congestion_zloss_w0001"
SEED="${SEED:-779}"
STEPS="${STEPS:-40}"
BATCH_SIZE="${BATCH_SIZE:-1}"
SEQ_LEN="${SEQ_LEN:-2048}"
LOG_FREQ="${LOG_FREQ:-1}"
KEEP_PROFILE_CHECKPOINTS="${KEEP_PROFILE_CHECKPOINTS:-0}"

MEM_WARMUP="${MEM_WARMUP:-10}"
MEM_STEPS="${MEM_STEPS:-2}"
PROFILE_WARMUP="${PROFILE_WARMUP:-20}"
PROFILE_STEPS="${PROFILE_STEPS:-4}"

export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export BLT_ALLOW_MISSING_FLEX_ATTENTION="${BLT_ALLOW_MISSING_FLEX_ATTENTION:-1}"
export BLT_SUPPRESS_ATTN_ERROR="${BLT_SUPPRESS_ATTN_ERROR:-1}"

if [[ "${1:-}" == "--list" ]]; then
  printf "%s\n" $KNOWN_VARIANTS
  exit 0
fi

preprocessed_source_dir="$PREPROCESS_DIR/$SOURCE/$ENTROPY_MODEL_NAME"
if [[ ! -d "$preprocessed_source_dir" ]]; then
  echo "Missing training entropy-preprocessed data: $preprocessed_source_dir" >&2
  exit 1
fi
if ! find "$preprocessed_source_dir" -name '*.arrow.complete' -print -quit | grep -q .; then
  echo "No completed training entropy arrow shards found in: $preprocessed_source_dir" >&2
  exit 1
fi
if (( STEPS <= PROFILE_WARMUP + PROFILE_STEPS )); then
  echo "STEPS must be greater than PROFILE_WARMUP + PROFILE_STEPS" >&2
  exit 1
fi

variant_overrides() {
  local variant="$1"
  case "$variant" in
    dense_compute_matched)
      printf '%s\n' \
        model.moe_num_experts=0 \
        model.moe_top_k=1 \
        model.moe_balance_loss_weight=0.0 \
        model.moe_router_jitter=0.0 \
        model.moe_router_congestion_weight=0.0 \
        model.moe_router_z_loss_weight=0.0 \
        model.moe_router_use_patch_length=false \
        model.moe_router_use_patch_entropy=false \
        model.moe_router_use_patch_byte_features=false \
        model.moe_balance_cost=patch \
        model.ffn_dim_multiplier_global=2.0
      ;;
    byte_hidden_only_w005)
      printf '%s\n' \
        model.moe_num_experts=8 \
        model.moe_top_k=2 \
        model.moe_balance_loss_weight=0.05 \
        model.moe_router_jitter=0.01 \
        model.moe_router_congestion_weight=0.0 \
        model.moe_router_z_loss_weight=0.0 \
        model.moe_router_use_patch_length=false \
        model.moe_router_use_patch_entropy=false \
        model.moe_router_use_patch_byte_features=false \
        model.moe_balance_cost=byte
      ;;
    byte_entropy_w005)
      printf '%s\n' \
        model.moe_num_experts=8 \
        model.moe_top_k=2 \
        model.moe_balance_loss_weight=0.05 \
        model.moe_router_jitter=0.01 \
        model.moe_router_congestion_weight=0.0 \
        model.moe_router_z_loss_weight=0.0 \
        model.moe_router_use_patch_length=false \
        model.moe_router_use_patch_entropy=true \
        model.moe_router_use_patch_byte_features=false \
        model.moe_balance_cost=byte
      ;;
    byte_entropy_type_w005)
      printf "%s\n" \
        model.moe_num_experts=8 \
        model.moe_top_k=2 \
        model.moe_balance_loss_weight=0.05 \
        model.moe_router_jitter=0.01 \
        model.moe_router_congestion_weight=0.0 \
        model.moe_router_z_loss_weight=0.0 \
        model.moe_router_use_patch_length=false \
        model.moe_router_use_patch_entropy=true \
        model.moe_router_use_patch_byte_features=true \
        model.moe_balance_cost=byte
      ;;
    byte_entropy_type_congestion_w05)
      printf "%s\n" \
        model.moe_num_experts=8 \
        model.moe_top_k=2 \
        model.moe_balance_loss_weight=0.05 \
        model.moe_router_jitter=0.01 \
        model.moe_router_congestion_weight=0.5 \
        model.moe_router_z_loss_weight=0.0 \
        model.moe_router_use_patch_length=false \
        model.moe_router_use_patch_entropy=true \
        model.moe_router_use_patch_byte_features=true \
        model.moe_balance_cost=byte
      ;;
    byte_entropy_type_congestion_zloss_w0001)
      printf "%s\n" \
        model.moe_num_experts=8 \
        model.moe_top_k=2 \
        model.moe_balance_loss_weight=0.05 \
        model.moe_router_jitter=0.01 \
        model.moe_router_congestion_weight=0.5 \
        model.moe_router_z_loss_weight=0.001 \
        model.moe_router_use_patch_length=false \
        model.moe_router_use_patch_entropy=true \
        model.moe_router_use_patch_byte_features=true \
        model.moe_balance_cost=byte
      ;;
    *)
      echo "Unknown profiling variant: $variant" >&2
      echo "Known variants: $KNOWN_VARIANTS" >&2
      return 1
      ;;
  esac
}

if [[ "${1:-}" == "--print-overrides" ]]; then
  if [[ "${2:-}" == "" || "${3:-}" != "" ]]; then
    echo "usage: $0 --print-overrides VARIANT" >&2
    exit 2
  fi
  variant_overrides "$2"
  exit 0
fi

run_variant() {
  local variant="$1"
  local run_dir="$OUT_ROOT/$variant"

  mkdir -p "$run_dir"
  mapfile -t overrides < <(variant_overrides "$variant")

  echo "Profiling matched variant: $variant"
  "$UV_BIN" run torchrun --standalone --nproc-per-node=1 \
    -m bytelatent.train \
    "config=$CONFIG" \
    "dump_dir=$run_dir" \
    "name=profile_$variant" \
    "steps=$STEPS" \
    "max_steps=$STEPS" \
    "seed=$SEED" \
    "model.seed=$SEED" \
    "data.seed=$SEED" \
    "grad_acc_steps=1" \
    "data.root_dir=$DATA_ROOT" \
    "data.sources={$SOURCE: 1.0}" \
    "data.batch_size=$BATCH_SIZE" \
    "data.seq_len=$SEQ_LEN" \
    "data.load_async=false" \
    "logging.freq=$LOG_FREQ" \
    "data.preprocess_dir=$PREPROCESS_DIR" \
    "data.entropy_model_name=$ENTROPY_MODEL_NAME" \
    "data.tokenizer_args.init_kwargs.bpe_tokenizer_path=$TOKENIZER_PATH" \
    "checkpoint.path=$run_dir/checkpoints" \
    "checkpoint.dump.every=1000000" \
    "checkpoint.dump.keep=1" \
    "checkpoint.eval.every=1000000" \
    "checkpoint.eval.keep=1" \
    "distributed.dp_shard=1" \
    "distributed.dp_replicate=1" \
    "eval_on_gpus=1" \
    "profiling.run=true" \
    "profiling.trace_folder=profiling" \
    "profiling.mem_warmup=$MEM_WARMUP" \
    "profiling.mem_steps=$MEM_STEPS" \
    "profiling.profile_warmup=$PROFILE_WARMUP" \
    "profiling.profile_steps=$PROFILE_STEPS" \
    "env.ENABLE_INTRA_NODE_COMM=\"1\"" \
    "env.NCCL_DEBUG=WARN" \
    "${overrides[@]}"

  if [[ "$KEEP_PROFILE_CHECKPOINTS" != "1" ]]; then
    rm -rf "$run_dir/checkpoints"
  fi
}

mkdir -p "$OUT_ROOT"
for variant in $VARIANTS; do
  run_variant "$variant"
done

"$UV_BIN" run python - "$OUT_ROOT" $VARIANTS <<'PY'
import csv
import json
import re
import sys
from pathlib import Path
from statistics import fmean

out_root = Path(sys.argv[1])
variants = sys.argv[2:]
rows = []
for variant in variants:
    run_dir = out_root / variant
    train_log = (run_dir / "train.log").read_text(errors="replace")
    metrics = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text().splitlines()
        if line.strip()
    ]

    def extract(pattern):
        matches = re.findall(pattern, train_log)
        if not matches:
            raise RuntimeError(f"Missing {pattern!r} in {run_dir / 'train.log'}")
        return float(matches[-1])

    # Use the stable pre-profiler interval so trace overhead does not distort WPS.
    steady = [
        row for row in metrics
        if 5 <= row["global_step"] < 10
    ]
    rows.append(
        {
            "variant": variant,
            "measured_tflop_per_step": extract(r"TFlop/step\s*:\s*([0-9.]+)"),
            "profile_step_time_ms": extract(r"Step time \(ms\)\s*:\s*([0-9.]+)"),
            "profile_tflops": extract(r"(?m)^\s*TFlops\s*:\s*([0-9.]+)"),
            "profile_hfu": extract(r"HFU\s*:\s*([0-9.]+)"),
            "profile_mfu": extract(r"MFU\s*:\s*([0-9.]+)"),
            "steady_wps": fmean(row["speed/wps"] for row in steady),
            "steady_iter_time_ms": 1000
            * fmean(row["speed/curr_iter_time"] for row in steady),
            "max_active_gib": max(row["memory/max_active_gib"] for row in metrics),
            "max_reserved_gib": max(row["memory/max_reserved_gib"] for row in metrics),
        }
    )

summary_path = out_root / "summary.csv"
with summary_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
print(f"Wrote profiler summary: {summary_path}")
PY

echo "Matched profiling complete."
echo "Profile root: $OUT_ROOT"
echo "Summary:      $OUT_ROOT/summary.csv"
