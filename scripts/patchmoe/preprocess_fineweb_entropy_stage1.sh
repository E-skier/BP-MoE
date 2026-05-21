#!/usr/bin/env bash
set -euo pipefail

# Generate entropy arrow shards for the Stage-1 PatchMoE candidate.
#
# With the current training data iterator and NPROC_PER_NODE=2, only the first
# two fineweb_edu_10bt chunks are consumed. The default MAX_FILES=2 prepares
# exactly those chunks; set MAX_FILES=all to prepare every local chunk.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

UV_BIN="${UV_BIN:-/home/ubuntu/.local/bin/uv}"
SOURCE_DIR="${SOURCE_DIR:-$ROOT_DIR/data/fineweb_edu_10bt}"
SOURCE_NAME="${SOURCE_NAME:-$(basename "$SOURCE_DIR")}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/data/entropy_preprocessed_stage1}"
ENTROPY_MODEL_NAME="${ENTROPY_MODEL_NAME:-transformer_100m}"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/$SOURCE_NAME/$ENTROPY_MODEL_NAME}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/runs/preprocess_entropy_stage1/$SOURCE_NAME}"

ENTROPY_MODEL_CHECKPOINT_DIR="${ENTROPY_MODEL_CHECKPOINT_DIR:-$ROOT_DIR/hf-weights/entropy_model}"
ENTROPY_MODEL_STATE_DICT_PATH="${ENTROPY_MODEL_STATE_DICT_PATH:-$ROOT_DIR/hf-weights/entropy_model/consolidated.pth}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/tmp/unused.tokenizer.model}"

NPROC="${NPROC:-2}"
MAX_FILES="${MAX_FILES:-2}"
DRY_RUN="${DRY_RUN:-0}"

export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export BLT_ALLOW_MISSING_FLEX_ATTENTION="${BLT_ALLOW_MISSING_FLEX_ATTENTION:-1}"
export BLT_SUPPRESS_ATTN_ERROR="${BLT_SUPPRESS_ATTN_ERROR:-1}"

if [[ ! -d "$SOURCE_DIR" ]]; then
  echo "Missing SOURCE_DIR: $SOURCE_DIR" >&2
  exit 1
fi
if [[ ! -f "$ENTROPY_MODEL_CHECKPOINT_DIR/params.json" ]]; then
  echo "Missing entropy params.json in: $ENTROPY_MODEL_CHECKPOINT_DIR" >&2
  exit 1
fi
if [[ ! -f "$ENTROPY_MODEL_STATE_DICT_PATH" ]]; then
  echo "Missing entropy state dict: $ENTROPY_MODEL_STATE_DICT_PATH" >&2
  exit 1
fi

mapfile -t files < <(find "$SOURCE_DIR" -maxdepth 1 -type f -name '*.chunk.*.jsonl' | sort)
if [[ "${#files[@]}" -eq 0 ]]; then
  echo "No .chunk.*.jsonl files found in: $SOURCE_DIR" >&2
  exit 1
fi
if [[ "$MAX_FILES" != "all" ]]; then
  files=("${files[@]:0:$MAX_FILES}")
fi

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

echo "Preprocessing ${#files[@]} file(s)"
echo "  source: $SOURCE_DIR"
echo "  output: $OUTPUT_DIR"
echo "  logs:   $LOG_DIR"
echo "  nproc:  $NPROC"

running=0
status=0
launched=0

wait_for_one() {
  if ! wait -n; then
    status=1
  fi
  running=$((running - 1))
}

for input_file in "${files[@]}"; do
  basename="$(basename "$input_file")"
  output_file="$OUTPUT_DIR/${basename}.shard_00.arrow"
  complete_file="${output_file}.complete"
  log_file="$LOG_DIR/${basename}.shard_00.log"

  if [[ -f "$complete_file" ]]; then
    echo "Skipping completed shard: $complete_file"
    continue
  fi

  gpu_id=$((launched % NPROC))
  echo "Launching $basename on visible GPU $gpu_id"

  args=(
    python -u -m bytelatent.preprocess.preprocess_entropies
    "$input_file"
    "$output_file"
    --patching-device cuda
    --entropy-model-checkpoint-dir "$ENTROPY_MODEL_CHECKPOINT_DIR"
    --entropy-model-state-dict-path "$ENTROPY_MODEL_STATE_DICT_PATH"
    --bpe-tokenizer-path "$TOKENIZER_PATH"
  )
  if [[ "$DRY_RUN" == "1" ]]; then
    args+=(--dry-run)
  fi

  (
    export CUDA_VISIBLE_DEVICES="$gpu_id"
    "${UV_BIN}" run "${args[@]}"
  ) >"$log_file" 2>&1 &

  launched=$((launched + 1))
  running=$((running + 1))
  if [[ "$running" -ge "$NPROC" ]]; then
    wait_for_one
  fi
done

while [[ "$running" -gt 0 ]]; do
  wait_for_one
done

if [[ "$status" -ne 0 ]]; then
  echo "One or more preprocessing jobs failed. Check logs in: $LOG_DIR" >&2
  exit "$status"
fi

echo "Entropy preprocessing complete."
