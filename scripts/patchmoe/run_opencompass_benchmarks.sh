#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

CKPT_DIR="${CKPT_DIR:-${1:-}}"
if [[ -z "$CKPT_DIR" ]]; then
  echo "usage: CKPT_DIR=/path/to/consolidated_or_dcp_checkpoint $0" >&2
  echo "       or: $0 /path/to/consolidated_or_dcp_checkpoint" >&2
  exit 2
fi

OPENCOMPASS_BIN="${OPENCOMPASS_BIN:-opencompass}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CONFIG="${CONFIG:-apps/opencompass/patchmoe_benchmarks.py}"
RUN_NAME="${RUN_NAME:-$(basename "$(dirname "$CKPT_DIR")")_$(basename "$CKPT_DIR")}"
WORKDIR="${WORKDIR:-$ROOT_DIR/runs/opencompass/$RUN_NAME}"
MODE="${MODE:-all}"
DATASETS="${DATASETS:-mmlu_ppl hellaswag_ppl ARC_c_ppl ARC_e_ppl obqa_ppl piqa_ppl SuperGLUE_BoolQ_ppl}"
MAX_NUM_WORKER="${MAX_NUM_WORKER:-1}"
DEBUG="${DEBUG:-0}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_PREFLIGHT="${SKIP_PREFLIGHT:-0}"
ALLOW_MISSING_OPENCOMPASS="${ALLOW_MISSING_OPENCOMPASS:-0}"
OPENCOMPASS_ROOT="${OPENCOMPASS_ROOT:-}"

export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export PATCHMOE_CKPT_DIR="$CKPT_DIR"
export PATCHMOE_OC_ABBR="${PATCHMOE_OC_ABBR:-$RUN_NAME}"
export PATCHMOE_OC_DATASETS="${PATCHMOE_OC_DATASETS:-$DATASETS}"
export BLT_ALLOW_MISSING_FLEX_ATTENTION="${BLT_ALLOW_MISSING_FLEX_ATTENTION:-1}"
export BLT_SUPPRESS_ATTN_ERROR="${BLT_SUPPRESS_ATTN_ERROR:-1}"

if [[ "$SKIP_PREFLIGHT" != "1" ]]; then
  preflight_cmd=(
    "$PYTHON_BIN" scripts/patchmoe/preflight_opencompass_benchmarks.py
    "$CKPT_DIR"
    --config "$CONFIG"
    --datasets "$DATASETS"
    --opencompass-bin "$OPENCOMPASS_BIN"
  )
  if [[ -n "$OPENCOMPASS_ROOT" ]]; then
    preflight_cmd+=(--opencompass-root "$OPENCOMPASS_ROOT")
  fi
  if [[ "$ALLOW_MISSING_OPENCOMPASS" == "1" ]]; then
    preflight_cmd+=(--allow-missing-opencompass)
  fi
  "${preflight_cmd[@]}"
fi

# Allow OPENCOMPASS_BIN="python /path/to/opencompass/run.py" as documented.
# shellcheck disable=SC2206
_opencompass_cmd=($OPENCOMPASS_BIN)
cmd=(
  "${_opencompass_cmd[@]}"
  "$CONFIG"
  --datasets
)
# shellcheck disable=SC2206
_dataset_args=($DATASETS)
cmd+=("${_dataset_args[@]}")
cmd+=(
  -w "$WORKDIR"
  -m "$MODE"
  --max-num-worker "$MAX_NUM_WORKER"
)
if [[ "$DEBUG" == "1" ]]; then
  cmd+=(--debug)
fi
if [[ "$DRY_RUN" == "1" ]]; then
  cmd+=(--dry-run)
fi

printf 'OpenCompass command:\n'
printf '%q ' "${cmd[@]}"
printf '\n'
exec "${cmd[@]}"
