#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data1/pengfeigao/BP-MoE}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/experiments/aaai_layer1_e8_top1}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}"
NPROC_PER_NODE="${NPROC_PER_NODE:-6}"
MODE="${MODE:-print}"
SEEDS="${SEEDS:-42 43 44}"
METHODS="${METHODS:-A-Dense A-H8 A-E8 A-HE8}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"

TRAIN_MANIFEST="$PROJECT_ROOT/manifests/aaai_series_a_train.json"
HELDOUT_MANIFEST="$PROJECT_ROOT/manifests/heldout_natural_core.json"
FINAL_TEST_MANIFEST="$PROJECT_ROOT/manifests/heldout_natural_extended.json"

INIT_A_DENSE="${INIT_A_DENSE:-$PROJECT_ROOT/blt1b_warmstart/initial_weights/aaai_layer1_dense_blt1b_official_dcp}"
INIT_A_H8="${INIT_A_H8:-$PROJECT_ROOT/blt1b_warmstart/initial_weights/aaai_layer1_e8_top1_hidden_dcp}"
INIT_A_E8="${INIT_A_E8:-$PROJECT_ROOT/blt1b_warmstart/initial_weights/aaai_layer1_e8_top1_entropy_mlp_dcp}"
INIT_A_HE8="${INIT_A_HE8:-$PROJECT_ROOT/blt1b_warmstart/initial_weights/aaai_layer1_e8_top1_entropy_mlp_dcp}"
COMPUTE_BUDGET_JSON="${COMPUTE_BUDGET_JSON:-$RUN_ROOT/compute_budgets.json}"

cd "$PROJECT_ROOT"

if [[ "$MODE" != "print" && "$MODE" != "run" ]]; then
  echo "MODE must be print or run" >&2
  exit 2
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  else
    echo "No executable Python found. Set PYTHON_BIN." >&2
    exit 2
  fi
fi

export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export BLT_ALLOW_MISSING_FLEX_ATTENTION="${BLT_ALLOW_MISSING_FLEX_ATTENTION:-1}"
export BLT_SUPPRESS_ATTN_ERROR="${BLT_SUPPRESS_ATTN_ERROR:-1}"
export ENABLE_INTRA_NODE_COMM="${ENABLE_INTRA_NODE_COMM:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

config_for_method() {
  case "$1" in
    A-Dense) echo "$PROJECT_ROOT/configs/aaai_layer1/aaai_dense.yaml" ;;
    A-H8) echo "$PROJECT_ROOT/configs/aaai_layer1/aaai_e8_hidden.yaml" ;;
    A-E8) echo "$PROJECT_ROOT/configs/aaai_layer1/aaai_e8_entropy.yaml" ;;
    A-HE8) echo "$PROJECT_ROOT/configs/aaai_layer1/aaai_e8_hidden_entropy.yaml" ;;
    *) echo "Unknown method: $1" >&2; return 2 ;;
  esac
}

init_for_method() {
  case "$1" in
    A-Dense) echo "$INIT_A_DENSE" ;;
    A-H8) echo "$INIT_A_H8" ;;
    A-E8) echo "$INIT_A_E8" ;;
    A-HE8) echo "$INIT_A_HE8" ;;
    *) echo "" ;;
  esac
}

quote_cmd() {
  printf '%q ' "$@"
  printf '\n'
}

require_file() {
  [[ -f "$1" ]] || { echo "FAIL missing file: $1" >&2; return 1; }
}

require_dcp() {
  local path="$1"
  local label="$2"
  if [[ -z "$path" ]]; then
    echo "FAIL missing $label. Set ${label}=<torch.distributed.checkpoint dir>." >&2
    return 1
  fi
  [[ -d "$path" ]] || { echo "FAIL $label is not a directory: $path" >&2; return 1; }
  [[ -f "$path/.metadata" ]] || { echo "FAIL $label is not DCP format: $path/.metadata missing" >&2; return 1; }
}

preflight_common() {
  require_file "$TRAIN_MANIFEST"
  require_file "$HELDOUT_MANIFEST"
  require_file "$FINAL_TEST_MANIFEST"
  "$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path

for manifest in [
    "manifests/aaai_series_a_train.json",
    "manifests/heldout_natural_core.json",
    "manifests/heldout_natural_extended.json",
]:
    data = json.loads(Path(manifest).read_text())
    for key in ("raw_files", "arrow_files"):
        missing = [path for path in data[key] if not Path(path).exists()]
        if missing:
            raise SystemExit(f"{manifest} missing {key}: {missing}")
print("OK manifests and files")
PY
  "$PYTHON_BIN" - <<'PY'
from bytelatent.config_parser import parse_file_config, parse_args_with_default
for path in [
    "configs/aaai_layer1/aaai_dense.yaml",
    "configs/aaai_layer1/aaai_e8_hidden.yaml",
    "configs/aaai_layer1/aaai_e8_entropy.yaml",
    "configs/aaai_layer1/aaai_e8_hidden_entropy.yaml",
]:
    parse_args_with_default(cli_args=parse_file_config(path))
print("OK configs parse")
PY
}

write_run_environment() {
  local run_dir="$1"
  mkdir -p "$run_dir"
  {
    date --iso-8601=seconds
    git rev-parse HEAD
    nvidia-smi
    nvidia-smi topo -m
  } >"$run_dir/environment.txt"
}

write_run_manifest() {
  local run_dir="$1"
  local method="$2"
  local seed="$3"
  local config="$4"
  local init_ckpt="$5"
  local train_sha heldout_sha final_sha config_sha git_head
  train_sha="$(sha256sum "$TRAIN_MANIFEST" | awk '{print $1}')"
  heldout_sha="$(sha256sum "$HELDOUT_MANIFEST" | awk '{print $1}')"
  final_sha="$(sha256sum "$FINAL_TEST_MANIFEST" | awk '{print $1}')"
  config_sha="$(sha256sum "$config" | awk '{print $1}')"
  git_head="$(git rev-parse HEAD)"
  cat >"$run_dir/run_manifest.json" <<EOF
{
  "method": "$method",
  "seed": $seed,
  "git_head": "$git_head",
  "config": "$config",
  "config_sha256": "$config_sha",
  "init_checkpoint": "$init_ckpt",
  "train_manifest": "$TRAIN_MANIFEST",
  "train_manifest_sha256": "$train_sha",
  "heldout_manifest": "$HELDOUT_MANIFEST",
  "heldout_manifest_sha256": "$heldout_sha",
  "final_test_manifest": "$FINAL_TEST_MANIFEST",
  "final_test_manifest_sha256": "$final_sha",
  "cuda_visible_devices": "$CUDA_VISIBLE_DEVICES",
  "nproc_per_node": $NPROC_PER_NODE,
  "moe_ep_size": 2
}
EOF
}

build_cmd() {
  local method="$1"
  local seed="$2"
  local config="$3"
  local init_ckpt="$4"
  local run_dir="$RUN_ROOT/$method/seed$seed"
  cmd=(
    env "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    "$PYTHON_BIN" -m torch.distributed.run
    --standalone "--nproc-per-node=$NPROC_PER_NODE"
    -m bytelatent.train
    "config=$config"
    "seed=$seed"
    "data.seed=$seed"
    "dump_dir=$run_dir"
    "name=${method}_seed${seed}"
    "checkpoint.path=$run_dir/checkpoints"
    "checkpoint.init_ckpt_path=$init_ckpt"
    "compute_budget_json=$COMPUTE_BUDGET_JSON"
    "save_budget_boundaries=true"
  )
}

preflight_common
if [[ ! -f "$COMPUTE_BUDGET_JSON" ]]; then
  echo "FAIL missing compute budget file: $COMPUTE_BUDGET_JSON" >&2
  echo "Run dense 200-step profiling and write C_low/C_mid/C_high before MODE=run." >&2
  if [[ "$MODE" == "run" ]]; then
    exit 1
  fi
fi

for seed in $SEEDS; do
  for method in $METHODS; do
    config="$(config_for_method "$method")"
    init_ckpt="$(init_for_method "$method")"
    label="INIT_${method//-/_}"
    label="${label//8/E8}"
    require_file "$config"
    require_dcp "$init_ckpt" "$label"

    run_dir="$RUN_ROOT/$method/seed$seed"
    write_run_environment "$run_dir"
    write_run_manifest "$run_dir" "$method" "$seed" "$config" "$init_ckpt"
    build_cmd "$method" "$seed" "$config" "$init_ckpt"

    echo "Prepared $method seed=$seed"
    quote_cmd "${cmd[@]}"
    echo "Run dir: $run_dir"

    if [[ "$MODE" == "run" ]]; then
      mkdir -p "$run_dir"
      quote_cmd "${cmd[@]}" >"$run_dir/launch_command.txt"
      "${cmd[@]}" 2>&1 | tee "$run_dir/train.log"
    fi
  done
done
