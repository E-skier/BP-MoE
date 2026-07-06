#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/data1/pengfeigao/BP-MoE"
GRID_DIR="$ROOT_DIR/experiments/bcfp_cpt/D_strong_bias_grid_2k_6gpu_nockpt"
LAUNCH="$GRID_DIR/launch_one.sh"

cd "$ROOT_DIR"

last_step() {
  local metrics="$1"
  "$ROOT_DIR/.venv/bin/python" - "$metrics" <<'PY'
import json
import sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.exists():
    print(0)
    raise SystemExit
last = 0
for line in p.read_text().splitlines():
    if not line.strip():
        continue
    try:
        last = int(json.loads(line).get("global_step") or last)
    except json.JSONDecodeError:
        pass
print(last)
PY
}

fmt() {
  printf '%s' "$1" | tr -d '.'
}

run_one() {
  local ema="$1"
  local lr="$2"
  local max_frac="$3"
  local run_name="D_strongbias_ema$(fmt "$ema")_lr$(fmt "$lr")_max$(fmt "$max_frac")_2k_6gpu_nockpt"
  local run_dir="$GRID_DIR/$run_name"
  local metrics="$run_dir/metrics.jsonl"
  local step
  step="$(last_step "$metrics")"
  if (( step >= 2000 )); then
    echo "$(date) queue: skip completed $run_name step=$step"
    return
  fi

  mkdir -p "$run_dir"
  echo "$(date) queue: start $run_name"
  "$LAUNCH" "$ema" "$lr" "$max_frac" > "$run_dir/pipeline.log" 2>&1
  echo "$(date) queue: finished $run_name"
  rm -rf "$run_dir/checkpoints"
}

for ema in 0.8 0.9; do
  for lr in 0.2 0.5; do
    for max_frac in 0.45 0.50; do
      run_one "$ema" "$lr" "$max_frac"
    done
  done
done

echo "$(date) queue: all strong-bias 2k grid runs finished"
