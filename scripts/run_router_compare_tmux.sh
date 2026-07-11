#!/usr/bin/env bash
set -euo pipefail

cd /data1/pengfeigao/BP-MoE

export PYTHONPATH=/data1/pengfeigao/BP-MoE:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=""

mkdir -p logs
mkdir -p /data1/pengfeigao/BP-MoE/blt1b_warmstart/analysis_results/router_compare

echo "============================================================"
echo "Router compare started at $(date)"
echo "PWD=$(pwd)"
echo "PYTHONPATH=${PYTHONPATH}"
echo "============================================================"

# ---------------------------------------------------------------------
# 0. Check scripts
# ---------------------------------------------------------------------
if [[ ! -f scripts/analysis/analyze_patch_router_feature_grid_dcp.py ]]; then
  echo "[ERROR] Missing scripts/analysis/analyze_patch_router_feature_grid_dcp.py"
  echo "Please create this script first."
  exit 1
fi

if [[ ! -f scripts/analysis/check_hidden_only_router_dcp.py ]]; then
  echo "[WARN] Missing scripts/analysis/check_hidden_only_router_dcp.py"
  echo "Hidden-only check will be skipped."
fi

# ---------------------------------------------------------------------
# 1. Natural-granularity AC1 total20k
# ---------------------------------------------------------------------
export CKPT_NATURAL=/data1/pengfeigao/BP-MoE/blt1b_warmstart/natural_granularity_AC1_A5k_B15k_6gpu_c00000_c00005/phaseB_resume_from4000_to_total20k/checkpoints/0000011000
export OUT_NATURAL=/data1/pengfeigao/BP-MoE/blt1b_warmstart/analysis_results/router_compare/natural_granularity_AC1_total20k

mkdir -p "${OUT_NATURAL}"

echo
echo "============================================================"
echo "[1/5] Natural-granularity AC1 total20k"
echo "CKPT=${CKPT_NATURAL}"
echo "OUT=${OUT_NATURAL}"
echo "============================================================"

if [[ -d "${CKPT_NATURAL}" ]]; then
  .venv/bin/python scripts/analysis/analyze_patch_router_feature_grid_dcp.py \
    --ckpt "${CKPT_NATURAL}" \
    --out "${OUT_NATURAL}" \
    --tag natural_granularity_AC1_total20k \
    --mode natural_2d \
    --length-min 1 \
    --length-max 16 \
    --length-steps 16 \
    --entropy-min 0.0 \
    --entropy-max 4.0 \
    --entropy-steps 81 \
    2>&1 | tee "${OUT_NATURAL}/router_feature_grid.log"
else
  echo "[ERROR] Missing natural checkpoint: ${CKPT_NATURAL}"
fi

# ---------------------------------------------------------------------
# 2. Entropy-randominit
# ---------------------------------------------------------------------
export CKPT_ENTROPY_RANDOMINIT=/data1/pengfeigao/BP-MoE/blt1b_warmstart/entropy_randominit_A1k_B50k_6gpu_c00000_c00005/phaseB_resume_from29000_to_total50k/checkpoints/0000019000
export OUT_ENTROPY_RANDOMINIT=/data1/pengfeigao/BP-MoE/blt1b_warmstart/analysis_results/router_compare/entropy_randominit

mkdir -p "${OUT_ENTROPY_RANDOMINIT}"

echo
echo "============================================================"
echo "[2/5] Entropy-randominit"
echo "CKPT=${CKPT_ENTROPY_RANDOMINIT}"
echo "OUT=${OUT_ENTROPY_RANDOMINIT}"
echo "============================================================"

if [[ -d "${CKPT_ENTROPY_RANDOMINIT}" ]]; then
  .venv/bin/python scripts/analysis/analyze_patch_router_feature_grid_dcp.py \
    --ckpt "${CKPT_ENTROPY_RANDOMINIT}" \
    --out "${OUT_ENTROPY_RANDOMINIT}" \
    --tag entropy_randominit \
    --mode entropy_only \
    --length-min 1 \
    --length-max 16 \
    --length-steps 16 \
    --entropy-min 0.0 \
    --entropy-max 4.0 \
    --entropy-steps 81 \
    2>&1 | tee "${OUT_ENTROPY_RANDOMINIT}/router_feature_grid.log"
else
  echo "[WARN] Missing fixed entropy-randominit checkpoint."
  echo "[INFO] Searching candidates..."
  find /data1/pengfeigao/BP-MoE/blt1b_warmstart \
    -path "*entropy*random*checkpoints/[0-9]*" \
    -type d | sort -V | tail -20
fi

# ---------------------------------------------------------------------
# 3. Entropy-only
# ---------------------------------------------------------------------
export OUT_ENTROPY_ONLY=/data1/pengfeigao/BP-MoE/blt1b_warmstart/analysis_results/router_compare/entropy_only
mkdir -p "${OUT_ENTROPY_ONLY}"

echo
echo "============================================================"
echo "[3/5] Entropy-only"
echo "============================================================"

CKPT_ENTROPY_ONLY=$(
  find /data1/pengfeigao/BP-MoE/blt1b_warmstart \
    -path "*entropy*only*checkpoints/[0-9]*" \
    -type d | sort -V | tail -1 || true
)

if [[ -n "${CKPT_ENTROPY_ONLY}" && -d "${CKPT_ENTROPY_ONLY}" ]]; then
  echo "CKPT_ENTROPY_ONLY=${CKPT_ENTROPY_ONLY}"
  echo "OUT=${OUT_ENTROPY_ONLY}"

  .venv/bin/python scripts/analysis/analyze_patch_router_feature_grid_dcp.py \
    --ckpt "${CKPT_ENTROPY_ONLY}" \
    --out "${OUT_ENTROPY_ONLY}" \
    --tag entropy_only \
    --mode entropy_only \
    --length-min 1 \
    --length-max 16 \
    --length-steps 16 \
    --entropy-min 0.0 \
    --entropy-max 4.0 \
    --entropy-steps 81 \
    2>&1 | tee "${OUT_ENTROPY_ONLY}/router_feature_grid.log"
else
  echo "[WARN] No entropy-only checkpoint found."
  echo "[INFO] Candidates from broader search:"
  find /data1/pengfeigao/BP-MoE/blt1b_warmstart \
    -path "*entropy*checkpoints/[0-9]*" \
    -type d | sort -V | tail -30 || true
fi

# ---------------------------------------------------------------------
# 4. Hidden-only structure check
# ---------------------------------------------------------------------
echo
echo "============================================================"
echo "[4/5] Hidden-only structure check"
echo "============================================================"

if [[ -f scripts/analysis/check_hidden_only_router_dcp.py ]]; then
  mkdir -p /data1/pengfeigao/BP-MoE/blt1b_warmstart/analysis_results/router_compare/hidden_only

  CKPT_HIDDEN_ONLY=$(
    find /data1/pengfeigao/BP-MoE/blt1b_warmstart \
      -path "*hidden*only*checkpoints/[0-9]*" \
      -type d | sort -V | tail -1 || true
  )

  if [[ -n "${CKPT_HIDDEN_ONLY}" && -d "${CKPT_HIDDEN_ONLY}" ]]; then
    echo "CKPT_HIDDEN_ONLY=${CKPT_HIDDEN_ONLY}"

    .venv/bin/python scripts/analysis/check_hidden_only_router_dcp.py \
      --ckpt "${CKPT_HIDDEN_ONLY}" \
      2>&1 | tee /data1/pengfeigao/BP-MoE/blt1b_warmstart/analysis_results/router_compare/hidden_only/hidden_only_check.log
  else
    echo "[WARN] No hidden-only checkpoint found."
    echo "[INFO] Candidates from broader search:"
    find /data1/pengfeigao/BP-MoE/blt1b_warmstart \
      -path "*hidden*checkpoints/[0-9]*" \
      -type d | sort -V | tail -30 || true
  fi
else
  echo "[SKIP] scripts/analysis/check_hidden_only_router_dcp.py not found."
fi

# ---------------------------------------------------------------------
# 5. Summary table
# ---------------------------------------------------------------------
echo
echo "============================================================"
echo "[5/5] Summary table"
echo "============================================================"

cat > scripts/analysis/summarize_patch_router_feature_grid_compare.py <<'PY'
#!/usr/bin/env python3
import json
from pathlib import Path

items = {
    "natural_granularity_AC1_total20k": Path("/data1/pengfeigao/BP-MoE/blt1b_warmstart/analysis_results/router_compare/natural_granularity_AC1_total20k/natural_granularity_AC1_total20k_feature_grid_summary.json"),
    "entropy_randominit": Path("/data1/pengfeigao/BP-MoE/blt1b_warmstart/analysis_results/router_compare/entropy_randominit/entropy_randominit_feature_grid_summary.json"),
    "entropy_only": Path("/data1/pengfeigao/BP-MoE/blt1b_warmstart/analysis_results/router_compare/entropy_only/entropy_only_feature_grid_summary.json"),
}

def agg(rows):
    n = len(rows)
    active2 = [x["active_top2_count"] for x in rows]
    effs = [x["top2_effective_experts"] for x in rows]
    same = [x["same_pair_fraction"] for x in rows]

    ratios = [
        x["entropy_to_length_l2_ratio"]
        for x in rows
        if x.get("entropy_to_length_l2_ratio") is not None
    ]
    cosines = [
        x["length_entropy_axis_cosine"]
        for x in rows
        if x.get("length_entropy_axis_cosine") is not None
    ]

    return {
        "layers": n,
        "feature_names": rows[0].get("feature_names"),
        "mean_same_pair_frac": sum(same) / n,
        "mean_active_top2_experts": sum(active2) / n,
        "mean_top2_effective_experts": sum(effs) / n,
        "layers_le2_active_top2": sum(c <= 2 for c in active2),
        "layers_ge4_active_top2": sum(c >= 4 for c in active2),
        "mean_entropy_to_length_l2_ratio": sum(ratios) / len(ratios) if ratios else None,
        "mean_length_entropy_axis_cosine": sum(cosines) / len(cosines) if cosines else None,
    }

print("method                         features              same_pair  active_top2  eff_top2  <=2_layers  >=4_layers  ent/len  axis_cos")
print("-" * 125)

for name, path in items.items():
    if not path.exists():
        print(f"{name:<30} MISSING: {path}")
        continue

    rows = json.loads(path.read_text())
    x = agg(rows)

    ent_len = x["mean_entropy_to_length_l2_ratio"]
    axis_cos = x["mean_length_entropy_axis_cosine"]

    ent_len_s = f"{ent_len:.3f}" if ent_len is not None else "NA"
    axis_cos_s = f"{axis_cos:+.3f}" if axis_cos is not None else "NA"

    print(
        f"{name:<30} "
        f"{str(x['feature_names']):<21} "
        f"{x['mean_same_pair_frac']:<10.3f} "
        f"{x['mean_active_top2_experts']:<12.2f} "
        f"{x['mean_top2_effective_experts']:<9.2f} "
        f"{x['layers_le2_active_top2']:<10} "
        f"{x['layers_ge4_active_top2']:<10} "
        f"{ent_len_s:<8} "
        f"{axis_cos_s:<8}"
    )
PY

chmod +x scripts/analysis/summarize_patch_router_feature_grid_compare.py

.venv/bin/python scripts/analysis/summarize_patch_router_feature_grid_compare.py \
  2>&1 | tee /data1/pengfeigao/BP-MoE/blt1b_warmstart/analysis_results/router_compare/summary_table.log

echo
echo "============================================================"
echo "Router compare finished at $(date)"
echo "Summary saved to:"
echo "  /data1/pengfeigao/BP-MoE/blt1b_warmstart/analysis_results/router_compare/summary_table.log"
echo "============================================================"
