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
