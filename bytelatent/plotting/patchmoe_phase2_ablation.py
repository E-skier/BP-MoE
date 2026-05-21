import argparse
import json
from pathlib import Path

import altair as alt
import pandas as pd


BPB_KEY = "bpb/interval_across_gpus"
LOSS_KEY = "loss/interval_across_gpu"
WPS_KEY = "speed/wps"


def read_metrics(path: Path, variant: str) -> pd.DataFrame:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                row["variant"] = variant
                rows.append(row)
    if not rows:
        raise ValueError(f"No metrics found in {path}")
    return pd.DataFrame(rows)


def read_run_root(run_root: Path) -> pd.DataFrame:
    frames = []
    for run_dir in sorted(p for p in run_root.iterdir() if p.is_dir()):
        metrics_path = run_dir / "metrics.jsonl"
        if metrics_path.exists():
            frames.append(read_metrics(metrics_path, run_dir.name))
    if not frames:
        raise ValueError(f"No metrics.jsonl files found under {run_root}")
    return pd.concat(frames, ignore_index=True, sort=False)


def last_non_null(series: pd.Series):
    values = series.dropna()
    if values.empty:
        return None
    return values.iloc[-1]


def tail_mean(group: pd.DataFrame, key: str, tail_steps: int):
    if key not in group:
        return None
    values = group.tail(tail_steps)[key].dropna()
    if values.empty:
        return None
    return float(values.mean())


def build_summary(df: pd.DataFrame, tail_steps: int) -> pd.DataFrame:
    rows = []
    for variant, group in df.groupby("variant", sort=True):
        group = group.sort_values("global_step")
        final = group.iloc[-1]
        row = {
            "variant": variant,
            "steps": int(group["global_step"].max()),
            "metric_rows": int(len(group)),
            "final_bpb": last_non_null(group[BPB_KEY]) if BPB_KEY in group else None,
            "last_tail_bpb": tail_mean(group, BPB_KEY, tail_steps),
            "last_tail_loss": tail_mean(group, LOSS_KEY, tail_steps),
            "last_tail_wps": tail_mean(group, WPS_KEY, tail_steps),
            "patch_length_mean": final.get("patch/length_mean_across_gpus"),
            "patch_entropy_mean": final.get("patch/entropy_mean_across_gpus"),
            "moe_load_imbalance": final.get("moe/load_imbalance_mean"),
            "moe_max_load_fraction": final.get("moe/max_load_fraction_mean"),
            "moe_min_load_fraction": final.get("moe/min_load_fraction_mean"),
            "moe_side_feature_count": final.get("moe/side_feature_count_mean"),
        }
        if BPB_KEY in group and group[BPB_KEY].notna().any():
            best_idx = group[BPB_KEY].idxmin()
            best = group.loc[best_idx]
            row["best_bpb"] = best[BPB_KEY]
            row["best_bpb_step"] = int(best["global_step"])
        for expert_id in range(16):
            entropy_key = f"moe/expert_{expert_id}_patch_entropy_mean_mean"
            length_key = f"moe/expert_{expert_id}_patch_length_mean_mean"
            load_key = f"moe/expert_{expert_id}_load_fraction_mean"
            if entropy_key in group:
                row[f"expert_{expert_id}_patch_entropy_mean"] = final.get(entropy_key)
            if length_key in group:
                row[f"expert_{expert_id}_patch_length_mean"] = final.get(length_key)
            if load_key in group:
                row[f"expert_{expert_id}_load_fraction"] = final.get(load_key)
        rows.append(row)
    return pd.DataFrame(rows)


def save_bpb_curve(df: pd.DataFrame, output_dir: Path) -> None:
    if BPB_KEY not in df:
        return
    plot_df = df[["variant", "global_step", BPB_KEY]].dropna()
    if plot_df.empty:
        return
    chart = (
        alt.Chart(plot_df)
        .mark_line(point=False)
        .encode(
            x=alt.X("global_step:Q", title="Global Step"),
            y=alt.Y(f"{BPB_KEY}:Q", title="Bits per Byte").scale(zero=False),
            color=alt.Color("variant:N", title="Variant"),
            tooltip=["variant:N", "global_step:Q", alt.Tooltip(f"{BPB_KEY}:Q", format=".4f")],
        )
        .properties(width=860, height=420, title="Phase-2 BPB Training Curve")
    )
    chart.save(output_dir / "bpb_training_curve.html")


def final_rows(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.sort_values("global_step")
        .groupby("variant", sort=True, as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )


def bucket_distribution(
    df: pd.DataFrame,
    *,
    feature: str,
    buckets: tuple[str, ...],
) -> pd.DataFrame:
    rows = []
    finals = final_rows(df)
    for _, row in finals.iterrows():
        for bucket in buckets:
            key = f"moe/{feature}_bucket_{bucket}_unit_fraction_mean"
            if key in finals and pd.notna(row.get(key)):
                rows.append(
                    {
                        "variant": row["variant"],
                        "feature": feature,
                        "bucket": bucket,
                        "unit_fraction": row[key],
                    }
                )
    return pd.DataFrame(rows)


def save_bucket_chart(
    bucket_df: pd.DataFrame,
    *,
    title: str,
    output_path: Path,
) -> None:
    if bucket_df.empty:
        return
    chart = (
        alt.Chart(bucket_df)
        .mark_bar()
        .encode(
            x=alt.X("variant:N", title="Variant", sort=sorted(bucket_df["variant"].unique())),
            y=alt.Y("unit_fraction:Q", title="Patch Fraction"),
            color=alt.Color("bucket:N", title="Bucket"),
            column=alt.Column("feature:N", title=None),
            tooltip=["variant:N", "feature:N", "bucket:N", alt.Tooltip("unit_fraction:Q", format=".3f")],
        )
        .properties(width=300, height=320, title=title)
        .resolve_scale(y="shared")
    )
    chart.save(output_path)


def expert_load_distribution(df: pd.DataFrame) -> pd.DataFrame:
    finals = final_rows(df)
    rows = []
    for _, row in finals.iterrows():
        for col in finals.columns:
            if col.startswith("moe/expert_") and col.endswith("_load_fraction_mean"):
                value = row.get(col)
                if pd.notna(value):
                    expert = col.split("/expert_", 1)[1].split("_", 1)[0]
                    rows.append(
                        {
                            "variant": row["variant"],
                            "expert": f"expert_{expert}",
                            "load_fraction": value,
                        }
                    )
    return pd.DataFrame(rows)


def save_expert_load_chart(load_df: pd.DataFrame, output_dir: Path) -> None:
    if load_df.empty:
        return
    chart = (
        alt.Chart(load_df)
        .mark_bar()
        .encode(
            x=alt.X("expert:N", title="Expert"),
            y=alt.Y("load_fraction:Q", title="Load Fraction"),
            color=alt.Color("expert:N", title="Expert"),
            column=alt.Column("variant:N", title=None),
            tooltip=["variant:N", "expert:N", alt.Tooltip("load_fraction:Q", format=".3f")],
        )
        .properties(width=110, height=300, title="Final Expert Load Distribution")
    )
    chart.save(output_dir / "expert_load_distribution.html")


def specialization_heatmap(
    df: pd.DataFrame,
    *,
    feature: str,
    buckets: tuple[str, ...],
) -> pd.DataFrame:
    finals = final_rows(df)
    rows = []
    for _, row in finals.iterrows():
        for col in finals.columns:
            prefix = f"moe/{feature}_bucket_"
            if not col.startswith(prefix) or "_expert_" not in col:
                continue
            suffix = col.removeprefix(prefix)
            bucket, expert_part = suffix.split("_expert_", 1)
            if bucket not in buckets:
                continue
            expert = expert_part.split("_assignment_fraction_mean", 1)[0]
            value = row.get(col)
            if pd.notna(value):
                rows.append(
                    {
                        "variant": row["variant"],
                        "feature": feature,
                        "bucket": bucket,
                        "expert": f"expert_{expert}",
                        "assignment_fraction": value,
                    }
                )
    return pd.DataFrame(rows)


def save_heatmap(df: pd.DataFrame, *, title: str, output_path: Path) -> None:
    if df.empty:
        return
    chart = (
        alt.Chart(df)
        .mark_rect()
        .encode(
            x=alt.X("bucket:N", title="Bucket"),
            y=alt.Y("expert:N", title="Expert"),
            color=alt.Color("assignment_fraction:Q", title="Assignment Fraction", scale=alt.Scale(scheme="viridis")),
            facet=alt.Facet("variant:N", columns=3, title=None),
            tooltip=["variant:N", "expert:N", "bucket:N", alt.Tooltip("assignment_fraction:Q", format=".3f")],
        )
        .properties(width=180, height=120, title=title)
    )
    chart.save(output_path)


def expert_feature_means(df: pd.DataFrame) -> pd.DataFrame:
    finals = final_rows(df)
    rows = []
    for _, row in finals.iterrows():
        for col in finals.columns:
            if not col.startswith("moe/expert_"):
                continue
            if not (
                col.endswith("_patch_entropy_mean_mean")
                or col.endswith("_patch_length_mean_mean")
            ):
                continue
            expert = col.split("/expert_", 1)[1].split("_", 1)[0]
            if "patch_entropy" in col:
                feature = "patch_entropy"
            else:
                feature = "patch_length"
            value = row.get(col)
            if pd.notna(value):
                rows.append(
                    {
                        "variant": row["variant"],
                        "expert": f"expert_{expert}",
                        "feature": feature,
                        "mean_value": value,
                    }
                )
    return pd.DataFrame(rows)


def save_expert_feature_chart(feature_df: pd.DataFrame, output_dir: Path) -> None:
    if feature_df.empty:
        return
    chart = (
        alt.Chart(feature_df)
        .mark_bar()
        .encode(
            x=alt.X("expert:N", title="Expert"),
            y=alt.Y("mean_value:Q", title="Mean Routed Patch Feature"),
            color=alt.Color("expert:N", title="Expert"),
            column=alt.Column("feature:N", title=None),
            row=alt.Row("variant:N", title=None),
            tooltip=["variant:N", "expert:N", "feature:N", alt.Tooltip("mean_value:Q", format=".3f")],
        )
        .properties(width=120, height=95, title="Per-Expert Routed Patch Feature Means")
    )
    chart.save(output_dir / "expert_feature_means.html")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize and plot Phase-2 PatchMoE ablation metrics."
    )
    parser.add_argument("run_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--tail-steps", type=int, default=20)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = read_run_root(args.run_root)

    summary = build_summary(df, args.tail_steps)
    summary.to_csv(args.output_dir / "summary.csv", index=False)

    save_bpb_curve(df, args.output_dir)

    length_buckets = ("short", "medium", "long")
    entropy_buckets = ("low", "medium", "high")
    patch_dist = pd.concat(
        [
            bucket_distribution(df, feature="length", buckets=length_buckets),
            bucket_distribution(df, feature="entropy", buckets=entropy_buckets),
        ],
        ignore_index=True,
    )
    if not patch_dist.empty:
        patch_dist.to_csv(args.output_dir / "patch_bucket_distribution.csv", index=False)
        save_bucket_chart(
            patch_dist,
            title="Final Patch Length/Entropy Bucket Distribution",
            output_path=args.output_dir / "patch_bucket_distribution.html",
        )

    load_df = expert_load_distribution(df)
    if not load_df.empty:
        load_df.to_csv(args.output_dir / "expert_load_distribution.csv", index=False)
        save_expert_load_chart(load_df, args.output_dir)

    for feature, buckets in (("length", length_buckets), ("entropy", entropy_buckets)):
        spec_df = specialization_heatmap(df, feature=feature, buckets=buckets)
        if spec_df.empty:
            continue
        spec_df.to_csv(args.output_dir / f"{feature}_specialization_heatmap.csv", index=False)
        save_heatmap(
            spec_df,
            title=f"Expert Assignment by Patch {feature.title()} Bucket",
            output_path=args.output_dir / f"{feature}_specialization_heatmap.html",
        )

    feature_df = expert_feature_means(df)
    if not feature_df.empty:
        feature_df.to_csv(args.output_dir / "expert_feature_means.csv", index=False)
        save_expert_feature_chart(feature_df, args.output_dir)


if __name__ == "__main__":
    main()
