import json
import sys
from pathlib import Path

import altair as alt
import pandas as pd


def read_metrics(path: Path) -> pd.DataFrame:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No metrics found in {path}")
    return pd.DataFrame(rows)


def save_line_chart(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    title: str,
    output_path: Path,
) -> None:
    chart = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X(x, title="Global Step"),
            y=alt.Y(y, title=title).scale(zero=False),
        )
        .properties(width=640, height=320)
    )
    chart.save(output_path)


def save_expert_load_chart(df: pd.DataFrame, output_path: Path) -> None:
    expert_cols = [
        col
        for col in df.columns
        if col.startswith("moe/expert_") and col.endswith("_load_fraction_mean")
    ]
    if not expert_cols:
        return

    load_df = df[["global_step", *expert_cols]].melt(
        id_vars=["global_step"],
        var_name="expert",
        value_name="load_fraction",
    )
    load_df["expert"] = load_df["expert"].str.extract(r"moe/(expert_[0-9]+)")
    chart = (
        alt.Chart(load_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("global_step", title="Global Step"),
            y=alt.Y("load_fraction", title="Expert Load Fraction"),
            color=alt.Color("expert", title="Expert"),
        )
        .properties(width=720, height=360)
    )
    chart.save(output_path)


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(
            "Usage: python -m bytelatent.plotting.patchmoe_stage0_metrics "
            "<metrics.jsonl> <output_dir>"
        )

    metrics_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(exist_ok=True, parents=True)
    df = read_metrics(metrics_path)

    if "bpb/interval_across_gpus" in df:
        save_line_chart(
            df,
            x="global_step",
            y="bpb/interval_across_gpus",
            title="Bits per Byte",
            output_path=output_dir / "bpb_curve.html",
        )
    if "patch/length_mean_across_gpus" in df:
        save_line_chart(
            df,
            x="global_step",
            y="patch/length_mean_across_gpus",
            title="Mean Patch Length",
            output_path=output_dir / "patch_length_mean.html",
        )
    if "moe/load_imbalance_mean" in df:
        save_line_chart(
            df,
            x="global_step",
            y="moe/load_imbalance_mean",
            title="Expert Load Imbalance",
            output_path=output_dir / "expert_load_imbalance.html",
        )
    save_expert_load_chart(df, output_dir / "expert_load_fraction.html")


if __name__ == "__main__":
    main()
