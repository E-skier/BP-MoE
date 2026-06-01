#!/usr/bin/env python3
"""Report formal PatchMoE Stage-1/200k run status.

This is intentionally lightweight and read-only. It summarizes whether each
planned formal variant has training metrics, final checkpoints, and held-out
validation output, then writes machine-readable CSV/JSON status files.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

STAGE1_VARIANTS = (
    "dense",
    "byte_hidden_only_w005",
    "byte_entropy_w005",
    "byte_type_w005",
    "byte_entropy_type_w005",
    "byte_entropy_type_congestion_w05",
    "byte_entropy_type_congestion_zloss_w0001",
    "byte_entropy_length_type_w005",
)

MATCHED_200K_VARIANTS = (
    "dense",
    "byte_hidden_only_w005",
    "byte_entropy_w005",
    "byte_type_w005",
    "byte_entropy_type_w005",
    "byte_entropy_type_congestion_w05",
    "byte_entropy_type_congestion_zloss_w0001",
    "byte_entropy_length_type_w005",
)


def read_last_jsonl(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    last_line = ""
    with path.open() as f:
        for line in f:
            if line.strip():
                last_line = line
    if not last_line:
        return None
    return json.loads(last_line)


def read_validation(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open() as f:
        results = json.load(f)
    total_bytes = 0.0
    total_loss = 0.0
    source_count = 0
    for metrics in results.values():
        n_bytes = float(metrics.get("n_bytes") or 0.0)
        loss_sum = metrics.get("loss_sum")
        if loss_sum is None and n_bytes > 0 and metrics.get("loss_mean") is not None:
            loss_sum = float(metrics["loss_mean"]) * n_bytes
        if loss_sum is not None:
            total_loss += float(loss_sum)
        total_bytes += n_bytes
        source_count += 1
    heldout_bpb = None
    if total_bytes > 0:
        import math

        heldout_bpb = total_loss / math.log(2) / total_bytes
    return {
        "source_count": source_count,
        "heldout_n_bytes": total_bytes,
        "heldout_loss_sum": total_loss,
        "heldout_bpb": heldout_bpb,
    }


def status_from_artifacts(
    *,
    pipeline: str,
    variant: str,
    seed: str | None,
    run_name: str,
    run_dir: Path,
    eval_root: Path,
    final_step: int,
) -> dict[str, Any]:
    final_step_dir = f"{final_step:010d}"
    metrics_path = run_dir / "metrics.jsonl"
    checkpoint_dir = run_dir / "checkpoints" / final_step_dir
    eval_path = eval_root / run_name / final_step_dir / "validation.json"

    last_metrics = read_last_jsonl(metrics_path)
    validation = read_validation(eval_path)
    last_step = int(last_metrics.get("global_step", 0)) if last_metrics else 0
    has_final_checkpoint = checkpoint_dir.is_dir()
    has_eval = validation is not None

    if has_final_checkpoint and has_eval:
        status = "done"
    elif last_metrics or has_final_checkpoint or has_eval:
        status = "partial"
    else:
        status = "todo"

    row: dict[str, Any] = {
        "pipeline": pipeline,
        "variant": variant,
        "seed": seed or "",
        "run_name": run_name,
        "status": status,
        "last_step": last_step,
        "target_step": final_step,
        "progress_fraction": (last_step / final_step) if final_step > 0 else None,
        "has_metrics": metrics_path.exists(),
        "has_final_checkpoint": has_final_checkpoint,
        "has_eval": has_eval,
        "metrics_path": str(metrics_path),
        "checkpoint_dir": str(checkpoint_dir),
        "eval_path": str(eval_path),
        "final_train_bpb": None,
        "last_tail_wps": None,
        "moe_load_imbalance": None,
        "moe_side_feature_count": None,
        "moe_congestion_weight": None,
        "moe_router_z_loss_weight": None,
        "heldout_bpb": None,
        "heldout_n_bytes": None,
    }
    if last_metrics:
        row.update(
            {
                "final_train_bpb": last_metrics.get("bpb/interval_across_gpus"),
                "last_tail_wps": last_metrics.get("speed/wps"),
                "moe_load_imbalance": last_metrics.get("moe/load_imbalance_mean"),
                "moe_side_feature_count": last_metrics.get(
                    "moe/side_feature_count_mean"
                ),
                "moe_congestion_weight": last_metrics.get("moe/congestion_weight_mean"),
                "moe_router_z_loss_weight": last_metrics.get(
                    "moe/router_z_loss_weight_mean"
                ),
            }
        )
    if validation:
        row.update(
            {
                "heldout_bpb": validation["heldout_bpb"],
                "heldout_n_bytes": validation["heldout_n_bytes"],
            }
        )
    return row


def collect_external_dense_compute_rows(
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in args.external_dense_compute:
        parts = spec.split(":", 3)
        if len(parts) != 4:
            raise ValueError(
                "--external-dense-compute entries must be seed:run_root:eval_root:steps"
            )
        seed, run_root, eval_root, steps = parts
        final_step = int(steps)
        run_name = f"dense_compute_matched_seed{seed}_{final_step}step"
        rows.append(
            status_from_artifacts(
                pipeline="200k_dense_compute",
                variant="dense_compute_matched",
                seed=seed,
                run_name=run_name,
                run_dir=Path(run_root) / run_name,
                eval_root=Path(eval_root),
                final_step=final_step,
            )
        )
    return rows


def collect_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant in args.stage1_variants:
        run_name = f"stage1_{variant}_matched"
        rows.append(
            status_from_artifacts(
                pipeline="stage1",
                variant=variant,
                seed=None,
                run_name=run_name,
                run_dir=args.stage1_run_root / run_name,
                eval_root=args.stage1_eval_root,
                final_step=args.stage1_steps,
            )
        )
    for seed in args.seeds:
        for variant in args.matched_200k_variants:
            run_name = f"{variant}_seed{seed}_{args.matched_200k_steps}step"
            rows.append(
                status_from_artifacts(
                    pipeline="200k",
                    variant=variant,
                    seed=str(seed),
                    run_name=run_name,
                    run_dir=args.matched_200k_run_root / run_name,
                    eval_root=args.matched_200k_eval_root,
                    final_step=args.matched_200k_steps,
                )
            )
    rows.extend(collect_external_dense_compute_rows(args))
    return rows


def write_outputs(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "formal_status.json"
    csv_path = output_dir / "formal_status.csv"
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    if not rows:
        csv_path.write_text("")
        return
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, Any]]) -> None:
    by_pipeline: dict[str, dict[str, int]] = {}
    for row in rows:
        counts = by_pipeline.setdefault(
            row["pipeline"], {"done": 0, "partial": 0, "todo": 0}
        )
        counts[row["status"]] += 1
    for pipeline, counts in sorted(by_pipeline.items()):
        print(
            f"{pipeline}: done={counts['done']} partial={counts['partial']} todo={counts['todo']}"
        )
    pending = [row for row in rows if row["status"] != "done"]
    if pending:
        print("Pending formal runs:")
        for row in pending:
            print(
                f"  {row['pipeline']} {row['variant']} seed={row['seed'] or '-'} "
                f"status={row['status']} last_step={row['last_step']}/{row['target_step']}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/formal_status"))
    parser.add_argument(
        "--stage1-run-root", type=Path, default=Path("runs/stage1_matched_budget")
    )
    parser.add_argument(
        "--stage1-eval-root", type=Path, default=Path("runs/stage1_heldout_eval")
    )
    parser.add_argument("--stage1-steps", type=int, default=100000)
    parser.add_argument("--stage1-variants", nargs="+", default=list(STAGE1_VARIANTS))
    parser.add_argument(
        "--matched-200k-run-root",
        type=Path,
        default=Path("runs/byte_entropy_200k_matched_controls"),
    )
    parser.add_argument(
        "--matched-200k-eval-root",
        type=Path,
        default=Path("runs/byte_entropy_200k_matched_controls_heldout_eval"),
    )
    parser.add_argument("--matched-200k-steps", type=int, default=200000)
    parser.add_argument(
        "--matched-200k-variants", nargs="+", default=list(MATCHED_200K_VARIANTS)
    )
    parser.add_argument("--seeds", nargs="+", default=["779"])
    parser.add_argument(
        "--external-dense-compute",
        nargs="*",
        default=[
            "779:runs/dense_compute_matched_200k:runs/dense_compute_matched_200k_heldout_eval:200000",
            "778:runs/dense_compute_matched_seed778_200k:runs/dense_compute_matched_seed778_200k_heldout_eval:200000",
        ],
        help="Extra dense-compute controls as seed:run_root:eval_root:steps.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = collect_rows(args)
    write_outputs(rows, args.output_dir)
    print_summary(rows)
    print(f"Wrote: {args.output_dir / 'formal_status.csv'}")
    print(f"Wrote: {args.output_dir / 'formal_status.json'}")


if __name__ == "__main__":
    main()
