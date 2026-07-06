#!/usr/bin/env python3
import argparse
import glob
import hashlib
import json
from pathlib import Path
import sys
from typing import Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bytelatent.data.patcher import (
    find_entropy_patch_start_ids,
    patch_lengths_from_start_ids,
)


def _as_1d_float_tensor(values: torch.Tensor | Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32).reshape(-1)
    if tensor.numel() == 0:
        raise ValueError("calibration tensors must be non-empty")
    return tensor


def byte_weighted_quantiles(
    values: torch.Tensor,
    weights: torch.Tensor,
    quantiles: Sequence[float],
) -> torch.Tensor:
    if values.numel() != weights.numel():
        raise ValueError("values and weights must have the same length")
    if torch.any(weights < 0):
        raise ValueError("weights must be non-negative")
    total_weight = weights.sum()
    if total_weight <= 0:
        raise ValueError("weights must contain positive mass")

    order = torch.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = torch.cumsum(sorted_weights, dim=0) / total_weight
    out = []
    for quantile in quantiles:
        if not 0.0 <= quantile <= 1.0:
            raise ValueError(f"quantile must be in [0, 1], got {quantile}")
        idx = torch.searchsorted(cumulative, torch.tensor(float(quantile)))
        idx = idx.clamp(max=sorted_values.numel() - 1)
        out.append(sorted_values[idx])
    return torch.stack(out)


def _initial_widths(centers: torch.Tensor, min_width: float) -> torch.Tensor:
    if centers.numel() == 1:
        return centers.new_full((1,), max(float(min_width), 1.0))
    sorted_centers = centers.sort().values
    widths = torch.empty_like(sorted_centers)
    widths[0] = (sorted_centers[1] - sorted_centers[0]).abs()
    widths[-1] = (sorted_centers[-1] - sorted_centers[-2]).abs()
    if sorted_centers.numel() > 2:
        widths[1:-1] = (
            sorted_centers[2:] - sorted_centers[:-2]
        ).abs() * 0.5
    return widths.clamp_min(float(min_width))


def _soft_byte_share(
    patch_lengths: torch.Tensor,
    patch_entropies: torch.Tensor,
    centers: torch.Tensor,
    widths: torch.Tensor,
    static_bias: torch.Tensor,
    prior_scale: float,
) -> torch.Tensor:
    entropies = patch_entropies.unsqueeze(-1)
    logits = -((entropies - centers) ** 2) / (2.0 * widths.clamp_min(1e-6).square())
    logits = logits * float(prior_scale) + static_bias
    probs = torch.softmax(logits, dim=-1)
    byte_mass = (probs * patch_lengths.unsqueeze(-1)).sum(dim=0)
    return byte_mass / patch_lengths.sum().clamp_min(1.0)


def _fingerprint(patch_lengths: torch.Tensor, patch_entropies: torch.Tensor) -> str:
    hasher = hashlib.sha256()
    hasher.update(patch_lengths.cpu().contiguous().numpy().tobytes())
    hasher.update(patch_entropies.cpu().contiguous().numpy().tobytes())
    return hasher.hexdigest()[:16]


def calibrate_entropy_prior(
    *,
    patch_lengths: torch.Tensor | Sequence[float],
    patch_entropies: torch.Tensor | Sequence[float],
    num_pairs: int = 4,
    pair_size: int = 2,
    prior_scale: float = 0.5,
    min_width: float = 0.05,
    bias_steps: int = 500,
    bias_lr: float = 0.25,
    data_fingerprint: str | None = None,
) -> dict:
    if num_pairs <= 0:
        raise ValueError("num_pairs must be positive")
    if pair_size <= 0:
        raise ValueError("pair_size must be positive")

    lengths = _as_1d_float_tensor(patch_lengths)
    entropies = _as_1d_float_tensor(patch_entropies)
    if lengths.numel() != entropies.numel():
        raise ValueError("patch_lengths and patch_entropies must have the same length")

    valid_mask = lengths > 0
    valid_lengths = lengths[valid_mask]
    valid_entropies = entropies[valid_mask].clamp_min(0.0)
    if valid_lengths.numel() == 0:
        raise ValueError("calibration requires at least one valid patch")

    quantiles = [(idx + 0.5) / num_pairs for idx in range(num_pairs)]
    centers = byte_weighted_quantiles(valid_entropies, valid_lengths, quantiles)
    widths = _initial_widths(centers, min_width=min_width)

    static_bias = torch.zeros(num_pairs, dtype=torch.float32, requires_grad=True)
    target = torch.full((num_pairs,), 1.0 / num_pairs, dtype=torch.float32)
    optimizer = torch.optim.SGD([static_bias], lr=float(bias_lr))
    for _ in range(int(bias_steps)):
        optimizer.zero_grad()
        share = _soft_byte_share(
            valid_lengths,
            valid_entropies,
            centers,
            widths,
            static_bias,
            prior_scale,
        )
        loss = (share - target).square().sum()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            static_bias -= static_bias.mean()

    final_bias = static_bias.detach()
    calibrated_share = _soft_byte_share(
        valid_lengths,
        valid_entropies,
        centers,
        widths,
        final_bias,
        prior_scale,
    )

    return {
        "version": 1,
        "num_pairs": int(num_pairs),
        "num_experts": int(num_pairs * pair_size),
        "pair_size": int(pair_size),
        "entropy_centers": [float(value) for value in centers.tolist()],
        "entropy_widths": [float(value) for value in widths.tolist()],
        "static_pair_bias": [float(value) for value in final_bias.tolist()],
        "target_byte_share": [float(value) for value in target.tolist()],
        "calibrated_byte_share": [
            float(value) for value in calibrated_share.detach().tolist()
        ],
        "entropy_prior_scale": float(prior_scale),
        "calibration_num_valid_patches": int(valid_lengths.numel()),
        "calibration_total_bytes": int(valid_lengths.sum().item()),
        "data_fingerprint": data_fingerprint
        or _fingerprint(valid_lengths, valid_entropies),
    }


def _mean_patch_entropies(
    token_entropies: Sequence[float], patch_lengths: Sequence[int]
) -> list[float]:
    patch_entropies: list[float] = []
    start = 0
    for patch_length in patch_lengths:
        if patch_length <= 0:
            patch_entropies.append(0.0)
            continue
        end = start + int(patch_length)
        patch_values = token_entropies[start:end]
        if len(patch_values) != patch_length:
            raise ValueError(
                f"patch length {patch_length} exceeds token entropies at {start}:{end}"
            )
        patch_entropies.append(float(sum(patch_values) / patch_length))
        start = end
    if start != len(token_entropies):
        raise ValueError(f"patch lengths consume {start} tokens, expected {len(token_entropies)}")
    return patch_entropies


def _iter_arrow_record_batches(path: str | Path):
    import pyarrow as pa

    path = Path(path)
    try:
        with pa.memory_map(str(path), "r") as source:
            reader = pa.ipc.open_file(source)
            for batch_idx in range(reader.num_record_batches):
                yield reader.get_batch(batch_idx)
            return
    except (pa.ArrowInvalid, OSError):
        pass

    with pa.memory_map(str(path), "r") as source:
        reader = pa.ipc.open_stream(source)
        for batch in reader:
            yield batch


def _entropy_patch_lengths(
    token_entropies: Sequence[float],
    *,
    patch_size: float | None,
    threshold: float | None,
    threshold_add: float | None,
    monotonicity: bool,
    include_next_token: bool,
) -> list[int]:
    entropies = torch.tensor([token_entropies], dtype=torch.float32)
    patch_start_ids = find_entropy_patch_start_ids(
        entropies,
        patch_size=patch_size,
        threshold=threshold,
        threshold_add=threshold_add,
        monotonicity=monotonicity,
        include_next_token=include_next_token,
    )
    seq_len_next_tok = entropies.shape[1] + (1 if include_next_token else 0)
    return patch_lengths_from_start_ids(patch_start_ids, seq_len_next_tok)[0].tolist()


def collect_entropy_patch_statistics_from_arrow(
    arrow_files: Sequence[str | Path],
    *,
    patch_size: float | None = 4.5,
    threshold: float | None = 1.335442066192627,
    threshold_add: float | None = None,
    monotonicity: bool = False,
    include_next_token: bool = False,
    max_rows: int | None = None,
    max_patches: int | None = None,
    max_bytes: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    if not arrow_files:
        raise ValueError("at least one Arrow file is required")
    if include_next_token:
        raise ValueError(
            "Arrow entropy calibration requires include_next_token=False because "
            "the shard stores one entropy value per input token"
        )

    all_lengths: list[float] = []
    all_entropies: list[float] = []
    rows_seen = 0
    rows_used = 0
    files_seen = 0
    total_bytes = 0.0

    for arrow_file in arrow_files:
        files_seen += 1
        for batch in _iter_arrow_record_batches(arrow_file):
            if "entropies" not in batch.schema.names:
                raise ValueError(f"{arrow_file} does not contain an entropies column")
            entropies_column = batch.column(batch.schema.get_field_index("entropies"))
            for row_entropies in entropies_column.to_pylist():
                if max_rows is not None and rows_seen >= max_rows:
                    break
                rows_seen += 1
                if not row_entropies:
                    continue

                token_entropies = [float(value) for value in row_entropies]
                patch_lengths = _entropy_patch_lengths(
                    token_entropies,
                    patch_size=patch_size,
                    threshold=threshold,
                    threshold_add=threshold_add,
                    monotonicity=monotonicity,
                    include_next_token=include_next_token,
                )
                patch_entropies = _mean_patch_entropies(token_entropies, patch_lengths)
                valid_pairs = [
                    (float(length), float(entropy))
                    for length, entropy in zip(patch_lengths, patch_entropies)
                    if length > 0
                ]
                if not valid_pairs:
                    continue

                rows_used += 1
                for length, entropy in valid_pairs:
                    all_lengths.append(length)
                    all_entropies.append(entropy)
                    total_bytes += length
                    if max_patches is not None and len(all_lengths) >= max_patches:
                        break
                    if max_bytes is not None and total_bytes >= max_bytes:
                        break
                if (
                    (max_patches is not None and len(all_lengths) >= max_patches)
                    or (max_bytes is not None and total_bytes >= max_bytes)
                ):
                    break
            if (
                (max_rows is not None and rows_seen >= max_rows)
                or (max_patches is not None and len(all_lengths) >= max_patches)
                or (max_bytes is not None and total_bytes >= max_bytes)
            ):
                break
        if (
            (max_rows is not None and rows_seen >= max_rows)
            or (max_patches is not None and len(all_lengths) >= max_patches)
            or (max_bytes is not None and total_bytes >= max_bytes)
        ):
            break

    if not all_lengths:
        raise ValueError("no valid entropy patches collected from Arrow input")

    lengths_tensor = torch.tensor(all_lengths, dtype=torch.float32)
    entropies_tensor = torch.tensor(all_entropies, dtype=torch.float32)
    metadata = {
        "num_arrow_files_seen": files_seen,
        "num_rows_seen": rows_seen,
        "num_rows_used": rows_used,
        "num_valid_patches": int(lengths_tensor.numel()),
        "total_bytes": int(total_bytes),
    }
    return lengths_tensor, entropies_tensor, metadata


def write_calibration(report: dict, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def _load_tensor(path: str | Path) -> torch.Tensor:
    path = Path(path)
    if path.suffix == ".pt":
        value = torch.load(path, map_location="cpu", weights_only=True)
        return torch.as_tensor(value)
    if path.suffix == ".npy":
        import numpy as np

        return torch.from_numpy(np.load(path))
    raise ValueError(f"Unsupported tensor input extension: {path.suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate byte-weighted gaussian pair entropy prior."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--patch-lengths")
    input_group.add_argument("--arrow-glob")
    input_group.add_argument("--arrow-file", action="append")
    parser.add_argument("--patch-entropies")
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-pairs", type=int, default=4)
    parser.add_argument("--pair-size", type=int, default=2)
    parser.add_argument("--prior-scale", type=float, default=0.5)
    parser.add_argument("--min-width", type=float, default=0.05)
    parser.add_argument("--bias-steps", type=int, default=500)
    parser.add_argument("--bias-lr", type=float, default=0.25)
    parser.add_argument("--patch-size", type=float, default=4.5)
    parser.add_argument("--threshold", type=float, default=1.335442066192627)
    parser.add_argument("--threshold-add", type=float)
    parser.add_argument("--monotonicity", action="store_true")
    parser.add_argument("--include-next-token", action="store_true")
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--max-patches", type=int)
    parser.add_argument("--max-bytes", type=int)
    args = parser.parse_args()

    input_metadata = {}
    if args.patch_lengths is not None:
        if args.patch_entropies is None:
            parser.error("--patch-entropies is required with --patch-lengths")
        patch_lengths = _load_tensor(args.patch_lengths)
        patch_entropies = _load_tensor(args.patch_entropies)
    else:
        arrow_files = [Path(path) for path in args.arrow_file or []]
        if args.arrow_glob is not None:
            arrow_files.extend(Path(path) for path in sorted(glob.glob(args.arrow_glob)))
        if not arrow_files:
            parser.error("Arrow input did not match any files")
        patch_lengths, patch_entropies, input_metadata = (
            collect_entropy_patch_statistics_from_arrow(
                arrow_files,
                patch_size=args.patch_size,
                threshold=args.threshold,
                threshold_add=args.threshold_add,
                monotonicity=args.monotonicity,
                include_next_token=args.include_next_token,
                max_rows=args.max_rows,
                max_patches=args.max_patches,
                max_bytes=args.max_bytes,
            )
        )

    report = calibrate_entropy_prior(
        patch_lengths=patch_lengths,
        patch_entropies=patch_entropies,
        num_pairs=args.num_pairs,
        pair_size=args.pair_size,
        prior_scale=args.prior_scale,
        min_width=args.min_width,
        bias_steps=args.bias_steps,
        bias_lr=args.bias_lr,
    )
    if input_metadata:
        report["calibration_input"] = input_metadata
    write_calibration(report, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
