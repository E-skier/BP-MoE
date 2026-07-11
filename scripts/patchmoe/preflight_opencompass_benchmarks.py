#!/usr/bin/env python3
"""Preflight checks for PatchMoE OpenCompass benchmark runs."""

from __future__ import annotations

import argparse
import importlib.util
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


DEFAULT_DATASETS = (
    "mmlu_ppl hellaswag_ppl ARC_c_ppl ARC_e_ppl obqa_ppl piqa_ppl SuperGLUE_BoolQ_ppl"
)


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str
    warning: bool = False


def has_consolidated_files(path: Path) -> bool:
    return (path / "params.json").is_file() and any(path.glob("*.pth"))


def has_dcp_files(path: Path) -> bool:
    return (path / "params.json").is_file() and (path / ".metadata").is_file()


def check_checkpoint(path: Path) -> CheckResult:
    if not path.exists():
        return CheckResult("checkpoint", False, f"missing path: {path}")
    if not path.is_dir():
        return CheckResult(
            "checkpoint", False, f"expected checkpoint directory: {path}"
        )
    if has_consolidated_files(path):
        return CheckResult("checkpoint", True, f"consolidated checkpoint: {path}")
    consolidated = path / "consolidated"
    if has_consolidated_files(consolidated):
        return CheckResult(
            "checkpoint", True, f"nested consolidated checkpoint: {consolidated}"
        )
    if has_dcp_files(path):
        return CheckResult(
            "checkpoint",
            True,
            f"DCP checkpoint with params.json; launcher can consolidate: {path}",
        )
    return CheckResult(
        "checkpoint",
        False,
        "expected params.json plus *.pth, params.json plus .metadata, or a "
        f"consolidated/ child under {path}",
    )


def check_project_import(root: Path) -> CheckResult:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{root}:{env.get('PYTHONPATH', '')}"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from bytelatent.opencompass import ByteLatentOpenCompassModel; print(ByteLatentOpenCompassModel.__name__)",
        ],
        text=True,
        capture_output=True,
        env=env,
    )
    if result.returncode == 0:
        return CheckResult("adapter import", True, result.stdout.strip())
    detail = (result.stderr or result.stdout).strip()
    return CheckResult("adapter import", False, detail or "import failed")


def check_opencompass_import(allow_missing: bool) -> CheckResult:
    spec = importlib.util.find_spec("opencompass")
    if spec is not None:
        return CheckResult("opencompass import", True, f"found at {spec.origin}")
    detail = "Python package 'opencompass' is not installed in this environment"
    return CheckResult(
        "opencompass import", allow_missing, detail, warning=allow_missing
    )


def check_model_registration(root: Path, allow_missing: bool) -> CheckResult:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{root}:{env.get('PYTHONPATH', '')}"
    script = """
import sys
try:
    from opencompass.registry import MODELS
except Exception as exc:
    print(exc)
    raise SystemExit(2)
from bytelatent.opencompass import ByteLatentOpenCompassModel
registered = None
if hasattr(MODELS, '_module_dict'):
    registered = MODELS._module_dict.get('ByteLatentOpenCompassModel')
elif hasattr(MODELS, 'module_dict'):
    registered = MODELS.module_dict.get('ByteLatentOpenCompassModel')
elif hasattr(MODELS, 'registered'):
    registered = MODELS.registered.get('ByteLatentOpenCompassModel')
if registered is None:
    print('ByteLatentOpenCompassModel is not visible in OpenCompass MODELS registry')
    raise SystemExit(1)
print('ByteLatentOpenCompassModel')
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        text=True,
        capture_output=True,
        env=env,
    )
    if result.returncode == 0:
        return CheckResult("model registry", True, result.stdout.strip())
    detail = (result.stderr or result.stdout).strip()
    if result.returncode == 2:
        return CheckResult(
            "model registry",
            allow_missing,
            detail or "OpenCompass registry is unavailable",
            warning=allow_missing,
        )
    return CheckResult("model registry", False, detail or "model is not registered")


def check_opencompass_command(opencompass_bin: str, allow_missing: bool) -> CheckResult:
    parts = shlex.split(opencompass_bin)
    if not parts:
        return CheckResult("opencompass command", False, "empty OPENCOMPASS_BIN")
    executable = parts[0]
    if shutil.which(executable) or Path(executable).exists():
        return CheckResult("opencompass command", True, opencompass_bin)
    detail = f"command not found: {executable}"
    return CheckResult(
        "opencompass command", allow_missing, detail, warning=allow_missing
    )


def find_list_configs(opencompass_root: Path | None) -> Path | None:
    if opencompass_root is None:
        return None
    candidate = opencompass_root / "tools" / "list_configs.py"
    return candidate if candidate.is_file() else None


def check_dataset_configs(
    datasets: list[str],
    opencompass_root: Path | None,
    allow_missing_opencompass: bool,
) -> CheckResult:
    list_configs = find_list_configs(opencompass_root)
    if list_configs is None:
        if opencompass_root is None:
            return CheckResult(
                "dataset configs",
                True,
                "skipped exact dataset-name check; pass --opencompass-root to enable tools/list_configs.py",
            )
        return CheckResult(
            "dataset configs",
            allow_missing_opencompass,
            f"missing tools/list_configs.py under {opencompass_root}",
            warning=allow_missing_opencompass,
        )

    result = subprocess.run(
        [sys.executable, str(list_configs), *datasets],
        text=True,
        capture_output=True,
        cwd=opencompass_root,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        return CheckResult("dataset configs", False, detail or "list_configs.py failed")
    output = result.stdout.lower()
    missing = [dataset for dataset in datasets if dataset.lower() not in output]
    if missing:
        return CheckResult(
            "dataset configs",
            False,
            "not found by tools/list_configs.py: " + ", ".join(missing),
        )
    return CheckResult(
        "dataset configs", True, "all requested dataset names were listed"
    )


def check_config(path: Path) -> CheckResult:
    if path.is_file():
        return CheckResult("config", True, str(path))
    return CheckResult("config", False, f"missing config file: {path}")


def check_extra_data_files(datasets: list[str], root: Path) -> CheckResult:
    cache_root = Path(
        os.environ.get("COMPASS_DATA_CACHE", Path.home() / ".cache" / "opencompass")
    )
    required: list[Path] = []
    if "obqa_ppl" in datasets:
        required.extend(
            [
                cache_root / "data" / "openbookqa" / "Main" / "test.jsonl",
                cache_root
                / "data"
                / "openbookqa"
                / "Additional"
                / "test_complete.jsonl",
            ]
        )
    if "piqa_ppl" in datasets:
        required.extend(
            [
                cache_root / "data" / "piqa" / "train.jsonl",
                cache_root / "data" / "piqa" / "train-labels.lst",
                cache_root / "data" / "piqa" / "dev.jsonl",
                cache_root / "data" / "piqa" / "dev-labels.lst",
            ]
        )
    if "SuperGLUE_BoolQ_ppl" in datasets:
        required.append(root / "opencompass" / "boolq")

    missing = [path for path in required if not path.exists()]
    if missing:
        return CheckResult(
            "extra local data",
            False,
            "missing files: "
            + ", ".join(str(path) for path in missing)
            + "; run ./venv/bin/python scripts/patchmoe/prepare_opencompass_extra_data.py",
        )
    if required:
        return CheckResult(
            "extra local data", True, f"found {len(required)} required local files"
        )
    return CheckResult("extra local data", True, "no extra local files required")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ckpt_dir", type=Path, help="BLT/PatchMoE checkpoint directory")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("apps/opencompass/patchmoe_benchmarks.py"),
        help="OpenCompass config file used by the launcher.",
    )
    parser.add_argument(
        "--datasets",
        default=DEFAULT_DATASETS,
        help="Whitespace-separated OpenCompass dataset config names.",
    )
    parser.add_argument(
        "--opencompass-bin",
        default="opencompass",
        help="OpenCompass entry point, matching OPENCOMPASS_BIN.",
    )
    parser.add_argument(
        "--opencompass-root",
        type=Path,
        default=None,
        help="Optional OpenCompass source checkout for exact tools/list_configs.py checks.",
    )
    parser.add_argument(
        "--allow-missing-opencompass",
        action="store_true",
        help="Allow OpenCompass package/CLI checks to warn instead of fail.",
    )
    return parser.parse_args()


def print_result(result: CheckResult) -> None:
    if result.warning:
        status = "WARN"
    else:
        status = "OK" if result.ok else "FAIL"
    print(f"[{status}] {result.name}: {result.detail}")


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[2]
    datasets = shlex.split(args.datasets)
    checks = [
        check_config(args.config),
        check_project_import(root),
        check_checkpoint(args.ckpt_dir),
        check_opencompass_import(args.allow_missing_opencompass),
        check_model_registration(root, args.allow_missing_opencompass),
        check_opencompass_command(args.opencompass_bin, args.allow_missing_opencompass),
        check_dataset_configs(
            datasets,
            args.opencompass_root,
            args.allow_missing_opencompass,
        ),
        check_extra_data_files(datasets, root),
    ]
    for result in checks:
        print_result(result)
    return 0 if all(result.ok for result in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
