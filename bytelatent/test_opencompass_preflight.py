import importlib.util
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "patchmoe" / "preflight_opencompass_benchmarks.py"


def load_preflight_module():
    spec = importlib.util.spec_from_file_location("opencompass_preflight", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_checkpoint_check_accepts_consolidated_directory(tmp_path):
    module = load_preflight_module()
    (tmp_path / "params.json").write_text("{}")
    (tmp_path / "consolidated.pth").write_bytes(b"")

    result = module.check_checkpoint(tmp_path)

    assert result.ok
    assert "consolidated" in result.detail


def test_checkpoint_check_accepts_dcp_directory(tmp_path):
    module = load_preflight_module()
    (tmp_path / "params.json").write_text("{}")
    (tmp_path / ".metadata").write_text("{}")

    result = module.check_checkpoint(tmp_path)

    assert result.ok
    assert "DCP" in result.detail


def test_checkpoint_check_rejects_incomplete_directory(tmp_path):
    module = load_preflight_module()
    (tmp_path / "params.json").write_text("{}")

    result = module.check_checkpoint(tmp_path)

    assert not result.ok
    assert "expected params.json" in result.detail


def test_model_registration_check_accepts_fake_opencompass_registry(
    tmp_path, monkeypatch
):
    import textwrap

    module = load_preflight_module()
    fake_pkg = tmp_path / "fake_opencompass"
    base_dir = fake_pkg / "opencompass" / "models"
    base_dir.mkdir(parents=True)
    (fake_pkg / "opencompass" / "__init__.py").write_text("")
    (base_dir / "__init__.py").write_text("")
    (base_dir / "base.py").write_text(
        textwrap.dedent(
            """
            class BaseModel:
                def __init__(self, *args, **kwargs):
                    pass
            """
        )
    )
    (fake_pkg / "opencompass" / "registry.py").write_text(
        textwrap.dedent(
            """
            class Registry:
                def __init__(self):
                    self.registered = {}
                def register_module(self):
                    def decorate(cls):
                        self.registered[cls.__name__] = cls
                        return cls
                    return decorate
                def get(self, name):
                    return self.registered.get(name)
            MODELS = Registry()
            """
        )
    )
    monkeypatch.setenv("PYTHONPATH", str(fake_pkg))

    result = module.check_model_registration(ROOT, allow_missing=False)

    assert result.ok
    assert result.detail == "ByteLatentOpenCompassModel"


def test_preflight_cli_can_warn_for_missing_opencompass(tmp_path):
    (tmp_path / "params.json").write_text("{}")
    (tmp_path / "consolidated.pth").write_bytes(b"")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(tmp_path),
            "--config",
            str(ROOT / "apps" / "opencompass" / "patchmoe_benchmarks.py"),
            "--opencompass-bin",
            "echo",
            "--datasets",
            "mmlu_ppl",
            "--allow-missing-opencompass",
        ],
        check=True,
        text=True,
        capture_output=True,
        cwd=ROOT,
    )

    assert "[OK] checkpoint" in result.stdout
    assert (
        "[WARN] opencompass import" in result.stdout
        or "[OK] opencompass import" in result.stdout
    )
    assert (
        "[WARN] model registry" in result.stdout
        or "[OK] model registry" in result.stdout
    )


def test_opencompass_config_loads_with_mmengine_when_available(monkeypatch):
    pytest = __import__("pytest")
    mmengine_config = pytest.importorskip("mmengine.config")
    pytest.importorskip("opencompass")

    monkeypatch.setenv("PATCHMOE_CKPT_DIR", "/tmp/ckpt")
    monkeypatch.setenv("PATCHMOE_OC_DATASETS", "mmlu_ppl")
    monkeypatch.setenv("PATCHMOE_OC_TEST_RANGE", "[0:1]")

    cfg = mmengine_config.Config.fromfile(
        ROOT / "apps" / "opencompass" / "patchmoe_benchmarks.py",
        format_python_code=False,
    )

    assert cfg.models[0]["ckpt_dir"] == "/tmp/ckpt"
    assert len(cfg.datasets) > 0
    assert cfg.datasets[0]["reader_cfg"]["test_range"] == "[0:1]"


def test_extra_data_check_reports_missing_files(tmp_path, monkeypatch):
    module = load_preflight_module()
    monkeypatch.setenv("COMPASS_DATA_CACHE", str(tmp_path / "cache"))

    result = module.check_extra_data_files(["obqa_ppl", "piqa_ppl"], tmp_path)

    assert not result.ok
    assert "prepare_opencompass_extra_data.py" in result.detail


def test_extra_data_check_accepts_present_files(tmp_path, monkeypatch):
    module = load_preflight_module()
    cache = tmp_path / "cache"
    monkeypatch.setenv("COMPASS_DATA_CACHE", str(cache))
    required = [
        cache / "data" / "openbookqa" / "Main" / "test.jsonl",
        cache / "data" / "openbookqa" / "Additional" / "test_complete.jsonl",
        cache / "data" / "piqa" / "train.jsonl",
        cache / "data" / "piqa" / "train-labels.lst",
        cache / "data" / "piqa" / "dev.jsonl",
        cache / "data" / "piqa" / "dev-labels.lst",
        tmp_path / "opencompass" / "boolq",
    ]
    for path in required:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n")

    result = module.check_extra_data_files(
        ["obqa_ppl", "piqa_ppl", "SuperGLUE_BoolQ_ppl"], tmp_path
    )

    assert result.ok
    assert "7 required" in result.detail
