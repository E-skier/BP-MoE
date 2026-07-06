import json
from pathlib import Path

from bytelatent.checkpoint import (
    COMPLETE_NAME,
    LATEST_NAME,
    CheckpointArgs,
    CheckpointManager,
    SaveEvery,
)


def _make_checkpoint(root: Path, step: int, *, complete: bool = True) -> Path:
    ckpt = root / f"{step:010d}"
    ckpt.mkdir(parents=True)
    (ckpt / ".metadata").write_text("{}")
    (ckpt / "train_state_00000.json").write_text("{}")
    if complete:
        (ckpt / COMPLETE_NAME).write_text(json.dumps({"step": step, "verified": True}))
    return ckpt


def test_checkpoint_manager_resumes_from_latest_complete_pointer(tmp_path):
    _make_checkpoint(tmp_path, 1000, complete=True)
    latest = _make_checkpoint(tmp_path, 2000, complete=True)
    _make_checkpoint(tmp_path, 3000, complete=False)
    (tmp_path / LATEST_NAME).write_text(
        json.dumps({"path": str(latest), "step": 2000, "verified": True})
    )

    manager = CheckpointManager(
        CheckpointArgs(path=str(tmp_path), dump=SaveEvery(every=1000, keep=2))
    )

    assert manager.get_last_step_path(dp_rank=0) == str(latest)


def test_checkpoint_retention_keeps_latest_and_fallback_complete_checkpoints(tmp_path):
    older = _make_checkpoint(tmp_path, 1000, complete=True)
    fallback = _make_checkpoint(tmp_path, 2000, complete=True)
    latest = _make_checkpoint(tmp_path, 3000, complete=True)
    (tmp_path / LATEST_NAME).write_text(
        json.dumps({"path": str(latest), "step": 3000, "verified": True})
    )
    manager = CheckpointManager(
        CheckpointArgs(
            path=str(tmp_path),
            dump=SaveEvery(every=1000, keep=1),
            eval=SaveEvery(every=1000, keep=1),
        )
    )
    manager.clean_up()

    assert not older.exists()
    assert fallback.exists()
    assert latest.exists()
    latest_data = json.loads((tmp_path / LATEST_NAME).read_text())
    assert latest_data["path"] == str(latest)
