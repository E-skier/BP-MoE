import sys
import types
from importlib.machinery import ModuleSpec


sys.modules.setdefault("s3fs", types.SimpleNamespace(S3FileSystem=object))
sys.modules.setdefault("jsonlines", types.SimpleNamespace())
wandb_stub = types.ModuleType("wandb")
wandb_stub.__spec__ = ModuleSpec("wandb", loader=None)
sys.modules.setdefault("wandb", wandb_stub)

from bytelatent.args import DataloaderArgs


def test_dataloader_dataset_files_are_distributed_without_world_size_truncation(monkeypatch):
    captured = []

    class FakeArrowIterator:
        def __init__(self, **kwargs):
            captured.append(kwargs)

    monkeypatch.setattr("bytelatent.args.ArrowFileIterator", FakeArrowIterator)

    args = DataloaderArgs(
        root_dir="/data",
        sources={"fineweb_edu_10bt": 1.0},
        preprocess_dir="/preprocessed",
        dataset_files=[
            f"/data/fineweb_edu_10bt/fineweb_edu_10bt.chunk.{idx:05d}.jsonl"
            for idx in range(7)
        ],
    )

    iterator = args._build_arrow_iterator_for_source(
        dataset_path="/data/fineweb_edu_10bt",
        rank=5,
        world_size=6,
    )

    assert isinstance(iterator, FakeArrowIterator)
    assert captured[0]["file_path"] is None
    assert captured[0]["dataset_files"] == args.dataset_files
    assert captured[0]["worker_id"] == 5
    assert captured[0]["num_workers"] == 6
