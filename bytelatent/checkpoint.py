# Copyright (c) Meta Platforms, Inc. and affiliates.

import json
import logging
import os
import re
import time

import fsspec
import s3fs
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn
import torch.optim.optimizer
import typer
from pydantic import BaseModel, ConfigDict
from torch.distributed._tensor import DeviceMesh
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_state_dict,
    set_state_dict,
)

from bytelatent.data.file_util import get_fs
from bytelatent.distributed import get_is_master

logger = logging.getLogger("CHECKPOINT")

FOLDER_NAME = "{:010d}"
RE_FOLDER = r"\d{10}"

RE_CKPT = r"__\d_\d\.distcp"

CONSOLIDATE_FOLDER = "consolidated"
CONSOLIDATE_NAME = "consolidated.pth"

CONFIG_NAME = "params.json"
TRAIN_STATE_NAME = "train_state_{:05d}.json"
LATEST_NAME = "LATEST.json"
COMPLETE_NAME = "complete.json"
RE_DIGITS = re.compile(r"\d+")


def _dist_barrier() -> None:
    if not dist.is_initialized():
        return
    if torch.cuda.is_available():
        dist.barrier(device_ids=[torch.cuda.current_device()])
    else:
        dist.barrier()


class SaveEvery(BaseModel):
    model_config = ConfigDict(extra="forbid")
    every: int = 1000
    keep: int = 0


class CheckpointArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dump: SaveEvery = SaveEvery()
    eval: SaveEvery = SaveEvery()
    path: str | None = None
    init_ckpt_path: str | None = None
    continue_training_from_init: bool = False
    s3_profile: str | None = None


def _get_key_step(name: str):
    return int(re.findall(RE_DIGITS, name)[-1])


def consolidate_checkpoints(fs: fsspec.AbstractFileSystem, ckpt_dir: str):
    """
    Consolidates all FSDP checkpoints in a directory to a single file
    Consolidate checkpoint is saved in a subdirectory of ckpt_dir

    Parameters:
        ckpt_dir: str - path to the directory containing the checkpoints

    Returns the path to the consolidated checkpoint
    """
    consolidate_path = os.path.join(ckpt_dir, CONSOLIDATE_FOLDER)
    consolidate_name = os.path.join(consolidate_path, CONSOLIDATE_NAME)
    if not fs.exists(consolidate_name):
        fs.mkdirs(consolidate_path, exist_ok=True)
        logger.info(f"Consolidating to: {consolidate_path}")
        dcp_to_torch_save(ckpt_dir, consolidate_name)
        fs.write_text(
            os.path.join(consolidate_path, CONFIG_NAME),
            fs.read_text(os.path.join(ckpt_dir, CONFIG_NAME)),
        )
        logger.info("Consolidated !")
    return consolidate_path


def load_from_checkpoint(
    fs: fsspec.AbstractFileSystem,
    ckpt_dir: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    model_key: str = "model",
    optim_key: str = "optim",
):
    if not fs.exists(os.path.join(ckpt_dir, ".metadata")):
        raise ValueError(
            f"Please convert the checkpoint distcp format using `torch.distributed.checkpoint.format_utils.torch_save_to_dcp` before loading it"
        )

    state_dict = {}
    if optimizer is not None:
        state_dict[model_key], state_dict[optim_key] = get_state_dict(model, optimizer)
    else:
        state_dict[model_key] = get_model_state_dict(model)
        if model_key == "":  # If only loading a model directly, the key should be empty
            state_dict = state_dict.pop(model_key)

    dcp.load(state_dict, checkpoint_id=ckpt_dir)


# TODO: Rewrite the file operations here to use fsspec to enable s3 writing.
class CheckpointManager:
    def __init__(self, args: CheckpointArgs):
        self.path = args.path
        self.fs = get_fs(self.path, s3_profile=args.s3_profile)
        self.dump_every = args.dump
        self.eval_every = args.eval
        self.init_ckpt_path = args.init_ckpt_path
        self.continue_training_from_init = args.continue_training_from_init

        if not isinstance(self.fs, s3fs.S3FileSystem):
            # S3 does not have a concept of directories
            assert self.fs.exists(
                self.path
            ), f"Path {self.path} does not exist and needs to be created before using CheckpointManager (use instantiate_and_make_dir)"

        self.existing_saves = self.get_existing_saves()

    def get_existing_saves(self) -> list[str]:
        if self.fs.exists(self.path) and self.fs.isdir(self.path):
            folders = [
                p
                for p in self.fs.ls(self.path)
                if self.fs.isdir(p) and re.match(RE_FOLDER, os.path.basename(p))
            ]
        else:
            folders = []
        folders.sort(key=lambda p: _get_key_step(os.path.basename(p)))
        return folders

    def _is_complete_checkpoint(self, path: str) -> bool:
        return self.fs.isfile(os.path.join(path, COMPLETE_NAME))

    def _read_latest_path(self, dp_rank: int = 0) -> str | None:
        latest_file = os.path.join(self.path, LATEST_NAME)
        if not self.fs.isfile(latest_file):
            return None
        try:
            data = json.loads(self.fs.read_text(latest_file))
        except Exception:
            logger.warning("Ignoring unreadable latest checkpoint pointer: %s", latest_file)
            return None
        path = data.get("path")
        if not path:
            return None
        train_state_name = TRAIN_STATE_NAME.format(dp_rank)
        if (
            self.fs.isdir(path)
            and self._is_complete_checkpoint(path)
            and self.fs.isfile(os.path.join(path, train_state_name))
        ):
            return path
        logger.warning("Ignoring invalid latest checkpoint pointer: %s", path)
        return None

    def _write_complete_marker(self, path: str, step: int) -> None:
        if not get_is_master():
            return
        marker = {
            "step": int(step),
            "verified": True,
            "path": path,
            "created_at": time.time(),
        }
        self.fs.write_text(os.path.join(path, COMPLETE_NAME), json.dumps(marker))

    def _write_latest_pointer(self, path: str, step: int) -> None:
        if not get_is_master():
            return
        latest_path = os.path.join(self.path, LATEST_NAME)
        tmp_path = os.path.join(self.path, f".{LATEST_NAME}.tmp")
        payload = {
            "step": int(step),
            "verified": True,
            "path": path,
            "complete_marker": os.path.join(path, COMPLETE_NAME),
        }
        self.fs.write_text(tmp_path, json.dumps(payload, indent=2, sort_keys=True))
        if hasattr(self.fs, "mv"):
            self.fs.mv(tmp_path, latest_path)
        else:
            os.replace(tmp_path, latest_path)

    def _remove_checkpoint_folder(self, folder: str) -> None:
        for file in self.fs.ls(folder):
            if self.fs.isfile(file):
                self.fs.rm_file(file)
            elif self.fs.isdir(file):
                for f in self.fs.ls(file):
                    self.fs.rm(f)
                self.fs.rmdir(file)
        self.fs.rmdir(folder)

    def clean_up(self):
        logger.info("Cleaning up checkpoints...")
        dump_folders = []
        eval_folders = []
        other_folders = []
        for p in self.existing_saves:
            assert isinstance(p, str), f"Base path type: {p}"
            is_dump = _get_key_step(os.path.basename(p)) % self.dump_every.every == 0
            is_eval = _get_key_step(os.path.basename(p)) % self.eval_every.every == 0
            if is_dump:
                dump_folders.append(p)
            if is_eval:
                eval_folders.append(p)
            if not (is_dump or is_eval):
                other_folders.append(p)

        logger.info(f"Dump folders: {dump_folders}")
        logger.info(f"Eval folders: {eval_folders}")
        logger.info(f"Other folders: {other_folders}")

        if self.dump_every.keep > 0:
            dump_folders = dump_folders[-self.dump_every.keep :]
        if self.eval_every.keep > 0:
            eval_folders = eval_folders[-self.eval_every.keep :]

        folder_to_keep = set(other_folders + dump_folders + eval_folders)
        complete_folders = [
            p for p in self.existing_saves if self._is_complete_checkpoint(p)
        ]
        # Formal runs must always keep a latest checkpoint plus one fallback.
        folder_to_keep.update(complete_folders[-2:])
        folder_to_remove = set(self.existing_saves) - folder_to_keep

        logger.info(f"Removing folders: {folder_to_remove}")

        if get_is_master():
            for folder in folder_to_remove:
                self._remove_checkpoint_folder(folder)

        _dist_barrier()

        self.existing_saves = list(folder_to_keep)
        self.existing_saves.sort(key=lambda p: _get_key_step(os.path.basename(p)))

    def get_last_step_path(self, dp_rank: int = 0) -> str | None:
        latest_path = self._read_latest_path(dp_rank=dp_rank)
        if latest_path is not None:
            return latest_path
        path = None
        for p in reversed(self.existing_saves):
            if self._is_complete_checkpoint(p) and self.fs.isfile(
                os.path.join(p, TRAIN_STATE_NAME.format(dp_rank))
            ):
                path = p
                break
        return path

    def _create_folder(self, base_path: str, folder_name: str) -> str:
        folder = os.path.join(base_path, folder_name)
        if get_is_master():
            self.fs.mkdirs(folder, exist_ok=True)
        if dist.is_initialized():
            _dist_barrier()
        return folder

    def _get_dp_tp_mesh(self, device_mesh: DeviceMesh | None = None) -> tuple[int, int]:
        dp_rank = 0
        tp_rank = 0
        if device_mesh is not None:
            if "dp_replicate" in device_mesh.mesh_dim_names:
                dp_rank = device_mesh.get_local_rank("dp_replicate")
                if "dp_shard" in device_mesh.mesh_dim_names:
                    dp_rank = dp_rank * device_mesh[
                        "dp_replicate"
                    ].size() + device_mesh.get_local_rank("dp_shard")
            if "tp" in device_mesh.mesh_dim_names:
                tp_rank = device_mesh.get_local_rank("tp")
        return dp_rank, tp_rank

    @torch.no_grad()
    def get_state_dict(
        self,
        model,
        optimizer,
    ):
        model_sd, optim_sd = get_state_dict(model, optimizer)
        return {"model": model_sd, "optim": optim_sd}

    def save(
        self,
        model,
        optimizer,
        train_state,
        config: BaseModel,
        device_mesh: DeviceMesh | None = None,
    ) -> bool:

        # When creating directory check if only rank0 or is there other solution
        path = self.path
        curr_save_dir = self._create_folder(path, FOLDER_NAME.format(train_state.step))
        logger.info(f"Saving to: {curr_save_dir}")

        if dist.is_initialized():
            _dist_barrier()

        logger.info("Saving...")
        state_dict = self.get_state_dict(model, optimizer)
        dcp.save(state_dict, checkpoint_id=curr_save_dir)
        logger.info("State dict saved!")

        if dist.is_initialized():
            _dist_barrier()

        print("config type", type(config))
        if get_is_master():
            self.fs.write_text(
                os.path.join(curr_save_dir, CONFIG_NAME), config.model_dump_json()
            )

        # Add json dump here
        dp_rank, tp_rank = self._get_dp_tp_mesh(device_mesh)
        if tp_rank == 0:
            train_state_name = TRAIN_STATE_NAME.format(dp_rank)
            train_state_full_path = os.path.join(curr_save_dir, train_state_name)
            logger.info(f"Saving train state to: {train_state_full_path}")
            with self.fs.open(train_state_full_path, "w") as f:
                json.dump(train_state.state_dict(), f)
            logger.info("Train state saved !")

        if dist.is_initialized():
            _dist_barrier()

        self._write_complete_marker(curr_save_dir, train_state.step)
        self._write_latest_pointer(curr_save_dir, train_state.step)

        self.existing_saves.append(curr_save_dir)

        self.clean_up()

        if dist.is_initialized():
            _dist_barrier()
        return True

    @torch.no_grad()
    def load(
        self,
        model: nn.Module,
        optimizer,
        train_state,
        device_mesh: DeviceMesh,
        path: str | None = None,
    ):
        dp_rank, tp_rank = self._get_dp_tp_mesh(device_mesh)
        # Loading tries to load the provided path, if not available the last saved step and finally from the init path
        path = path or self.get_last_step_path(dp_rank=dp_rank)
        # If none of those are available don't do anything
        if path is None:
            # If no checkpoints exist do nothing
            return

        # Only load train state if it's provided, the files exist and we're not loading from init path
        train_state_name = TRAIN_STATE_NAME.format(dp_rank)
        logger.info("Reloading train state")
        with self.fs.open(os.path.join(path, train_state_name), "r") as f:
            train_state_dict = json.load(f)
        train_state.load_state_dict(train_state_dict)
        logger.info("Train state reloaded")

        logger.info(f"Loading from: {path}")
        state_dict = self.get_state_dict(
            model=model,
            optimizer=optimizer,
        )
        dcp.load(state_dict, checkpoint_id=path)
        logger.info("State dict loaded.")

        logger.info("Reloading model and optim")

        set_state_dict(
            model,
            optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optim"],
        )
        logger.info("Model and optim reloaded")

    @classmethod
    def instantiate_and_make_dir(cls, args: CheckpointArgs):
        if get_is_master():
            os.makedirs(args.path, exist_ok=True)
        deadline = time.monotonic() + 60.0
        while args.path is not None and not os.path.isdir(args.path):
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"Timed out waiting for checkpoint path: {args.path}"
                )
            time.sleep(0.1)

        return cls(args)


def main(
    command: str,
    model_checkpoint_dir: str,
):
    if command == "consolidate":
        print(
            f"Consolidating {model_checkpoint_dir}. Output will be in the {CONSOLIDATE_FOLDER} folder."
        )
        consolidate_checkpoints(fsspec.filesystem("file"), model_checkpoint_dir)
    else:
        raise ValueError("Invalid command")


if __name__ == "__main__":
    typer.run(main)
