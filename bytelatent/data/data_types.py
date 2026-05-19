# Copyright (c) Meta Platforms, Inc. and affiliates.
import json
from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np
from pydantic import BaseModel, ConfigDict


class BltExample(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sample_id: str
    text: str
    tokens: list[int] | None
    entropies: list[float] | None
    patch_lengths: list[int] | None
    mask: list[bool] | None


class MultiChoiceState(BaseModel):
    model_config = ConfigDict(extra="forbid")
    root_dir: str
    sources: dict[str, float]
    source_to_state: dict[str, Any]
    rng_state: dict[str, Any]


class PrefetchState(BaseModel):
    model_config = ConfigDict(extra="forbid")
    seq_idx: int
    rng_state: dict[str, Any]
    prefetch_size: int
    batch_size: int


class BltPackTokensState(BaseModel):
    model_config = ConfigDict(extra="forbid")
    start_token: int
    output_seq_len: int
    n_views: int = 2


class BltSequence(BaseModel):
    tokens: list[int]
    mask: list[bool]
    patch_lengths: list[int] | None
    entropies: list[float] | None = None


@dataclass
class Batch:
    x: np.ndarray
    y: np.ndarray
    mask: np.ndarray | None = None
    patch_lengths: np.ndarray | None = None
    patch_entropies: np.ndarray | None = None
    ngram_ids: np.ndarray | None = None
    is_final: bool = False

    def to_python_dict(self) -> dict:
        x = self.x.tolist()
        y = self.y.tolist()
        if self.mask is None:
            mask = None
        else:
            mask = self.mask.tolist()
        if self.patch_lengths is None:
            patch_lengths = None
        else:
            patch_lengths = self.patch_lengths.tolist()
        if self.patch_entropies is None:
            patch_entropies = None
        else:
            patch_entropies = self.patch_entropies.tolist()
        if self.ngram_ids is None:
            ngram_ids = None
        else:
            ngram_ids = self.ngram_ids.tolist()
        return {
            "x": x,
            "y": y,
            "mask": mask,
            "patch_lengths": patch_lengths,
            "patch_entropies": patch_entropies,
            "ngram_ids": ngram_ids,
            "is_final": self.is_final,
        }

    @classmethod
    def from_python_dict(cls, data: dict) -> "Batch":
        x = np.array(data["x"])
        y = np.array(data["y"])
        if data["mask"] is None:
            mask = None
        else:
            mask = np.array(data["mask"])
        if data["patch_lengths"] is None:
            patch_lengths = None
        else:
            patch_lengths = np.array(data["patch_lengths"])
        if data.get("patch_entropies") is None:
            patch_entropies = None
        else:
            patch_entropies = np.array(data["patch_entropies"])
        if data["ngram_ids"] is None:
            ngram_ids = None
        else:
            ngram_ids = np.array(data["ngram_ids"])
        return Batch(
            x=x,
            y=y,
            mask=mask,
            patch_lengths=patch_lengths,
            patch_entropies=patch_entropies,
            ngram_ids=ngram_ids,
            is_final=data["is_final"],
        )
