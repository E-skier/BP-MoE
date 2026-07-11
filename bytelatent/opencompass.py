"""OpenCompass adapter for BLT/PatchMoE checkpoints.

This module is intentionally optional: importing it without OpenCompass installed
still works, but instantiating the model inside OpenCompass requires the
``opencompass`` package to provide its normal ``BaseModel`` interface.
"""

from __future__ import annotations

import math
import os
from typing import Any

try:  # OpenCompass 0.4/0.5 style import.
    from opencompass.models.base import BaseModel as _OpenCompassBaseModel
except Exception:  # pragma: no cover - exercised only when OpenCompass is absent.
    try:
        from opencompass.models.base_model import BaseModel as _OpenCompassBaseModel
    except Exception:

        class _OpenCompassBaseModel:  # type: ignore[no-redef]
            pass


try:  # Optional registry path used by OpenCompass' config builder.
    from opencompass.registry import MODELS as _OpenCompassModels
except Exception:  # pragma: no cover - exercised only when OpenCompass is absent.
    _OpenCompassModels = None


def _register_opencompass_model(cls):
    if _OpenCompassModels is None:
        return cls
    try:
        return _OpenCompassModels.register_module()(cls)
    except Exception:
        return cls


@_register_opencompass_model
class ByteLatentOpenCompassModel(_OpenCompassBaseModel):
    """OpenCompass ``BaseModel`` wrapper for BLT/PatchMoE checkpoints.

    The wrapper supports both generative tasks through ``generate`` and PPL-style
    multiple-choice tasks through ``get_ppl``. It reuses the repository's native
    checkpoint loader and packed generator so OpenCompass benchmark scores are
    produced by the same inference path as ``bytelatent.eval`` task evaluation.
    """

    def __init__(
        self,
        ckpt_dir: str,
        abbr: str | None = None,
        max_seq_len: int = 2048,
        max_out_len: int = 128,
        batch_size: int = 1,
        run_cfg: dict[str, Any] | None = None,
        generator_kwargs: dict[str, Any] | None = None,
        generation_kwargs: dict[str, Any] | None = None,
        meta_template: dict[str, Any] | None = None,
        sync_rank: bool = False,
        consolidate_if_needed: bool = True,
        consolidate_folder: str = "consolidated",
        tokenizer_only: bool = False,
        device: str = "cuda",
        dtype: str = "bf16",
        entropy_ckpt_dir: str | None = None,
        entropy_state_dict_path: str | None = None,
        entropy_attn_impl: str = "sdpa",
        suppress_attn_error: bool = True,
        allow_missing_flex_attention: bool = True,
        **_: Any,
    ) -> None:
        try:
            super().__init__(
                path=ckpt_dir,
                max_seq_len=max_seq_len,
                tokenizer_only=tokenizer_only,
                meta_template=meta_template,
                generation_kwargs=generation_kwargs or {},
                sync_rank=sync_rank,
            )
        except TypeError:
            # Fallback base class used when OpenCompass is not installed.
            pass

        self.path = ckpt_dir
        self.ckpt_dir = ckpt_dir
        self.abbr = abbr or os.path.basename(os.path.normpath(ckpt_dir))
        self.max_seq_len = max_seq_len
        self.max_out_len = max_out_len
        self.batch_size = batch_size
        self.run_cfg = run_cfg or dict(num_gpus=1, num_procs=1)
        self.generator_kwargs = dict(generator_kwargs or {})
        self.generation_kwargs = dict(generation_kwargs or {})
        self.sync_rank = sync_rank
        self.consolidate_if_needed = consolidate_if_needed
        self.consolidate_folder = consolidate_folder
        self.device = device
        self.dtype = dtype
        self.entropy_ckpt_dir = entropy_ckpt_dir
        self.entropy_state_dict_path = entropy_state_dict_path
        self.entropy_attn_impl = entropy_attn_impl
        self._model = None
        self._tokenizer = None
        self._train_cfg = None
        self._generator = None

        if suppress_attn_error:
            os.environ.setdefault("BLT_SUPPRESS_ATTN_ERROR", "1")
        if allow_missing_flex_attention:
            os.environ.setdefault("BLT_ALLOW_MISSING_FLEX_ATTENTION", "1")
        os.environ.setdefault("ENTROPY_MODEL_ATTN_IMPL", entropy_attn_impl)

        if tokenizer_only:
            self._load_tokenizer_only()

    def _has_consolidated_files(self, path: str) -> bool:
        from bytelatent.data.file_util import get_fs

        fs = get_fs(path)
        return fs.exists(os.path.join(path, "params.json")) and bool(
            fs.glob(os.path.join(path, "*.pth"))
        )

    def _resolve_consolidated_path(self) -> str:
        if self._has_consolidated_files(self.ckpt_dir):
            return self.ckpt_dir

        consolidated = os.path.join(self.ckpt_dir, self.consolidate_folder)
        if self._has_consolidated_files(consolidated):
            return consolidated

        if not self.consolidate_if_needed:
            raise FileNotFoundError(
                "OpenCompass requires a consolidated checkpoint with params.json "
                f"and *.pth files. Missing under {self.ckpt_dir!r} and "
                f"{consolidated!r}."
            )

        from bytelatent.checkpoint import consolidate_checkpoints
        from bytelatent.data.file_util import get_fs

        return consolidate_checkpoints(get_fs(self.ckpt_dir), self.ckpt_dir)

    def _load_tokenizer_only(self) -> None:
        if self._tokenizer is not None:
            return

        from bytelatent.args import TrainArgs
        from bytelatent.data.file_util import get_fs

        consolidated_path = self._resolve_consolidated_path()
        params_path = os.path.join(consolidated_path, "params.json")
        train_cfg = TrainArgs.model_validate_json(
            get_fs(params_path).read_text(params_path)
        )
        self._train_cfg = train_cfg
        self._tokenizer = train_cfg.data.tokenizer_args.build()

    def _ensure_runtime_patcher(self, model, train_cfg) -> None:
        if getattr(model, "patcher", None) is not None:
            return
        data_cfg = getattr(train_cfg, "data", None)
        patcher_args = getattr(data_cfg, "patcher_args", None)
        if patcher_args is None:
            return

        patcher_args = patcher_args.model_copy(deep=True)
        patcher_args.realtime_patching = False
        patcher_args.device = self.device
        patcher_args.patching_device = self.device
        model.patcher = patcher_args.build()

    def _ensure_entropy_patcher(self, model) -> None:
        patcher = getattr(model, "patcher", None)
        if patcher is None:
            return
        patching_mode = getattr(getattr(patcher, "patching_mode", None), "value", None)
        if patching_mode is None:
            patching_mode = getattr(patcher, "patching_mode", None)
        if patching_mode != "entropy":
            return
        if getattr(patcher, "entropy_model", None) is not None:
            return

        ckpt_dir = (
            self.entropy_ckpt_dir
            or os.environ.get("PATCHMOE_ENTROPY_CKPT_DIR")
            or os.environ.get("ENTROPY_MODEL_CHECKPOINT_DIR")
            or os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "hf-weights",
                "entropy_model",
            )
        )
        state_path = (
            self.entropy_state_dict_path
            or os.environ.get("PATCHMOE_ENTROPY_STATE_DICT_PATH")
            or os.environ.get("ENTROPY_MODEL_STATE_DICT_PATH")
        )
        if state_path is None:
            candidates = [
                os.path.join(ckpt_dir, "consolidated", "consolidated.pth"),
                os.path.join(ckpt_dir, "consolidated.pth"),
            ]
            state_path = next((item for item in candidates if os.path.exists(item)), "")
        if not ckpt_dir or not os.path.exists(ckpt_dir) or not state_path:
            raise FileNotFoundError(
                "Entropy patching needs an entropy model checkpoint. Set "
                "PATCHMOE_ENTROPY_CKPT_DIR and optionally "
                "PATCHMOE_ENTROPY_STATE_DICT_PATH."
            )
        if not os.path.exists(state_path):
            raise FileNotFoundError(f"Entropy model state dict not found: {state_path}")

        from bytelatent.data.patcher import to_device
        from bytelatent.entropy_model import load_entropy_model

        entropy_model, _ = load_entropy_model(ckpt_dir, state_path)
        patching_device = getattr(
            getattr(patcher, "patcher_args", None), "patching_device", self.device
        )
        entropy_model, _ = to_device(entropy_model, patching_device)
        patcher.entropy_model = entropy_model
        if getattr(patcher, "patcher_args", None) is not None:
            patcher.patcher_args.entropy_model_checkpoint_dir = ckpt_dir
            patcher.patcher_args.realtime_patching = True

    def _load_model(self) -> None:
        if self._model is not None and self._generator is not None:
            return

        from bytelatent.args import PackedCausalTransformerGeneratorArgs
        from bytelatent.eval import configure_eval_expert_parallel
        from bytelatent.generate import (
            PackedCausalTransformerGenerator,
            load_consolidated_model_and_tokenizer,
        )

        consolidated_path = self._resolve_consolidated_path()
        model, tokenizer, train_cfg = load_consolidated_model_and_tokenizer(
            consolidated_path
        )
        configure_eval_expert_parallel(model, train_cfg)
        self._ensure_runtime_patcher(model, train_cfg)
        self._ensure_entropy_patcher(model)

        generator_args = {
            "temperature": 0.0,
            "top_p": None,
            "top_k": None,
            "max_gen_len": self.max_out_len,
            "max_tokens": self.max_seq_len + self.max_out_len,
            "max_prompt_len": self.max_seq_len,
            "until": [],
            "compile_prefilling": False,
            "reduce_generation_overhead": False,
            "show_progress": False,
            "dtype": self.dtype,
            "device": self.device,
        }
        generator_args.update(self.generator_kwargs)

        self._model = model
        self._tokenizer = tokenizer
        self._train_cfg = train_cfg
        self._generator = PackedCausalTransformerGenerator(
            PackedCausalTransformerGeneratorArgs(**generator_args), model, tokenizer
        )

    def get_token_len(self, prompt: str) -> int:
        """Return BLT tokenizer length for OpenCompass prompt truncation."""
        if self._tokenizer is None:
            self._load_tokenizer_only()
        return len(self._tokenizer.encode(str(prompt), add_bos=True, add_eos=False))

    def generate(
        self,
        inputs: list[str],
        max_out_len: int | None = None,
        min_out_len: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        stop: list[str] | str | None = None,
        stopping_criteria: list[str] | str | None = None,
        **_: Any,
    ) -> list[str]:
        """Generate completions for OpenCompass generative datasets."""
        self._load_model()
        assert self._generator is not None

        if max_out_len is not None:
            self._generator.max_gen_len = int(max_out_len)
        if min_out_len is not None:
            self._generator.min_gen_len = int(min_out_len)
        if temperature is not None:
            self._generator.temperature = temperature
        if top_p is not None:
            self._generator.top_p = top_p
        if top_k is not None:
            self._generator.top_k = top_k
        stop_words: list[str] = []
        for item in (stop, stopping_criteria):
            if item is None:
                continue
            stop_words.extend([item] if isinstance(item, str) else list(item))
        if stop_words:
            self._generator.until = stop_words
            self._generator.max_until_size = max(
                [len(item) for item in self._generator.until], default=1
            )

        generations, _, _ = self._generator.generate([str(item) for item in inputs])
        return generations

    def _score_tokenwise_nll(self, inputs: list[str]):
        """Score next-token NLL with the native BLT/LM forward path."""
        import torch
        from torch.nn import functional as F

        self._load_model()
        if self._tokenizer is None:
            self._load_tokenizer_only()
        assert self._model is not None
        assert self._tokenizer is not None

        try:
            device = next(self._model.parameters()).device
        except StopIteration:
            device = torch.device(self.device)

        max_seq_len = self.max_seq_len
        if hasattr(self._model, "get_output_seq_len"):
            max_seq_len = min(max_seq_len, int(self._model.get_output_seq_len()))

        encoded_inputs = []
        removed_prefix_lengths = []
        for text in inputs:
            token_ids = self._tokenizer.encode(str(text), add_bos=True, add_eos=False)
            removed = max(len(token_ids) - (max_seq_len + 1), 0)
            if removed > 0:
                token_ids = token_ids[removed:]
            encoded_inputs.append(token_ids)
            removed_prefix_lengths.append(removed)

        nll_by_input = [torch.empty(0, dtype=torch.float32) for _ in encoded_inputs]
        pad_id = getattr(self._tokenizer, "pad_id", 0)
        batch_size = max(int(self.batch_size or 1), 1)
        with torch.inference_mode():
            for batch_start in range(0, len(encoded_inputs), batch_size):
                batch = encoded_inputs[batch_start : batch_start + batch_size]
                valid_rows = [idx for idx, token_ids in enumerate(batch) if len(token_ids) >= 2]
                if not valid_rows:
                    continue

                max_input_len = max(len(batch[idx]) - 1 for idx in valid_rows)
                x = torch.full(
                    (len(valid_rows), max_input_len),
                    int(pad_id),
                    dtype=torch.long,
                    device=device,
                )
                y = torch.full_like(x, int(pad_id))
                valid_mask = torch.zeros_like(x, dtype=torch.bool)
                for row, batch_idx in enumerate(valid_rows):
                    token_ids = batch[batch_idx]
                    input_len = len(token_ids) - 1
                    x[row, :input_len] = torch.tensor(
                        token_ids[:-1], dtype=torch.long, device=device
                    )
                    y[row, :input_len] = torch.tensor(
                        token_ids[1:], dtype=torch.long, device=device
                    )
                    valid_mask[row, :input_len] = True

                logits = self._model(x)
                if isinstance(logits, tuple):
                    logits = logits[0]
                if hasattr(logits, "logits"):
                    logits = logits.logits
                logits = logits[:, :max_input_len, :]
                loss = F.cross_entropy(
                    logits.float().reshape(-1, logits.shape[-1]),
                    y.reshape(-1),
                    reduction="none",
                ).view(len(valid_rows), max_input_len)
                for row, batch_idx in enumerate(valid_rows):
                    output_idx = batch_start + batch_idx
                    nll_by_input[output_idx] = loss[row][valid_mask[row]].detach().cpu()
        return nll_by_input, removed_prefix_lengths

    def get_ppl(
        self, inputs: list[str], mask_length: list[int] | None = None
    ):
        """Return per-input mean NLL for OpenCompass PPL datasets.

        OpenCompass' HuggingFace backend names this method ``get_ppl``, but its
        implementation returns mean cross-entropy loss. Keep the same convention
        so multiple-choice PPL evaluators compare lower-is-better scores.

        ``mask_length`` is interpreted as a token count prefix to ignore. If a
        dataset provides prompt+choice inputs with prompt token lengths, this
        scores only the answer suffix.
        """
        import numpy as np

        token_nlls, removed_prefix_lengths = self._score_tokenwise_nll(
            [str(item) for item in inputs]
        )

        results: list[float] = []
        for idx, token_nll in enumerate(token_nlls):
            start = 0
            if mask_length is not None:
                # token_nll[j] scores original token j+1, so a masked prefix of
                # M tokens starts scoring at index M-1. Account for any left
                # truncation applied before model forward.
                start = max(
                    int(mask_length[idx]) - 1 - int(removed_prefix_lengths[idx]), 0
                )
            selected = token_nll[start:]
            if selected.numel() == 0:
                results.append(math.inf)
                continue
            results.append(selected.float().mean().item())
        return np.asarray(results, dtype=float)

    def get_ppl_tokenwise(
        self,
        inputs: list[str],
        label: list[list[list[int]]] | list[int] | None = None,
        mask_length: list[int] | None = None,
        **_: Any,
    ) -> list[list[float]] | tuple[list[float], list[int]]:
        """Return token-wise NLL scores for OpenCompass diagnostics.

        Newer OpenCompass builds call this as
        ``get_ppl_tokenwise(inputs, label, mask_length)`` for advanced
        token-wise PPL tasks. If ``label`` is provided, return the
        HuggingFace-compatible ``(loss_sum, token_count)`` tuple. Otherwise
        return per-token NLL lists after applying the optional prefix mask.
        """
        if label is not None and mask_length is None and self._looks_like_mask(label):
            mask_length = label  # type: ignore[assignment]
            label = None

        token_nlls, removed_prefix_lengths = self._score_tokenwise_nll(
            [str(item) for item in inputs]
        )

        results: list[list[float]] = []
        for idx, token_nll in enumerate(token_nlls):
            start = 0
            if mask_length is not None:
                start = max(
                    int(mask_length[idx]) - 1 - int(removed_prefix_lengths[idx]), 0
                )
            results.append(token_nll[start:].float().tolist())

        if label is None:
            return results

        loss_sums: list[float] = []
        token_counts: list[int] = []
        for text, nll_values, spans in zip(inputs, results, label):
            span_indices = self._loss_indices_for_char_spans(str(text), spans)
            selected_values = [
                nll_values[index] for index in span_indices if index < len(nll_values)
            ]
            loss_sums.append(float(sum(selected_values)))
            token_counts.append(len(selected_values))
        return loss_sums, token_counts

    def encode(self, prompt: str):
        """Encode text for OpenCompass input synchronization helpers."""
        import torch

        if self._tokenizer is None:
            self._load_tokenizer_only()
        return torch.tensor(
            self._tokenizer.encode(str(prompt), add_bos=True, add_eos=False),
            dtype=torch.long,
        )

    def decode(self, tokens) -> str:
        """Decode token ids for OpenCompass input synchronization helpers."""
        if self._tokenizer is None:
            self._load_tokenizer_only()
        if hasattr(tokens, "detach"):
            tokens = tokens.detach().cpu().tolist()
        return self._tokenizer.decode(tokens)

    def _looks_like_mask(self, label: object) -> bool:
        return isinstance(label, list) and all(isinstance(item, int) for item in label)

    def _loss_indices_for_char_spans(
        self, text: str, spans: list[list[int]] | None
    ) -> list[int]:
        if self._tokenizer is None:
            self._load_tokenizer_only()
        if not spans:
            return []

        indices: set[int] = set()
        for span in spans:
            if len(span) < 2:
                continue
            left = max(int(span[0]), 0)
            right = min(int(span[1]), len(text))
            if right <= left:
                continue
            prefix_tokens = self._tokenizer.encode(
                text[:left], add_bos=True, add_eos=False
            )
            prefix_span_tokens = self._tokenizer.encode(
                text[:right], add_bos=True, add_eos=False
            )
            start = max(len(prefix_tokens) - 1, 0)
            end = max(len(prefix_span_tokens) - 1, start)
            indices.update(range(start, end))
        return sorted(indices)
