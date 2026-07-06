# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import gc
import json
import logging
import math
import os
import sys
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict, dataclass
from timeit import default_timer as timer
from typing import Any, TypeVar

import numpy as np
import pyarrow
import torch
import torch.distributed
import torch.nn.functional
import torch.nn.functional as F
import wandb

try:
    import xformers.profiler as xformers_profiler
except ImportError:
    xformers_profiler = None
from torch.distributed._tensor import DTensor
from torch.distributed.checkpoint.stateful import Stateful
from torch.optim import lr_scheduler

from bytelatent.args import TrainArgs
from bytelatent.base_transformer import (
    get_moe_balance_loss_weight_for_step,
    get_moe_aux_loss,
    get_moe_metrics,
    get_moe_router_z_loss,
    set_moe_router_step,
)
from bytelatent.checkpoint import CheckpointManager, load_from_checkpoint
from bytelatent.config_parser import parse_args_to_pydantic_model
from bytelatent.data.file_util import get_fs
from bytelatent.data.iterators.abstract_iterator import get_state_and_refresh
from bytelatent.data.iterators.multiprocess_iterator import (
    MultiprocessIterator,
    MultiprocessIteratorState,
    PersistType,
)
from bytelatent.data.iterators.packing_iterator import PackingIteratorState
from bytelatent.distributed import (
    check_model_value_range,
    clean_env,
    dist_max,
    dist_mean,
    dist_sum,
    get_device_mesh,
    get_is_master,
    get_world_size,
    init_signal_handler,
    parallelize_model,
    requeue_slurm_job,
    setup_env,
    setup_torch_distributed,
    to_py_num,
)
from bytelatent.eval import EVAL_FOLDER_NAME, launch_eval
from bytelatent.logger import init_logger
from bytelatent.metrics import GPUMemoryMonitor, MetricLogger, get_num_params
from bytelatent.model.blt import ByteLatentTransformer
from bytelatent.norms import fixed_clip_grad_norm_
from bytelatent.optim import build_optimizer
from bytelatent.probe import AutoProbeD
from bytelatent.profiling import maybe_run_profiler
from bytelatent.stool import StoolArgs, launch_job
from bytelatent.transformer import (
    LMTransformer,
    build_fsdp_grouping_plan,
    get_no_recompute_ops,
    get_num_flop_per_token,
    tp_parallelize,
)

logger = logging.getLogger()

T = TypeVar("T")


def flatten_dict(d, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def get_iterator_state_name(iterator_state):
    if isinstance(iterator_state, MultiprocessIteratorState):
        return "multiprocess"
    elif isinstance(iterator_state, PackingIteratorState):
        return "packing"
    else:
        raise ValueError(f"Unsupported iterator to get name from: {iterator_state}")


# TODO: Make this pydantic based instead of data class based
# TODO: Generalize this to any iterator state
@dataclass
class TrainState(Stateful):
    step: int  # Nb of steps taken by the optimizer
    acc_step: int  # Nb of accumulation steps done since last optimizer step
    scheduler: lr_scheduler.LambdaLR
    data_loader_state: MultiprocessIteratorState | PackingIteratorState
    scale: float = 1.0
    data_loader_class: str | None = None
    cumulative_active_flops: float = 0.0
    cumulative_valid_bytes: int = 0
    cumulative_valid_patches: int = 0
    saved_budget_boundaries: list[str] | None = None

    def state_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "acc_step": self.acc_step,
            "data_loader_state": (
                self.data_loader_state.model_dump()
                if self.data_loader_state is not None
                else None
            ),
            "data_loader_class": (
                get_iterator_state_name(self.data_loader_state)
                if self.data_loader_state is not None
                else None
            ),
            "scheduler": self.scheduler.state_dict(),
            "cumulative_active_flops": self.cumulative_active_flops,
            "cumulative_valid_bytes": self.cumulative_valid_bytes,
            "cumulative_valid_patches": self.cumulative_valid_patches,
            "saved_budget_boundaries": self.saved_budget_boundaries or [],
        }

    def load_state_dict(self, state_dict):
        self.step = state_dict["step"]
        self.acc_step = state_dict["acc_step"]
        self.data_loader_class = state_dict["data_loader_class"]
        if self.data_loader_class is None:
            self.data_loader_state = None
        elif self.data_loader_class == "multiprocess":
            self.data_loader_state = MultiprocessIteratorState(
                **state_dict["data_loader_state"]
            )
        elif self.data_loader_class == "packing":
            self.data_loader_state = PackingIteratorState(
                **state_dict["data_loader_state"]
            )
        else:
            raise ValueError(f"invalid data loader class: {self.data_loader_class}")
        self.scheduler.load_state_dict(state_dict["scheduler"])
        self.cumulative_active_flops = float(
            state_dict.get("cumulative_active_flops", 0.0)
        )
        self.cumulative_valid_bytes = int(state_dict.get("cumulative_valid_bytes", 0))
        self.cumulative_valid_patches = int(
            state_dict.get("cumulative_valid_patches", 0)
        )
        self.saved_budget_boundaries = list(
            state_dict.get("saved_budget_boundaries", [])
        )


def load_compute_budgets(path: str | None) -> dict[str, float]:
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    budgets: dict[str, float] = {}
    for key in ("C_low", "C_mid", "C_high"):
        if key in raw:
            budgets[key] = float(raw[key])
    return budgets


def crossed_compute_budgets(
    *,
    previous_flops: float,
    current_flops: float,
    budgets: dict[str, float],
    saved_boundaries: set[str],
) -> list[str]:
    crossed = []
    for key in ("C_low", "C_mid", "C_high"):
        budget = budgets.get(key)
        if budget is None or key in saved_boundaries:
            continue
        if previous_flops < budget <= current_flops:
            crossed.append(key)
    return crossed


def update_active_compute_counters(
    train_state: TrainState,
    *,
    step_active_flops: float,
    step_valid_bytes: int,
    step_valid_patches: int,
    budgets: dict[str, float],
) -> list[str]:
    previous_flops = train_state.cumulative_active_flops
    train_state.cumulative_active_flops += float(step_active_flops)
    train_state.cumulative_valid_bytes += int(step_valid_bytes)
    train_state.cumulative_valid_patches += int(step_valid_patches)
    saved_boundaries = set(train_state.saved_budget_boundaries or [])
    crossed = crossed_compute_budgets(
        previous_flops=previous_flops,
        current_flops=train_state.cumulative_active_flops,
        budgets=budgets,
        saved_boundaries=saved_boundaries,
    )
    if crossed:
        train_state.saved_budget_boundaries = list(
            (train_state.saved_budget_boundaries or []) + crossed
        )
    return crossed


def reduce_step_valid_counts(
    step_valid_bytes: int,
    step_valid_patches: int,
) -> tuple[int, int]:
    global_step_valid_bytes = int(
        to_py_num(dist_sum(step_valid_bytes, reduce_dtype=torch.bfloat16))
    )
    global_step_valid_patches = int(
        to_py_num(dist_sum(step_valid_patches, reduce_dtype=torch.bfloat16))
    )
    return global_step_valid_bytes, global_step_valid_patches


def validate_train_args(args: TrainArgs, output_size: int):
    assert args.model is not None or args.entropy_model is not None
    if args.model is not None:
        logger.info(f"Setting model output size to {args.model.vocab_size}")
        args.model.vocab_size = output_size
        assert (
            args.model.max_encoder_seq_length == args.data.max_encoder_seq_length
        ), "max_encoder_seq_length for model and data should match"

    if args.entropy_model is not None:
        logger.info(f"Setting model output size to {args.entropy_model.vocab_size}")
        args.entropy_model.vocab_size = output_size

    assert args.dump_dir, "Dump dir not set"

    if args.checkpoint.path is None:
        logger.info(f"Setting checkpoint path to {args.checkpoint.path}")
        args.checkpoint.path = os.path.join(args.dump_dir, "checkpoints")

    if args.data.root_dir is not None:
        data_fs = get_fs(args.data.root_dir, s3_profile=args.data.s3_profile)
        for source in args.data.sources:
            data_path = os.path.join(args.data.root_dir, source)
            assert data_fs.exists(data_path), f"{data_path} doesn't exist"

    args.distributed.configure_world()

    if args.model is not None:
        args.model.max_seqlen = args.data.seq_len
    if args.entropy_model is not None:
        args.entropy_model.max_seqlen = args.data.seq_len

    if args.distributed.tp_size == 1:
        logger.warning(
            "Tensor parallelism has not been tested for a while, use at your own risk"
        )

    assert (
        args.probe_freq != args.profiling.mem_steps
    ), "Don't profile during probe step"
    assert (
        args.probe_freq != args.profiling.profile_steps
    ), "Don't profile during probe step"
    if args.logging.wandb is not None:
        args.logging.wandb.name = args.name

    if args.probe_freq is not None:
        assert (
            args.distributed.tp_size == 1
        ), "Probing not supported with tensor parallelism"
        assert (
            args.distributed.selective_activation_checkpointing is False
        ), "Probing not supported with selective activation checkpointing"


preemption_flag = dict(flag=False)


def set_preemption_flag(signum, frame):
    logger.warning("Signal handler called with signal " + str(signum))
    logger.warning("Preemption ! checkpointing asap and exiting.")
    preemption_flag["flag"] = True


def every_n_steps(train_state, freq: int, acc_step=None, acc_freq=None):
    if freq < 0:
        return False
    test = train_state.step % freq == 0
    if acc_step is not None:
        test = test and (train_state.acc_step == acc_step)
    elif acc_freq is not None:
        test = test and ((train_state.acc_step % acc_freq) == 0)
    return test


def trace_train_phase(step: int, acc_step: int, phase: str):
    if os.environ.get("BLT_STAGE1_TRACE", "").lower() not in {"1", "true", "yes"}:
        return
    if (
        os.environ.get("BLT_STAGE1_TRACE_SYNC", "").lower() in {"1", "true", "yes"}
        and torch.cuda.is_available()
    ):
        torch.cuda.synchronize()
    rank = (
        torch.distributed.get_rank()
        if torch.distributed.is_available() and torch.distributed.is_initialized()
        else 0
    )
    message = f"stage1_trace rank={rank} step={step} acc={acc_step} phase={phase}"
    trace_dir = os.environ.get("BLT_STAGE1_TRACE_DIR")
    if trace_dir:
        try:
            os.makedirs(trace_dir, exist_ok=True)
            trace_path = os.path.join(trace_dir, f"stage1_trace_rank{rank}.log")
            with open(trace_path, "a", encoding="utf-8") as trace_file:
                trace_file.write(message + "\n")
        except Exception as exc:
            logger.warning("failed to write stage1 trace file: %s", exc)
    logger.warning(message)


def compute_loss(p, y, mask, scale):
    tok_loss = scale * F.cross_entropy(
        p.flatten(0, 1), y.flatten(0, 1), reduction="none"
    )
    if mask is None:
        loss = tok_loss.mean()
    else:
        mask = mask.flatten(0, 1)
        tok_loss = tok_loss * mask
        loss = tok_loss.sum() / (mask.sum() + 1e-6)
    return loss, tok_loss


def train(args: TrainArgs):
    with ExitStack() as context_stack:
        pyarrow.set_io_thread_count(4)
        pyarrow.set_cpu_count(4)
        tokenizer = args.data.tokenizer_args.build()
        validate_train_args(
            args,
            tokenizer.get_vocab_size(),
        )
        dump_fs = get_fs(args.dump_dir, s3_profile=args.checkpoint.s3_profile)
        if get_is_master():
            dump_fs.mkdirs(args.dump_dir, exist_ok=True)
            config_yaml_str = args.dump_to_yaml_str()
            logging.info("TrainArgs: \n%s", config_yaml_str)
            dump_fs.write_text(
                os.path.join(args.dump_dir, "config.yaml"), config_yaml_str
            )
        init_logger(os.path.join(args.dump_dir, "train.log"), fs=dump_fs)
        init_signal_handler(set_preemption_flag)  # For handling preemption signals.
        setup_env(args.env)
        setup_torch_distributed(args.distributed)
        world_mesh = get_device_mesh(args.distributed)
        logger.info(f"Starting job: {args.name}")

        # build dataloader
        # need dp world size and rank
        dp_mesh = world_mesh["dp_replicate"]
        dp_degree = dp_mesh.size()
        dp_rank = dp_mesh.get_local_rank()
        if args.distributed.dp_shard > 1:
            dp_rank = dp_rank * dp_degree + world_mesh["dp_shard"].get_local_rank()
            dp_degree *= world_mesh["dp_shard"].size()

        logger.info(f"Running on dp rank : {dp_rank}")
        logger.info(f"Running on dp size : {dp_degree}")

        torch.manual_seed(args.seed)
        logger.info("Building model")

        # Initializing Model in meta device allows us to initialize models much bigger than 1 gpu's memory
        with torch.device("meta"):
            if args.train_entropy_model:
                assert args.entropy_model is not None
                model = LMTransformer(args.entropy_model)
                model_args = args.entropy_model
            else:
                assert args.model is not None
                model = ByteLatentTransformer(args.model)
                model_args = args.model
        logger.info("Model is built !")

        model_param_count = get_num_params(model)

        model = parallelize_model(
            model,
            world_mesh,
            model_args,
            args.distributed,
            fsdp_grouping_plan=build_fsdp_grouping_plan(model_args),
            tp_parallelize=tp_parallelize,
            no_recompute_ops=get_no_recompute_ops(),
        )

        # Once we shard the model on different gpus we can actually initialize the model
        # First we create empty tensors of the correct shapes
        model = model.to_empty(device="cuda")
        # Then we init the model. Please make sure this function initializes *ALL* parameters
        # and buffers, otherwise you will have random values in the unitialized tensors
        # which will silently fail (give nan gradients for example)

        if args.checkpoint.init_ckpt_path:
            logger.info(f"Loading initial model from {args.checkpoint.init_ckpt_path}")
            ckpt_fs = get_fs(
                args.checkpoint.init_ckpt_path, s3_profile=args.checkpoint.s3_profile
            )
            load_from_checkpoint(
                ckpt_fs, args.checkpoint.init_ckpt_path, model, model_key="model"
            )  # Put model_key="" if its directly the model checkpoint
            # RoPE buffers are non-persistent and therefore absent from DCP.
            model.reset_rope_embeddings()
        else:
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                torch.manual_seed(model_args.seed)
                model.init_weights()
        check_model_value_range(model, range=10.0, std=1.0)

        # log model size

        logger.info(model)
        logger.info(f"Model size: {model_param_count:,} total parameters")

        gpu_memory_monitor = GPUMemoryMonitor("cuda")
        logger.info(
            f"GPU capacity: {gpu_memory_monitor.device_name} ({gpu_memory_monitor.device_index}) "
            f"with {gpu_memory_monitor.device_capacity_gib:.2f}GiB memory"
        )
        logger.info(f"GPU memory usage: {gpu_memory_monitor}")

        # build optimizer after apply parallelisms to the model
        optimizer, scheduler = build_optimizer(model, args.optim, args.steps)
        data_loader = args.data.build_from_rank(dp_rank, dp_degree)
        data_loader_state = data_loader.get_state()

        train_state = TrainState(
            step=0,
            acc_step=0,
            data_loader_state=data_loader_state,
            scheduler=scheduler,
            scale=1.0,
        )

        checkpoint = CheckpointManager.instantiate_and_make_dir(args.checkpoint)
        checkpoint.load(model, optimizer, train_state, world_mesh)
        compute_budgets = load_compute_budgets(args.compute_budget_json)
        # Either load from latest checkpoint or start from scratch
        if args.probe_freq is not None:
            # TODO: Convert this to fsspec compatible
            if get_is_master():
                os.makedirs(os.path.join(args.dump_dir, "probe"), exist_ok=True)
            torch.distributed.barrier()
            probe = AutoProbeD(
                model,
                (
                    os.path.join(args.dump_dir, "probe", f"probe.{dp_rank}.jsonl")
                    if (dp_rank % 128 == 0)
                    else None
                ),
            )
            probe_mod = model._orig_mod if args.distributed.compile else model

        gc.disable()

        # train loop
        model.train()
        metric_logger = context_stack.enter_context(
            MetricLogger(os.path.join(args.dump_dir, "metrics.jsonl"), args, fs=dump_fs)
        )
        data_loader = train_state.data_loader_state.build()
        batch_iterator = data_loader.create_iter()

        torch_profiler = context_stack.enter_context(
            maybe_run_profiler(args.dump_dir, model, args.profiling)
        )

        nwords_since_last_log = 0
        time_last_log = timer()
        gc.collect()
        saved = False
        step_losses: list[float] = []
        step_tok_losses: list[float] = []
        n_bytes: int = 0
        patch_length_sum = 0.0
        patch_length_count = 0
        patch_length_max = 0.0
        patch_entropy_sum = 0.0
        patch_entropy_count = 0
        patch_entropy_max = 0.0
        patches_per_seq_sum = 0.0
        patch_seq_count = 0
        accum_valid_bytes = 0
        accum_valid_patches = 0
        while train_state.step < args.steps and (
            args.max_steps is None or train_state.step < args.max_steps
        ):
            # We constrain train_state.acc_step to be in range 0 to args.grad_acc_steps - 1
            train_state.acc_step += 1
            train_state.acc_step = train_state.acc_step % args.grad_acc_steps
            trace_step = train_state.step + 1
            trace_train_phase(trace_step, train_state.acc_step, "loop_start")

            # get batch
            curr_lr = float(optimizer.param_groups[0]["lr"])
            data_load_start = timer()
            trace_train_phase(trace_step, train_state.acc_step, "before_next_batch")
            batch = next(batch_iterator)
            trace_train_phase(trace_step, train_state.acc_step, "after_next_batch")
            batch_x = torch.from_numpy(
                batch.x,
            ).cuda()
            batch_y = torch.from_numpy(batch.y).cuda()
            if batch.patch_lengths is None:
                batch_patch_lengths = None
                batch_valid_patches = 0
            else:
                batch_patch_lengths = torch.from_numpy(batch.patch_lengths).cuda()
                active_patch_lengths = batch_patch_lengths[batch_patch_lengths > 0]
                batch_valid_patches = int(active_patch_lengths.numel())
                if active_patch_lengths.numel() > 0:
                    patch_length_sum += active_patch_lengths.sum().item()
                    patch_length_count += active_patch_lengths.numel()
                    patch_length_max = max(
                        patch_length_max, active_patch_lengths.max().item()
                    )
                patches_per_seq = (batch_patch_lengths > 0).sum(dim=-1)
                patches_per_seq_sum += patches_per_seq.sum().item()
                patch_seq_count += patches_per_seq.numel()
            if batch.patch_entropies is None:
                batch_patch_entropies = None
            else:
                batch_patch_entropies = torch.from_numpy(batch.patch_entropies).cuda()
                active_patch_entropies = batch_patch_entropies
                if batch_patch_lengths is not None:
                    active_patch_entropies = active_patch_entropies[
                        batch_patch_lengths > 0
                    ]
                if active_patch_entropies.numel() > 0:
                    patch_entropy_sum += active_patch_entropies.sum().item()
                    patch_entropy_count += active_patch_entropies.numel()
                    patch_entropy_max = max(
                        patch_entropy_max, active_patch_entropies.max().item()
                    )
            mask = None if batch.mask is None else torch.from_numpy(batch.mask).cuda()

            if args.data.tokenizer_args.name in ["bytes", "blt"]:
                batch_valid_bytes = (
                    batch_y.numel() if mask is None else int(mask.sum().item())
                )
                n_bytes += batch_valid_bytes
            elif args.data.tokenizer_args.name in ["sp", "tiktoken"]:
                batch_valid_bytes = 0
                for example in batch.y:
                    target_tokens = tokenizer.decode(example.tolist(), cut_at_eos=False)
                    batch_valid_bytes += (
                        len(bytes(target_tokens, encoding="utf-8", errors="ignore"))
                        + sum(example == tokenizer.eos_id)
                        + sum(example == tokenizer.bos_id)
                    )
                n_bytes += batch_valid_bytes
            else:
                raise ValueError(
                    f"Unexpected tokenizer to count n_bytes for: {args.data.tokenizer_args.name}"
                )
            accum_valid_bytes += int(batch_valid_bytes)
            accum_valid_patches += int(batch_valid_patches)

            if (
                not args.train_entropy_model
                and args.model.encoder_enable_byte_ngrams
                and batch.ngram_ids is None
            ):
                raise ValueError(
                    "Cannot enable byte ngrams and have batch.ngram_ids be None"
                )
            ngram_ids = (
                None
                if batch.ngram_ids is None
                else torch.from_numpy(batch.ngram_ids).cuda()
            )
            trace_train_phase(trace_step, train_state.acc_step, "after_batch_to_cuda")

            if every_n_steps(train_state, args.gc_collect_freq, acc_step=0):
                logger.info("garbage collection")
                # we do garbage collection manually otherwise different processes
                # run the GC at different times so they slow down the whole pipeline
                gc.collect()

            data_load_time = round(timer() - data_load_start, 4)
            nwords_since_last_log += batch_x.numel()

            bsz, seqlen = batch_y.shape

            # forward
            start_timer = torch.cuda.Event(enable_timing=True)
            end_timer = torch.cuda.Event(enable_timing=True)
            start_timer.record()

            # This is an automatic probe that will compute statistics
            # of all linears' inputs, weights and outputs
            # along with attention logits and entropy
            # both in forward and backward pass
            tok_loss = None
            if (args.probe_freq is not None) and every_n_steps(
                train_state, args.probe_freq, acc_step=1 % args.grad_acc_steps
            ):
                # Here we do a fake forward and backward pass on a smaller
                # batch size to avoid OOM
                # This assumes the model has no stateful layers (batch norm..)
                assert (
                    next(probe_mod.parameters()).grad is None
                ), "Can't probe model if grads are not reset"

                with probe:
                    probe.metadata = {
                        "it": train_state.step,
                        "global_step": train_state.step,
                        "loop": "lingua",
                    }
                    # Non compiled model uses roughly 2x memory in our exps
                    # So we divide bsz by 2 or seqlen by 2
                    probe_bsz = max(1, bsz // 2)
                    probe_seq = seqlen if (bsz // 2 >= 1) else (seqlen // 2)
                    probe_loss = probe_mod(
                        batch_x[:probe_bsz, :probe_seq],
                        batch_y[:probe_bsz, :probe_seq],
                    )
                    probe_loss.backward()
                    # We zero grads to cancel this fake step
                    optimizer.zero_grad()

                assert (
                    next(probe_mod.parameters()).grad is None
                ), "Probe model shouldn't have grads at this point"

            moe_aux_loss_log = None
            moe_router_z_loss_log = None
            trace_train_phase(trace_step, train_state.acc_step, "before_forward")
            if not args.train_entropy_model:
                set_moe_router_step(model, train_state.step)
            if args.train_entropy_model:
                pred = model(batch_x)
            else:
                pred = model(
                    batch_x,
                    patch_lengths=batch_patch_lengths,
                    patch_entropies=batch_patch_entropies,
                    ngram_ids=ngram_ids,
                )
            trace_train_phase(trace_step, train_state.acc_step, "after_forward")

            loss, tok_loss = compute_loss(pred, batch_y, mask, train_state.scale)
            if not args.train_entropy_model:
                moe_aux_loss = get_moe_aux_loss(model)
                moe_balance_loss_weight = get_moe_balance_loss_weight_for_step(
                    args.model, train_state.step
                )
                if moe_aux_loss is not None and moe_balance_loss_weight > 0:
                    moe_aux_loss_log = moe_aux_loss.detach()
                    loss = loss + moe_balance_loss_weight * moe_aux_loss
                elif moe_aux_loss is not None:
                    moe_aux_loss_log = moe_aux_loss.detach()

                moe_router_z_loss = get_moe_router_z_loss(model)
                if moe_router_z_loss is not None:
                    moe_router_z_loss_log = moe_router_z_loss.detach()
                    if args.model.moe_router_z_loss_weight > 0:
                        loss = (
                            loss
                            + args.model.moe_router_z_loss_weight * moe_router_z_loss
                        )

            # We scale loss with grad_acc_steps so the gradient is the same
            # regardless of grad_acc_steps
            loss = loss / args.grad_acc_steps

            # backward on scaled loss to create scaled gradients
            trace_train_phase(trace_step, train_state.acc_step, "before_backward")
            loss.backward()
            trace_train_phase(trace_step, train_state.acc_step, "after_backward")
            # For logging we undo that scaling
            loss = loss.detach() * args.grad_acc_steps

            # Undo loss scaling so downstream down't need to worry about it
            step_losses.append((loss / train_state.scale).item())
            step_tok_losses.append(tok_loss / train_state.scale)

            trace_train_phase(trace_step, train_state.acc_step, "before_grad_clip")
            if args.optim.clip <= 0:
                grad_norm = 0.0
            else:
                world_size = get_world_size()
                if 1 < world_size <= 8:
                    # For some reason, there are errors in reduces due to
                    # not working for non-bf16 numbers. This function is a patched
                    # version that converts gradients to bf16 before computing norms.
                    # The error only happens in distributed training on one node,
                    # hence the guard
                    grad_norm = fixed_clip_grad_norm_(
                        model.parameters(), max_norm=args.optim.clip, foreach=True
                    )
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=args.optim.clip, foreach=True
                    )

                grad_norm = (
                    grad_norm.full_tensor()
                    if isinstance(grad_norm, DTensor)
                    else grad_norm
                ).item()
            trace_train_phase(trace_step, train_state.acc_step, "after_grad_clip")

            # optimizer step
            if train_state.acc_step == 0:
                trace_train_phase(
                    trace_step, train_state.acc_step, "before_optimizer_step"
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                train_state.step += 1
                global_step_valid_bytes, global_step_valid_patches = (
                    reduce_step_valid_counts(accum_valid_bytes, accum_valid_patches)
                )
                previous_active_flops = train_state.cumulative_active_flops
                active_flops_per_token = get_num_flop_per_token(
                    model_param_count - model_args.vocab_size * model_args.dim,
                    model_args.n_layers,
                    model_args.dim,
                    args.data.seq_len,
                )
                step_active_flops = active_flops_per_token * global_step_valid_bytes
                train_state.cumulative_active_flops += float(step_active_flops)
                train_state.cumulative_valid_bytes += global_step_valid_bytes
                train_state.cumulative_valid_patches += global_step_valid_patches
                accum_valid_bytes = 0
                accum_valid_patches = 0
                trace_train_phase(
                    trace_step, train_state.acc_step, "after_optimizer_step"
                )

            # updates the scale for next iteration
            # training iteration complete
            end_timer.record()

            trace_train_phase(
                trace_step, train_state.acc_step, "before_cuda_synchronize"
            )
            torch.cuda.synchronize()
            trace_train_phase(
                trace_step, train_state.acc_step, "after_cuda_synchronize"
            )

            curr_iter_time = round(start_timer.elapsed_time(end_timer) * 1e-3, 4)

            # if profiler is active
            if torch_profiler and xformers_profiler is not None:
                xformers_profiler.step()

            # log metrics
            if every_n_steps(
                train_state,
                args.logging.freq,
                acc_step=None if args.logging.acc_freq else 0,
                acc_freq=args.logging.acc_freq,
            ):
                trace_train_phase(
                    trace_step, train_state.acc_step, "before_log_metrics"
                )
                time_delta = timer() - time_last_log
                wps = nwords_since_last_log / (time_delta * args.distributed.tp_size)

                gpu_mem_stats = gpu_memory_monitor.get_peak_stats()

                total_acc_steps = (
                    args.grad_acc_steps * train_state.step + train_state.acc_step
                )
                tokens_per_gpu = (
                    total_acc_steps * args.data.batch_size * args.data.seq_len
                )
                total_tokens = dp_degree * tokens_per_gpu
                # This is an estimate and the correct values may change
                # if you change the architecture
                # Use xformer's analyze profile trace to get actual measurement
                FLOPS = (
                    get_num_flop_per_token(
                        model_param_count - model_args.vocab_size * model_args.dim,
                        model_args.n_layers,
                        model_args.dim,
                        args.data.seq_len,
                    )
                    * wps
                )

                # Below, semantics are:
                # per_gpu: Metrics on a given rank
                # across_gpus: Metrics averaged/summed across all ranks
                # step: Metric at a step
                # interval: Metric averaged/summed across all steps since the last log interval.
                #     Typically, this is 10
                step_loss_per_gpu = loss
                step_loss_across_gpus = dist_mean(step_loss_per_gpu)
                interval_loss_per_gpu = np.mean(step_losses)
                interval_loss_across_gpus = dist_mean(interval_loss_per_gpu)

                stacked_tok_loss = torch.cat(step_tok_losses, dim=0)
                interval_total_tok_loss_per_gpu = stacked_tok_loss.sum()
                interval_total_tok_loss_across_gpus = dist_sum(
                    interval_total_tok_loss_per_gpu, reduce_dtype=torch.bfloat16
                )
                interval_total_n_bytes_per_gpu = n_bytes
                interval_total_n_bytes_across_gpus = dist_sum(
                    n_bytes, reduce_dtype=torch.bfloat16
                )

                interval_bpb_per_gpu = (
                    interval_total_tok_loss_per_gpu
                    / math.log(2)
                    / interval_total_n_bytes_per_gpu
                )
                interval_bpb_across_gpus = (
                    interval_total_tok_loss_across_gpus
                    / math.log(2)
                    / interval_total_n_bytes_across_gpus
                )

                metric_dict = {
                    "global_step": train_state.step,
                    "acc_step": train_state.acc_step,
                    "speed": {
                        "wps": wps,
                        "FLOPS": FLOPS,
                        "curr_iter_time": curr_iter_time,
                        "data_load_time": data_load_time,
                    },
                    "optim": {
                        "grad_norm": grad_norm,
                        "lr": curr_lr,
                        "total_tokens": total_tokens,
                        "cumulative_active_flops": train_state.cumulative_active_flops,
                        "cumulative_valid_bytes": train_state.cumulative_valid_bytes,
                        "cumulative_valid_patches": train_state.cumulative_valid_patches,
                    },
                    "memory": gpu_mem_stats._asdict(),
                    "loss": {
                        "step_per_gpu": to_py_num(step_loss_per_gpu),
                        "step_across_gpu": to_py_num(step_loss_across_gpus),
                        "interval_per_gpu": to_py_num(interval_loss_per_gpu),
                        "interval_across_gpu": to_py_num(interval_loss_across_gpus),
                    },
                    "bpb": {
                        "interval_per_gpu": to_py_num(interval_bpb_per_gpu),
                        "interval_across_gpus": to_py_num(interval_bpb_across_gpus),
                    },
                    "n_bytes": {
                        "interval_per_gpu": to_py_num(interval_total_n_bytes_per_gpu),
                        "interval_across_gpus": to_py_num(
                            interval_total_n_bytes_across_gpus
                        ),
                    },
                }
                if patch_length_count > 0:
                    patch_length_count_across_gpus = dist_sum(
                        patch_length_count, reduce_dtype=torch.bfloat16
                    )
                    patch_seq_count_across_gpus = dist_sum(
                        patch_seq_count, reduce_dtype=torch.bfloat16
                    )
                    patch_length_sum_across_gpus = dist_sum(
                        patch_length_sum, reduce_dtype=torch.bfloat16
                    )
                    patches_per_seq_sum_across_gpus = dist_sum(
                        patches_per_seq_sum, reduce_dtype=torch.bfloat16
                    )
                    patch_length_max_across_gpus = dist_max(
                        patch_length_max, reduce_dtype=torch.bfloat16
                    )
                    metric_dict["patch"] = {
                        "length_mean_per_gpu": patch_length_sum
                        / max(patch_length_count, 1),
                        "length_mean_across_gpus": to_py_num(
                            patch_length_sum_across_gpus
                            / patch_length_count_across_gpus.clamp_min(1)
                        ),
                        "length_max_per_gpu": patch_length_max,
                        "length_max_across_gpus": to_py_num(
                            patch_length_max_across_gpus
                        ),
                        "patches_per_seq_mean_per_gpu": patches_per_seq_sum
                        / max(patch_seq_count, 1),
                        "patches_per_seq_mean_across_gpus": to_py_num(
                            patches_per_seq_sum_across_gpus
                            / patch_seq_count_across_gpus.clamp_min(1)
                        ),
                    }
                    if patch_entropy_count > 0:
                        patch_entropy_count_across_gpus = dist_sum(
                            patch_entropy_count, reduce_dtype=torch.bfloat16
                        )
                        patch_entropy_sum_across_gpus = dist_sum(
                            patch_entropy_sum, reduce_dtype=torch.bfloat16
                        )
                        patch_entropy_max_across_gpus = dist_max(
                            patch_entropy_max, reduce_dtype=torch.bfloat16
                        )
                        metric_dict["patch"].update(
                            {
                                "entropy_mean_per_gpu": patch_entropy_sum
                                / max(patch_entropy_count, 1),
                                "entropy_mean_across_gpus": to_py_num(
                                    patch_entropy_sum_across_gpus
                                    / patch_entropy_count_across_gpus.clamp_min(1)
                                ),
                                "entropy_max_per_gpu": patch_entropy_max,
                                "entropy_max_across_gpus": to_py_num(
                                    patch_entropy_max_across_gpus
                                ),
                            }
                        )
                if not args.train_entropy_model:
                    moe_metrics = get_moe_metrics(model)
                    if moe_aux_loss_log is not None:
                        moe_metrics["aux_loss"] = to_py_num(moe_aux_loss_log)
                        moe_metrics["aux_loss_weight"] = moe_balance_loss_weight
                    if moe_router_z_loss_log is not None:
                        moe_metrics["router_z_loss_aux"] = to_py_num(
                            moe_router_z_loss_log
                        )
                    if len(moe_metrics) > 0:
                        metric_dict["moe"] = moe_metrics

                metrics = flatten_dict(
                    metric_dict,
                    sep="/",
                )

                if get_is_master():
                    metric_logger.log(metrics)

                # Below semantics are:
                # step=Metrics at a step
                # interval=Metrics averaged across the logging interval
                # local=On one rank
                # global=Across all ranks
                logger.info(
                    f"step: {train_state.step}"
                    f"  acc: {train_state.acc_step}"
                    f"  loss_gpu: {round(to_py_num(interval_loss_per_gpu), 4):>7}"
                    f"  loss_avg: {round(to_py_num(interval_loss_across_gpus), 4):>7}"
                    f"  bpb_gpu: {interval_bpb_per_gpu:3f}"
                    f"  bpb_avg: {interval_bpb_across_gpus:3f}"
                    f"  grad: {grad_norm:.2e}"
                    f"  flops: {FLOPS:.2e}"
                    f"  wps: {wps:.2e}"
                    f"  iter: {curr_iter_time:>7}"
                    f"  data: {data_load_time:>5}"
                    f"  lr: {curr_lr:.2e}"
                    f"  n_bytes_gpu: {int(interval_total_n_bytes_per_gpu)}"
                    f"  n_bytes_sum: {int(interval_total_n_bytes_across_gpus)}"
                    f"  mem: {gpu_mem_stats.max_active_pct:.0f}%"
                    f"  pow: {gpu_mem_stats.power_draw/1000} W"
                )

                n_bytes = 0
                patch_length_sum = 0.0
                patch_length_count = 0
                patch_length_max = 0.0
                patch_entropy_sum = 0.0
                patch_entropy_count = 0
                patch_entropy_max = 0.0
                patches_per_seq_sum = 0.0
                patch_seq_count = 0
                step_losses = []
                step_tok_losses = []
                gpu_memory_monitor.reset_peak_stats()
                nwords_since_last_log = 0
                time_last_log = timer()

            crossed_budgets = []
            if args.save_budget_boundaries and compute_budgets:
                crossed_budgets = crossed_compute_budgets(
                    previous_flops=previous_active_flops
                    if "previous_active_flops" in locals()
                    else train_state.cumulative_active_flops,
                    current_flops=train_state.cumulative_active_flops,
                    budgets=compute_budgets,
                    saved_boundaries=set(train_state.saved_budget_boundaries or []),
                )
                if crossed_budgets:
                    saved_budget_boundaries = train_state.saved_budget_boundaries or []
                    saved_budget_boundaries.extend(crossed_budgets)
                    train_state.saved_budget_boundaries = saved_budget_boundaries

            if crossed_budgets or every_n_steps(
                train_state, args.checkpoint.dump.every, acc_step=0
            ) or every_n_steps(train_state, args.checkpoint.eval.every, acc_step=0):
                if (
                    args.data.load_async
                    and args.data.async_persist_type == PersistType.EXACT
                ):
                    train_state.data_loader_state, data_loader, batch_iterator = (
                        get_state_and_refresh(data_loader)
                    )
                else:
                    train_state.data_loader_state = data_loader.get_state()
                saved = checkpoint.save(
                    model,
                    optimizer,
                    train_state,
                    args,
                    device_mesh=world_mesh,
                )

            if args.eval is not None and every_n_steps(
                train_state, args.checkpoint.eval.every, acc_step=0
            ):
                eval_args = args.eval

                eval_args.global_step = train_state.step
                eval_args.ckpt_dir = str(checkpoint.existing_saves[-1])
                eval_args.dump_dir = os.path.join(
                    args.dump_dir,
                    "evals",
                    EVAL_FOLDER_NAME.format(train_state.step),
                )
                eval_args.metric_log_dir = args.dump_dir
                if args.async_eval_gpus is None:
                    launch_eval(eval_args)
                elif get_is_master():
                    if wandb.run is not None and args.logging.wandb is not None:
                        eval_args.wandb = deepcopy(args.logging.wandb)
                    assert args.async_eval_gpus > 0
                    logger.info(f"Launching evals on {args.async_eval_gpus} gpus")
                    with clean_env():
                        launch_job(
                            StoolArgs(
                                asdict(eval_args),
                                script="apps.main.eval",
                                copy_code=False,
                                nodes=args.async_eval_gpus // 8,
                                qos="lowest",
                            )
                        )

            if preemption_flag["flag"]:
                if not saved:
                    if (
                        args.data.load_async
                        and args.data.async_persist_type == PersistType.EXACT
                    ):
                        train_state.data_loader_state, data_loader, batch_iterator = (
                            get_state_and_refresh(data_loader)
                        )
                    else:
                        train_state.data_loader_state = data_loader.get_state()

                    checkpoint.save(
                        model,
                        optimizer,
                        train_state,
                        args,
                        device_mesh=world_mesh,
                    )
                requeue_slurm_job()
                sys.exit(0)

        if not saved:
            if (
                args.data.load_async
                and args.data.async_persist_type == PersistType.EXACT
            ):
                train_state.data_loader_state, data_loader, batch_iterator = (
                    get_state_and_refresh(data_loader)
                )
            else:
                train_state.data_loader_state = data_loader.get_state()
            checkpoint.save(
                model,
                optimizer,
                train_state,
                args,
                device_mesh=world_mesh,
            )
        if isinstance(data_loader, MultiprocessIterator):
            logger.info("Closing MP iterator before exiting")
            data_loader.shutdown()
        gc.collect()


def main():
    """
    The command line interface here uses OmegaConf https://omegaconf.readthedocs.io/en/2.3_branch/usage.html#from-command-line-arguments
    This accepts arguments as a dot list
    So if the dataclass looks like

    @dataclass
    class DummyArgs:
        name: str
        model: LMTransformerArgsgs

    @dataclass
    class LMTransformerArgsgs:
        dim: int

    Then you can pass model.dim=32 to change values in LMTransformerArgsgs
    or just name=tictac for top level attributes.

    The behavior here is as follows:
    1. We instantiate TrainArgs with its default values
    2. We override those default values with the ones in the provided config file
    3. We override the result with the additional arguments provided through command line

    For example, if the config is the following

    model:
        dim: 128
        n_layers: 4

    and you call train.py with train.py model.dim=64

    Then the final TrainArgs will have

    model:
        dim: 64
        n_layers: 4

    Plus all the default values in TrainArgs dataclass.
    """
    train_args = parse_args_to_pydantic_model(TrainArgs)
    if train_args.debug_dynamo:
        import torch._dynamo

        torch._dynamo.config.suppress_errors = True
    train(train_args)


if __name__ == "__main__":
    main()
