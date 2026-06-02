from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_state_dict,
    set_state_dict,
)

from bytelatent.base_transformer import SparseMoEFeedForward


def _build_moe(expert_parallel_size: int) -> SparseMoEFeedForward:
    return SparseMoEFeedForward(
        dim=8,
        hidden_dim=16,
        multiple_of=1,
        ffn_dim_multiplier=1.0,
        num_experts=4,
        top_k=2,
        expert_parallel_size=expert_parallel_size,
    )


def _run_ep_equivalence(rank: int, world_size: int, init_file: str, ckpt_dir: str):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        torch.manual_seed(1234)
        reference = _build_moe(expert_parallel_size=1)
        expert_parallel = _build_moe(expert_parallel_size=world_size)
        expert_parallel.load_state_dict(reference.state_dict())
        expert_parallel.configure_expert_parallel(dist.group.WORLD)

        local_expert_ids = tuple(
            expert_id
            for expert_id, expert in enumerate(expert_parallel.experts)
            if expert is not None
        )
        assert local_expert_ids == expert_parallel.local_expert_ids
        assert len(local_expert_ids) == 2

        torch.manual_seed(5678 + rank)
        reference_x = torch.randn(2, 3, 8, requires_grad=True)
        expert_parallel_x = reference_x.detach().clone().requires_grad_(True)
        reference_out = reference(reference_x)
        expert_parallel_out = expert_parallel(expert_parallel_x)
        torch.testing.assert_close(expert_parallel_out, reference_out)

        reference_out.square().sum().backward()
        expert_parallel_out.square().sum().backward()
        torch.testing.assert_close(expert_parallel_x.grad, reference_x.grad)
        torch.testing.assert_close(
            expert_parallel.router.weight.grad,
            reference.router.weight.grad,
        )

        for expert_id in range(reference.num_experts):
            ep_expert = expert_parallel.experts[expert_id]
            for param_id, reference_param in enumerate(
                reference.experts[expert_id].parameters()
            ):
                reference_grad = reference_param.grad.detach().clone()
                dist.all_reduce(reference_grad)
                if ep_expert is not None:
                    ep_param = tuple(ep_expert.parameters())[param_id]
                    torch.testing.assert_close(ep_param.grad, reference_grad)

        assert expert_parallel.last_metrics["ep_size"] == float(world_size)
        assert expert_parallel.last_metrics["ep_dispatched_assignments"] == 12.0

        optimizer = torch.optim.AdamW(expert_parallel.parameters(), lr=0.01)
        optimizer.step()
        model_state_dict, optim_state_dict = get_state_dict(expert_parallel, optimizer)
        state_dict = {"model": model_state_dict, "optim": optim_state_dict}
        dcp.save(state_dict, checkpoint_id=ckpt_dir)

        for expert_id in local_expert_ids:
            for param in expert_parallel.experts[expert_id].parameters():
                param.detach().zero_()
        dcp.load(state_dict, checkpoint_id=ckpt_dir)
        set_state_dict(
            expert_parallel,
            optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optim"],
        )
        for expert_id in local_expert_ids:
            for param in expert_parallel.experts[expert_id].parameters():
                assert torch.count_nonzero(param).item() > 0
    finally:
        dist.destroy_process_group()


def test_expert_parallel_matches_local_moe_and_saves_all_experts(tmp_path: Path):
    init_file = tmp_path / "dist_init"
    ckpt_dir = tmp_path / "checkpoint"
    mp.start_processes(
        _run_ep_equivalence,
        args=(2, str(init_file), str(ckpt_dir)),
        nprocs=2,
        start_method="spawn",
        join=True,
    )

    metadata = dcp.FileSystemReader(ckpt_dir).read_metadata()
    checkpoint_keys = set(metadata.state_dict_metadata)
    for expert_id in range(4):
        assert f"model.experts.{expert_id}.w1.weight" in checkpoint_keys
        assert f"model.experts.{expert_id}.w2.weight" in checkpoint_keys
        assert f"model.experts.{expert_id}.w3.weight" in checkpoint_keys


def test_expert_parallel_requires_divisible_expert_count():
    try:
        SparseMoEFeedForward(
            dim=8,
            hidden_dim=16,
            multiple_of=1,
            ffn_dim_multiplier=1.0,
            num_experts=4,
            top_k=2,
            expert_parallel_size=3,
        )
    except ValueError as exc:
        assert "must be divisible" in str(exc)
    else:
        raise AssertionError("Expected invalid expert-parallel size to raise")
