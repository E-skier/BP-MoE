from pathlib import Path
import tempfile
import json

import torch

from bytelatent.patchmoe_warmstart import (
    PatchMoEWarmStartSpec,
    convert_dense_state_dict_to_patchmoe,
    load_patchmoe_warmstart_manifest,
    write_patchmoe_warmstart_dcp,
)
from bytelatent.base_transformer import SparseMoEFeedForward
from bytelatent.base_transformer import FeedForward


def _dense_global_state_dict(dim=8, hidden_dim=16):
    prefix = "global_transformer.layers.0.feed_forward"
    return {
        f"{prefix}.w1.weight": torch.randn(hidden_dim, dim),
        f"{prefix}.w2.weight": torch.randn(dim, hidden_dim),
        f"{prefix}.w3.weight": torch.randn(hidden_dim, dim),
    }


def test_paired_partition_manifest_records_bcfp_invariants():
    dense = _dense_global_state_dict()
    spec = PatchMoEWarmStartSpec(
        num_experts=8,
        top_k=2,
        patch_features=("entropy",),
        expert_ffn_dim_multiplier=0.5,
        expert_init_mode="paired_partition",
        patch_feature_bias=True,
        routing_granularity="pair",
    )
    converted, report = convert_dense_state_dict_to_patchmoe(dense, spec)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "warmstart"
        write_patchmoe_warmstart_dcp(
            output_dir,
            converted,
            source_checkpoint=Path(tmpdir) / "dense.pth",
            spec=spec,
            report=report,
        )
        manifest = load_patchmoe_warmstart_manifest(output_dir)

    assert manifest["num_experts"] == 8
    assert manifest["pair_size"] == 2
    assert manifest["num_pairs"] == 4
    assert manifest["top_k"] == 2
    assert manifest["expert_ffn_dim_multiplier"] == 0.5
    assert manifest["expert_init"] == "paired_partition_replicated_per_pair"
    assert manifest["pair_member_output_scale"] == 2.0
    assert manifest["routing_granularity"] == "pair"


def test_pair_routing_warmstart_exports_pair_sized_router_tensors():
    dense = _dense_global_state_dict(hidden_dim=24)
    spec = PatchMoEWarmStartSpec(
        num_experts=8,
        top_k=2,
        patch_features=("entropy",),
        expert_ffn_dim_multiplier=0.5,
        expert_init_mode="paired_partition",
        patch_feature_bias=True,
        routing_granularity="pair",
    )

    converted, _ = convert_dense_state_dict_to_patchmoe(dense, spec)

    prefix = "global_transformer.layers.0.feed_forward"
    assert converted[f"{prefix}.router.weight"].shape == (4, 8)
    assert converted[f"{prefix}.patch_feature_router.weight"].shape == (4, 1)
    assert converted[f"{prefix}.patch_feature_router.bias"].shape == (4,)

    moe = SparseMoEFeedForward(
        dim=8,
        hidden_dim=24,
        multiple_of=1,
        ffn_dim_multiplier=0.5,
        num_experts=8,
        top_k=2,
        routing_granularity="pair",
        pair_size=2,
        router_patch_feature_bias=True,
        router_use_patch_entropy=True,
        balance_cost="byte",
    )
    moe.router.weight.data.copy_(converted[f"{prefix}.router.weight"])
    moe.patch_feature_router.weight.data.copy_(
        converted[f"{prefix}.patch_feature_router.weight"]
    )
    moe.patch_feature_router.bias.data.copy_(
        converted[f"{prefix}.patch_feature_router.bias"]
    )


def test_pair_routing_warmstart_exports_bcfp_router_state_buffers():
    dense = _dense_global_state_dict(hidden_dim=24)
    calibration = {
        "version": 1,
        "num_pairs": 4,
        "num_experts": 8,
        "pair_size": 2,
        "entropy_centers": [0.25, 1.0, 2.0, 3.0],
        "entropy_widths": [0.5, 0.5, 0.5, 0.5],
        "static_pair_bias": [0.1, 0.0, -0.1, 0.0],
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        calibration_path = Path(tmpdir) / "router_calibration.json"
        calibration_path.write_text(json.dumps(calibration))
        spec = PatchMoEWarmStartSpec(
            num_experts=8,
            top_k=2,
            patch_features=("entropy",),
            expert_ffn_dim_multiplier=0.5,
            expert_init_mode="paired_partition",
            patch_feature_bias=True,
            routing_granularity="pair",
            entropy_prior_mode="gaussian_pairs",
            entropy_prior_calibration_path=str(calibration_path),
            pair_bias_mode="ema_byte_floor",
            hidden_residual_ramp_start_step=100,
            hidden_residual_ramp_end_step=300,
        )
        converted, _ = convert_dense_state_dict_to_patchmoe(dense, spec)

    prefix = "global_transformer.layers.0.feed_forward"
    torch.testing.assert_close(
        converted[f"{prefix}.entropy_prior_centers"],
        torch.tensor([0.25, 1.0, 2.0, 3.0]),
    )
    torch.testing.assert_close(
        converted[f"{prefix}.entropy_prior_widths"],
        torch.tensor([0.5, 0.5, 0.5, 0.5]),
    )
    torch.testing.assert_close(
        converted[f"{prefix}.entropy_prior_static_bias"],
        torch.tensor([0.1, 0.0, -0.1, 0.0]),
    )
    assert torch.equal(converted[f"{prefix}.dynamic_pair_bias"], torch.zeros(4))
    assert torch.equal(converted[f"{prefix}.pair_load_ema"], torch.zeros(4))
    assert converted[f"{prefix}.pair_bias_step"].shape == ()
    assert converted[f"{prefix}.router_step"].shape == ()


def test_pair_routing_paired_partition_is_function_preserving_at_t0():
    torch.manual_seed(0)
    dense_ffn = FeedForward(
        dim=8,
        hidden_dim=24,
        multiple_of=1,
        ffn_dim_multiplier=1.0,
    )
    prefix = "global_transformer.layers.0.feed_forward"
    dense = {
        f"{prefix}.{name}": value.detach().clone()
        for name, value in dense_ffn.state_dict().items()
    }
    spec = PatchMoEWarmStartSpec(
        num_experts=8,
        top_k=2,
        patch_features=("entropy",),
        expert_ffn_dim_multiplier=0.5,
        expert_init_mode="paired_partition",
        patch_feature_bias=True,
        routing_granularity="pair",
    )
    converted, _ = convert_dense_state_dict_to_patchmoe(dense, spec)

    moe = SparseMoEFeedForward(
        dim=8,
        hidden_dim=24,
        multiple_of=1,
        ffn_dim_multiplier=0.5,
        num_experts=8,
        top_k=2,
        routing_granularity="pair",
        pair_size=2,
        router_patch_feature_bias=True,
        router_use_patch_entropy=True,
        router_use_hidden_state=False,
        balance_cost="byte",
    )
    moe.router.weight.data.copy_(converted[f"{prefix}.router.weight"])
    moe.patch_feature_router.weight.data.zero_()
    moe.patch_feature_router.bias.data.zero_()
    for expert_idx, expert in enumerate(moe.experts):
        expert.load_state_dict(
            {
                name: converted[f"{prefix}.experts.{expert_idx}.{name}"]
                for name in dense_ffn.state_dict()
            }
        )

    x = torch.randn(2, 3, 8)
    patch_lengths = torch.tensor([[1, 4, 2], [3, 1, 5]])
    patch_entropies = torch.rand(2, 3)

    expected = dense_ffn(x)
    actual = moe(x, patch_lengths=patch_lengths, patch_entropies=patch_entropies)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
