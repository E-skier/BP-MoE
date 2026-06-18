from pathlib import Path

import pytest
import torch
import torch.distributed.checkpoint as dcp

from bytelatent.base_transformer import FeedForward, SparseMoEFeedForward
from bytelatent.patchmoe_warmstart import (
    MANIFEST_NAME,
    PatchMoEWarmStartSpec,
    convert_dense_state_dict_to_patchmoe,
    load_patchmoe_warmstart_manifest,
    verify_patchmoe_warmstart_dcp,
    write_patchmoe_warmstart_dcp,
)


def dense_global_state_dict(dim=8, hidden_dim=16, n_layers=3):
    state_dict = {"outside.weight": torch.randn(2, 2)}
    for layer_idx in range(n_layers):
        prefix = f"global_transformer.layers.{layer_idx}.feed_forward"
        state_dict[f"{prefix}.w1.weight"] = torch.randn(hidden_dim, dim)
        state_dict[f"{prefix}.w2.weight"] = torch.randn(dim, hidden_dim)
        state_dict[f"{prefix}.w3.weight"] = torch.randn(hidden_dim, dim)
    return state_dict


def test_dense_to_patchmoe_maps_frequency_layers_and_preserves_trunk():
    dense = dense_global_state_dict()
    spec = PatchMoEWarmStartSpec(
        num_experts=4,
        top_k=2,
        layer_frequency=2,
        patch_features=("entropy",),
    )

    converted, report = convert_dense_state_dict_to_patchmoe(dense, spec)

    assert report.selected_layers == (0, 2)
    assert report.global_layers == (0, 1, 2)
    assert report.patch_feature_count == 1
    assert converted["outside.weight"] is dense["outside.weight"]
    assert "global_transformer.layers.0.feed_forward.w1.weight" not in converted
    assert "global_transformer.layers.1.feed_forward.w1.weight" in converted
    for layer_idx in (0, 2):
        prefix = f"global_transformer.layers.{layer_idx}.feed_forward"
        assert converted[f"{prefix}.router.weight"].shape == (4, 8)
        assert converted[f"{prefix}.patch_feature_router.weight"].shape == (4, 1)
        assert (
            converted[f"{prefix}.patch_length_router.weight"]
            is converted[f"{prefix}.patch_feature_router.weight"]
        )
        for expert_idx in range(4):
            for weight_name in ("w1", "w2", "w3"):
                assert (
                    converted[f"{prefix}.experts.{expert_idx}.{weight_name}.weight"]
                    is dense[f"{prefix}.{weight_name}.weight"]
                )


def test_dense_to_patchmoe_initial_ffn_is_functionally_equivalent():
    torch.manual_seed(0)
    dense = FeedForward(
        dim=8,
        hidden_dim=24,
        multiple_of=1,
        ffn_dim_multiplier=1.0,
    )
    moe = SparseMoEFeedForward(
        dim=8,
        hidden_dim=24,
        multiple_of=1,
        ffn_dim_multiplier=1.0,
        num_experts=4,
        top_k=2,
        router_patch_feature_bias=True,
        router_use_patch_entropy=True,
        balance_cost="byte",
    )
    for expert in moe.experts:
        expert.load_state_dict(dense.state_dict())

    x = torch.randn(2, 3, 8)
    patch_lengths = torch.tensor([[1, 4, 2], [3, 1, 5]])
    patch_entropies = torch.rand(2, 3)

    expected = dense(x)
    actual = moe(
        x,
        patch_lengths=patch_lengths,
        patch_entropies=patch_entropies,
    )

    torch.testing.assert_close(actual, expected)


def test_paired_partition_is_functionally_equivalent_per_sample():
    torch.manual_seed(0)
    dense = FeedForward(
        dim=8,
        hidden_dim=24,
        multiple_of=1,
        ffn_dim_multiplier=1.0,
    )
    prefix = "global_transformer.layers.0.feed_forward"
    dense_state = {
        f"{prefix}.{name}": value for name, value in dense.state_dict().items()
    }
    spec = PatchMoEWarmStartSpec(
        num_experts=8,
        top_k=2,
        patch_features=("entropy",),
        expert_ffn_dim_multiplier=0.5,
        expert_init_mode="paired_partition",
        patch_feature_bias=True,
    )
    converted, _ = convert_dense_state_dict_to_patchmoe(dense_state, spec)

    moe = SparseMoEFeedForward(
        dim=8,
        hidden_dim=24,
        multiple_of=1,
        ffn_dim_multiplier=0.5,
        num_experts=8,
        top_k=2,
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
    for expert_idx, expert in enumerate(moe.experts):
        expert.load_state_dict(
            {
                name: converted[f"{prefix}.experts.{expert_idx}.{name}"]
                for name in dense.state_dict()
            }
        )

    router_weight = moe.router.weight.detach()
    feature_weight = moe.patch_feature_router.weight.detach()
    feature_bias = moe.patch_feature_router.bias.detach()
    for pair_start in range(0, 8, 2):
        torch.testing.assert_close(
            router_weight[pair_start], router_weight[pair_start + 1]
        )
        torch.testing.assert_close(
            feature_weight[pair_start], feature_weight[pair_start + 1]
        )
        torch.testing.assert_close(
            feature_bias[pair_start], feature_bias[pair_start + 1]
        )

    x = torch.randn(2, 3, 8)
    patch_lengths = torch.tensor([[1, 4, 2], [3, 1, 5]])
    patch_entropies = torch.rand(2, 3) * 4
    flat_x = x.reshape(-1, x.shape[-1])
    normalized_entropy = (
        patch_entropies.reshape(-1) / patch_entropies.mean().clamp_min(1.0)
    ).unsqueeze(-1)
    logits = moe.router(flat_x) + moe.patch_feature_router(normalized_entropy)
    top_indices = logits.topk(k=2, dim=-1).indices
    assert torch.equal(top_indices[:, 0] // 2, top_indices[:, 1] // 2)

    expected = dense(x)
    actual = moe(
        x,
        patch_lengths=patch_lengths,
        patch_entropies=patch_entropies,
    )
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_entropy_band_patch_feature_init_orders_expert_pairs():
    dense = dense_global_state_dict(n_layers=1)
    spec = PatchMoEWarmStartSpec(
        num_experts=8,
        top_k=2,
        patch_features=("entropy",),
        expert_ffn_dim_multiplier=0.5,
        expert_init_mode="paired_partition",
        patch_feature_bias=True,
        patch_feature_init="entropy_bands",
        entropy_band_logit_scale=2.0,
    )

    converted, _ = convert_dense_state_dict_to_patchmoe(dense, spec)

    prefix = "global_transformer.layers.0.feed_forward"
    weight = converted[f"{prefix}.patch_feature_router.weight"]
    bias = converted[f"{prefix}.patch_feature_router.bias"]
    entropies = torch.tensor([[0.25], [1.25], [2.25], [3.75]])
    logits = entropies @ weight.float().T + bias.float()
    top_indices = logits.topk(k=2, dim=-1).indices
    selected_pairs = top_indices.div(2, rounding_mode="floor")

    assert torch.equal(selected_pairs[:, 0], selected_pairs[:, 1])
    assert torch.equal(
        selected_pairs[:, 0],
        torch.tensor([0, 1, 2, 3]),
    )


def test_entropy_band_patch_feature_init_rejects_incompatible_specs():
    with pytest.raises(ValueError, match="requires patch_feature_bias"):
        PatchMoEWarmStartSpec(
            patch_features=("entropy",),
            expert_ffn_dim_multiplier=0.5,
            expert_init_mode="paired_partition",
            patch_feature_init="entropy_bands",
        )
    with pytest.raises(ValueError, match="requires entropy"):
        PatchMoEWarmStartSpec(
            patch_features=("length",),
            patch_feature_bias=True,
            patch_feature_init="entropy_bands",
        )


def test_paired_partition_rejects_incompatible_specs():
    with pytest.raises(ValueError, match="requires top_k=2"):
        PatchMoEWarmStartSpec(
            top_k=1,
            expert_ffn_dim_multiplier=0.5,
            expert_init_mode="paired_partition",
        )
    with pytest.raises(ValueError, match="requires expert_ffn_dim_multiplier=0.5"):
        PatchMoEWarmStartSpec(expert_init_mode="paired_partition")


def test_dense_to_patchmoe_router_initialization_is_deterministic():
    dense = dense_global_state_dict()
    spec = PatchMoEWarmStartSpec(router_seed=123)

    first, _ = convert_dense_state_dict_to_patchmoe(dense, spec)
    second, _ = convert_dense_state_dict_to_patchmoe(dense, spec)

    key = "global_transformer.layers.0.feed_forward.router.weight"
    assert torch.equal(first[key], second[key])


def test_dense_to_patchmoe_default_maps_every_global_ffn_layer():
    dense = dense_global_state_dict(n_layers=4)

    converted, report = convert_dense_state_dict_to_patchmoe(
        dense, PatchMoEWarmStartSpec()
    )

    assert report.selected_layers == (0, 1, 2, 3)
    for layer_idx in report.selected_layers:
        prefix = f"global_transformer.layers.{layer_idx}.feed_forward"
        assert f"{prefix}.experts.0.w1.weight" in converted
        assert f"{prefix}.w1.weight" not in converted


def test_write_patchmoe_warmstart_dcp_and_manifest(tmp_path: Path):
    dense = dense_global_state_dict(n_layers=1)
    spec = PatchMoEWarmStartSpec(num_experts=2, top_k=1, layer_frequency=1)
    converted, report = convert_dense_state_dict_to_patchmoe(dense, spec)
    output_dir = tmp_path / "warmstart"

    write_patchmoe_warmstart_dcp(
        output_dir,
        converted,
        source_checkpoint=tmp_path / "dense.pth",
        spec=spec,
        report=report,
    )

    assert (output_dir / ".metadata").is_file()
    assert (output_dir / MANIFEST_NAME).is_file()
    manifest = load_patchmoe_warmstart_manifest(output_dir)
    assert manifest["report"]["selected_layers"] == [0]

    loaded = {key: torch.empty_like(value) for key, value in converted.items()}
    dcp.load({"model": loaded}, checkpoint_id=output_dir)
    for key, value in converted.items():
        assert torch.equal(loaded[key], value)


def test_verify_patchmoe_warmstart_dcp_checks_keys_and_sample_values(tmp_path: Path):
    dense = dense_global_state_dict(n_layers=2)
    spec = PatchMoEWarmStartSpec(num_experts=2, top_k=1, layer_frequency=2)
    converted, report = convert_dense_state_dict_to_patchmoe(dense, spec)
    output_dir = tmp_path / "warmstart"
    write_patchmoe_warmstart_dcp(
        output_dir,
        converted,
        source_checkpoint=tmp_path / "dense.pth",
        spec=spec,
        report=report,
    )

    verification = verify_patchmoe_warmstart_dcp(output_dir, converted)
    assert verification.dcp_key_count == len(converted)
    assert "outside.weight" in verification.sample_keys
    assert any("experts.0.w1.weight" in key for key in verification.sample_keys)
    assert any(
        "layers.1.feed_forward.w1.weight" in key for key in verification.sample_keys
    )

    wrong = dict(converted)
    wrong["outside.weight"] = wrong["outside.weight"] + 1
    with pytest.raises(ValueError, match="DCP tensor mismatch: outside.weight"):
        verify_patchmoe_warmstart_dcp(
            output_dir, wrong, sample_keys=("outside.weight",)
        )
