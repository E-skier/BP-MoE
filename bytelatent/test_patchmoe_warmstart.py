from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp

from bytelatent.base_transformer import FeedForward, SparseMoEFeedForward
from bytelatent.patchmoe_warmstart import (
    MANIFEST_NAME,
    PatchMoEWarmStartSpec,
    convert_dense_state_dict_to_patchmoe,
    load_patchmoe_warmstart_manifest,
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


def test_dense_to_patchmoe_router_initialization_is_deterministic():
    dense = dense_global_state_dict()
    spec = PatchMoEWarmStartSpec(router_seed=123)

    first, _ = convert_dense_state_dict_to_patchmoe(dense, spec)
    second, _ = convert_dense_state_dict_to_patchmoe(dense, spec)

    key = "global_transformer.layers.0.feed_forward.router.weight"
    assert torch.equal(first[key], second[key])


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
