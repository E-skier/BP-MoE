import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import torch

from bytelatent.base_transformer import (
    PATCH_BYTE_TYPE_NAMES,
    BaseTransformer,
    BaseTransformerArgs,
    SparseMoEFeedForward,
    get_moe_aux_loss,
    get_moe_metrics,
    get_moe_router_z_loss,
    get_moe_balance_loss_weight_for_step,
)


def test_sparse_moe_feed_forward_forward_backward():
    moe = SparseMoEFeedForward(
        dim=32,
        hidden_dim=128,
        multiple_of=8,
        ffn_dim_multiplier=1.0,
        num_experts=4,
        top_k=2,
    )
    x = torch.randn(2, 5, 32, requires_grad=True)

    out = moe(x)
    aux_loss = get_moe_aux_loss(moe)
    assert aux_loss is not None

    loss = out.square().mean() + 0.01 * aux_loss
    loss.backward()

    assert out.shape == x.shape
    assert x.grad is not None
    assert moe.router.weight.grad is not None
    assert len(get_moe_metrics(moe)) > 0


def test_base_transformer_replaces_frequency_layers_with_moe():
    args = BaseTransformerArgs(
        dim=32,
        n_layers=3,
        n_heads=4,
        max_seqlen=8,
        multiple_of=8,
        moe_num_experts=4,
        moe_top_k=1,
        moe_layer_frequency=2,
    )
    model = BaseTransformer(args)

    assert isinstance(model.layers[0].feed_forward, SparseMoEFeedForward)
    assert not isinstance(model.layers[1].feed_forward, SparseMoEFeedForward)
    assert isinstance(model.layers[2].feed_forward, SparseMoEFeedForward)


def test_moe_balance_linear_decay_schedule():
    args = BaseTransformerArgs(
        dim=32,
        n_layers=1,
        n_heads=4,
        moe_balance_loss_weight=0.01,
        moe_balance_schedule="linear_decay",
        moe_balance_start_step=0,
        moe_balance_peak_step=10,
        moe_balance_decay_start_step=20,
        moe_balance_end_step=30,
        moe_balance_final_weight=0.002,
    )

    assert get_moe_balance_loss_weight_for_step(args, -1) == 0.0
    assert get_moe_balance_loss_weight_for_step(args, 0) == 0.0
    assert get_moe_balance_loss_weight_for_step(args, 5) == 0.005
    assert get_moe_balance_loss_weight_for_step(args, 10) == 0.01
    assert get_moe_balance_loss_weight_for_step(args, 20) == 0.01
    assert get_moe_balance_loss_weight_for_step(args, 25) == 0.006
    assert get_moe_balance_loss_weight_for_step(args, 30) == 0.002
    assert get_moe_balance_loss_weight_for_step(args, 40) == 0.002


def test_sparse_moe_patch_length_routing_and_byte_balance():
    moe = SparseMoEFeedForward(
        dim=32,
        hidden_dim=128,
        multiple_of=8,
        ffn_dim_multiplier=1.0,
        num_experts=4,
        top_k=1,
        router_use_patch_length=True,
        balance_cost="byte",
    )
    x = torch.randn(1, 3, 32, requires_grad=True)
    patch_lengths = torch.tensor([[1, 4, 0]])

    out = moe(x, patch_lengths=patch_lengths)
    aux_loss = get_moe_aux_loss(moe)
    assert aux_loss is not None

    loss = out.square().mean() + 0.01 * aux_loss
    loss.backward()

    assert out.shape == x.shape
    assert x.grad is not None
    assert moe.router.weight.grad is not None
    assert moe.patch_feature_router is not None
    assert moe.patch_feature_router.weight.grad is not None
    assert moe.last_metrics["routed_units"] == 2.0
    assert moe.last_metrics["total_units"] == 3.0
    assert moe.last_metrics["load_weight_max"] == 4.0


def test_sparse_moe_skips_invalid_patches_before_routing_and_experts():
    moe = SparseMoEFeedForward(
        dim=8,
        hidden_dim=16,
        multiple_of=1,
        ffn_dim_multiplier=1.0,
        num_experts=2,
        top_k=1,
        router_use_patch_length=True,
        balance_cost="byte",
    )
    with torch.no_grad():
        moe.router.weight.zero_()
        moe.router.weight[0, 0] = 1.0
        moe.router.weight[1, 0] = -1.0

    x_a = torch.randn(1, 4, 8, requires_grad=True)
    x_b = x_a.detach().clone()
    x_b[:, 2:, :] = torch.randn(1, 2, 8) * 10000.0
    x_b.requires_grad_(True)
    patch_lengths = torch.tensor([[4, 3, 0, 0]])

    out_a = moe(x_a, patch_lengths=patch_lengths)
    metrics_a = dict(moe.last_metrics)
    assignments_a = {
        key: value
        for key, value in metrics_a.items()
        if key.endswith("_unit_assignment_fraction")
    }
    out_b = moe(x_b, patch_lengths=patch_lengths)
    metrics_b = dict(moe.last_metrics)
    assignments_b = {
        key: value
        for key, value in metrics_b.items()
        if key.endswith("_unit_assignment_fraction")
    }

    torch.testing.assert_close(out_a[:, :2, :], out_b[:, :2, :])
    assert torch.equal(out_a[:, 2:, :], torch.zeros_like(out_a[:, 2:, :]))
    assert torch.equal(out_b[:, 2:, :], torch.zeros_like(out_b[:, 2:, :]))
    assert metrics_b["routed_units"] == 2.0
    assert metrics_b["total_units"] == 4.0
    assert metrics_b["invalid_positions_skipped"] == 2.0
    assert assignments_a == assignments_b

    out_b.square().sum().backward()
    assert torch.equal(x_b.grad[:, 2:, :], torch.zeros_like(x_b.grad[:, 2:, :]))


def test_sparse_moe_pair_router_selects_same_pair_with_equal_weights():
    moe = SparseMoEFeedForward(
        dim=8,
        hidden_dim=16,
        multiple_of=1,
        ffn_dim_multiplier=0.5,
        num_experts=4,
        top_k=2,
        routing_granularity="pair",
        pair_size=2,
        router_use_patch_entropy=True,
        router_patch_feature_bias=True,
        router_normalize_patch_entropy=False,
        router_use_hidden_state=False,
    )
    with torch.no_grad():
        moe.router.weight.zero_()
        assert moe.patch_feature_router is not None
        moe.patch_feature_router.weight.copy_(torch.tensor([[-1.0], [1.0]]))
        moe.patch_feature_router.bias.copy_(torch.tensor([1.0, -1.0]))

    x = torch.randn(1, 4, 8)
    patch_lengths = torch.ones(1, 4)
    patch_entropies = torch.tensor([[0.1, 0.2, 3.0, 4.0]])

    out = moe(x, patch_lengths=patch_lengths, patch_entropies=patch_entropies)

    assert out.shape == x.shape
    assert moe.last_metrics["top2_same_pair_fraction"] == 1.0
    assert moe.last_metrics["pair_top_weight_min"] == 0.5
    assert moe.last_metrics["pair_top_weight_max"] == 0.5
    assert moe.last_metrics["pair_0_unit_fraction"] == 0.5
    assert moe.last_metrics["pair_1_unit_fraction"] == 0.5


def test_sparse_moe_gaussian_pair_entropy_prior_loads_calibration_and_routes_pairs():
    calibration = {
        "version": 1,
        "num_pairs": 2,
        "num_experts": 4,
        "pair_size": 2,
        "entropy_centers": [0.25, 3.0],
        "entropy_widths": [0.5, 0.5],
        "static_pair_bias": [0.0, 0.0],
        "target_byte_share": [0.5, 0.5],
        "calibrated_byte_share": [0.5, 0.5],
        "entropy_prior_scale": 1.0,
        "calibration_num_valid_patches": 4,
        "calibration_total_bytes": 10,
        "data_fingerprint": "unit-test",
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        calibration_path = Path(tmpdir) / "router_calibration.json"
        calibration_path.write_text(json.dumps(calibration))
        moe = SparseMoEFeedForward(
            dim=8,
            hidden_dim=16,
            multiple_of=1,
            ffn_dim_multiplier=0.5,
            num_experts=4,
            top_k=2,
            routing_granularity="pair",
            pair_size=2,
            router_use_hidden_state=False,
            entropy_prior_mode="gaussian_pairs",
            entropy_prior_calibration_path=str(calibration_path),
            entropy_prior_scale=1.0,
        )

    x = torch.randn(1, 4, 8)
    patch_lengths = torch.ones(1, 4)
    patch_entropies = torch.tensor([[0.1, 0.2, 3.0, 3.2]])

    out = moe(x, patch_lengths=patch_lengths, patch_entropies=patch_entropies)

    assert out.shape == x.shape
    assert moe.last_metrics["top2_same_pair_fraction"] == 1.0
    assert moe.last_metrics["pair_0_unit_fraction"] == 0.5
    assert moe.last_metrics["pair_1_unit_fraction"] == 0.5
    assert torch.equal(
        moe.entropy_prior_centers.cpu(), torch.tensor([0.25, 3.0])
    )


def test_sparse_moe_gaussian_pair_entropy_prior_round_trips_state_dict():
    calibration = {
        "version": 1,
        "num_pairs": 2,
        "num_experts": 4,
        "pair_size": 2,
        "entropy_centers": [0.25, 3.0],
        "entropy_widths": [0.5, 0.75],
        "static_pair_bias": [0.1, -0.1],
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        calibration_path = Path(tmpdir) / "router_calibration.json"
        calibration_path.write_text(json.dumps(calibration))
        first = SparseMoEFeedForward(
            dim=8,
            hidden_dim=16,
            multiple_of=1,
            ffn_dim_multiplier=0.5,
            num_experts=4,
            top_k=2,
            routing_granularity="pair",
            pair_size=2,
            router_use_hidden_state=False,
            entropy_prior_mode="gaussian_pairs",
            entropy_prior_calibration_path=str(calibration_path),
        )
        second = SparseMoEFeedForward(
            dim=8,
            hidden_dim=16,
            multiple_of=1,
            ffn_dim_multiplier=0.5,
            num_experts=4,
            top_k=2,
            routing_granularity="pair",
            pair_size=2,
            router_use_hidden_state=False,
            entropy_prior_mode="gaussian_pairs",
            entropy_prior_calibration_path=str(calibration_path),
        )

    first.entropy_prior_static_bias.add_(torch.tensor([0.2, -0.2]))
    second.load_state_dict(first.state_dict())

    torch.testing.assert_close(
        second.entropy_prior_centers, first.entropy_prior_centers
    )
    torch.testing.assert_close(second.entropy_prior_widths, first.entropy_prior_widths)
    torch.testing.assert_close(
        second.entropy_prior_static_bias, first.entropy_prior_static_bias
    )


def test_sparse_moe_gaussian_pair_entropy_prior_loads_under_meta_device():
    calibration = {
        "version": 1,
        "num_pairs": 2,
        "num_experts": 4,
        "pair_size": 2,
        "entropy_centers": [0.25, 3.0],
        "entropy_widths": [0.5, 0.75],
        "static_pair_bias": [0.1, -0.1],
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        calibration_path = Path(tmpdir) / "router_calibration.json"
        calibration_path.write_text(json.dumps(calibration))
        with torch.device("meta"):
            moe = SparseMoEFeedForward(
                dim=8,
                hidden_dim=16,
                multiple_of=1,
                ffn_dim_multiplier=0.5,
                num_experts=4,
                top_k=2,
                routing_granularity="pair",
                pair_size=2,
                router_use_hidden_state=False,
                entropy_prior_mode="gaussian_pairs",
                entropy_prior_calibration_path=str(calibration_path),
            )

    assert moe.entropy_prior_centers.shape == (2,)
    assert moe.entropy_prior_widths.shape == (2,)
    assert moe.entropy_prior_static_bias.shape == (2,)


def test_sparse_moe_ema_pair_bias_updates_and_round_trips_state_dict():
    moe = SparseMoEFeedForward(
        dim=8,
        hidden_dim=16,
        multiple_of=1,
        ffn_dim_multiplier=0.5,
        num_experts=4,
        top_k=2,
        routing_granularity="pair",
        pair_size=2,
        router_use_patch_entropy=True,
        router_patch_feature_bias=True,
        router_normalize_patch_entropy=False,
        router_use_hidden_state=False,
        pair_bias_mode="ema_byte_floor",
        pair_bias_ema=0.0,
        pair_bias_update_interval=1,
        pair_bias_lr=0.5,
        pair_bias_clip=1.5,
        pair_min_byte_fraction=0.4,
        pair_max_byte_fraction=0.6,
    )
    with torch.no_grad():
        assert moe.patch_feature_router is not None
        moe.patch_feature_router.weight.zero_()
        moe.patch_feature_router.bias.copy_(torch.tensor([2.0, -2.0]))

    x = torch.randn(1, 4, 8)
    patch_lengths = torch.tensor([[4, 4, 4, 4]])
    patch_entropies = torch.ones(1, 4)

    moe(x, patch_lengths=patch_lengths, patch_entropies=patch_entropies)

    assert not moe.dynamic_pair_bias.requires_grad
    assert moe.dynamic_pair_bias[0] < 0
    assert moe.dynamic_pair_bias[1] > 0
    assert moe.pair_load_ema.tolist() == [1.0, 0.0]

    restored = SparseMoEFeedForward(
        dim=8,
        hidden_dim=16,
        multiple_of=1,
        ffn_dim_multiplier=0.5,
        num_experts=4,
        top_k=2,
        routing_granularity="pair",
        pair_size=2,
        router_use_patch_entropy=True,
        router_patch_feature_bias=True,
        router_normalize_patch_entropy=False,
        router_use_hidden_state=False,
        pair_bias_mode="ema_byte_floor",
    )
    restored.load_state_dict(moe.state_dict())

    torch.testing.assert_close(restored.dynamic_pair_bias, moe.dynamic_pair_bias)
    torch.testing.assert_close(restored.pair_load_ema, moe.pair_load_ema)
    assert restored.pair_bias_step.item() == moe.pair_bias_step.item()


def test_sparse_moe_ema_pair_bias_uses_bf16_distributed_gather():
    moe = SparseMoEFeedForward(
        dim=8,
        hidden_dim=16,
        multiple_of=1,
        ffn_dim_multiplier=0.5,
        num_experts=4,
        top_k=2,
        routing_granularity="pair",
        pair_size=2,
        router_use_patch_entropy=True,
        router_use_hidden_state=False,
        pair_bias_mode="ema_byte_floor",
        pair_bias_ema=0.0,
        pair_bias_update_interval=1,
    )
    gathered_dtypes = []

    def fake_all_gather_into_tensor(output_tensor, input_tensor, group=None):
        gathered_dtypes.append(input_tensor.dtype)
        chunk_size = input_tensor.numel()
        output_tensor[:chunk_size].copy_(input_tensor)
        output_tensor[chunk_size:].copy_(input_tensor)

    top_indices = torch.tensor([[0, 1], [2, 3], [2, 3]])
    load_weights = torch.tensor([2.0, 4.0, 6.0], dtype=torch.float32)

    with patch("bytelatent.base_transformer.dist.is_initialized", return_value=True):
        with patch("bytelatent.base_transformer.dist.get_world_size", return_value=2):
            with patch(
                "bytelatent.base_transformer.dist.all_reduce",
                side_effect=AssertionError("EMA pair bias should not use all_reduce"),
            ):
                with patch(
                    "bytelatent.base_transformer.dist.all_gather_into_tensor",
                    fake_all_gather_into_tensor,
                ):
                    moe._maybe_update_dynamic_pair_bias(top_indices, load_weights)

    assert gathered_dtypes == [torch.bfloat16]
    assert moe.pair_load_ema.dtype == torch.float32


def test_sparse_moe_hidden_residual_ramp_controls_hidden_router_scale():
    moe = SparseMoEFeedForward(
        dim=4,
        hidden_dim=8,
        multiple_of=1,
        ffn_dim_multiplier=0.5,
        num_experts=4,
        top_k=2,
        routing_granularity="pair",
        pair_size=2,
        router_use_patch_entropy=True,
        router_patch_feature_bias=True,
        router_normalize_patch_entropy=False,
        router_use_hidden_state=True,
        entropy_prior_hidden_scale=0.0,
        hidden_residual_ramp_start_step=10,
        hidden_residual_ramp_end_step=20,
        hidden_residual_final_scale=0.25,
    )
    with torch.no_grad():
        moe.router.weight.zero_()
        moe.router.weight[1, 0] = 10.0
        assert moe.patch_feature_router is not None
        moe.patch_feature_router.weight.zero_()
        moe.patch_feature_router.bias.copy_(torch.tensor([1.0, -1.0]))

    x = torch.zeros(1, 1, 4)
    x[..., 0] = 1.0
    patch_lengths = torch.ones(1, 1)
    patch_entropies = torch.ones(1, 1)

    moe.set_router_step(0)
    moe(x, patch_lengths=patch_lengths, patch_entropies=patch_entropies)
    assert moe.last_metrics["hidden_residual_scale"] == 0.0
    assert moe.last_metrics["pair_0_unit_fraction"] == 1.0

    moe.set_router_step(20)
    moe(x, patch_lengths=patch_lengths, patch_entropies=patch_entropies)
    assert moe.last_metrics["hidden_residual_scale"] == 0.25
    assert moe.last_metrics["pair_1_unit_fraction"] == 1.0


def test_sparse_moe_default_state_dict_omits_router_schedule_state():
    moe = SparseMoEFeedForward(
        dim=8,
        hidden_dim=16,
        multiple_of=1,
        ffn_dim_multiplier=1.0,
        num_experts=4,
        top_k=2,
    )
    assert "router_step" not in moe.state_dict()

    ramped = SparseMoEFeedForward(
        dim=8,
        hidden_dim=16,
        multiple_of=1,
        ffn_dim_multiplier=0.5,
        num_experts=4,
        top_k=2,
        routing_granularity="pair",
        pair_size=2,
        router_use_patch_entropy=True,
        router_patch_feature_bias=True,
        hidden_residual_ramp_start_step=10,
        hidden_residual_ramp_end_step=20,
    )
    assert "router_step" in ramped.state_dict()


def test_sparse_moe_length_routing_requires_patch_lengths():
    moe = SparseMoEFeedForward(
        dim=32,
        hidden_dim=128,
        multiple_of=8,
        ffn_dim_multiplier=1.0,
        num_experts=4,
        top_k=1,
        router_use_patch_length=True,
    )
    x = torch.randn(1, 3, 32)

    try:
        moe(x)
    except ValueError as exc:
        assert "patch_lengths must be provided" in str(exc)
    else:
        raise AssertionError("Expected missing patch_lengths to raise ValueError")


def test_sparse_moe_entropy_routing_and_entropy_byte_balance():
    moe = SparseMoEFeedForward(
        dim=32,
        hidden_dim=128,
        multiple_of=8,
        ffn_dim_multiplier=1.0,
        num_experts=4,
        top_k=1,
        router_use_patch_length=True,
        router_use_patch_entropy=True,
        balance_cost="entropy_byte",
    )
    x = torch.randn(1, 3, 32, requires_grad=True)
    patch_lengths = torch.tensor([[1, 4, 0]])
    patch_entropies = torch.tensor([[0.5, 2.0, 0.0]])

    out = moe(
        x,
        patch_lengths=patch_lengths,
        patch_entropies=patch_entropies,
    )
    aux_loss = get_moe_aux_loss(moe)
    assert aux_loss is not None

    loss = out.square().mean() + 0.01 * aux_loss
    loss.backward()

    assert out.shape == x.shape
    assert x.grad is not None
    assert moe.router.weight.grad is not None
    assert moe.patch_feature_router is not None
    assert moe.patch_feature_router.weight.shape[1] == 2
    assert moe.patch_feature_router.weight.grad is not None
    assert moe.last_metrics["routed_units"] == 2.0
    assert moe.last_metrics["total_units"] == 3.0
    assert moe.last_metrics["load_weight_max"] == 8.0
    assert moe.last_metrics["side_feature_count"] == 2.0
    assert moe.last_metrics["patch_entropy_mean"] == 1.25


def test_sparse_moe_entropy_routing_requires_patch_entropies():
    moe = SparseMoEFeedForward(
        dim=32,
        hidden_dim=128,
        multiple_of=8,
        ffn_dim_multiplier=1.0,
        num_experts=4,
        top_k=1,
        router_use_patch_entropy=True,
    )
    x = torch.randn(1, 3, 32)
    patch_lengths = torch.tensor([[1, 1, 1]])

    try:
        moe(x, patch_lengths=patch_lengths)
    except ValueError as exc:
        assert "patch_entropies must be provided" in str(exc)
    else:
        raise AssertionError("Expected missing patch_entropies to raise ValueError")


def test_sparse_moe_entropy_only_routing_ignores_hidden_state():
    moe = SparseMoEFeedForward(
        dim=4,
        hidden_dim=8,
        multiple_of=1,
        ffn_dim_multiplier=1.0,
        num_experts=2,
        top_k=1,
        router_use_hidden_state=False,
        router_patch_feature_bias=True,
        router_normalize_patch_entropy=False,
        router_use_patch_entropy=True,
    )
    with torch.no_grad():
        moe.router.weight.fill_(100.0)
        assert moe.patch_feature_router is not None
        moe.patch_feature_router.weight.copy_(torch.tensor([[-1.0], [1.0]]))
        moe.patch_feature_router.bias.copy_(torch.tensor([2.0, -2.0]))

    patch_lengths = torch.ones(1, 2, dtype=torch.long)
    patch_entropies = torch.tensor([[1.0, 3.0]])
    first_x = torch.randn(1, 2, 4)
    second_x = torch.randn(1, 2, 4) * 1000

    moe(
        first_x,
        patch_lengths=patch_lengths,
        patch_entropies=patch_entropies,
    )
    first_assignments = {
        key: value
        for key, value in moe.last_metrics.items()
        if key.endswith("_unit_assignment_fraction")
    }
    moe(
        second_x,
        patch_lengths=patch_lengths,
        patch_entropies=patch_entropies,
    )
    second_assignments = {
        key: value
        for key, value in moe.last_metrics.items()
        if key.endswith("_unit_assignment_fraction")
    }

    assert first_assignments == second_assignments
    assert moe.last_metrics["hidden_state_routing"] == 0.0
    assert moe.last_metrics["expert_0_unit_assignment_fraction"] == 0.5
    assert moe.last_metrics["expert_1_unit_assignment_fraction"] == 0.5


def test_sparse_moe_entropy_mlp_router_uses_zscore_clip_without_hidden_router():
    moe = SparseMoEFeedForward(
        dim=4,
        hidden_dim=8,
        multiple_of=1,
        ffn_dim_multiplier=1.0,
        num_experts=2,
        top_k=1,
        router_use_hidden_state=False,
        router_use_patch_entropy=True,
        router_entropy_mlp_hidden_dim=1,
        router_entropy_mean=2.0,
        router_entropy_std=0.5,
        router_entropy_clip=4.0,
    )
    with torch.no_grad():
        moe.entropy_router[0].weight.fill_(1.0)
        moe.entropy_router[0].bias.zero_()
        moe.entropy_router[2].weight.copy_(torch.tensor([[-1.0], [1.0]]))
        moe.entropy_router[2].bias.zero_()

    seen_entropy_features = []
    moe.entropy_router.register_forward_hook(
        lambda _module, inputs, _output: seen_entropy_features.append(
            inputs[0].detach().clone()
        )
    )

    patch_lengths = torch.tensor([[1, 1, 0]])
    patch_entropies = torch.tensor([[-100.0, 100.0, 9999.0]])
    x = torch.randn(1, 3, 4)

    with patch.object(moe.router, "forward", side_effect=AssertionError("hidden router called")):
        out = moe(x, patch_lengths=patch_lengths, patch_entropies=patch_entropies)

    assert out.shape == x.shape
    assert torch.equal(out[:, 2:, :], torch.zeros_like(out[:, 2:, :]))
    assert len(seen_entropy_features) == 1
    torch.testing.assert_close(
        seen_entropy_features[0],
        torch.tensor([[-4.0], [4.0]]),
    )
    assert moe.last_metrics["hidden_state_routing"] == 0.0
    assert moe.last_metrics["invalid_positions_skipped"] == 1.0
    assert moe.last_metrics["expert_0_unit_assignment_fraction"] == 0.5
    assert moe.last_metrics["expert_1_unit_assignment_fraction"] == 0.5


def test_base_transformer_passes_entropy_mlp_router_args_to_moe_layers():
    args = BaseTransformerArgs(
        dim=32,
        n_layers=1,
        n_heads=4,
        max_seqlen=8,
        multiple_of=8,
        moe_num_experts=8,
        moe_top_k=1,
        moe_router_use_hidden_state=False,
        moe_router_use_patch_entropy=True,
        moe_router_entropy_mlp_hidden_dim=32,
        moe_router_entropy_mean=1.5,
        moe_router_entropy_std=0.25,
        moe_router_entropy_clip=4.0,
    )
    model = BaseTransformer(args)
    moe = model.layers[0].feed_forward

    assert isinstance(moe, SparseMoEFeedForward)
    assert moe.entropy_router[0].in_features == 1
    assert moe.entropy_router[0].out_features == 32
    assert moe.entropy_router[2].out_features == 8
    assert moe.router_entropy_mean == 1.5
    assert moe.router_entropy_std == 0.25
    assert moe.router_entropy_clip == 4.0


def test_sparse_moe_rejects_router_without_any_input_features():
    try:
        SparseMoEFeedForward(
            dim=4,
            hidden_dim=8,
            multiple_of=1,
            ffn_dim_multiplier=1.0,
            num_experts=2,
            top_k=1,
            router_use_hidden_state=False,
        )
    except ValueError as exc:
        assert "patch routing feature is required" in str(exc)
    else:
        raise AssertionError("Expected a router without input features to fail")


def test_sparse_moe_patch_byte_feature_routing():
    moe = SparseMoEFeedForward(
        dim=32,
        hidden_dim=128,
        multiple_of=8,
        ffn_dim_multiplier=1.0,
        num_experts=4,
        top_k=1,
        router_use_patch_byte_features=True,
    )
    x = torch.randn(1, 3, 32, requires_grad=True)
    patch_lengths = torch.tensor([[2, 3, 0]])
    patch_byte_features = torch.tensor(
        [[[0.5, 0.5, 0.0, 0.0, 0.0], [0.0, 0.0, 0.5, 0.5, 0.0], [0.0] * 5]]
    )

    out = moe(
        x,
        patch_lengths=patch_lengths,
        patch_byte_features=patch_byte_features,
    )
    loss = out.square().mean() + 0.01 * get_moe_aux_loss(moe)
    loss.backward()

    assert out.shape == x.shape
    assert moe.patch_feature_router is not None
    assert moe.patch_feature_router.weight.shape[1] == len(PATCH_BYTE_TYPE_NAMES)
    assert moe.patch_feature_router.weight.grad is not None
    assert moe.last_metrics["side_feature_count"] == 5.0
    assert moe.last_metrics["patch_byte_alpha_fraction_mean"] == 0.25
    assert "byte_type_bucket_alpha_unit_fraction" in moe.last_metrics


def test_sparse_moe_patch_byte_feature_routing_requires_features():
    moe = SparseMoEFeedForward(
        dim=32,
        hidden_dim=128,
        multiple_of=8,
        ffn_dim_multiplier=1.0,
        num_experts=4,
        top_k=1,
        router_use_patch_byte_features=True,
    )

    try:
        moe(torch.randn(1, 3, 32))
    except ValueError as exc:
        assert "patch_byte_features must be provided" in str(exc)
    else:
        raise AssertionError("Expected missing patch_byte_features to raise ValueError")


def test_sparse_moe_specialization_metrics_bucket_usage():
    moe = SparseMoEFeedForward(
        dim=32,
        hidden_dim=128,
        multiple_of=8,
        ffn_dim_multiplier=1.0,
        num_experts=2,
        top_k=1,
    )
    router_probs = torch.full((4, 2), 0.5)
    top_indices = torch.tensor([[0], [0], [1], [1]])
    patch_lengths = torch.tensor([2, 6, 10, 0])
    patch_entropies = torch.tensor([0.5, 1.5, 2.5, 0.0])
    load_weights = (patch_lengths > 0).float()

    moe._record_metrics(
        router_probs,
        top_indices,
        None,
        load_weights,
        patch_lengths,
        patch_entropies,
    )

    metrics = moe.last_metrics

    def assert_close(key, expected):
        assert abs(metrics[key] - expected) < 1e-6, (key, metrics[key], expected)

    assert_close("expert_0_patch_length_mean", 4.0)
    assert_close("expert_1_patch_length_mean", 10.0)
    assert_close("expert_0_patch_entropy_mean", 1.0)
    assert_close("expert_1_patch_entropy_mean", 2.5)
    assert_close("expert_0_unit_assignment_fraction", 2.0 / 3.0)
    assert_close("expert_1_unit_assignment_fraction", 1.0 / 3.0)

    for bucket in ("short", "medium", "long"):
        assert_close(f"length_bucket_{bucket}_unit_fraction", 1.0 / 3.0)
        assert_close(f"length_bucket_{bucket}_assignment_fraction", 1.0 / 3.0)

    assert_close("length_bucket_short_expert_0_assignment_fraction", 1.0)
    assert_close("length_bucket_medium_expert_0_assignment_fraction", 1.0)
    assert_close("length_bucket_long_expert_1_assignment_fraction", 1.0)
    assert_close("expert_0_length_bucket_short_assignment_fraction", 0.5)
    assert_close("expert_0_length_bucket_medium_assignment_fraction", 0.5)
    assert_close("expert_1_length_bucket_long_assignment_fraction", 1.0)

    for bucket in ("low", "medium", "high"):
        assert_close(f"entropy_bucket_{bucket}_unit_fraction", 1.0 / 3.0)
        assert_close(f"entropy_bucket_{bucket}_assignment_fraction", 1.0 / 3.0)

    assert_close("entropy_bucket_low_expert_0_assignment_fraction", 1.0)
    assert_close("entropy_bucket_medium_expert_0_assignment_fraction", 1.0)
    assert_close("entropy_bucket_high_expert_1_assignment_fraction", 1.0)
    assert_close("expert_0_entropy_bucket_low_assignment_fraction", 0.5)
    assert_close("expert_0_entropy_bucket_medium_assignment_fraction", 0.5)
    assert_close("expert_1_entropy_bucket_high_assignment_fraction", 1.0)


def test_sparse_moe_entropy_correlation_metrics():
    moe = SparseMoEFeedForward(
        dim=4,
        hidden_dim=8,
        multiple_of=1,
        ffn_dim_multiplier=1.0,
        num_experts=4,
        top_k=2,
        router_use_hidden_state=False,
        router_use_patch_entropy=True,
        router_patch_feature_bias=True,
        router_normalize_patch_entropy=False,
    )
    with torch.no_grad():
        moe.router.weight.zero_()
        moe.patch_feature_router.weight.zero_()
        moe.patch_feature_router.bias.zero_()
        moe.patch_feature_router.weight[:, 0] = torch.tensor([0.0, 0.0, 1.0, 1.0])
        moe.patch_feature_router.bias[:] = torch.tensor([0.0, 0.0, -1.0, -1.0])

    x = torch.randn(1, 4, 4)
    patch_lengths = torch.ones(1, 4)
    patch_entropies = torch.tensor([[0.1, 0.5, 2.0, 3.0]])

    moe(x, patch_lengths=patch_lengths, patch_entropies=patch_entropies)
    metrics = moe.last_metrics

    assert metrics["entropy_expected_pair_corr"] > 0.9
    assert metrics["entropy_selected_pair_corr"] > 0.9
    assert metrics["top2_same_pair_fraction"] == 1.0


def test_sparse_moe_cost_aware_congestion_price_reroutes_overloaded_expert():
    moe = SparseMoEFeedForward(
        dim=4,
        hidden_dim=8,
        multiple_of=1,
        ffn_dim_multiplier=1.0,
        num_experts=2,
        top_k=1,
        router_congestion_weight=2.0,
        balance_cost="byte",
    )
    with torch.no_grad():
        moe.router.weight.zero_()
        moe.router.weight[0, 0] = 1.0

    x = torch.zeros(1, 2, 4)
    x[..., 0] = 1.0
    out = moe(x, patch_lengths=torch.tensor([[1, 3]]))
    metrics = moe.last_metrics

    assert out.shape == x.shape
    assert metrics["congestion_weight"] == 2.0
    assert metrics["congestion_price_max"] > 0.0
    assert metrics["congestion_price_min"] < 0.0
    assert metrics["congestion_top1_reroute_fraction"] == 1.0
    assert metrics["expert_1_unit_assignment_fraction"] == 1.0
    assert (
        metrics["post_congestion_prob_load_imbalance"]
        < metrics["pre_congestion_prob_load_imbalance"]
    )


def test_sparse_moe_congestion_price_is_zero_without_active_load():
    moe = SparseMoEFeedForward(
        dim=4,
        hidden_dim=8,
        multiple_of=1,
        ffn_dim_multiplier=1.0,
        num_experts=2,
        top_k=1,
    )
    price = moe._congestion_price(torch.tensor([[0.9, 0.1]]), torch.tensor([0.0]))

    assert torch.equal(price, torch.zeros_like(price))


def test_sparse_moe_rejects_negative_congestion_weight():
    try:
        SparseMoEFeedForward(
            dim=4,
            hidden_dim=8,
            multiple_of=1,
            ffn_dim_multiplier=1.0,
            num_experts=2,
            top_k=1,
            router_congestion_weight=-0.1,
        )
    except ValueError as exc:
        assert "router_congestion_weight must be non-negative" in str(exc)
    else:
        raise AssertionError("Expected negative congestion weight to raise ValueError")


def test_sparse_moe_router_z_loss_backward_and_metrics():
    moe = SparseMoEFeedForward(
        dim=8,
        hidden_dim=16,
        multiple_of=1,
        ffn_dim_multiplier=1.0,
        num_experts=3,
        top_k=1,
        router_z_loss_weight=0.001,
    )
    x = torch.randn(2, 4, 8, requires_grad=True)

    out = moe(x)
    router_z_loss = get_moe_router_z_loss(moe)
    assert router_z_loss is not None
    assert router_z_loss.item() > 0.0

    loss = out.square().mean() + 0.001 * router_z_loss
    loss.backward()

    assert moe.router.weight.grad is not None
    assert moe.last_metrics["router_z_loss"] == router_z_loss.detach().item()
    assert moe.last_metrics["router_z_loss_weight"] == 0.001


def test_sparse_moe_rejects_negative_router_z_loss_weight():
    try:
        SparseMoEFeedForward(
            dim=4,
            hidden_dim=8,
            multiple_of=1,
            ffn_dim_multiplier=1.0,
            num_experts=2,
            top_k=1,
            router_z_loss_weight=-0.1,
        )
    except ValueError as exc:
        assert "router_z_loss_weight must be non-negative" in str(exc)
    else:
        raise AssertionError(
            "Expected negative router z-loss weight to raise ValueError"
        )
