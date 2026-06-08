import torch

from bytelatent.base_transformer import (
    PATCH_BYTE_TYPE_NAMES,
    BaseTransformer,
    BaseTransformerArgs,
    SparseMoEFeedForward,
    get_moe_assignment_balance_loss,
    get_moe_aux_loss,
    get_moe_metrics,
    get_moe_router_z_loss,
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


def test_sparse_moe_assignment_balance_loss_penalizes_topk_tie_skew():
    moe = SparseMoEFeedForward(
        dim=8,
        hidden_dim=16,
        multiple_of=1,
        ffn_dim_multiplier=1.0,
        num_experts=4,
        top_k=2,
        assignment_balance_loss_weight=0.05,
    )
    with torch.no_grad():
        moe.router.weight.zero_()

    x = torch.randn(2, 3, 8, requires_grad=True)
    out = moe(x)
    assignment_loss = get_moe_assignment_balance_loss(moe)

    assert assignment_loss is not None
    assert assignment_loss.item() > 0.0
    assert moe.last_metrics["assignment_balance_loss"] == assignment_loss.item()
    assert moe.last_metrics["assignment_balance_loss_weight"] == 0.05

    loss = out.square().mean() + 0.05 * assignment_loss
    loss.backward()

    assert x.grad is not None
    assert moe.router.weight.grad is not None


def test_sparse_moe_rejects_negative_assignment_balance_loss_weight():
    try:
        SparseMoEFeedForward(
            dim=4,
            hidden_dim=8,
            multiple_of=1,
            ffn_dim_multiplier=1.0,
            num_experts=2,
            top_k=1,
            assignment_balance_loss_weight=-0.1,
        )
    except ValueError as exc:
        assert "assignment_balance_loss_weight must be non-negative" in str(exc)
    else:
        raise AssertionError(
            "Expected negative assignment balance loss weight to raise ValueError"
        )


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
