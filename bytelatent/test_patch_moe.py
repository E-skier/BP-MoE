import torch

from bytelatent.base_transformer import (
    BaseTransformer,
    BaseTransformerArgs,
    SparseMoEFeedForward,
    get_moe_aux_loss,
    get_moe_metrics,
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
