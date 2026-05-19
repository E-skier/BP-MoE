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
