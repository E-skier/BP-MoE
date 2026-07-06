import signal
from unittest import mock

import torch

from bytelatent.base_transformer import SparseMoEFeedForward
from bytelatent.distributed import (
    check_model_value_range,
    init_signal_handler,
    requeue_slurm_job,
    should_activation_checkpoint_layer,
)


def test_check_model_value_range_skips_integer_buffers_for_std_checks():
    module = torch.nn.Linear(2, 2)
    module.register_buffer("integer_state", torch.zeros((), dtype=torch.long))

    check_model_value_range(module)


def test_activation_checkpoint_policy_skips_sparse_moe_layers():
    layer = torch.nn.Module()
    layer.feed_forward = SparseMoEFeedForward(
        dim=8,
        hidden_dim=16,
        multiple_of=1,
        ffn_dim_multiplier=0.5,
        num_experts=4,
        top_k=2,
        routing_granularity="pair",
        pair_size=2,
    )

    assert should_activation_checkpoint_layer(layer) is False
    assert should_activation_checkpoint_layer(torch.nn.Linear(2, 2)) is True


def test_init_signal_handler_registers_slurm_and_termination_signals():
    old_usr2 = signal.getsignal(signal.SIGUSR2)
    old_term = signal.getsignal(signal.SIGTERM)

    def handler(signum, frame):
        return None

    try:
        init_signal_handler(handler)

        assert signal.getsignal(signal.SIGUSR2) is handler
        assert signal.getsignal(signal.SIGTERM) is handler
    finally:
        signal.signal(signal.SIGUSR2, old_usr2)
        signal.signal(signal.SIGTERM, old_term)


def test_requeue_slurm_job_exits_cleanly_outside_slurm():
    with mock.patch.dict("os.environ", {}, clear=True):
        try:
            requeue_slurm_job()
        except SystemExit as exc:
            assert exc.code == 0
        else:
            raise AssertionError("requeue_slurm_job should exit")
