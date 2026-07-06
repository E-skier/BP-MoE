from bytelatent.train import (
    TrainState,
    crossed_compute_budgets,
    load_compute_budgets,
    reduce_step_valid_counts,
    update_active_compute_counters,
)


class DummyScheduler:
    def state_dict(self):
        return {"lr": 1.0}

    def load_state_dict(self, state_dict):
        self.loaded = state_dict


def test_train_state_round_trips_active_compute_counters():
    state = TrainState(
        step=7,
        acc_step=0,
        scheduler=DummyScheduler(),
        data_loader_state=None,
        cumulative_active_flops=123.5,
        cumulative_valid_bytes=456,
        cumulative_valid_patches=78,
        saved_budget_boundaries=["C_low"],
    )

    restored = TrainState(
        step=0,
        acc_step=0,
        scheduler=DummyScheduler(),
        data_loader_state=None,
    )
    restored.load_state_dict(state.state_dict())

    assert restored.cumulative_active_flops == 123.5
    assert restored.cumulative_valid_bytes == 456
    assert restored.cumulative_valid_patches == 78
    assert restored.saved_budget_boundaries == ["C_low"]


def test_crossed_compute_budgets_returns_unsaved_boundaries_in_order():
    crossed = crossed_compute_budgets(
        previous_flops=90.0,
        current_flops=250.0,
        budgets={"C_low": 100.0, "C_mid": 200.0, "C_high": 300.0},
        saved_boundaries={"C_low"},
    )

    assert crossed == ["C_mid"]


def test_load_compute_budgets_reads_required_budget_keys(tmp_path):
    budget_path = tmp_path / "compute_budgets.json"
    budget_path.write_text(
        '{"C_low": 1.5, "C_mid": 2.5, "C_high": 3.5, "metadata": {"ignored": true}}'
    )

    assert load_compute_budgets(str(budget_path)) == {
        "C_low": 1.5,
        "C_mid": 2.5,
        "C_high": 3.5,
    }


def test_update_active_compute_counters_accumulates_and_reports_crossed_budgets():
    state = TrainState(
        step=0,
        acc_step=0,
        scheduler=DummyScheduler(),
        data_loader_state=None,
        cumulative_active_flops=90.0,
        cumulative_valid_bytes=10,
        cumulative_valid_patches=2,
        saved_budget_boundaries=["C_low"],
    )

    crossed = update_active_compute_counters(
        state,
        step_active_flops=160.0,
        step_valid_bytes=7,
        step_valid_patches=3,
        budgets={"C_low": 100.0, "C_mid": 200.0, "C_high": 300.0},
    )

    assert state.cumulative_active_flops == 250.0
    assert state.cumulative_valid_bytes == 17
    assert state.cumulative_valid_patches == 5
    assert crossed == ["C_mid"]
    assert state.saved_budget_boundaries == ["C_low", "C_mid"]


def test_reduce_step_valid_counts_uses_bfloat16_for_single_node_collective(monkeypatch):
    import torch

    reduce_dtypes = []

    def fake_dist_sum(value, reduce_dtype=None):
        reduce_dtypes.append(reduce_dtype)
        return torch.tensor(value, dtype=torch.float32)

    monkeypatch.setattr("bytelatent.train.dist_sum", fake_dist_sum)

    assert reduce_step_valid_counts(12, 3) == (12, 3)
    assert reduce_dtypes == [torch.bfloat16, torch.bfloat16]
