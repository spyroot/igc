"""Offline guards for M3 distributed metric logging.

The M3 trainer logs token/sample counters by reducing them across ranks. That
reduction is a distributed collective, so every rank must enter it in the same
order. These tests pin the CPU-inspectable contract that prevents a DDP
collective mismatch where rank 0 all-reduces metrics while other ranks enter
the next model-forward broadcast.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

import inspect
import types

import torch

from igc.modules.m3_goal_planner_train import (
    M3GoalPlannerSFTConfig,
    M3GoalPlannerSFTTrainer,
    _DistributedContext,
)


def _bare_trainer(rank: int) -> M3GoalPlannerSFTTrainer:
    """Build just enough trainer state to call ``_append_history``."""
    trainer = object.__new__(M3GoalPlannerSFTTrainer)
    trainer.config = M3GoalPlannerSFTConfig(output_dir="/tmp/m3-test")
    trainer.distributed = _DistributedContext(
        enabled=True,
        rank=rank,
        world_size=4,
        local_rank=rank,
    )
    trainer.device = torch.device("cpu")
    trainer.optimizer = types.SimpleNamespace(param_groups=[{"lr": 1e-4}])
    trainer.metric_logger = None
    return trainer


def test_append_history_participates_in_reductions_on_non_main_rank(monkeypatch):
    """Non-main ranks must reduce counters even though they do not append/log."""
    calls: list[int] = []

    def fake_sum(value: int, distributed: _DistributedContext) -> int:
        assert distributed.rank == 2
        calls.append(value)
        return value * distributed.world_size

    monkeypatch.setattr(
        "igc.modules.m3_goal_planner_train._sum_number_across_ranks",
        fake_sum,
    )
    history: list[dict[str, float]] = []

    _bare_trainer(rank=2)._append_history(
        history,
        epoch=0,
        global_step=3,
        epoch_loss=1.0,
        epoch_steps=1,
        epoch_tokens=11,
        epoch_positions=17,
        epoch_samples=2,
        epoch_start=0.0,
        grad_norm=None,
    )

    assert calls == [11, 17, 2]
    assert history == []


def test_train_loop_does_not_rank_gate_append_history_collectives():
    """Any train-loop call into ``_append_history`` must be reached by all ranks."""
    src = inspect.getsource(M3GoalPlannerSFTTrainer.train)
    assert "and self.distributed.is_main_process" not in src, (
        "train() must not rank-gate _append_history(); that helper performs "
        "distributed reductions, so all ranks must call it in the same order."
    )


# Author: Mus mbayramo@stanford.edu
