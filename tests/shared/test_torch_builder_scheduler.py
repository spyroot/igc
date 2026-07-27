"""Offline regression for ``TorchBuilder.constant_schedule_with_warmup``.

The builder referenced ``_get_constant_schedule_with_warmup_lr_lambda`` without importing
it, so the constant-with-warmup scheduler raised ``NameError`` when actually used. This
pins that the scheduler builds and steps on CPU. No GPU needed.

Author:
Mus mbayramo@stanford.edu
"""
import pytest
import torch
from torch import nn

from igc.shared.shared_torch_builder import TorchBuilder


def test_constant_schedule_with_warmup_builds_and_steps():
    """The scheduler constructs (import resolved) and a step does not raise."""
    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = TorchBuilder.constant_schedule_with_warmup(optimizer, num_warmup_steps=2)
    assert isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR)
    lrs = []
    for _ in range(4):
        optimizer.step()
        scheduler.step()
        lrs.append(optimizer.param_groups[0]["lr"])
    # warmup ramps up, then holds constant at the base lr (0.1)
    assert all(0.0 <= lr <= 0.1 + 1e-9 for lr in lrs)
    assert lrs[-1] == 0.1


def test_one_cycle_scheduler_translates_warmup_ratio_to_pct_start(monkeypatch):
    """OneCycleLR consumes profile warmup_ratio as pct_start, not as a raw kwarg."""
    captured = {}

    class FakeOneCycleLR:
        def __init__(self, optimizer, *, max_lr, total_steps, pct_start=0.3):
            captured.update({
                "optimizer": optimizer,
                "max_lr": max_lr,
                "total_steps": total_steps,
                "pct_start": pct_start,
            })

    optimizer = object()
    monkeypatch.setattr(torch.optim.lr_scheduler, "OneCycleLR", FakeOneCycleLR)

    scheduler = TorchBuilder.create_scheduler(
        "OneCycleLR",
        optimizer,
        max_lr=0.01,
        total_steps=100,
        warmup_ratio=0.2,
    )

    assert isinstance(scheduler, FakeOneCycleLR)
    assert captured == {
        "optimizer": optimizer,
        "max_lr": 0.01,
        "total_steps": 100,
        "pct_start": 0.2,
    }


@pytest.mark.parametrize("warmup_ratio", [0.0, -0.1, 1.0, 1.2])
def test_one_cycle_scheduler_rejects_invalid_warmup_ratio(
    monkeypatch,
    warmup_ratio: float,
):
    """OneCycleLR warmup_ratio is bounded to the open interval (0, 1)."""

    class FakeOneCycleLR:
        def __init__(self, optimizer, *, max_lr, total_steps, pct_start=0.3):
            self.optimizer = optimizer

    monkeypatch.setattr(torch.optim.lr_scheduler, "OneCycleLR", FakeOneCycleLR)

    with pytest.raises(ValueError, match="warmup_ratio"):
        TorchBuilder.create_scheduler(
            "OneCycleLR",
            object(),
            max_lr=0.01,
            total_steps=100,
            warmup_ratio=warmup_ratio,
        )


def test_non_one_cycle_schedulers_do_not_receive_warmup_ratio(monkeypatch):
    """warmup_ratio is OneCycleLR-only and must not leak into other schedulers."""
    captured = {}

    class FakeExponentialLR:
        def __init__(self, optimizer, *, gamma):
            captured.update({"optimizer": optimizer, "gamma": gamma})

    optimizer = object()
    monkeypatch.setattr(torch.optim.lr_scheduler, "ExponentialLR", FakeExponentialLR)

    scheduler = TorchBuilder.create_scheduler(
        "ExponentialLR",
        optimizer,
        gamma=0.95,
        warmup_ratio=0.2,
    )

    assert isinstance(scheduler, FakeExponentialLR)
    assert captured == {"optimizer": optimizer, "gamma": 0.95}


# Author: Mus mbayramo@stanford.edu
