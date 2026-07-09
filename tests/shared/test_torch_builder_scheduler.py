"""Offline regression for ``TorchBuilder.constant_schedule_with_warmup``.

The builder referenced ``_get_constant_schedule_with_warmup_lr_lambda`` without importing
it, so the constant-with-warmup scheduler raised ``NameError`` when actually used. This
pins that the scheduler builds and steps on CPU. No GPU needed.

Author:
Mus mbayramo@stanford.edu
"""
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


# Author: Mus mbayramo@stanford.edu
