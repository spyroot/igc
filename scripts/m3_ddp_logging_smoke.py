"""M3 DDP metric-logging smoke test.

Run this under ``torchrun`` on a GPU node to verify that M3 metric collection
uses the same collective order on every rank before entering the next DDP
forward pass. It intentionally avoids HuggingFace downloads and real datasets.

Example:

.. code-block:: bash

   torchrun --nproc_per_node=4 scripts/m3_ddp_logging_smoke.py

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

import time
import types

import torch
import torch.distributed as dist
from torch import nn

from igc.modules.m3_goal_planner_train import (
    M3GoalPlannerSFTConfig,
    M3GoalPlannerSFTTrainer,
    _DistributedContext,
)


class _TinyBufferedModel(nn.Module):
    """Small DDP model with a buffer so pre-forward broadcast is exercised."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.register_buffer("scale", torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a scalar loss for one backward pass."""
        return (self.linear(x) * self.scale).sum()


def _device_for(local_rank: int) -> torch.device:
    """Resolve the rank-local smoke-test device."""
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank)
    return torch.device("cpu")


def _bare_trainer(distributed: _DistributedContext, device: torch.device):
    """Build enough trainer state to execute the real metric reducer."""
    trainer = object.__new__(M3GoalPlannerSFTTrainer)
    trainer.config = M3GoalPlannerSFTConfig(output_dir="/tmp/m3-ddp-smoke")
    trainer.distributed = distributed
    trainer.device = device
    trainer.optimizer = types.SimpleNamespace(param_groups=[{"lr": 1e-4}])
    trainer.metric_logger = None
    return trainer


def main() -> None:
    """Run metric reductions followed by one DDP forward/backward."""
    distributed = _DistributedContext.from_env()
    device = _device_for(distributed.local_rank)
    history: list[dict[str, float]] = []

    _bare_trainer(distributed, device)._append_history(
        history,
        epoch=0,
        global_step=1,
        epoch_loss=1.0,
        epoch_steps=1,
        epoch_tokens=10 + distributed.rank,
        epoch_positions=20 + distributed.rank,
        epoch_samples=1,
        epoch_start=time.monotonic(),
        grad_norm=None,
    )

    expected_history_len = 1 if distributed.is_main_process else 0
    if len(history) != expected_history_len:
        raise RuntimeError(
            f"rank {distributed.rank}: expected history length "
            f"{expected_history_len}, got {len(history)}"
        )

    model = _TinyBufferedModel().to(device)
    if distributed.enabled:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
        )
    loss = model(torch.ones(2, 4, device=device))
    loss.backward()

    if distributed.enabled:
        dist.barrier()
    if distributed.is_main_process:
        print(
            "[m3-ddp-logging-smoke] PASS "
            f"world={distributed.world_size} device={device.type}"
        )
    if distributed.enabled:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()


# Author: Mus mbayramo@stanford.edu
