"""Offline regression for unwrapping accelerate-wrapped optimizer/scheduler.

The multi-GPU (FSDP) checkpoint save called ``Accelerator.unwrap_model`` on the
optimizer and scheduler. That helper is for ``nn.Module``s; in accelerate>=1.14 it
probes ``._modules`` and raises ``AttributeError('_modules')`` on an
``AcceleratedOptimizer`` / ``AcceleratedScheduler``, killing every 4-GPU run at the
first best-checkpoint save. ``unwrap_accelerate`` reads the wrapper's own inner
attribute instead. This drives it without a distributed launch.

Author:
Mus mbayramo@stanford.edu
"""

from igc.modules.llm_train_state_encoder import unwrap_accelerate


def test_unwrap_accelerate_returns_inner_for_wrapped():
    """A wrapper exposing ``.optimizer`` / ``.scheduler`` yields its inner object."""
    class AcceleratedOptimizer:
        def __init__(self, inner):
            self.optimizer = inner

    class AcceleratedScheduler:
        def __init__(self, inner):
            self.scheduler = inner

    base_opt = object()
    base_sched = object()
    assert unwrap_accelerate(AcceleratedOptimizer(base_opt), "optimizer") is base_opt
    assert unwrap_accelerate(AcceleratedScheduler(base_sched), "scheduler") is base_sched


def test_unwrap_accelerate_passes_through_unwrapped():
    """A plain optimizer/scheduler (single-GPU path) is returned unchanged."""
    plain = object()
    assert unwrap_accelerate(plain, "optimizer") is plain
    assert unwrap_accelerate(plain, "scheduler") is plain


# Author: Mus mbayramo@stanford.edu
