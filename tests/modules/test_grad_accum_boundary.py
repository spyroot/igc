"""
Offline regression for the plain-path gradient-accumulation boundary.

Pins that ``is_accum_boundary`` fires every ``accum`` micro-batches AND on the final batch
of the epoch (so a trailing partial window is flushed, not dropped), yielding exactly
``ceil(total_batches / accum)`` optimizer steps. This is the logic that makes
``--gradient_accumulation_steps`` actually honored on the non-accelerator path, where the
loop previously called optimizer.step() every micro-batch. Pure logic — no torch tensors,
no GPU, no network.

Author:
Mus mbayramo@stanford.edu
"""

import math

import pytest

from igc.modules.llm_train_state_encoder import is_accum_boundary, reached_max_steps


def _boundaries(total, accum):
    """Micro-batch indices at which an optimizer step fires over one epoch."""
    return [i for i in range(total) if is_accum_boundary(i, accum, total)]


def test_accum_one_steps_every_micro_batch():
    """accum=1 degenerates to stepping on every micro-batch."""
    assert _boundaries(5, 1) == [0, 1, 2, 3, 4]


def test_steps_on_window_and_flushes_partial_tail():
    """Boundaries land on each full window and always on the final batch."""
    # 10 batches, accum 4 -> windows end at 3 and 7; final batch 9 flushes the partial.
    assert _boundaries(10, 4) == [3, 7, 9]


def test_no_partial_tail_when_evenly_divisible():
    """When total is a multiple of accum, the last window IS the final batch (no dup)."""
    assert _boundaries(8, 4) == [3, 7]


@pytest.mark.parametrize("total,accum", [(10, 4), (8, 4), (10, 3), (6, 4), (1, 8), (7, 1)])
def test_optimizer_step_count_is_ceil(total, accum):
    """Optimizer steps per epoch == ceil(total_batches / accum) — the effective-batch fix."""
    assert len(_boundaries(total, accum)) == math.ceil(total / accum)


def test_accum_below_one_is_coerced():
    """accum <= 0 is coerced to 1 (step every batch) rather than dividing by zero."""
    assert _boundaries(3, 0) == [0, 1, 2]
    assert _boundaries(3, -5) == [0, 1, 2]


def test_final_batch_always_flushes():
    """The last micro-batch is always a boundary so no accumulated gradients are stranded."""
    assert is_accum_boundary(9, 4, 10) is True   # last batch, mid-window
    assert is_accum_boundary(5, 4, 6) is True     # last batch, partial window


def test_reached_max_steps_uncapped_when_none_or_nonpositive():
    """None / 0 / negative max_steps mean no cap, so training never stops on the counter."""
    assert reached_max_steps(10_000, None) is False
    assert reached_max_steps(10_000, 0) is False
    assert reached_max_steps(10_000, -5) is False


def test_reached_max_steps_stops_at_the_cap():
    """With a positive cap, stop once optimizer steps reach it (honors --max_train_steps)."""
    assert reached_max_steps(49, 50) is False
    assert reached_max_steps(50, 50) is True
    assert reached_max_steps(51, 50) is True


# Author: Mus mbayramo@stanford.edu


def test_optimizer_steps_per_epoch_scales_by_accum():
    """Scheduler steps are optimizer steps: ceil(micro/accum)."""
    from igc.modules.llm_train_state_encoder import optimizer_steps_per_epoch
    assert optimizer_steps_per_epoch(100, 1) == 100
    assert optimizer_steps_per_epoch(100, 4) == 25
    assert optimizer_steps_per_epoch(101, 4) == 26
    assert optimizer_steps_per_epoch(5, 0) == 5  # coerced accum


def test_measure_grad_norm_before_and_after_zero_grad():
    """Nonzero with live grads; 0.0 after zero_grad — the original bug's shape."""
    import torch
    from igc.modules.llm_train_state_encoder import measure_grad_norm

    model = torch.nn.Linear(3, 3)
    model(torch.ones(2, 3)).sum().backward()
    assert measure_grad_norm(model) > 0.0

    model.zero_grad()
    assert measure_grad_norm(model) == 0.0
