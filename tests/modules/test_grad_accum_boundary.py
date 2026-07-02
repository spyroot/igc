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

from igc.modules.llm_train_state_encoder import is_accum_boundary


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


# Author: Mus mbayramo@stanford.edu
