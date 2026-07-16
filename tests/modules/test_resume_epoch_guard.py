"""
Offline regression for the resumed-run scheduler-argument guard.

Pins ``remaining_epochs`` and ``scheduler_epoch_args``: a relaunch that resumes
from a checkpoint already at or beyond the configured epoch target owes zero
epochs, and scheduler construction must receive positive ``epochs`` /
``steps_per_epoch`` anyway (OneCycleLR raises ``ValueError: Expected positive
integer epochs, but got 0`` — the slot2 Phase 1 prebuild crash) while the
zero-iteration epoch loop still falls through to end-of-train finalization
(save/report — behavior pinned by tests/gpu/test_m1_train_step_live.py). Pure
logic — no torch models, no GPU, no network.

Author:
Mus mbayramo@stanford.edu
"""

from igc.modules.llm_train_state_encoder import (
    optimizer_steps_per_epoch,
    remaining_epochs,
    scheduler_epoch_args,
)


def test_fresh_run_owes_all_epochs():
    """A fresh run (no checkpoint) owes the full configured epoch count."""
    assert remaining_epochs(3, 0) == 3


def test_mid_run_resume_owes_the_tail():
    """Resuming a partially trained run owes only the unfinished epochs."""
    assert remaining_epochs(3, 2) == 1


def test_completed_checkpoint_owes_zero():
    """A checkpoint exactly at the target owes 0 remaining epochs."""
    assert remaining_epochs(3, 3) == 0


def test_checkpoint_beyond_target_owes_zero_not_negative():
    """A checkpoint from a longer prior run must floor at 0, never negative."""
    assert remaining_epochs(3, 5) == 0


def test_empty_dataloader_reports_zero_steps():
    """drop_last on a sub-batch dataset yields 0 raw optimizer steps."""
    assert optimizer_steps_per_epoch(0, 4) == 0


def test_scheduler_args_pass_real_work_through_unchanged():
    """Genuine remaining work reaches the scheduler exactly as computed."""
    # 2 epochs left, 10 micro-batches at accum 4 -> ceil(10/4) = 3 opt steps.
    assert scheduler_epoch_args(3, 1, 10, 4) == (2, 3)


def test_scheduler_args_clamp_completed_resume_to_valid_schedule():
    """A completed-checkpoint resume yields a valid (never stepped) schedule."""
    # The slot2 prebuild crash: remaining epochs 0 must become 1 for OneCycleLR;
    # the epoch loop still runs range(3, 3) == zero iterations.
    assert scheduler_epoch_args(3, 3, 10, 4) == (1, 3)


def test_scheduler_args_clamp_empty_dataloader_to_valid_schedule():
    """An empty per-rank dataloader yields steps_per_epoch=1, not a crash."""
    assert scheduler_epoch_args(1, 0, 0, 4) == (1, 1)


# Author: Mus mbayramo@stanford.edu
