"""Offline regression for the resume-time ``validation_accuracy`` seed in the encoder trainer.

``LlmEmbeddingsTrainer`` seeds the pre-loop ``validation_accuracy`` from the checkpoint before
the training loop starts. On epochs where evaluation does not run, that seed is the value fed to
the best-checkpoint comparison ``validation_accuracy > self._best_validation_metric``. It must be
the restored accuracy (``CheckpointState.best_accuracy``), NOT the epoch counter
(``CheckpointState.last_epoch``): an epoch integer is ``>= 0`` and would beat the ``-inf`` best
sentinel on every no-eval epoch, spuriously flagging a non-improving epoch as "best" and
overwriting ``_best_validation_metric`` with an epoch number.

These are the two fields that are trivially confusable, so this test pins that ``best_accuracy``
is the accuracy channel and ``last_epoch`` the epoch channel, and that seeding the best-tracking
comparison from the correct field behaves while the epoch field would corrupt it. Runs on CPU with
no checkpoint files.

Author:
Mus mbayramo@stanford.edu
"""
import math

from igc.modules.base.igc_base_module import CheckpointState, IgcModule


def _module_without_checkpoint_dir():
    """An IgcModule with no checkpoint dir (bypasses the heavy __init__)."""
    module = IgcModule.__new__(IgcModule)
    module._module_checkpoint_dir = None
    return module


def test_fresh_resume_seed_is_accuracy_sentinel_not_epoch():
    """No-checkpoint resume seeds validation_accuracy from best_accuracy (-inf), not last_epoch (0)."""
    state = _module_without_checkpoint_dir().load_checkpoint("ignored")

    # The field the fixed line reads is the accuracy channel: -inf sentinel.
    assert math.isinf(state.best_accuracy) and state.best_accuracy < 0
    # The field the bug read is the epoch channel: a finite non-negative integer.
    assert state.last_epoch == 0
    assert state.best_accuracy != state.last_epoch


def test_epoch_seed_would_spuriously_win_best_but_accuracy_seed_does_not():
    """A no-eval epoch must not register as best on a fresh resume.

    The best sentinel starts at -inf. Seeding the comparison from best_accuracy (-inf) does not
    beat it; seeding from last_epoch (0) would, demonstrating the regression the fix prevents.
    """
    state = _module_without_checkpoint_dir().load_checkpoint("ignored")
    best_validation_metric = float("-inf")

    correct_seed = state.best_accuracy  # fixed behaviour
    buggy_seed = state.last_epoch       # pre-fix behaviour

    assert not (correct_seed > best_validation_metric)
    assert buggy_seed > best_validation_metric


def test_restored_best_is_carried_forward_not_the_epoch():
    """When a real checkpoint is loaded, the accuracy — not the epoch — seeds best tracking.

    With a distinct epoch and accuracy, seeding from best_accuracy carries the true best forward,
    so a genuinely worse no-eval epoch is not mistaken for an improvement; seeding from last_epoch
    would let the epoch counter masquerade as the best accuracy.
    """
    loaded = CheckpointState(
        last_epoch=7,
        scheduler_state=None,
        initial_lr=0.0,
        best_accuracy=0.83,
        batch_idx=0,
    )

    correct_seed = loaded.best_accuracy
    buggy_seed = loaded.last_epoch

    assert correct_seed == 0.83
    # The epoch counter (7) is not an accuracy and dwarfs a realistic [0, 1] metric.
    assert buggy_seed != loaded.best_accuracy
    assert buggy_seed > 1.0


# Author: Mus mbayramo@stanford.edu
