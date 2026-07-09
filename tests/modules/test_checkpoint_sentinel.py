"""Offline regression for the ``best_accuracy`` sentinel in ``IgcModule.load_checkpoint``.

The early-return checkpoint states seed ``best_accuracy`` with the "no best yet" sentinel.
Best-accuracy tracking is higher-is-better (``validation_accuracy > best``), so the sentinel
must be ``-inf`` (the first real accuracy beats it). A prior ``-float('-inf')`` typo evaluated
to ``+inf``, which would make no accuracy ever register as best once the field is consumed on
resume. This test pins the sentinel to negative infinity, consistent with the loaded-value
default. Runs on CPU with no checkpoint files.

Author:
Mus mbayramo@stanford.edu
"""
import math

from igc.modules.base.igc_base_module import IgcModule


def _module_without_checkpoint_dir():
    """An IgcModule with no checkpoint dir (bypasses the heavy __init__)."""
    module = IgcModule.__new__(IgcModule)
    module._module_checkpoint_dir = None
    return module


def test_no_checkpoint_dir_seeds_negative_infinity_best_accuracy():
    """With no checkpoint dir, best_accuracy is -inf so the first accuracy wins."""
    state = _module_without_checkpoint_dir().load_checkpoint("ignored")
    assert math.isinf(state.best_accuracy)
    assert state.best_accuracy < 0
    assert state.best_accuracy == float("-inf")


def test_sentinel_loses_to_any_finite_accuracy():
    """The seeded sentinel is strictly less than any finite validation accuracy."""
    state = _module_without_checkpoint_dir().load_checkpoint("ignored")
    for accuracy in (-1.0, 0.0, 0.5, 1.0, 1e9):
        assert accuracy > state.best_accuracy


# Author: Mus mbayramo@stanford.edu
