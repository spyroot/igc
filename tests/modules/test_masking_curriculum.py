"""Offline regressions for the masking curriculum in the encoder trainer.

The curriculum was inert three ways: activation required a per-micro-batch
counter to equal zero (never after epoch one), the enable path indexed the
enum LIST with an enum (TypeError) instead of the dispatcher dict, and
deactivation compared the counter with ``==`` to a boundary it skips over.
Drives ``swap_masking_method``/``enable_masking_method`` on a stubbed trainer.

Author:
Mus mbayramo@stanford.edu
"""

import pytest

from igc.modules.llm_train_state_encoder import LlmEmbeddingsTrainer


class _Dataset:
    def __init__(self):
        self.disabled = 0

    def disable_masking(self):
        self.disabled += 1


def _trainer(freq=2, passes=10):
    trainer = LlmEmbeddingsTrainer.__new__(LlmEmbeddingsTrainer)
    trainer._masked_freq = freq
    trainer._num_mask_passed = passes
    trainer._current_mask_method_idx = 0
    trainer._current_mask_method_counter = 0
    trainer.dataset = _Dataset()
    trainer.calls = []
    trainer._masking_method_dispatcher = {
        "mask_a": lambda: trainer.calls.append("mask_a"),
        "mask_b": lambda: trainer.calls.append("mask_b"),
    }
    import loguru
    trainer.logger = loguru.logger
    return trainer


def test_mask_activates_on_frequency_boundary_even_mid_run():
    """Activation no longer requires the batch counter to be zero."""
    trainer = _trainer(freq=2)
    trainer._current_mask_method_counter = 4321  # mid-run counter state
    trainer.swap_masking_method(epoch=1, mask_type=["mask_a", "mask_b"])
    assert trainer.calls == ["mask_a"]
    assert trainer._current_mask_method_idx == 1
    assert trainer._current_mask_method_counter == 0


def test_curriculum_cycles_through_methods():
    """Consecutive boundaries walk the ordered method list cyclically."""
    trainer = _trainer(freq=1)
    for epoch in range(3):
        trainer.swap_masking_method(epoch, ["mask_a", "mask_b"])
    assert trainer.calls == ["mask_a", "mask_b", "mask_a"]


def test_mask_disables_after_enough_batches():
    """Off-boundary epochs disable masking once the pass budget is exceeded."""
    trainer = _trainer(freq=5, passes=10)
    trainer._current_mask_method_counter = 17  # advanced past the budget
    trainer.swap_masking_method(epoch=0, mask_type=["mask_a"])
    assert trainer.dataset.disabled == 1
    assert trainer._current_mask_method_counter == 0


def test_unknown_mask_raises():
    """An enum missing from the dispatcher fails loudly, not with TypeError."""
    trainer = _trainer()
    with pytest.raises(ValueError, match="Unknown masking type"):
        trainer.enable_masking_method("nope")


def test_no_mask_list_is_a_noop():
    """The smoke path passes mask_type=[] — nothing activates or disables."""
    trainer = _trainer(freq=1)
    trainer.swap_masking_method(epoch=0, mask_type=[])
    assert trainer.calls == [] and trainer.dataset.disabled == 0


# Author: Mus mbayramo@stanford.edu
