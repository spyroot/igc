"""Offline regression for consuming ``load_checkpoint`` as a resume epoch.

``IgcModule.load_checkpoint`` returns a ``CheckpointState`` namedtuple, and the
resume epoch is its ``last_epoch`` field — not the state object itself. The
autoencoder trainer's ``for epoch in range(last_epoch, self.num_epochs)`` loop
previously bound ``last_epoch`` to the whole namedtuple, raising
``TypeError: 'Checkpoint' object cannot be interpreted as an integer`` on every
launch. This pins the field to an int usable in ``range``. Runs on CPU with no
checkpoint files.

Author:
Mus mbayramo@stanford.edu
"""

from igc.modules.base.igc_base_module import IgcModule


def _module_without_checkpoint_dir():
    """An IgcModule with no checkpoint dir (bypasses the heavy __init__)."""
    module = IgcModule.__new__(IgcModule)
    module._module_checkpoint_dir = None
    return module


def test_no_checkpoint_last_epoch_is_int_zero():
    """With no checkpoint, last_epoch is the int 0, not the state object."""
    state = _module_without_checkpoint_dir().load_checkpoint("ignored")
    assert isinstance(state.last_epoch, int)
    assert state.last_epoch == 0


def test_last_epoch_is_usable_as_range_start():
    """range(state.last_epoch, n) must iterate — the crashed training loop."""
    state = _module_without_checkpoint_dir().load_checkpoint("ignored")
    epochs = list(range(state.last_epoch, 3))
    assert epochs == [0, 1, 2]


# Author: Mus mbayramo@stanford.edu
