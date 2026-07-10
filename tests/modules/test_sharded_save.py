"""Offline regressions for sharded-run checkpointing support.

Under ZeRO-3/FSDP a rank-0-only ``unwrap_model().state_dict()`` deadlocks (the
gather is collective) or writes shards. The trainer now gathers via
``accelerator.get_state_dict`` on every rank behind a rank-0 verdict made
uniform by ``broadcast_flag``, and ``IgcModule.save_checkpoint`` accepts the
pre-gathered ``model_state_dict`` instead of re-calling ``state_dict()``.
CPU-only; single-process behavior plus the saver override are pinned here —
the true multi-rank path runs on the cluster rung (gpu-marked).

Author:
Mus mbayramo@stanford.edu
"""

import argparse

import loguru
import torch

from igc.modules.base.igc_base_module import IgcModule
from igc.shared.shared_accelerator import broadcast_flag


class _SingleProcessAccelerator:
    num_processes = 1


def test_broadcast_flag_single_process_is_identity():
    """With one process there is nothing to synchronize."""
    accelerator = _SingleProcessAccelerator()
    assert broadcast_flag(accelerator, True) is True
    assert broadcast_flag(accelerator, False) is False


def _module(tmp_path):
    module = IgcModule.__new__(IgcModule)
    module.rank = 0
    module.module_name = "tmod"
    module.logger = loguru.logger
    module.model = torch.nn.Linear(2, 2)
    module.optimizer = None
    module.scheduler = None
    module._trainer_args = argparse.Namespace(seed=1, output_dir=str(tmp_path))
    return module


def test_save_checkpoint_uses_pregathered_state_dict(tmp_path):
    """A provided model_state_dict is saved verbatim, not model.state_dict()."""
    module = _module(tmp_path)
    gathered = {"weight": torch.ones(2, 2), "bias": torch.zeros(2)}

    path = module.save_checkpoint(
        str(tmp_path), epoch=1, model_state_dict=gathered
    )
    checkpoint = torch.load(path, weights_only=True)

    assert torch.equal(checkpoint["model_state_dict"]["weight"], torch.ones(2, 2))


def test_save_checkpoint_falls_back_to_model_state(tmp_path):
    """Without the override the model's own state dict is saved (plain path)."""
    module = _module(tmp_path)
    path = module.save_checkpoint(str(tmp_path), epoch=1)
    checkpoint = torch.load(path, weights_only=True)
    assert set(checkpoint["model_state_dict"]) == {"weight", "bias"}


# Author: Mus mbayramo@stanford.edu
