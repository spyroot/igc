"""Broad offline edges for ``IgcModule`` checkpoint save/resume behavior.

These tests extend the checkpoint cluster coverage without touching engine
code: checkpoint slot rotation, best-checkpoint naming, nonzero-rank no-op
saves, list scheduler state, and seeded split determinism across ratios.

Author:
Mus mbayramo@stanford.edu
"""

import argparse
from pathlib import Path

import loguru
import pytest
import torch
from torch.utils.data import TensorDataset

from igc.modules.base.igc_base_module import IgcModule


def _module(tmp_path: Path, *, rank: int = 0) -> IgcModule:
    """Build a minimal CPU-only ``IgcModule`` shell for checkpoint tests."""
    module = IgcModule.__new__(IgcModule)
    module.rank = rank
    module.module_name = "tmod"
    module.logger = loguru.logger
    module.model = torch.nn.Linear(2, 2)
    module.optimizer = torch.optim.SGD(module.model.parameters(), lr=0.1)
    module.scheduler = None
    module.dataset = TensorDataset(torch.arange(100, dtype=torch.float32))
    module._trainer_args = argparse.Namespace(seed=42, output_dir=str(tmp_path))
    module._module_checkpoint_dir = str(tmp_path)
    return module


def test_save_checkpoint_rotates_epoch_slots_by_keep_count(tmp_path: Path) -> None:
    """Epoch checkpoints wrap by ``num_check_points_to_keep`` and overwrite."""
    module = _module(tmp_path)

    first_slot = Path(
        module.save_checkpoint(str(tmp_path), epoch=1, num_check_points_to_keep=3)
    )
    second_slot = Path(
        module.save_checkpoint(str(tmp_path), epoch=2, num_check_points_to_keep=3)
    )
    wrapped_slot = Path(
        module.save_checkpoint(str(tmp_path), epoch=4, num_check_points_to_keep=3)
    )

    assert first_slot == wrapped_slot
    assert first_slot.name == "tmod_epoch_1.pt"
    assert second_slot.name == "tmod_epoch_2.pt"
    assert torch.load(first_slot, weights_only=True)["epoch"] == 4


def test_save_checkpoint_best_accuracy_uses_stable_best_filename(
        tmp_path: Path) -> None:
    """Best-accuracy saves use the stable best filename and keep the metric."""
    module = _module(tmp_path)

    checkpoint_path = Path(
        module.save_checkpoint(
            str(tmp_path),
            epoch=5,
            is_best_accuracy=True,
            last_accuracy=0.875,
        )
    )
    checkpoint = torch.load(checkpoint_path, weights_only=True)

    assert checkpoint_path.name == "tmod_epoch_best.pt"
    assert checkpoint["epoch"] == 5
    assert checkpoint["last_accuracy"] == pytest.approx(0.875)


def test_save_checkpoint_rank_greater_than_zero_is_noop(tmp_path: Path) -> None:
    """Nonzero ranks do not write checkpoint files from local save calls."""
    module = _module(tmp_path, rank=1)

    result = module.save_checkpoint(str(tmp_path), epoch=3)

    assert result == ""
    assert list(tmp_path.glob("*.pt")) == []


def test_load_checkpoint_restores_list_scheduler_state(tmp_path: Path) -> None:
    """List scheduler state dicts are saved and restored during resume."""
    source = _module(tmp_path)
    source_schedulers = [
        torch.optim.lr_scheduler.StepLR(source.optimizer, step_size=2, gamma=0.5),
        torch.optim.lr_scheduler.ExponentialLR(source.optimizer, gamma=0.9),
    ]
    source.optimizer.step()
    for scheduler in source_schedulers:
        scheduler.step()

    source.save_checkpoint(str(tmp_path), epoch=6, scheduler=source_schedulers)

    target = _module(tmp_path)
    target.scheduler = [
        torch.optim.lr_scheduler.StepLR(target.optimizer, step_size=2, gamma=0.5),
        torch.optim.lr_scheduler.ExponentialLR(target.optimizer, gamma=0.9),
    ]
    state = target.load_checkpoint(str(tmp_path), resuming=True)

    assert state.last_epoch == 6
    assert isinstance(state.scheduler_state, list)
    assert len(state.scheduler_state) == 2
    assert target.scheduler[0].state_dict()["last_epoch"] == 1
    assert target.scheduler[1].state_dict()["last_epoch"] == 1


@pytest.mark.parametrize("ratio", [0.25, 0.5, 0.75])
def test_split_dataset_is_seeded_across_ratios(
        tmp_path: Path, ratio: float) -> None:
    """Seeded train/eval partitions stay deterministic for common ratios."""
    first_train, first_eval = _module(tmp_path).split_dataset(ratio=ratio)
    second_train, second_eval = _module(tmp_path).split_dataset(ratio=ratio)

    assert list(first_train.indices) == list(second_train.indices)
    assert list(first_eval.indices) == list(second_eval.indices)
    assert len(first_train) == int(100 * ratio)
    assert len(first_train) + len(first_eval) == 100


# Author: Mus mbayramo@stanford.edu
