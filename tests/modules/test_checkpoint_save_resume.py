"""Offline regressions for the checkpoint save/resume path in ``IgcModule``.

Pins the fixes for the audit's checkpoint cluster: optimizer state must be
saved when the trainer owns an optimizer (the guard previously tested the
``optimizer`` parameter instead of the resolved ``_optimizer``, so plain-path
checkpoints could never resume); ``batch_idx`` must store the batch index (a
copy-paste stored ``initial_lr``); ``last_checkpoint`` must prefer a real
epoch checkpoint over the newer weights-only ``{module}_last.pt`` written by
``save_model``; ``can_resume`` must actually return True; and the train/eval
splits must be seeded (stable across runs/ranks) with ``split_slice_dataset``
sizing computed from the sampled subset, not the full dataset. CPU-only.

Author:
Mus mbayramo@stanford.edu
"""

import argparse
import os
import time

import loguru
import torch
from torch.utils.data import TensorDataset

from igc.modules.base.igc_base_module import IgcModule


def _module(tmp_path, with_optimizer=True):
    """A minimal IgcModule (bypasses the heavy __init__) for save/split calls."""
    m = IgcModule.__new__(IgcModule)
    m.rank = 0
    m.module_name = "tmod"
    m.logger = loguru.logger
    m.model = torch.nn.Linear(2, 2)
    m.optimizer = (
        torch.optim.SGD(m.model.parameters(), lr=0.1) if with_optimizer else None
    )
    m.scheduler = None
    m.dataset = TensorDataset(torch.arange(100, dtype=torch.float32))
    m._trainer_args = argparse.Namespace(seed=42, output_dir=str(tmp_path))
    return m


def test_save_checkpoint_includes_trainer_optimizer_state(tmp_path):
    """The trainer's own optimizer is saved even when none is passed in."""
    m = _module(tmp_path)
    path = m.save_checkpoint(str(tmp_path), epoch=1)
    checkpoint = torch.load(path, weights_only=True)
    assert "optimizer_state_dict" in checkpoint


def test_save_checkpoint_stores_batch_idx_not_lr(tmp_path):
    """batch_idx stores the batch index, not the initial_lr value."""
    m = _module(tmp_path)
    path = m.save_checkpoint(str(tmp_path), epoch=1, initial_lr=0.5, batch_idx=7)
    checkpoint = torch.load(path, weights_only=True)
    assert checkpoint["batch_idx"] == 7
    assert checkpoint["initial_lr"] == 0.5


def test_last_checkpoint_prefers_epoch_file_over_newer_last_pt(tmp_path):
    """Resume picks the epoch checkpoint even when {module}_last.pt is newer."""
    epoch_file = tmp_path / "tmod_epoch_1.pt"
    torch.save({"model_state_dict": {}, "epoch": 1}, epoch_file)
    time.sleep(0.02)
    last_file = tmp_path / "tmod_last.pt"
    torch.save({"model_state_dict": {}}, last_file)
    os.utime(last_file)  # ensure strictly newer mtime

    assert IgcModule.last_checkpoint(str(tmp_path)) == str(epoch_file)


def test_last_checkpoint_falls_back_to_last_pt(tmp_path):
    """With only the weights-only file present, it is still returned."""
    last_file = tmp_path / "tmod_last.pt"
    torch.save({"model_state_dict": {}}, last_file)
    assert IgcModule.last_checkpoint(str(tmp_path)) == str(last_file)


def test_can_resume_true_when_model_file_exists(tmp_path):
    """can_resume returns True (not None) once the module file exists."""
    torch.save({}, tmp_path / "tmod_last.pt")
    assert IgcModule.can_resume(str(tmp_path), "tmod") is True


def test_can_resume_false_when_missing(tmp_path):
    """can_resume stays False when no module file exists."""
    assert IgcModule.can_resume(str(tmp_path), "tmod") is False


def test_split_dataset_is_deterministic_for_a_seed(tmp_path):
    """Two split calls with the same spec seed produce identical partitions."""
    first = _module(tmp_path).split_dataset(ratio=0.8)
    second = _module(tmp_path).split_dataset(ratio=0.8)
    assert list(first[0].indices) == list(second[0].indices)
    assert list(first[1].indices) == list(second[1].indices)


def test_split_slice_dataset_sizes_come_from_the_subset(tmp_path):
    """The sampled-subset split no longer raises and sums to the sample size."""
    train, evals = _module(tmp_path).split_slice_dataset(
        train_ratio=0.8, sample_ratio=0.1
    )
    assert len(train) + len(evals) == 10  # 10% of 100
    assert len(train) == 8


# Author: Mus mbayramo@stanford.edu
