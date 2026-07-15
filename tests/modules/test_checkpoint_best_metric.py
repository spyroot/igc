"""Offline regressions for best-metric persistence across checkpoint save/resume.

Pins the Phase 1 resume fix: a loss-mode run must persist its best validation
metric (`best_metric` + `best_metric_mode` in the checkpoint, both created by
``IgcModule.save_checkpoint``) and restore it on resume via
``restored_best_metric`` — otherwise ``_best_validation_metric`` restarts at
``+inf`` and the first post-resume eval is always flagged "best", overwriting
the true best checkpoint and resetting early-stop patience. Also pins
``resolve_early_stopping`` honoring an explicit ``0`` (a bare ``or`` default
silently swallowed it). CPU-only, no GPU/network/downloads.

Author:
Mus mbayramo@stanford.edu
"""

import argparse

import loguru
import torch
from torch.utils.data import TensorDataset

from igc.modules.base.igc_base_module import CheckpointState, IgcModule
from igc.modules.llm_train_state_encoder import (
    resolve_early_stopping,
    restored_best_metric,
)


def _module(tmp_path):
    """A minimal IgcModule (bypasses the heavy __init__) for checkpoint calls."""
    m = IgcModule.__new__(IgcModule)
    m.rank = 0
    m.module_name = "tmod"
    m.logger = loguru.logger
    m.model = torch.nn.Linear(2, 2)
    m.optimizer = torch.optim.SGD(m.model.parameters(), lr=0.1)
    m.scheduler = None
    m.dataset = TensorDataset(torch.arange(10, dtype=torch.float32))
    m._trainer_args = argparse.Namespace(seed=42, output_dir=str(tmp_path))
    m._module_checkpoint_dir = str(tmp_path)
    return m


def test_save_checkpoint_persists_best_metric_and_mode(tmp_path):
    """best_metric/best_metric_mode round-trip through the checkpoint file."""
    m = _module(tmp_path)
    path = m.save_checkpoint(
        str(tmp_path), epoch=1, best_metric=1.25, best_metric_mode="loss")
    checkpoint = torch.load(path, weights_only=True)
    assert checkpoint["best_metric"] == 1.25
    assert checkpoint["best_metric_mode"] == "loss"


def test_load_checkpoint_returns_best_metric_fields(tmp_path):
    """load_checkpoint surfaces the persisted best metric in CheckpointState."""
    m = _module(tmp_path)
    m.save_checkpoint(str(tmp_path), epoch=1, best_metric=0.75, best_metric_mode="loss")
    state = m.load_checkpoint(str(tmp_path), map_location="cpu")
    assert state.best_metric == 0.75
    assert state.best_metric_mode == "loss"


def test_load_legacy_checkpoint_defaults_best_metric_none(tmp_path):
    """A checkpoint written without the new channel loads with None defaults."""
    m = _module(tmp_path)
    m.save_checkpoint(str(tmp_path), epoch=1, last_accuracy=42.0)
    state = m.load_checkpoint(str(tmp_path), map_location="cpu")
    assert state.best_metric is None
    assert state.best_metric_mode is None
    assert state.best_accuracy == 42.0


def test_checkpoint_state_five_positional_still_constructs():
    """Legacy 5-positional CheckpointState construction keeps working."""
    state = CheckpointState(0, None, 0, float('-inf'), 0)
    assert state.best_metric is None
    assert state.best_metric_mode is None


def test_restored_best_metric_loss_mode_matches():
    """Loss mode restores a loss-mode best_metric."""
    state = CheckpointState(3, None, 1e-4, 55.0, 0, 1.5, "loss")
    assert restored_best_metric(state, select_by_loss=True) == 1.5


def test_restored_best_metric_rejects_mode_mismatch():
    """An accuracy-mode best never seeds loss-mode tracking (and vice versa)."""
    acc_state = CheckpointState(3, None, 1e-4, 55.0, 0, 55.0, "accuracy")
    assert restored_best_metric(acc_state, select_by_loss=True) is None
    loss_state = CheckpointState(3, None, 1e-4, float('-inf'), 0, 1.5, "loss")
    assert restored_best_metric(loss_state, select_by_loss=False) is None


def test_restored_best_metric_loss_mode_legacy_starts_fresh():
    """Loss mode has no legacy channel: a legacy checkpoint restores nothing."""
    state = CheckpointState(3, None, 1e-4, 55.0, 0)
    assert restored_best_metric(state, select_by_loss=True) is None


def test_restored_best_metric_accuracy_mode_legacy_fallback():
    """Accuracy mode falls back to the legacy finite best_accuracy channel."""
    state = CheckpointState(3, None, 1e-4, 55.0, 0)
    assert restored_best_metric(state, select_by_loss=False) == 55.0


def test_restored_best_metric_filters_non_finite():
    """Sentinel -inf/+inf values never seed best tracking."""
    inf_best = CheckpointState(3, None, 1e-4, float('-inf'), 0, float('inf'), "loss")
    assert restored_best_metric(inf_best, select_by_loss=True) is None
    assert restored_best_metric(inf_best, select_by_loss=False) is None


def test_resolve_early_stopping_defaults():
    """Missing/None knobs fall back to the Phase 1 defaults."""
    assert resolve_early_stopping(argparse.Namespace()) == (3, 0.005)
    spec = argparse.Namespace(
        early_stopping_patience=None, early_stopping_min_delta=None)
    assert resolve_early_stopping(spec) == (3, 0.005)


def test_resolve_early_stopping_honors_explicit_zero():
    """An explicit 0 disables patience / zeroes min_delta instead of defaulting."""
    spec = argparse.Namespace(
        early_stopping_patience=0, early_stopping_min_delta=0.0)
    assert resolve_early_stopping(spec) == (0, 0.0)


def test_resolve_early_stopping_explicit_values():
    """Explicit non-default knobs pass through unchanged."""
    spec = argparse.Namespace(
        early_stopping_patience=7, early_stopping_min_delta=0.01)
    assert resolve_early_stopping(spec) == (7, 0.01)


# Author: Mus mbayramo@stanford.edu
