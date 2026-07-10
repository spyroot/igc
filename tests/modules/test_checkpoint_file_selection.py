"""Offline coverage for ``IgcModule`` checkpoint file selection.

These tests complement the save/resume regressions by pinning how explicit
checkpoint paths, checkpoint directories, and copy-to-final-model selection are
resolved before a checkpoint is loaded.
"""

import argparse
import os
from pathlib import Path

import loguru
import torch
import torch.nn as nn

from igc.modules.base.igc_base_module import IgcModule


def _module(tmp_path: Path) -> IgcModule:
    """Build a minimal CPU-only ``IgcModule`` shell for selection tests."""
    module = IgcModule.__new__(IgcModule)
    module._module_checkpoint_dir = str(tmp_path)
    module.module_name = "tmod"
    module.logger = loguru.logger
    module.model = nn.Linear(2, 2)
    module.optimizer = torch.optim.SGD(module.model.parameters(), lr=0.1)
    module.scheduler = None
    module.rank = 0
    return module


def _write_checkpoint(path: Path, model: nn.Module, *, epoch: int) -> None:
    """Write the minimal resumable checkpoint payload."""
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": torch.optim.SGD(
                model.parameters(), lr=0.1
            ).state_dict(),
            "epoch": epoch,
            "is_trained": True,
        },
        path,
    )


def test_last_checkpoint_chooses_newest_resumable_epoch_file(
        tmp_path: Path) -> None:
    """Among resumable files, the newest mtime wins."""
    model = nn.Linear(2, 2)
    older = tmp_path / "tmod_epoch_1.pt"
    newer = tmp_path / "tmod_epoch_2.pt"
    _write_checkpoint(older, model, epoch=1)
    _write_checkpoint(newer, model, epoch=2)
    os.utime(older, (10, 10))
    os.utime(newer, (20, 20))

    assert IgcModule.last_checkpoint(str(tmp_path)) == str(newer)


def test_last_checkpoint_returns_none_without_pt_candidates(
        tmp_path: Path) -> None:
    """Directories with no checkpoint candidates return ``None``."""
    (tmp_path / "notes.txt").write_text("not a checkpoint", encoding="utf-8")

    assert IgcModule.last_checkpoint(str(tmp_path)) is None


def test_checkpoint_file_returns_explicit_file_path(tmp_path: Path) -> None:
    """A direct checkpoint file path is returned without directory scanning."""
    module = _module(tmp_path)
    checkpoint = tmp_path / "manual.pt"
    _write_checkpoint(checkpoint, module.model, epoch=7)

    assert module.checkpoint_file(str(checkpoint)) == str(checkpoint)


def test_checkpoint_file_empty_directory_returns_none(tmp_path: Path) -> None:
    """An empty checkpoint directory resolves to ``None``."""
    module = _module(tmp_path)

    assert module.checkpoint_file(str(tmp_path)) is None


def test_checkpoint_file_non_resuming_still_selects_checkpoint(
        tmp_path: Path) -> None:
    """When a checkpoint exists, non-resume mode still returns that file."""
    module = _module(tmp_path)
    checkpoint = tmp_path / "tmod_epoch_1.pt"
    _write_checkpoint(checkpoint, module.model, epoch=1)

    assert module.checkpoint_file(str(tmp_path), resuming=False) == str(checkpoint)


def test_copy_checkpoint_uses_explicit_checkpoint_directory(
        tmp_path: Path) -> None:
    """An explicit checkpoint path writes the final model beside that file."""
    source = nn.Linear(2, 2)
    checkpoint = tmp_path / "picked.pt"
    _write_checkpoint(checkpoint, source, epoch=9)
    target = nn.Linear(2, 2)

    model_payload, epoch, last_model_path = IgcModule.copy_checkpoint(
        str(tmp_path),
        "tmod",
        target,
        checkpoint_file=str(checkpoint),
        device="cpu",
    )

    assert model_payload is not None
    assert epoch == 9
    assert last_model_path == str(tmp_path / "tmod_last.pt")
    assert os.path.exists(last_model_path)
    assert torch.allclose(target.weight, source.weight)
    assert torch.allclose(target.bias, source.bias)
    assert target.training is False
    assert not any(param.requires_grad for param in target.parameters())
    saved = torch.load(last_model_path, weights_only=True)
    assert "optimizer_state_dict" not in saved
    assert "scheduler_state_dict" not in saved


def test_copy_checkpoint_discovers_module_subdir_checkpoint(
        tmp_path: Path) -> None:
    """Without an explicit file, copy_checkpoint scans ``<output>/<module>``."""
    module_dir = tmp_path / "tmod"
    module_dir.mkdir()
    source = nn.Linear(2, 2)
    checkpoint = module_dir / "tmod_epoch_3.pt"
    _write_checkpoint(checkpoint, source, epoch=3)
    target = nn.Linear(2, 2)
    specs = argparse.Namespace(output_dir=str(tmp_path))

    model_payload, epoch, last_model_path = IgcModule.copy_checkpoint(
        specs, "tmod", target, device="cpu"
    )

    assert model_payload is not None
    assert epoch == 3
    assert last_model_path == str(tmp_path / "tmod_last.pt")
    assert os.path.exists(last_model_path)
    assert torch.allclose(target.weight, source.weight)
    assert torch.allclose(target.bias, source.bias)
    assert target.training is False


def test_copy_checkpoint_returns_none_without_module_candidates(
        tmp_path: Path) -> None:
    """An empty ``<output>/<module>`` directory has no checkpoint to copy."""
    (tmp_path / "tmod").mkdir()
    target = nn.Linear(2, 2)
    specs = argparse.Namespace(output_dir=str(tmp_path))

    model_payload, epoch, last_model_path = IgcModule.copy_checkpoint(
        specs, "tmod", target, device="cpu"
    )

    assert model_payload is None
    assert epoch == 0
    assert last_model_path == str(tmp_path / "tmod_last.pt")
