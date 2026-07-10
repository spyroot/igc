"""Regression tests for device resolution in ``shared_main``.

The ``--device`` argument defaults to the string ``"auto"``, which is not a
valid ``torch.device`` string. ``shared_main`` must resolve ``"auto"`` (and an
unset device) to a concrete cuda/mps/cpu device before any module calls
``torch.device(args.device)``; a real device string such as ``"cpu"`` must be
left untouched.

Author:
Mus mbayramo@stanford.edu
"""

import sys

import pytest
import torch

import igc.shared.shared_main as shared_main_mod
from igc.shared.shared_main import shared_main


@pytest.fixture
def resolved_device(monkeypatch: pytest.MonkeyPatch) -> torch.device:
    """Patch get_device with a sentinel so resolution is observable."""
    sentinel = torch.device("meta")
    monkeypatch.setattr(shared_main_mod, "get_device", lambda *a, **k: sentinel)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    return sentinel


def _run(monkeypatch, tmp_path, extra_argv):
    """Invoke shared_main with a minimal argv and a tmp output_dir."""
    argv = ["igc_main.py", "--output_dir", str(tmp_path), *extra_argv]
    monkeypatch.setattr(sys, "argv", argv)
    args, _ = shared_main(is_cuda_empty_cache=False)
    return args


def test_default_device_auto_is_resolved(monkeypatch, tmp_path, resolved_device):
    """The parser default 'auto' is replaced with a concrete device."""
    args = _run(monkeypatch, tmp_path, [])
    assert args.device == resolved_device
    assert args.device != "auto"


def test_explicit_auto_is_resolved(monkeypatch, tmp_path, resolved_device):
    """An explicit --device auto is resolved the same way as the default."""
    args = _run(monkeypatch, tmp_path, ["--device", "auto"])
    assert args.device == resolved_device


def test_explicit_cpu_is_left_untouched(monkeypatch, tmp_path, resolved_device):
    """A concrete --device cpu is not overwritten by get_device()."""
    args = _run(monkeypatch, tmp_path, ["--device", "cpu"])
    assert args.device == "cpu"
    assert args.device != resolved_device


def test_resolved_device_is_a_valid_torch_device(monkeypatch, tmp_path):
    """The resolved default is directly usable by torch.device()."""
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    args = _run(monkeypatch, tmp_path, [])
    # torch.device() must accept the resolved value without raising.
    assert isinstance(torch.device(args.device), torch.device)


# Author: Mus mbayramo@stanford.edu


def test_local_rank_launch_resolves_auto_via_rank(monkeypatch, tmp_path):
    """Under accelerate launch/torchrun (LOCAL_RANK set), 'auto' resolves per-rank."""
    seen = {}

    def fake_get_device(rank=None):
        seen["rank"] = rank
        return torch.device("meta")

    monkeypatch.setattr(shared_main_mod, "get_device", fake_get_device)
    monkeypatch.setenv("LOCAL_RANK", "1")
    args = _run(monkeypatch, tmp_path, [])
    assert args.device == torch.device("meta")
    assert seen["rank"] == 1


def test_local_rank_launch_keeps_explicit_device(monkeypatch, tmp_path, resolved_device):
    """An explicit --device cpu survives a distributed launch untouched."""
    monkeypatch.setenv("LOCAL_RANK", "0")
    args = _run(monkeypatch, tmp_path, ["--device", "cpu"])
    assert args.device == "cpu"
