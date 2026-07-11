"""Offline regressions for device resolution in ``IgcBaseState.__init__``.

Pins the resolution precedence: accelerator > explicit ``device`` argument >
``spec.device`` > ``get_device()`` fallback. Previously the explicit argument
was silently clobbered by ``spec.device`` (the shared parser always defines
it), and the no-device fallback branch crashed twice over — it logged
``self.device`` before ``_device`` was assigned (AttributeError) and ended in
a bare ``raise`` with no active exception. Runs on CPU only.

Author:
Mus mbayramo@stanford.edu
"""

import argparse

import pytest
import torch

import igc.modules.base.igc_state as igc_state_mod
from igc.modules.base.igc_state import IgcBaseState


def _spec(tmp_path, **kwargs) -> argparse.Namespace:
    """Minimal spec Namespace with a tmp log_dir plus the given fields."""
    return argparse.Namespace(log_dir=str(tmp_path), **kwargs)


def test_explicit_device_arg_wins_over_spec_device(tmp_path):
    """A caller-passed device is honored even when spec.device differs."""
    state = IgcBaseState("t", _spec(tmp_path, device="cpu"), device=torch.device("meta"))
    assert state.device == torch.device("meta")


def test_spec_device_used_when_no_explicit_arg(tmp_path):
    """Without an explicit arg, spec.device resolves to a torch.device."""
    state = IgcBaseState("t", _spec(tmp_path, device="cpu"))
    assert state.device == torch.device("cpu")
    assert isinstance(state.device, torch.device)


def test_missing_spec_device_falls_back_without_crash(tmp_path, monkeypatch):
    """A spec lacking a device attr resolves via get_device — no bare raise."""
    sentinel = torch.device("meta")
    monkeypatch.setattr(igc_state_mod, "get_device", lambda *a, **k: sentinel)
    monkeypatch.delenv("LOCAL_RANK", raising=False)

    state = IgcBaseState("t", _spec(tmp_path))

    assert state.device == sentinel


def test_none_spec_device_falls_back_without_crash(tmp_path, monkeypatch):
    """spec.device=None (accelerator-style spec) also takes the fallback."""
    sentinel = torch.device("meta")
    monkeypatch.setattr(igc_state_mod, "get_device", lambda *a, **k: sentinel)

    state = IgcBaseState("t", _spec(tmp_path, device=None))

    assert state.device == sentinel


def test_non_namespace_spec_rejected(tmp_path):
    """A non-Namespace spec still raises TypeError up front."""
    with pytest.raises(TypeError):
        IgcBaseState("t", {"device": "cpu"})


# Author: Mus mbayramo@stanford.edu


def test_device_property_tolerates_partial_construction():
    """A __new__-built instance (no _accelerator/_device) does not crash on .device."""
    obj = IgcBaseState.__new__(IgcBaseState)
    # neither _accelerator nor _device set — the property must return None, not raise.
    assert obj.device is None
