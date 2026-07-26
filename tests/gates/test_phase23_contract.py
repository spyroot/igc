"""Regression coverage for the Phase 2/3 project contract gate."""

from __future__ import annotations

import importlib.util
from pathlib import Path


GATE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "gates" / "phase23_contract.py"


def _gate_module():
    spec = importlib.util.spec_from_file_location("phase23_contract_gate", GATE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_phase23_contract_gate_has_no_violations() -> None:
    """All six forbidden regression families remain rejected."""
    assert _gate_module().run_gate() == []
