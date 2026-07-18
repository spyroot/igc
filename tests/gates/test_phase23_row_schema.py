"""Offline tests for the machine-readable Phase 2/3 row-schema gate.

The committed JSON schemas are the authoritative row shapes; these tests prove
the real builders conform and that representative drift (extra keys, leakage
into x, allowed_methods echoed into a Call, scalar/list unions) fails.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

SCRIPT = Path("scripts/gates/phase23_row_schema.py")


def _load_gate() -> ModuleType:
    """Load the gate script module for direct testing."""
    spec = importlib.util.spec_from_file_location("phase23_row_schema", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_builder_rows_conform_to_committed_schemas() -> None:
    """The canonical builder rows validate against both committed schemas."""
    gate = _load_gate()
    rows = gate.canonical_rows()

    assert gate.validate_schema(rows["phase2"], gate.load_schema(gate.PHASE2_SCHEMA)) == []
    assert gate.validate_schema(rows["phase3"], gate.load_schema(gate.PHASE3_SCHEMA)) == []
    assert gate.check() == 0


def test_extra_top_level_key_fails() -> None:
    """An unknown top-level key (schema drift) is rejected."""
    gate = _load_gate()
    row = copy.deepcopy(gate.canonical_rows()["phase2"])
    row["order_evidence"] = "explicit_then"

    violations = gate.validate_schema(row, gate.load_schema(gate.PHASE2_SCHEMA))
    assert any("order_evidence" in v for v in violations)


def test_phase2_x_leakage_fails() -> None:
    """json/allowed_methods inside Phase 2 x is input leakage and fails."""
    gate = _load_gate()
    row = copy.deepcopy(gate.canonical_rows()["phase2"])
    row["x"]["json"] = [{"@odata.id": "/redfish/v1/Systems"}]

    violations = gate.validate_schema(row, gate.load_schema(gate.PHASE2_SCHEMA))
    assert any("x.json" in v for v in violations)


def test_scalar_rest_api_variant_fails() -> None:
    """The forbidden scalar variant ({'rest_api': str}) never validates."""
    gate = _load_gate()
    row = copy.deepcopy(gate.canonical_rows()["phase2"])
    row["y_true"] = {"rest_api": "/redfish/v1/Systems"}

    violations = gate.validate_schema(row, gate.load_schema(gate.PHASE2_SCHEMA))
    assert violations  # missing rest_api_list + unknown rest_api key


def test_call_with_allowed_methods_fails() -> None:
    """allowed_methods echoed into a Call is context leakage and fails."""
    gate = _load_gate()
    row = copy.deepcopy(gate.canonical_rows()["phase3"])
    row["y_true"]["calls"][0]["allowed_methods"] = ["GET", "HEAD"]

    violations = gate.validate_schema(row, gate.load_schema(gate.PHASE3_SCHEMA))
    assert any("allowed_methods" in v for v in violations)


def test_missing_validation_field_fails() -> None:
    """Dropping one locked validation flag fails the Phase 2 schema."""
    gate = _load_gate()
    row = copy.deepcopy(gate.canonical_rows()["phase2"])
    del row["validation"]["method_semantics_valid"]

    violations = gate.validate_schema(row, gate.load_schema(gate.PHASE2_SCHEMA))
    assert any("method_semantics_valid" in v for v in violations)


def test_bool_does_not_satisfy_integer_const() -> None:
    """True must not satisfy the phase const 2 / integer checks (bool is an int subclass)."""
    gate = _load_gate()
    row = copy.deepcopy(gate.canonical_rows()["phase2"])
    row["phase"] = True

    violations = gate.validate_schema(row, gate.load_schema(gate.PHASE2_SCHEMA))
    assert any("phase" in v for v in violations)


# Author: Mus mbayramo@stanford.edu
