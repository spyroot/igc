#!/usr/bin/env python3
"""Gate: contract.phase23-row-schema — machine-readable Phase 2/3 row schemas.

The JSON schemas under ``configs/contracts/phase2_row.schema.json`` and
``configs/contracts/phase3_row.schema.json`` are the AUTHORITATIVE row shapes;
docs and docstring examples are illustrative. This gate renders one canonical
row through each real builder (``build_d1_rest_api_list_row`` and
``build_call_row`` from ``igc/ds/rest_goal_contract.py``) and validates it
against its schema, so a producer edit that drifts from the committed schema —
a renamed/added/dropped field, a scalar/list union, an ``allowed_methods``
echoed into a Call — fails CI instead of poisoning a dataset.

``validate_schema`` is a small self-contained validator for exactly the schema
subset these contracts use (``type``, ``const``, ``required``, ``properties``,
``additionalProperties`` as false-or-schema, ``items``): the CI image does not
ship the ``jsonschema`` package, and the gate must not gain a dependency for a
five-keyword subset.

Used by:
  tests/gates/test_phase23_row_schema.py  (offline gate; runs in `pytest -q`)
  CLI: python scripts/gates/phase23_row_schema.py

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

# Allow running as a plain script from the repo root without an editable install.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from igc.ds.rest_goal_contract import (  # noqa: E402
    RedfishContext,
    build_call_row,
    build_d1_rest_api_list_row,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PHASE2_SCHEMA = REPO_ROOT / "configs" / "contracts" / "phase2_row.schema.json"
PHASE3_SCHEMA = REPO_ROOT / "configs" / "contracts" / "phase3_row.schema.json"

# JSON-type name -> the Python types it admits. bool is checked before integer
# because bool is an int subclass and must never satisfy "integer".
_TYPES: dict[str, tuple[type, ...]] = {
    "object": (dict,),
    "array": (list,),
    "string": (str,),
    "integer": (int,),
    "boolean": (bool,),
}


def validate_schema(value: Any, schema: Mapping[str, Any], path: str = "$") -> list[str]:
    """Validate ``value`` against the supported JSON-schema subset.

    :param value: the JSON-compatible value to check.
    :param schema: schema node using type/const/required/properties/
        additionalProperties/items only.
    :param path: JSON-path prefix used in violation messages.
    :return: list of human-readable violations; empty means valid.
    """
    violations: list[str] = []

    if "const" in schema:
        if value != schema["const"] or isinstance(value, bool) != isinstance(schema["const"], bool):
            violations.append(f"{path}: expected const {schema['const']!r}, got {value!r}")
        return violations

    expected_type = schema.get("type")
    if expected_type is not None:
        allowed = _TYPES[expected_type]
        if expected_type == "integer" and isinstance(value, bool):
            violations.append(f"{path}: expected integer, got bool")
            return violations
        if not isinstance(value, allowed):
            violations.append(
                f"{path}: expected {expected_type}, got {type(value).__name__}"
            )
            return violations

    if expected_type == "object":
        for key in schema.get("required", []):
            if key not in value:
                violations.append(f"{path}.{key}: required key missing")
        properties = schema.get("properties", {})
        additional = schema.get("additionalProperties", True)
        for key, item in value.items():
            if key in properties:
                violations += validate_schema(item, properties[key], f"{path}.{key}")
            elif additional is False:
                violations.append(f"{path}.{key}: additional key not allowed")
            elif isinstance(additional, Mapping):
                violations += validate_schema(item, additional, f"{path}.{key}")
    elif expected_type == "array":
        items = schema.get("items")
        if isinstance(items, Mapping):
            for index, item in enumerate(value):
                violations += validate_schema(item, items, f"{path}[{index}]")

    return violations


def load_schema(path: Path) -> dict[str, Any]:
    """Load one committed schema file."""
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_rows() -> dict[str, dict[str, Any]]:
    """Render one canonical row per phase through the REAL builders.

    :return: ``{"phase2": <row>, "phase3": <row>}`` from the canonical builders.
    """
    read_context = RedfishContext(
        rest_api="/redfish/v1/Systems",
        allowed_methods=("GET", "HEAD"),
        json={"@odata.id": "/redfish/v1/Systems"},
    )
    write_context = RedfishContext(
        rest_api="/redfish/v1/Systems/1/Bios/Settings",
        allowed_methods=("GET", "PATCH"),
        json={"@odata.id": "/redfish/v1/Systems/1/Bios/Settings"},
    )
    phase2 = build_d1_rest_api_list_row(
        text="list systems and set the bios boot mode to Uefi",
        contexts=(read_context, write_context),
        rest_api_list=("/redfish/v1/Systems", "/redfish/v1/Systems/1/Bios/Settings"),
    )
    phase3 = build_call_row(
        text="list systems and set the bios boot mode to Uefi",
        contexts=(read_context, write_context),
        rest_api_list=("/redfish/v1/Systems", "/redfish/v1/Systems/1/Bios/Settings"),
        method_by_api={
            "/redfish/v1/Systems": "GET",
            "/redfish/v1/Systems/1/Bios/Settings": "PATCH",
        },
        arguments_by_api={
            "/redfish/v1/Systems/1/Bios/Settings": {"Attributes": {"BootMode": "Uefi"}},
        },
    )
    return {"phase2": phase2, "phase3": phase3}


def check() -> int:
    """Validate the canonical builder rows against the committed schemas.

    :return: process exit code (0 = pass).
    """
    rows = canonical_rows()
    failures = 0
    for name, schema_path in (("phase2", PHASE2_SCHEMA), ("phase3", PHASE3_SCHEMA)):
        violations = validate_schema(rows[name], load_schema(schema_path))
        if violations:
            failures += 1
            for violation in violations:
                print(f"BLOCKER: {name} row vs {schema_path.name}: {violation}", file=sys.stderr)
        else:
            print(f"OK: {name} builder row conforms to {schema_path.name}.")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    _ = argv
    return check()


if __name__ == "__main__":
    raise SystemExit(main())


# Author: Mus mbayramo@stanford.edu
