#!/usr/bin/env python3
"""Validate the strict Phase 1 D0 row invariants used before tokenization."""

from __future__ import annotations

import copy
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from igc.ds.phase1_render import build_phase1_row, validate_phase1_row  # noqa: E402


CONTRACT_PATH = Path(__file__).resolve().parents[2] / "configs" / "contracts" / "phase1.yaml"
SOURCE_REGISTRY_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "data" / "redfish_sources.yaml"
)
PROMOTION_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "inference"
    / "phase1_golden_acceptance.yaml"
)


def _check_machine_contract(failures: list[str]) -> None:
    """Require the checked-in YAML authority to match the Python validator."""
    value = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        failures.append("Phase 1 contract YAML must contain an object")
        return

    authority = value.get("authority", {})
    expected_authority = {
        "dataset": "D0",
        "phase": 1,
        "task": "redfish_json_reconstruction",
    }
    if authority != expected_authority:
        failures.append(
            f"Phase 1 authority: expected {expected_authority!r}, got {authority!r}"
        )

    row = value.get("row", {})
    expected_required = ["phase", "dataset", "task", "x", "y_true"]
    if row.get("required_top_level_fields") != expected_required:
        failures.append("Phase 1 required fields do not match the D0 validator")
    if row.get("optional_top_level_fields") != ["metadata"]:
        failures.append("Phase 1 optional fields must contain only metadata")
    if row.get("forbidden_top_level_fields") != ["y_pred"]:
        failures.append("Phase 1 forbidden fields must contain y_pred")

    objective = value.get("objective", {})
    expected_objective = {
        "family": "causal_lm_completion",
        "prompt_tokens_ignored": True,
        "completion_tokens_supervised": True,
        "overflow_policy": "reject",
    }
    if objective != expected_objective:
        failures.append("Phase 1 objective does not match completion-only training")

    materialization = value.get("materialization", {})
    if materialization != {
        "canonical_unit": "directory",
        "members": [
            "train/examples.jsonl",
            "train/manifest.json",
            "heldout/examples.jsonl",
            "heldout/manifest.json",
        ],
        "single_atomic_directory_rename": True,
        "immutable": True,
        "split_membership_unit": "original_resource_before_chunking",
        "exact_reassembly_required": True,
    }:
        failures.append("Phase 1 materialization must be one immutable directory release")

    source_registry = yaml.safe_load(SOURCE_REGISTRY_PATH.read_text(encoding="utf-8"))
    promotion = yaml.safe_load(PROMOTION_PATH.read_text(encoding="utf-8"))
    artifact_contract = source_registry.get("artifact_contract", {})
    api_map = artifact_contract.get("api_map", {})
    if artifact_contract.get("producer") != "redfish_ctl_discovery":
        failures.append("Phase 1 source producer must be redfish_ctl discovery")
    if set(api_map.get("accepted_names", [])) != {
        "rest_api_map.v1.json",
        "rest_api_map.npy",
    }:
        failures.append("Phase 1 source registry has invalid REST API map names")
    if api_map.get("required_keys") != [
        "url_file_mapping",
        "allowed_methods_mapping",
    ]:
        failures.append("Phase 1 source registry has invalid REST API map keys")
    if not artifact_contract.get("semantics_bundle"):
        failures.append("Phase 1 source registry requires a semantics bundle")
    source_floor = source_registry.get("evaluation", {}).get(
        "min_heldout_rows_per_source"
    )
    promotion_thresholds = promotion.get("thresholds", {})
    if source_floor != promotion_thresholds.get("min_heldout_rows_per_corpus"):
        failures.append(
            "Phase 1 source and promotion per-corpus held-out floors disagree"
        )
    if promotion_thresholds.get("small_corpus_policy") != "require_all_available_rows":
        failures.append("Phase 1 small-corpus policy must require all available rows")


def _must_reject(label: str, row: dict[str, Any], failures: list[str]) -> None:
    try:
        validate_phase1_row(row)
    except ValueError:
        return
    failures.append(f"{label}: invalid row was accepted")


def run_gate() -> list[str]:
    """Return Phase 1 invariant violations; an empty list is a pass."""
    failures: list[str] = []
    _check_machine_contract(failures)
    row = build_phase1_row(
        rest_api="/redfish/v1/Systems/1",
        allowed_methods=("GET", "PATCH"),
        input_json={"@odata.id": "/redfish/v1/Systems/1"},
        target_json={"@odata.id": "/redfish/v1/Systems/1"},
    )
    mutations: tuple[tuple[str, Callable[[dict[str, Any]], None]], ...] = (
        ("phase", lambda value: value.__setitem__("phase", 2)),
        ("dataset", lambda value: value.__setitem__("dataset", "other")),
        ("task", lambda value: value.__setitem__("task", "other")),
        ("empty rest_api", lambda value: value["x"].__setitem__("rest_api", "")),
        ("scalar methods", lambda value: value["x"].__setitem__("allowed_methods", "GET")),
        ("lowercase method", lambda value: value["x"].__setitem__("allowed_methods", ["get"])),
        ("duplicate methods", lambda value: value["x"].__setitem__("allowed_methods", ["GET", "GET"])),
        ("non-object x.json", lambda value: value["x"].__setitem__("json", [])),
        ("non-object y_true.json", lambda value: value["y_true"].__setitem__("json", [])),
        ("committed y_pred", lambda value: value.__setitem__("y_pred", {})),
        ("missing target", lambda value: value.__setitem__("y_true", {})),
    )
    for label, mutate in mutations:
        candidate = copy.deepcopy(row)
        mutate(candidate)
        _must_reject(label, candidate, failures)
    return failures


def main() -> int:
    """CLI entry point for the project contract gate."""
    failures = run_gate()
    if failures:
        print(json.dumps({"status": "fail", "failures": failures}, indent=2))
        return 1
    print(json.dumps({"status": "pass", "gate": "phase1.contract"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
