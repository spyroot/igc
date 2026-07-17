"""Prove the contract gates fail on the exact defect class PR #108 slipped past.

#108 shipped a parallel Phase 2/3 contract module with a forked namespace, and its
own isolated tests were green. These tests exercise the two repo-level gates:

* ``repo.contract-single-source`` passes on the live tree, and fails on a forked
  namespace token or a canonical symbol defined in a second module.
* ``repo.schema-snapshot`` bootstraps, round-trips, and detects drift.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, relpath: str):
    """Load a gate script (not an importable package) from its path."""
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relpath)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


css = _load("contract_single_source", "scripts/gates/contract_single_source.py")
ss = _load("schema_snapshot", "scripts/gates/schema_snapshot.py")


def _fixture_tree(root: Path) -> Path:
    """Write a minimal repo tree + registry the single-source gate can check."""
    (root / "igc" / "ds").mkdir(parents=True, exist_ok=True)
    (root / "configs" / "contracts").mkdir(parents=True, exist_ok=True)
    (root / "igc" / "ds" / "rest_goal_contract.py").write_text(
        "def build_d1_rest_api_list_row():\n    return {}\n\n\nclass RedfishContext:\n    pass\n",
        encoding="utf-8",
    )
    registry = root / "configs" / "contracts" / "rest_goal.yaml"
    registry.write_text(
        "version: 1\n"
        "canonical_module: igc/ds/rest_goal_contract.py\n"
        "search_root: igc\n"
        "canonical_namespaces: [phase2_goal_extraction]\n"
        "forbidden_namespace_literals: [phase2_goal_extract, phase3_argument_extract]\n"
        "canonical_symbols: [build_d1_rest_api_list_row, RedfishContext]\n"
        "canonical_tasks: [text_to_rest_api_list]\n",
        encoding="utf-8",
    )
    return registry


def test_single_source_passes_on_live_tree() -> None:
    """The real repository has one canonical contract and no forked namespaces."""
    violations = css.check(REPO_ROOT / "configs/contracts/rest_goal.yaml", REPO_ROOT)
    assert violations == [], violations


def test_single_source_flags_forked_namespace(tmp_path: Path) -> None:
    """A forked namespace token under igc/ fails the gate (the #108 namespace defect)."""
    registry = _fixture_tree(tmp_path)
    (tmp_path / "igc" / "island.py").write_text('NS = "phase2_goal_extract"\n', encoding="utf-8")
    violations = css.check(registry, tmp_path)
    assert any("forked" in v for v in violations), violations


def test_single_source_flags_parallel_island(tmp_path: Path) -> None:
    """A canonical symbol defined in a second module fails the gate (parallel island)."""
    registry = _fixture_tree(tmp_path)
    (tmp_path / "igc" / "mock_inference_contract.py").write_text(
        "class RedfishContext:\n    pass\n", encoding="utf-8"
    )
    violations = css.check(registry, tmp_path)
    assert any("parallel contract island" in v for v in violations), violations


def test_single_source_ignores_canonical_value(tmp_path: Path) -> None:
    """The canonical *_extraction value must not trip the shorter fork token."""
    registry = _fixture_tree(tmp_path)
    (tmp_path / "igc" / "ok.py").write_text('NS = "phase2_goal_extraction"\n', encoding="utf-8")
    violations = css.check(registry, tmp_path)
    assert violations == [], violations


def test_schema_snapshot_bootstrap_roundtrip_and_drift(tmp_path: Path) -> None:
    """Missing snapshot bootstraps; update then check matches; a mutated snapshot fails."""
    snap = tmp_path / "shape.json"
    assert ss.check(snap) == ss.EXIT_BOOTSTRAP
    assert ss.update(snap) == ss.EXIT_OK
    assert ss.check(snap) == ss.EXIT_OK
    snap.write_text(snap.read_text(encoding="utf-8").replace('"phase"', '"phase_RENAMED"'), encoding="utf-8")
    assert ss.check(snap) == ss.EXIT_FAIL


# Author: Mus mbayramo@stanford.edu
