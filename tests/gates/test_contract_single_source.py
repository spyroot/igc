"""Tests for the repo-level contract single-source gate."""
from __future__ import annotations

from pathlib import Path

from scripts.gates.contract_single_source import check


def _write_registry(root: Path) -> Path:
    """Write a tiny registry for a fixture repository."""

    registry = root / "configs/contracts/rest_goal.yaml"
    registry.parent.mkdir(parents=True)
    registry.write_text(
        "\n".join([
            "version: 1",
            "canonical_module: igc/ds/rest_goal_contract.py",
            "scan_paths: [igc, docs]",
            "file_suffixes: [.py, .md]",
            "allow_paths: []",
            "required_symbols:",
            "  - build_d1_rest_api_list_row",
            "forbidden_regex:",
            r"  - \"\\bmock_inference_contract\\b\"",
        ]),
        encoding="utf-8",
    )
    return registry


def test_contract_single_source_passes_canonical_only(tmp_path: Path) -> None:
    """A repo with the canonical definition only has no violations."""

    canonical = tmp_path / "igc/ds/rest_goal_contract.py"
    canonical.parent.mkdir(parents=True)
    canonical.write_text("def build_d1_rest_api_list_row():\n    pass\n", encoding="utf-8")

    assert check(_write_registry(tmp_path), tmp_path) == []


def test_contract_single_source_rejects_parallel_definition(tmp_path: Path) -> None:
    """The same canonical symbol in another module is a contract island."""

    canonical = tmp_path / "igc/ds/rest_goal_contract.py"
    other = tmp_path / "igc/ds/other_contract.py"
    canonical.parent.mkdir(parents=True)
    canonical.write_text("def build_d1_rest_api_list_row():\n    pass\n", encoding="utf-8")
    other.write_text("def build_d1_rest_api_list_row():\n    pass\n", encoding="utf-8")

    violations = check(_write_registry(tmp_path), tmp_path)

    assert any("expected only" in violation for violation in violations)


def test_contract_single_source_rejects_forbidden_fingerprint(tmp_path: Path) -> None:
    """Stale contract module names fail even when definitions are absent."""

    canonical = tmp_path / "igc/ds/rest_goal_contract.py"
    stale_doc = tmp_path / "docs/stale.md"
    canonical.parent.mkdir(parents=True)
    stale_doc.parent.mkdir(parents=True)
    canonical.write_text("def build_d1_rest_api_list_row():\n    pass\n", encoding="utf-8")
    stale_doc.write_text("use mock_inference_contract here\n", encoding="utf-8")

    violations = check(_write_registry(tmp_path), tmp_path)

    assert any("forbidden token" in violation for violation in violations)
