"""Offline tests for the documentation-consistency gate.

The real repo docs must pass every check; representative regressions (a stale
identifier, a dead link, an orphan diagram, a scalar call form, a missing
freeze statement) must fail. Fixture repos are built in tmp_path so the checks
are proven to generalize, not just to pass on today's docs.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path
from types import ModuleType

SCRIPT = Path("scripts/gates/docs_consistency.py")


def _load_gate() -> ModuleType:
    """Load the gate script module for direct testing."""
    spec = importlib.util.spec_from_file_location("docs_consistency", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_repo_docs_pass_all_consistency_checks() -> None:
    """The committed documentation passes the full consistency gate."""
    gate = _load_gate()
    violations = gate.check_docs()
    assert violations == []


def _mini_repo(tmp_path: Path) -> Path:
    """Copy the real docs + README into a scratch repo root for tamper tests."""
    root = tmp_path / "repo"
    (root / "docs").mkdir(parents=True)
    shutil.copytree(Path("docs"), root / "docs", dirs_exist_ok=True)
    shutil.copy(Path("README.md"), root / "README.md")
    return root


def test_stale_identifier_fails(tmp_path: Path) -> None:
    """A stale contract identifier in an active doc is a violation."""
    gate = _load_gate()
    root = _mini_repo(tmp_path)
    doc = root / "docs" / "phase_3.md"
    doc.write_text(doc.read_text(encoding="utf-8") + "\nUse ordered_goals here.\n", encoding="utf-8")

    violations = gate.check_docs(root)
    assert any("ordered_goals" in v for v in violations)


def test_stale_identifier_allowed_in_decisions(tmp_path: Path) -> None:
    """DECISIONS.md may carry superseded identifiers as marked history."""
    gate = _load_gate()
    root = _mini_repo(tmp_path)
    doc = root / "docs" / "DECISIONS.md"
    doc.write_text(
        doc.read_text(encoding="utf-8") + "\nSUPERSEDED: the GoalRef ordered_goals design.\n",
        encoding="utf-8",
    )

    violations = gate.check_docs(root)
    assert not any("DECISIONS.md" in v for v in violations)


def test_dead_nav_link_fails(tmp_path: Path) -> None:
    """A docs/README link to a missing file is a violation."""
    gate = _load_gate()
    root = _mini_repo(tmp_path)
    nav = root / "docs" / "README.md"
    nav.write_text(
        nav.read_text(encoding="utf-8") + "\n[gone](DOES_NOT_EXIST.md)\n",
        encoding="utf-8",
    )

    violations = gate.check_docs(root)
    assert any("DOES_NOT_EXIST.md" in v for v in violations)


def test_orphan_diagram_fails(tmp_path: Path) -> None:
    """An SVG no active doc references is an orphan violation."""
    gate = _load_gate()
    root = _mini_repo(tmp_path)
    (root / "docs" / "diagrams" / "99-orphan.svg").write_text("<svg/>", encoding="utf-8")

    violations = gate.check_docs(root)
    assert any("orphan diagram" in v for v in violations)


def test_scalar_call_form_fails(tmp_path: Path) -> None:
    """The forbidden scalar {"call": ...} union form is a violation."""
    gate = _load_gate()
    root = _mini_repo(tmp_path)
    doc = root / "docs" / "phase_3.md"
    doc.write_text(
        doc.read_text(encoding="utf-8") + '\nBad: {"call": {"rest_api": "/api/x"}}\n',
        encoding="utf-8",
    )

    violations = gate.check_docs(root)
    assert any("forbidden scalar call form" in v for v in violations)


def test_missing_freeze_statement_fails(tmp_path: Path) -> None:
    """Removing the RL freeze from the latent design doc is a violation."""
    gate = _load_gate()
    root = _mini_repo(tmp_path)
    doc = root / "docs" / "GOAL_LATENT_DESIGN.md"
    doc.write_text(
        doc.read_text(encoding="utf-8").replace("frozen", "trainable"),
        encoding="utf-8",
    )

    violations = gate.check_docs(root)
    assert any("RL encoder freeze" in v for v in violations)


# Author: Mus mbayramo@stanford.edu
