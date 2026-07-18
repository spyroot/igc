"""Offline tests for the checkpoint-isolation gate.

Prove that disjoint Phase 1 / Phase 2 / Phase 3 output directories pass, and that
every way a Phase 2/3 output could clobber the ``model_x`` checkpoint — writing
under it, being identical to it, or being a parent of it — is caught. Pure ``Path``
logic on path strings; no filesystem writes and no model loads.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

from scripts.gates.checkpoint_isolation import check_isolation

_MODEL_X = "/models/igc/phase1_runs/model_x"


def test_disjoint_dirs_pass() -> None:
    """Three sibling directories under /models never overlap -> no violations."""
    violations = check_isolation(
        _MODEL_X,
        "/models/igc/phase2_runs/goal_extractor",
        "/models/igc/phase3_runs/argument_extractor",
    )
    assert violations == [], violations


def test_phase2_under_model_x_fails() -> None:
    """A Phase 2 output nested under model_x would overwrite the checkpoint."""
    violations = check_isolation(
        _MODEL_X,
        f"{_MODEL_X}/phase2",
        "/models/igc/phase3_runs/argument_extractor",
    )
    assert len(violations) == 1
    assert "phase2_out" in violations[0]
    assert "nested under" in violations[0]


def test_phase3_equal_to_model_x_fails() -> None:
    """A Phase 3 output identical to model_x is a direct overwrite."""
    violations = check_isolation(
        _MODEL_X,
        "/models/igc/phase2_runs/goal_extractor",
        _MODEL_X,
    )
    assert len(violations) == 1
    assert "phase3_out" in violations[0]
    assert "identical to" in violations[0]


def test_model_x_under_phase2_parent_overwrite_fails() -> None:
    """A Phase 2 output that is a PARENT of model_x also clobbers it."""
    violations = check_isolation(
        _MODEL_X,
        "/models/igc/phase1_runs",  # parent of model_x
        "/models/igc/phase3_runs/argument_extractor",
    )
    assert len(violations) == 1
    assert "phase2_out" in violations[0]
    assert "parent of" in violations[0]


def test_both_phases_can_violate_at_once() -> None:
    """Both Phase 2 and Phase 3 overlapping model_x each report a violation."""
    violations = check_isolation(
        _MODEL_X,
        _MODEL_X,               # identical
        f"{_MODEL_X}/phase3",   # nested under
    )
    assert len(violations) == 2
    assert any("phase2_out" in v for v in violations)
    assert any("phase3_out" in v for v in violations)


def test_sibling_prefix_is_not_an_overlap() -> None:
    """A path that merely shares a name prefix (model_x_backup) is disjoint.

    Guards against a naive string ``startswith`` check: ``.../model_x_backup`` is
    NOT under ``.../model_x``. ``Path.is_relative_to`` compares path components,
    so the sibling correctly passes.
    """
    violations = check_isolation(
        _MODEL_X,
        f"{_MODEL_X}_backup",
        "/models/igc/phase3_runs/argument_extractor",
    )
    assert violations == [], violations


def test_relative_dot_segments_normalize_before_compare() -> None:
    """``.``/``..`` segments resolve so an obfuscated path still trips the gate."""
    violations = check_isolation(
        _MODEL_X,
        "/models/igc/phase1_runs/model_x/../model_x/phase2",  # -> under model_x
        "/models/igc/phase3_runs/argument_extractor",
    )
    assert len(violations) == 1
    assert "phase2_out" in violations[0]
    assert "nested under" in violations[0]


# Author: Mus mbayramo@stanford.edu
