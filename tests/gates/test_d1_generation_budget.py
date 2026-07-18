"""Offline tests for the D1 generation-budget gate.

Prove the committed budget config validates, that every unbounded/invalid form of a
required cap is rejected (missing, null, 0, negative, bool, non-finite/float, bad
``sample_widths``, non-bool ``require_single_api_coverage``), and that the stop
predicates fire exactly at their caps. Uses several synthetic configs so the gate is
exercised as general contract logic, not against a single fixture.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

import copy
from pathlib import Path

import pytest

from scripts.gates.d1_generation_budget import (
    api_full,
    combination_full,
    load_config,
    should_stop,
    validate_budget,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = REPO_ROOT / "configs/gates/d1_budget.yaml"


def _valid_config() -> dict:
    """A minimal, fully-specified budget config used as a mutation base."""
    return {
        "sample_widths": [1, 2, 3],
        "max_accepted_rows": 100,
        "max_candidates": 1000,
        "max_accepted_per_combination": 3,
        "max_attempts_per_combination": 8,
        "max_accepted_per_api": 50,
        "require_single_api_coverage": True,
    }


def test_committed_config_validates() -> None:
    """The committed configs/gates/d1_budget.yaml has no budget violations."""
    config = load_config(CONFIG)
    assert validate_budget(config) == []


def test_synthetic_valid_config_passes() -> None:
    """A hand-built config with all finite positive caps validates cleanly."""
    assert validate_budget(_valid_config()) == []


def test_missing_max_candidates_fails() -> None:
    """Dropping a required limit (max_candidates) fails the gate."""
    config = _valid_config()
    del config["max_candidates"]
    errors = validate_budget(config)
    assert any("max_candidates" in e for e in errors)


@pytest.mark.parametrize("bad", [0, None, -1, -100])
def test_zero_null_or_negative_max_accepted_rows_fails(bad: object) -> None:
    """max_accepted_rows of 0, null, or negative is unbounded/invalid and fails."""
    config = _valid_config()
    config["max_accepted_rows"] = bad
    errors = validate_budget(config)
    assert any("max_accepted_rows" in e for e in errors)


@pytest.mark.parametrize("key", [
    "max_accepted_rows",
    "max_candidates",
    "max_accepted_per_combination",
    "max_attempts_per_combination",
    "max_accepted_per_api",
])
def test_each_required_limit_must_be_present(key: str) -> None:
    """Removing any single required integer limit fails the gate."""
    config = _valid_config()
    del config[key]
    assert any(key in e for e in validate_budget(config))


@pytest.mark.parametrize("bad", [float("inf"), float("nan"), 2.5, True, False, "10"])
def test_non_finite_bool_or_stringy_limit_fails(bad: object) -> None:
    """A cap that is non-finite, a float, a bool, or a string is rejected."""
    config = _valid_config()
    config["max_candidates"] = bad
    assert any("max_candidates" in e for e in validate_budget(config))


@pytest.mark.parametrize("widths", [[], None, "1,2,3", [1, 0, 2], [1, -1], [1, True], [1, 2.0]])
def test_bad_sample_widths_fail(widths: object) -> None:
    """sample_widths must be a non-empty list of positive ints."""
    config = _valid_config()
    config["sample_widths"] = widths
    assert any("sample_widths" in e for e in validate_budget(config))


def test_missing_sample_widths_fails() -> None:
    """An absent sample_widths key fails the gate."""
    config = _valid_config()
    del config["sample_widths"]
    assert any("sample_widths" in e for e in validate_budget(config))


@pytest.mark.parametrize("bad", [None, 1, 0, "true", "false"])
def test_require_single_api_coverage_must_be_bool(bad: object) -> None:
    """require_single_api_coverage present but non-bool fails."""
    config = _valid_config()
    config["require_single_api_coverage"] = bad
    assert any("require_single_api_coverage" in e for e in validate_budget(config))


def test_require_single_api_coverage_missing_fails() -> None:
    """Omitting the coverage-policy flag fails the gate."""
    config = _valid_config()
    del config["require_single_api_coverage"]
    assert any("require_single_api_coverage" in e for e in validate_budget(config))


def test_require_single_api_coverage_false_is_valid() -> None:
    """An explicit False coverage policy is a valid bool, not a failure."""
    config = _valid_config()
    config["require_single_api_coverage"] = False
    assert validate_budget(config) == []


def test_multiple_violations_reported_together() -> None:
    """A config with several defects reports one message per defect."""
    config = _valid_config()
    del config["max_candidates"]
    config["max_accepted_rows"] = 0
    config["require_single_api_coverage"] = "yes"
    errors = validate_budget(config)
    assert any("max_candidates" in e for e in errors)
    assert any("max_accepted_rows" in e for e in errors)
    assert any("require_single_api_coverage" in e for e in errors)


def test_original_config_not_mutated_by_validate() -> None:
    """validate_budget must not mutate the config it inspects."""
    config = _valid_config()
    snapshot = copy.deepcopy(config)
    validate_budget(config)
    assert config == snapshot


def test_should_stop_fires_on_accepted_rows() -> None:
    """should_stop is True once accepted_rows reaches max_accepted_rows."""
    config = _valid_config()  # max_accepted_rows=100, max_candidates=1000
    assert not should_stop(99, 0, config)
    assert should_stop(100, 0, config)
    assert should_stop(101, 0, config)


def test_should_stop_fires_on_candidates() -> None:
    """should_stop is True once attempted_candidates reaches max_candidates."""
    config = _valid_config()
    assert not should_stop(0, 999, config)
    assert should_stop(0, 1000, config)
    assert should_stop(0, 5000, config)


def test_should_stop_false_below_both_limits() -> None:
    """should_stop stays False while both counters are under their caps."""
    config = _valid_config()
    assert not should_stop(0, 0, config)
    assert not should_stop(50, 500, config)


def test_combination_full_fires_on_accepted_or_attempts() -> None:
    """combination_full trips on the per-combination accepted OR attempt cap."""
    config = _valid_config()  # per-combo: 3 accepted, 8 attempts
    assert not combination_full(2, 7, config)
    assert combination_full(3, 0, config)   # accepted cap
    assert combination_full(0, 8, config)   # attempt cap


def test_api_full_fires_on_per_api_cap() -> None:
    """api_full trips once a single API reaches max_accepted_per_api."""
    config = _valid_config()  # max_accepted_per_api=50
    assert not api_full(49, config)
    assert api_full(50, config)
    assert api_full(51, config)


# Author: Mus mbayramo@stanford.edu
