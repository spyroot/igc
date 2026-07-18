"""Offline tests for the judge-calibration gate.

Prove the suite covers every failure mode, that a calibrated judge (the committed
reference responses) passes, and that a lenient judge that accepts everything is
caught by a dangerous false-accept.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

from pathlib import Path

from scripts.gates.judge_calibration import is_calibrated, load_suite, run

REPO_ROOT = Path(__file__).resolve().parents[2]
SUITE = REPO_ROOT / "configs/gates/judge_calibration_suite.yaml"

_REQUIRED_CATEGORIES = {
    "correct_single_api",
    "correct_multi_api",
    "reversed_explicit_order",
    "missing_api",
    "extra_api",
    "duplicate",
    "ambiguous",
    "method_incompatible",
    "nonsense",
}


def test_suite_covers_all_nine_categories() -> None:
    """The committed suite must exercise every calibration category."""
    suite = load_suite(SUITE)
    categories = {c["category"] for c in suite["cases"]}
    assert _REQUIRED_CATEGORIES <= categories, _REQUIRED_CATEGORIES - categories


def test_reference_suite_is_calibrated() -> None:
    """The hand-labelled reference responses agree with every expectation."""
    report = run(load_suite(SUITE))
    assert report["accuracy"] == 1.0, report["rows"]
    assert report["dangerous_false_accepts"] == []
    assert is_calibrated(report)


def test_lenient_judge_fails_the_gate() -> None:
    """A judge that accepts everything is caught by dangerous false-accepts."""
    accept_all = '{"accepted": true, "rest_api_list": [], "nonsense": false}'
    report = run(load_suite(SUITE), judge_fn=lambda _case: accept_all)
    assert report["dangerous_false_accepts"], "lenient judge must trip the gate"
    assert not is_calibrated(report)


def test_paranoid_judge_flags_false_rejects_but_no_false_accepts() -> None:
    """A judge that rejects everything has false-rejects, not dangerous accepts."""
    reject_all = '{"accepted": false, "rest_api_list": [], "nonsense": false}'
    report = run(load_suite(SUITE), judge_fn=lambda _case: reject_all)
    assert report["dangerous_false_accepts"] == []
    assert report["false_rejects"], "correct_* cases should be false-rejected here"
    assert not is_calibrated(report)  # accuracy < 1.0 due to false-rejects


# Author: Mus mbayramo@stanford.edu
