"""Offline tests for the holdout split-verification gate.

Prove that matching fair-comparison keys pass, that a differing ``data_manifest`` or
``eval_split`` fails (the golden producer scored a different corpus or split than the
model trained on), and that a missing/empty key fails. Pure dict comparison — no
cluster read, no model load, no network.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

from scripts.gates.holdout_split import REQUIRED_FIELDS, run, verify_split

# A matched pair: producer and training agree on both fair-comparison keys.
_MATCHED_PRODUCER = {"data_manifest": "0123456789abcdef", "eval_split": "floor=REAL:frac=0.15:seed=0"}
_MATCHED_TRAINING = {"data_manifest": "0123456789abcdef", "eval_split": "floor=REAL:frac=0.15:seed=0"}


def test_matching_fields_pass() -> None:
    """Identical data_manifest + eval_split on both sides yields no violations."""
    assert verify_split(_MATCHED_PRODUCER, _MATCHED_TRAINING) == []
    report = run(_MATCHED_PRODUCER, _MATCHED_TRAINING)
    assert report["verified"] is True
    assert report["violations"] == []


def test_mismatched_data_manifest_fails() -> None:
    """A different record mix (data_manifest) is caught as a mismatch."""
    training = {**_MATCHED_TRAINING, "data_manifest": "ffffffffffffffff"}
    violations = verify_split(_MATCHED_PRODUCER, training)
    assert violations, "differing data_manifest must fail"
    assert any("data_manifest mismatch" in v for v in violations)
    assert run(_MATCHED_PRODUCER, training)["verified"] is False


def test_mismatched_eval_split_fails() -> None:
    """A different split policy (eval_split) is caught as a mismatch."""
    training = {**_MATCHED_TRAINING, "eval_split": "floor=REAL:frac=0.20:seed=7"}
    violations = verify_split(_MATCHED_PRODUCER, training)
    assert violations, "differing eval_split must fail"
    assert any("eval_split mismatch" in v for v in violations)
    assert run(_MATCHED_PRODUCER, training)["verified"] is False


def test_missing_key_fails() -> None:
    """A wholly absent required key on the training side fails the gate."""
    training = {"data_manifest": "0123456789abcdef"}  # eval_split absent
    violations = verify_split(_MATCHED_PRODUCER, training)
    assert any("training manifest missing 'eval_split'" in v for v in violations)
    assert run(_MATCHED_PRODUCER, training)["verified"] is False


def test_empty_value_counts_as_missing() -> None:
    """An empty/whitespace value cannot certify a split and is treated as missing."""
    producer = {"data_manifest": "   ", "eval_split": "floor=REAL:frac=0.15:seed=0"}
    violations = verify_split(producer, _MATCHED_TRAINING)
    assert any("producer manifest missing 'data_manifest'" in v for v in violations)
    # A missing value must not also be double-reported as a mismatch.
    assert not any("mismatch" in v for v in violations)


def test_missing_on_both_sides_reports_each() -> None:
    """When a key is absent on both sides, each side is reported missing, no mismatch."""
    violations = verify_split({"eval_split": "s"}, {"eval_split": "s"})
    assert "producer manifest missing 'data_manifest'" in violations
    assert "training manifest missing 'data_manifest'" in violations
    assert not any("mismatch" in v for v in violations)


def test_required_fields_are_the_run_manifest_keys() -> None:
    """The gate checks exactly the two keys DataManifest.to_run_manifest_fields emits."""
    assert set(REQUIRED_FIELDS) == {"data_manifest", "eval_split"}


# Author: Mus mbayramo@stanford.edu
