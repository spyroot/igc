"""Focused tests for shared promotion evidence validators."""

from __future__ import annotations

import pytest

from igc.modules.train.promotion_evidence import (
    PromotionEvidenceError,
    validate_configured_hard_checks,
)


def test_validate_configured_hard_checks_accepts_required_subset() -> None:
    """Evaluators may emit extra checks, but every configured check must appear."""
    validate_configured_hard_checks(
        checks=[
            {"name": "immutable_full_manifest", "passed": True},
            {"name": "heldout_manifest_sha_matches_file", "passed": True},
            {"name": "threshold_specific_extra_check", "passed": False},
        ],
        required_names=[
            "immutable_full_manifest",
            "heldout_manifest_sha_matches_file",
        ],
    )


@pytest.mark.parametrize(
    "required_names",
    [
        None,
        (),
        [],
        ["immutable_full_manifest", ""],
        ["immutable_full_manifest", 1],
        ["immutable_full_manifest", "immutable_full_manifest"],
    ],
)
def test_validate_configured_hard_checks_rejects_malformed_configured_names(
    required_names: object,
) -> None:
    """Configured hard_checks must be a unique non-empty list[str]."""
    with pytest.raises(PromotionEvidenceError, match="hard_checks"):
        validate_configured_hard_checks(
            checks=[{"name": "immutable_full_manifest", "passed": True}],
            required_names=required_names,
        )


@pytest.mark.parametrize(
    "checks",
    [
        None,
        (),
        [{"name": ""}],
        [{"name": 1}],
        [{"passed": True}],
        ["immutable_full_manifest"],
    ],
)
def test_validate_configured_hard_checks_rejects_malformed_observed_checks(
    checks: object,
) -> None:
    """Observed promotion checks must be objects with valid string names."""
    with pytest.raises(PromotionEvidenceError, match="promotion result check|checks"):
        validate_configured_hard_checks(
            checks=checks,
            required_names=["immutable_full_manifest"],
        )


def test_validate_configured_hard_checks_rejects_duplicate_observed_names() -> None:
    """Duplicate observed check names make configured hard-check validation ambiguous."""
    with pytest.raises(PromotionEvidenceError, match="duplicate checks"):
        validate_configured_hard_checks(
            checks=[
                {"name": "immutable_full_manifest", "passed": True},
                {"name": "immutable_full_manifest", "passed": False},
            ],
            required_names=["immutable_full_manifest"],
        )


def test_validate_configured_hard_checks_rejects_missing_configured_check() -> None:
    """A configured hard check cannot disappear from evaluator output."""
    with pytest.raises(PromotionEvidenceError, match="missing configured hard checks"):
        validate_configured_hard_checks(
            checks=[{"name": "immutable_full_manifest", "passed": True}],
            required_names=[
                "immutable_full_manifest",
                "heldout_manifest_sha_matches_file",
            ],
        )
