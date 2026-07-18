"""Offline tests for the d1.balance-and-dedupe gate.

These exercise the corpus-level D1 balance + dedup contract with MULTIPLE synthetic
accepted-row sets — never a single hardcoded example — proving:

* ``normalize`` collapses case/whitespace/punctuation/stutters and is idempotent.
* exact and near-duplicate detection catch reworded re-drafts.
* ``balance_report`` counts k=1/2/3, per-API, per-combination, and single-API
  coverage, and surfaces generation rates only when telemetry is present.
* ``fail_balance`` blocks a missing single-command API and an out-of-band
  (too-low or suspiciously-high) judge acceptance rate.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

from typing import Any

from scripts.gates.d1_balance_and_dedupe import (
    balance_report,
    fail_balance,
    is_exact_duplicate,
    is_near_duplicate,
    normalize,
)

# A pool of realistic APIs so tests never depend on the operator's single
# BIOS/Certificates illustration.
BIOS = "/redfish/v1/Systems/1/Bios"
CERTS = "/redfish/v1/Managers/1/NetworkProtocol/HTTPS/Certificates"
THERMAL = "/redfish/v1/Chassis/1/Thermal"
POWER = "/redfish/v1/Chassis/1/Power"


def _row(text: str, apis: list[str], **extra: Any) -> dict[str, Any]:
    """Build a minimal locked-contract D1 row (text-only x, set-valued y_true)."""
    row: dict[str, Any] = {
        "phase": 2,
        "dataset": "D1",
        "x": {"text": text},                       # TEXT ONLY — no json/methods leak
        "y_true": {"rest_api_list": sorted(apis)},  # stored sorted; evaluated as a set
    }
    row.update(extra)
    return row


# --------------------------------------------------------------------------- #
# normalize
# --------------------------------------------------------------------------- #
def test_normalize_collapses_case_space_punctuation() -> None:
    """Case, extra whitespace, and punctuation all fold to one canonical form."""
    assert normalize("  Check   the BOOT-config!! ") == "check the boot config"
    assert normalize("Reboot, please.") == "reboot please"


def test_normalize_collapses_repeated_words_and_is_idempotent() -> None:
    """Stuttered consecutive words collapse; re-normalising changes nothing."""
    once = normalize("check check the the boot config")
    assert once == "check the boot config"
    assert normalize(once) == once  # idempotent


def test_normalize_non_consecutive_repeats_are_kept() -> None:
    """Only consecutive duplicates collapse — a real repeated word survives."""
    assert normalize("read system then read manager") == "read system then read manager"


# --------------------------------------------------------------------------- #
# duplicate detection
# --------------------------------------------------------------------------- #
def test_exact_duplicate_ignores_cosmetics() -> None:
    """A cosmetic variant of a seen text is an exact duplicate; a new one is not."""
    seen = ["Check the boot configuration"]
    assert is_exact_duplicate("check   the boot   configuration!", seen) is True
    assert is_exact_duplicate("check the certificate expiration", seen) is False


def test_exact_duplicate_empty_index() -> None:
    """Nothing is a duplicate against an empty index."""
    assert is_exact_duplicate("anything", []) is False


def test_near_duplicate_by_token_overlap() -> None:
    """A reworded draft with high token overlap is a near-duplicate; a distinct one is not."""
    accepted = ["check the boot configuration and certificate expiration"]
    # High overlap (one extra/changed word) -> near duplicate at 0.6.
    assert is_near_duplicate(
        "please check the boot configuration and certificate expiration",
        accepted,
        token_overlap_threshold=0.6,
    ) is True
    # Unrelated request -> not a near duplicate.
    assert is_near_duplicate(
        "show the thermal sensor readings",
        accepted,
        token_overlap_threshold=0.6,
    ) is False


def test_near_duplicate_threshold_one_needs_identical_token_set() -> None:
    """At threshold 1.0 only an identical (order-independent) token set matches."""
    accepted = ["power on the system"]
    assert is_near_duplicate("system the on power", accepted, 1.0) is True   # same tokens
    assert is_near_duplicate("power on the manager", accepted, 1.0) is False  # one token differs


def test_blank_text_is_never_a_near_duplicate() -> None:
    """An empty/punctuation-only draft never counts as a near-duplicate."""
    assert is_near_duplicate("!!!", ["power on the system"], 0.1) is False


# --------------------------------------------------------------------------- #
# balance_report
# --------------------------------------------------------------------------- #
def _balanced_rows() -> list[dict[str, Any]]:
    """A fully-covered, balanced accepted set over four APIs (k=1,2,3 present)."""
    return [
        _row("check the bios settings", [BIOS]),
        _row("check the https certificates", [CERTS]),
        _row("read the thermal sensors", [THERMAL]),
        _row("read the power readings", [POWER]),
        _row("check bios and certificate expiration", [BIOS, CERTS]),
        _row("read thermal and power", [THERMAL, POWER]),
        _row("check bios, certs, and thermal", [BIOS, CERTS, THERMAL]),
    ]


def test_balance_report_counts_k_and_coverage() -> None:
    """k=1/2/3 counts, per-API, per-combination, and single-API coverage are correct."""
    report = balance_report(_balanced_rows())
    assert report["total_accepted"] == 7
    assert report["k_counts"] == {1: 4, 2: 2, 3: 1}
    assert report["single_api_coverage"] == sorted([BIOS, CERTS, THERMAL, POWER])
    # BIOS appears in 3 rows (single, pair, triple); THERMAL in 3 as well.
    assert report["per_rest_api"][BIOS] == 3
    assert report["per_rest_api"][POWER] == 2
    # Combination key is order-independent and canonical.
    assert report["per_combination"][" + ".join(sorted([BIOS, CERTS]))] == 1
    assert report["malformed"] == 0


def test_balance_report_combination_is_order_independent() -> None:
    """[B, A] and [A, B] land in the same combination bucket."""
    rows = [
        _row("a then b", [BIOS, CERTS]),
        {  # deliberately store reversed order to prove set/canonical handling
            "phase": 2, "dataset": "D1",
            "x": {"text": "b then a"},
            "y_true": {"rest_api_list": [CERTS, BIOS]},
        },
    ]
    report = balance_report(rows)
    assert report["per_combination"][" + ".join(sorted([BIOS, CERTS]))] == 2
    assert list(report["per_combination"]) == [" + ".join(sorted([BIOS, CERTS]))]


def test_balance_report_flags_duplicate_list_as_malformed() -> None:
    """A stored rest_api_list with a duplicate breaks unordered-unique-set and is malformed."""
    rows = [{
        "phase": 2, "dataset": "D1",
        "x": {"text": "check bios twice"},
        "y_true": {"rest_api_list": [BIOS, BIOS]},  # duplicate -> malformed
    }]
    report = balance_report(rows)
    assert report["malformed"] == 1
    assert report["k_counts"] == {1: 1}  # collapses to one unique API


def test_balance_report_rates_absent_without_generation_block() -> None:
    """Without generation telemetry the corpus rates are None ('if present')."""
    report = balance_report(_balanced_rows())
    for key in ("acceptance_rate", "retry_rate", "duplicate_rate", "near_duplicate_rate"):
        assert report[key] is None


def test_balance_report_rates_present_with_generation_block() -> None:
    """Generation telemetry yields acceptance/retry/duplicate rates."""
    rows = [
        _row("check the bios settings", [BIOS], generation={"attempts": 1}),
        _row("check the https certificates", [CERTS], generation={"attempts": 3}),
        _row("read the thermal sensors", [THERMAL],
             generation={"attempts": 1, "near_duplicate": True}),
    ]
    report = balance_report(rows)
    # 3 accepted rows over 1+3+1 = 5 total attempts.
    assert abs(report["acceptance_rate"] - (3 / 5)) < 1e-9
    assert abs(report["retry_rate"] - (1 / 3)) < 1e-9   # one row needed >1 attempt
    assert abs(report["near_duplicate_rate"] - (1 / 3)) < 1e-9
    assert report["duplicate_rate"] == 0.0


# --------------------------------------------------------------------------- #
# fail_balance
# --------------------------------------------------------------------------- #
def test_fail_balance_passes_when_covered_and_in_band() -> None:
    """Full single-command coverage + in-band acceptance rate -> no violations."""
    rows = [
        _row("check the bios settings", [BIOS], generation={"attempts": 2}),
        _row("check the https certificates", [CERTS], generation={"attempts": 2}),
    ]
    report = balance_report(rows)
    failures = fail_balance(report, [BIOS, CERTS], min_acceptance=0.1, max_acceptance=0.95)
    assert failures == []


def test_fail_balance_flags_missing_single_command_coverage() -> None:
    """An eligible API present only inside a multi-API row still fails coverage."""
    rows = [
        _row("check the bios settings", [BIOS]),
        _row("check bios and certificate expiration", [BIOS, CERTS]),  # CERTS never solo
    ]
    report = balance_report(rows)
    failures = fail_balance(report, [BIOS, CERTS], min_acceptance=0.0, max_acceptance=1.0)
    assert any(CERTS in f for f in failures)
    assert all(BIOS not in f or "acceptance" in f for f in failures)  # BIOS is covered


def test_fail_balance_flags_acceptance_rate_too_low() -> None:
    """A judge accepting almost nothing (many retries per row) trips the floor."""
    rows = [
        _row("check the bios settings", [BIOS], generation={"attempts": 50}),
        _row("check the https certificates", [CERTS], generation={"attempts": 50}),
    ]
    report = balance_report(rows)  # acceptance_rate = 2/100 = 0.02
    failures = fail_balance(report, [BIOS, CERTS], min_acceptance=0.1, max_acceptance=0.95)
    assert any("< min" in f for f in failures)


def test_fail_balance_flags_acceptance_rate_too_high() -> None:
    """A rubber-stamp judge (every draft accepted first try) trips the ceiling."""
    rows = [
        _row("check the bios settings", [BIOS], generation={"attempts": 1}),
        _row("check the https certificates", [CERTS], generation={"attempts": 1}),
    ]
    report = balance_report(rows)  # acceptance_rate = 1.0
    failures = fail_balance(report, [BIOS, CERTS], min_acceptance=0.1, max_acceptance=0.95)
    assert any("> max" in f for f in failures)


def test_fail_balance_skips_rate_check_without_telemetry() -> None:
    """Absent acceptance telemetry -> only coverage is enforced, not the rate."""
    report = balance_report(_balanced_rows())
    failures = fail_balance(
        report, [BIOS, CERTS, THERMAL, POWER], min_acceptance=0.9, max_acceptance=0.95
    )
    assert failures == []  # every API covered; rate is None so bounds are not applied


# Author: Mus mbayramo@stanford.edu
