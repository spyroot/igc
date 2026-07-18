"""Offline tests for the D1 judge-semantics gate.

Prove the locked structured-judge acceptance formula: a valid judge result parses
and type-checks, and ``decide_accept`` reproduces the operator's accept/reject table
across MULTIPLE synthetic domains (power/thermal, network, storage, BIOS/certs) — not
one hardcoded example. Covers the accept paths (correct multi-API, reordered same set)
and every reject mode (missing intent, extra intent, duplicate intent, ambiguous,
method-semantics-invalid, unsupported coverage item), plus strict parse validation and
the set-based target eval.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.gates.d1_judge_semantics import (
    D1JudgeResultError,
    REASON_AMBIGUOUS,
    REASON_COVERAGE_MISMATCH,
    REASON_DUPLICATE_INTENT,
    REASON_EXTRA_INTENTS,
    REASON_METHOD_SEMANTICS_INVALID,
    REASON_NONSENSE,
    REASON_NOT_ACCEPTED,
    REASON_NOT_NATURAL,
    decide_accept,
    is_consistent,
    load_suite,
    parse_judge_result,
    run,
    target_set_matches,
)


def _coverage(*pairs: tuple[str, bool]) -> list[dict[str, object]]:
    """Build a coverage list from ``(rest_api, supported)`` pairs with a generic span."""
    return [
        {"rest_api": api, "text_span": f"span for {api}", "supported": supported}
        for api, supported in pairs
    ]


def _verdict_dict(
    *,
    coverage: list[dict[str, object]],
    accepted: bool = True,
    natural: bool = True,
    nonsense: bool = False,
    ambiguous: bool = False,
    duplicate_intent: bool = False,
    method_semantics_valid: bool = True,
    extra_intents: list[str] | None = None,
    reason: str = "",
) -> dict[str, object]:
    """Assemble a full structured judge result with sensible accepting defaults."""
    return {
        "accepted": accepted,
        "natural": natural,
        "nonsense": nonsense,
        "ambiguous": ambiguous,
        "duplicate_intent": duplicate_intent,
        "method_semantics_valid": method_semantics_valid,
        "coverage": coverage,
        "extra_intents": extra_intents if extra_intents is not None else [],
        "reason": reason,
    }


# --- parse / validate -------------------------------------------------------


def test_parse_valid_result_from_mapping_and_json() -> None:
    """A well-formed judge result parses identically from a dict or a JSON string."""
    raw = _verdict_dict(coverage=_coverage(("/redfish/v1/Systems/1", True)))
    from_map = parse_judge_result(raw)
    from_json = parse_judge_result(json.dumps(raw))
    assert from_map == from_json
    assert from_map.covered_api_set == {"/redfish/v1/Systems/1"}
    assert from_map.coverage[0].text_span  # span preserved for auditability.


def test_parse_rejects_invalid_json() -> None:
    """A non-JSON judge string raises rather than silently accepting."""
    with pytest.raises(D1JudgeResultError):
        parse_judge_result("{not valid json")


@pytest.mark.parametrize("missing", ["accepted", "natural", "method_semantics_valid", "coverage", "extra_intents"])
def test_parse_requires_all_fields(missing: str) -> None:
    """Dropping any required field is a parse error, never a default-accept."""
    raw = _verdict_dict(coverage=_coverage(("/redfish/v1/Systems/1", True)))
    del raw[missing]
    with pytest.raises(D1JudgeResultError):
        parse_judge_result(raw)


def test_parse_rejects_mistyped_bool_field() -> None:
    """A string where a bool is required (e.g. accepted=\"true\") is rejected."""
    raw = _verdict_dict(coverage=_coverage(("/redfish/v1/Systems/1", True)))
    raw["accepted"] = "true"
    with pytest.raises(D1JudgeResultError):
        parse_judge_result(raw)


def test_parse_rejects_malformed_coverage_item() -> None:
    """A coverage item missing 'supported' is rejected."""
    raw = _verdict_dict(coverage=[{"rest_api": "/redfish/v1/Systems/1", "text_span": "x"}])
    with pytest.raises(D1JudgeResultError):
        parse_judge_result(raw)


def test_parse_rejects_non_string_extra_intents() -> None:
    """extra_intents must contain only strings."""
    raw = _verdict_dict(coverage=_coverage(("/redfish/v1/Systems/1", True)), extra_intents=None)
    raw["extra_intents"] = [123]
    with pytest.raises(D1JudgeResultError):
        parse_judge_result(raw)


# --- decide_accept: accept paths -------------------------------------------


def test_correct_multi_api_accepts() -> None:
    """A thermal+power request whose supported coverage equals the selected set accepts."""
    selected = ["/redfish/v1/Chassis/1/Thermal", "/redfish/v1/Systems/1"]
    verdict = parse_judge_result(_verdict_dict(coverage=_coverage(
        ("/redfish/v1/Chassis/1/Thermal", True),
        ("/redfish/v1/Systems/1", True),
    )))
    accepted, reasons = decide_accept(verdict, selected)
    assert accepted, reasons
    assert reasons == []


def test_reordered_selected_set_still_accepts() -> None:
    """Selected set compared as a set: reversing the label list changes nothing."""
    apis = ["/redfish/v1/Managers/1/EthernetInterfaces/1", "/redfish/v1/Systems/1/Storage/1"]
    verdict = parse_judge_result(_verdict_dict(coverage=_coverage(
        (apis[0], True),
        (apis[1], True),
    )))
    accepted_forward, _ = decide_accept(verdict, apis)
    accepted_reversed, _ = decide_accept(verdict, list(reversed(apis)))
    assert accepted_forward and accepted_reversed


def test_single_api_bios_certificates_accepts() -> None:
    """The operator's illustrative BIOS+Certificates case is ONE accepted case, not the only one."""
    selected = ["/redfish/v1/Systems/1/Bios", "/redfish/v1/CertificateService/CertificateLocations"]
    verdict = parse_judge_result(_verdict_dict(coverage=_coverage(
        ("/redfish/v1/Systems/1/Bios", True),
        ("/redfish/v1/CertificateService/CertificateLocations", True),
    )))
    accepted, reasons = decide_accept(verdict, selected)
    assert accepted, reasons


# --- decide_accept: reject modes -------------------------------------------


def test_missing_intent_rejects_with_coverage_mismatch() -> None:
    """A selected API with no supported coverage item is a coverage mismatch (missing)."""
    selected = ["/redfish/v1/Systems/1", "/redfish/v1/Managers/1"]
    verdict = parse_judge_result(_verdict_dict(coverage=_coverage(
        ("/redfish/v1/Systems/1", True),
    )))
    accepted, reasons = decide_accept(verdict, selected)
    assert not accepted
    assert any(r.startswith(REASON_COVERAGE_MISMATCH) and "/redfish/v1/Managers/1" in r for r in reasons)


def test_extra_supported_api_rejects_as_coverage_mismatch() -> None:
    """A supported coverage item outside the selected set makes covered != selected."""
    selected = ["/redfish/v1/Systems/1"]
    verdict = parse_judge_result(_verdict_dict(coverage=_coverage(
        ("/redfish/v1/Systems/1", True),
        ("/redfish/v1/Chassis/1/Power", True),
    )))
    accepted, reasons = decide_accept(verdict, selected)
    assert not accepted
    assert any(r.startswith(REASON_COVERAGE_MISMATCH) and "unselected" in r for r in reasons)


def test_extra_intents_field_rejects() -> None:
    """A non-empty extra_intents list rejects even when coverage equals the selected set."""
    selected = ["/redfish/v1/Systems/1"]
    verdict = parse_judge_result(_verdict_dict(
        coverage=_coverage(("/redfish/v1/Systems/1", True)),
        extra_intents=["also delete all logs"],
    ))
    accepted, reasons = decide_accept(verdict, selected)
    assert not accepted
    assert any(r.startswith(REASON_EXTRA_INTENTS) for r in reasons)


def test_duplicate_intent_rejects() -> None:
    """duplicate_intent set by the judge rejects the row."""
    selected = ["/redfish/v1/Systems/1"]
    verdict = parse_judge_result(_verdict_dict(
        coverage=_coverage(("/redfish/v1/Systems/1", True)),
        duplicate_intent=True,
    ))
    accepted, reasons = decide_accept(verdict, selected)
    assert not accepted
    assert REASON_DUPLICATE_INTENT in reasons


def test_ambiguous_rejects() -> None:
    """ambiguous text rejects even with matching coverage."""
    selected = ["/redfish/v1/Systems/1"]
    verdict = parse_judge_result(_verdict_dict(
        coverage=_coverage(("/redfish/v1/Systems/1", True)),
        ambiguous=True,
    ))
    accepted, reasons = decide_accept(verdict, selected)
    assert not accepted
    assert REASON_AMBIGUOUS in reasons


def test_method_semantics_invalid_rejects() -> None:
    """A read-phrased request implying a write (method_semantics_valid=False) rejects."""
    selected = ["/redfish/v1/Systems/1"]
    verdict = parse_judge_result(_verdict_dict(
        coverage=_coverage(("/redfish/v1/Systems/1", True)),
        method_semantics_valid=False,
    ))
    accepted, reasons = decide_accept(verdict, selected)
    assert not accepted
    assert REASON_METHOD_SEMANTICS_INVALID in reasons


def test_unsupported_coverage_item_rejects() -> None:
    """A coverage item with supported=False does not count toward covered -> mismatch."""
    selected = ["/redfish/v1/Systems/1", "/redfish/v1/Chassis/1/Thermal"]
    verdict = parse_judge_result(_verdict_dict(coverage=_coverage(
        ("/redfish/v1/Systems/1", True),
        ("/redfish/v1/Chassis/1/Thermal", False),  # judge could not ground this one.
    )))
    accepted, reasons = decide_accept(verdict, selected)
    assert not accepted
    assert any(r.startswith(REASON_COVERAGE_MISMATCH) for r in reasons)


def test_nonsense_and_not_accepted_and_not_natural_all_reject() -> None:
    """The remaining single-flag guards each independently reject."""
    selected = ["/redfish/v1/Systems/1"]
    cov = _coverage(("/redfish/v1/Systems/1", True))
    nonsense = parse_judge_result(_verdict_dict(coverage=cov, nonsense=True))
    not_accepted = parse_judge_result(_verdict_dict(coverage=cov, accepted=False))
    not_natural = parse_judge_result(_verdict_dict(coverage=cov, natural=False))
    assert REASON_NONSENSE in decide_accept(nonsense, selected)[1]
    assert REASON_NOT_ACCEPTED in decide_accept(not_accepted, selected)[1]
    assert REASON_NOT_NATURAL in decide_accept(not_natural, selected)[1]


def test_multiple_failures_reported_together() -> None:
    """A row failing several guards reports all of them, not just the first."""
    selected = ["/redfish/v1/Systems/1"]
    verdict = parse_judge_result(_verdict_dict(
        coverage=_coverage(("/redfish/v1/Managers/1", True)),  # wrong API -> mismatch.
        accepted=False,
        ambiguous=True,
    ))
    accepted, reasons = decide_accept(verdict, selected)
    assert not accepted
    assert REASON_NOT_ACCEPTED in reasons
    assert REASON_AMBIGUOUS in reasons
    assert any(r.startswith(REASON_COVERAGE_MISMATCH) for r in reasons)


# --- target eval (set semantics) -------------------------------------------


def test_target_set_matches_ignores_order() -> None:
    """Target eval treats the API list as an unordered set."""
    assert target_set_matches(["b", "a"], ["a", "b"])


def test_target_set_matches_fails_on_duplicates() -> None:
    """A duplicated prediction fails target eval even when the set is correct."""
    assert not target_set_matches(["a", "a"], ["a"])
    assert target_set_matches([], [])


# --- suite run / CLI plumbing ----------------------------------------------


def test_run_over_labelled_suite_agrees(tmp_path: Path) -> None:
    """A hand-labelled YAML suite of mixed accept/reject cases matches the formula."""
    suite = {
        "cases": [
            {
                "id": "accept-thermal-power",
                "category": "correct_multi_api",
                "judge_result": _verdict_dict(coverage=_coverage(
                    ("/redfish/v1/Chassis/1/Thermal", True),
                    ("/redfish/v1/Systems/1", True),
                )),
                "selected_rest_api_list": ["/redfish/v1/Systems/1", "/redfish/v1/Chassis/1/Thermal"],
                "expected_accepted": True,
            },
            {
                "id": "reject-missing",
                "category": "missing_api",
                "judge_result": _verdict_dict(coverage=_coverage(("/redfish/v1/Systems/1", True))),
                "selected_rest_api_list": ["/redfish/v1/Systems/1", "/redfish/v1/Managers/1"],
                "expected_accepted": False,
            },
            {
                "id": "reject-method-json-string",
                "category": "method_incompatible",
                # judge_result supplied as a JSON string to exercise string parsing.
                "judge_result": json.dumps(_verdict_dict(
                    coverage=_coverage(("/redfish/v1/Systems/1", True)),
                    method_semantics_valid=False,
                )),
                "selected_rest_api_list": ["/redfish/v1/Systems/1"],
                "expected_accepted": False,
            },
        ]
    }
    path = tmp_path / "suite.yaml"
    path.write_text(json.dumps(suite), encoding="utf-8")  # JSON is valid YAML.
    report = run(load_suite(path))
    assert report["accuracy"] == 1.0
    assert report["false_accepts"] == []
    assert is_consistent(report)


def test_run_flags_a_lenient_formula_expectation(tmp_path: Path) -> None:
    """If a case that must reject is mislabelled as accept, the suite is inconsistent."""
    suite = {
        "cases": [
            {
                "id": "bad-label",
                "category": "extra_api",
                "judge_result": _verdict_dict(coverage=_coverage(
                    ("/redfish/v1/Systems/1", True),
                    ("/redfish/v1/Chassis/1/Power", True),
                )),
                "selected_rest_api_list": ["/redfish/v1/Systems/1"],  # extra supported -> reject.
                "expected_accepted": True,  # wrong expectation.
            }
        ]
    }
    path = tmp_path / "suite.yaml"
    path.write_text(json.dumps(suite), encoding="utf-8")
    report = run(load_suite(path))
    assert report["accuracy"] < 1.0
    assert not is_consistent(report)


def test_load_suite_rejects_malformed(tmp_path: Path) -> None:
    """A YAML file without a 'cases' list is rejected."""
    path = tmp_path / "bad.yaml"
    path.write_text("version: 1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_suite(path)


# Author: Mus mbayramo@stanford.edu
