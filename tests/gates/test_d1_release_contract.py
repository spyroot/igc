"""Offline tests for the Phase 2 D1 release-contract gate.

These tests encode the GENERAL D1 contract, not a single example: every fixture
here is a synthetic row/list built in-test, so passing depends on the contract
logic (text-only ``x``, unordered-unique-set target, exact acceptance logic), not
on the one illustrative BIOS/Certificates row committed in the YAML. The committed
reference is only checked for self-consistency via the gate ``check``.

Covered: a valid row passes; ``x`` carrying json/allowed_methods fails as leakage;
a duplicate ``y_true.rest_api_list`` fails; wrong dataset / target_semantics fail;
a set-equal-but-reordered prediction still matches; a duplicate prediction fails
eval; and the exact judge acceptance logic accepts/rejects per field. Pure
fixtures: no corpus, model, or Phase render import.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

from pathlib import Path

from scripts.gates.d1_release_contract import (
    EXIT_OK,
    check,
    compute_acceptance,
    evaluate_target,
    validate_d1_row,
    validate_judge_result,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACT = REPO_ROOT / "configs/contracts/d1_contract.yaml"


def _valid_row() -> dict:
    """Build a synthetic valid D1 row from scratch (not the YAML reference)."""
    return {
        "phase": 2,
        "dataset": "D1",
        "source_dataset": "D0",
        "model_x": "model_x",
        "task": "text_to_rest_api_list",
        "target_semantics": "unordered_unique_set",
        "x": {"text": "reset the manager and read the thermal sensors"},
        "y_true": {
            "rest_api_list": [
                "/redfish/v1/Managers/1/Actions/Manager.Reset",
                "/redfish/v1/Chassis/1/Thermal",
            ]
        },
        "validation": {
            "text_source": "model_x_then_pro_judge",
            "review_judged": True,
            "natural": True,
            "exact_api_coverage": True,
            "extra_intent": False,
            "duplicate_intent": False,
            "ambiguous": False,
            "nonsense": False,
            "method_semantics_valid": True,
        },
    }


def _valid_verdict() -> dict:
    """Build a synthetic well-formed accepting judge result for the row above."""
    return {
        "accepted": True,
        "natural": True,
        "nonsense": False,
        "ambiguous": False,
        "duplicate_intent": False,
        "method_semantics_valid": True,
        "coverage": [
            {
                "rest_api": "/redfish/v1/Managers/1/Actions/Manager.Reset",
                "text_span": "reset the manager",
                "supported": True,
            },
            {
                "rest_api": "/redfish/v1/Chassis/1/Thermal",
                "text_span": "read the thermal sensors",
                "supported": True,
            },
        ],
        "extra_intents": [],
        "reason": "text maps to exactly the two selected APIs",
    }


# --------------------------------------------------------------------------- #
# validate_d1_row
# --------------------------------------------------------------------------- #
def test_valid_row_passes() -> None:
    """A synthetic well-formed D1 row yields zero violations."""
    assert validate_d1_row(_valid_row()) == []


def test_empty_row_is_valid_hard_negative() -> None:
    """An empty rest_api_list (no-action / hard-negative row) is still valid."""
    row = _valid_row()
    row["x"]["text"] = "everything looks fine, do nothing"
    row["y_true"]["rest_api_list"] = []
    assert validate_d1_row(row) == []


def test_x_carrying_json_fails_as_leakage() -> None:
    """A ``json`` body inside ``x`` is reported as Phase-2 input leakage."""
    row = _valid_row()
    row["x"]["json"] = {"@odata.id": "/redfish/v1/Systems/1"}
    violations = validate_d1_row(row)
    assert any("input leakage" in v and "x.json" in v for v in violations), violations


def test_x_carrying_allowed_methods_fails_as_leakage() -> None:
    """``allowed_methods`` inside ``x`` is Phase-2 input leakage."""
    row = _valid_row()
    row["x"]["allowed_methods"] = ["GET", "PATCH"]
    violations = validate_d1_row(row)
    assert any("input leakage" in v and "x.allowed_methods" in v for v in violations), violations


def test_x_carrying_rest_api_fails_as_leakage() -> None:
    """A raw ``rest_api`` path inside ``x`` is Phase-2 input leakage."""
    row = _valid_row()
    row["x"]["rest_api"] = "/redfish/v1/Systems/1"
    violations = validate_d1_row(row)
    assert any("input leakage" in v and "x.rest_api" in v for v in violations), violations


def test_x_text_must_be_string() -> None:
    """A non-string ``x.text`` is a type violation."""
    row = _valid_row()
    row["x"]["text"] = 123
    violations = validate_d1_row(row)
    assert any("'x.text' must be str" in v for v in violations), violations


def test_duplicate_rest_api_list_fails() -> None:
    """A duplicated entry in ``y_true.rest_api_list`` violates the unique-set target."""
    row = _valid_row()
    row["y_true"]["rest_api_list"] = [
        "/redfish/v1/Systems/1",
        "/redfish/v1/Systems/1",
    ]
    violations = validate_d1_row(row)
    assert any("duplicate entries" in v for v in violations), violations


def test_non_string_api_entry_fails() -> None:
    """A non-string entry in the API list is reported by index."""
    row = _valid_row()
    row["y_true"]["rest_api_list"] = ["/redfish/v1/Systems/1", 42]
    violations = validate_d1_row(row)
    assert any("'y_true.rest_api_list[1]' must be str" in v for v in violations), violations


def test_wrong_dataset_value_fails() -> None:
    """A ``dataset`` other than ``D1`` is a value violation."""
    row = _valid_row()
    row["dataset"] = "D0"
    violations = validate_d1_row(row)
    assert any("'dataset' must be 'D1'" in v for v in violations), violations


def test_wrong_target_semantics_fails() -> None:
    """A ``target_semantics`` other than the locked value fails."""
    row = _valid_row()
    row["target_semantics"] = "ordered_list"
    violations = validate_d1_row(row)
    assert any("'target_semantics' must be 'unordered_unique_set'" in v for v in violations), violations


def test_wrong_source_dataset_fails() -> None:
    """A ``source_dataset`` other than ``D0`` fails (D1 is drafted from D0)."""
    row = _valid_row()
    row["source_dataset"] = "D2"
    violations = validate_d1_row(row)
    assert any("'source_dataset' must be 'D0'" in v for v in violations), violations


def test_wrong_phase_type_fails() -> None:
    """A boolean ``phase`` is rejected even though bool subclasses int."""
    row = _valid_row()
    row["phase"] = True
    violations = validate_d1_row(row)
    assert any("'phase' must be int" in v for v in violations), violations


def test_missing_validation_field_fails() -> None:
    """Dropping a validation flag is reported as a missing key."""
    row = _valid_row()
    del row["validation"]["method_semantics_valid"]
    violations = validate_d1_row(row)
    assert any("missing validation key: 'method_semantics_valid'" in v for v in violations), violations


def test_non_bool_validation_flag_fails() -> None:
    """A validation flag that is not a real bool is a type violation."""
    row = _valid_row()
    row["validation"]["natural"] = "true"
    violations = validate_d1_row(row)
    assert any("'validation.natural' must be bool" in v for v in violations), violations


def test_extra_top_level_key_fails() -> None:
    """An unexpected extra top-level key is rejected."""
    row = _valid_row()
    row["allowed_methods"] = ["GET"]
    violations = validate_d1_row(row)
    assert any("unexpected top-level key: 'allowed_methods'" in v for v in violations), violations


def test_non_dict_row_reported() -> None:
    """A non-dict row is reported instead of raising."""
    assert validate_d1_row(["not", "a", "row"]) == ["row must be a dict, got list"]


# --------------------------------------------------------------------------- #
# evaluate_target
# --------------------------------------------------------------------------- #
def test_target_reordered_still_matches() -> None:
    """A set-equal but reordered prediction still matches (order-agnostic)."""
    expected = ["/redfish/v1/A", "/redfish/v1/B"]
    pred = ["/redfish/v1/B", "/redfish/v1/A"]
    assert evaluate_target(pred, expected) is True


def test_target_exact_match() -> None:
    """An identical prediction matches."""
    apis = ["/redfish/v1/Systems/1"]
    assert evaluate_target(apis, apis) is True


def test_target_empty_matches_empty() -> None:
    """A no-action prediction matches a no-action expected set."""
    assert evaluate_target([], []) is True


def test_target_duplicate_prediction_fails() -> None:
    """A duplicated prediction fails even though the underlying set matches."""
    expected = ["/redfish/v1/A", "/redfish/v1/B"]
    pred = ["/redfish/v1/A", "/redfish/v1/B", "/redfish/v1/B"]
    assert evaluate_target(pred, expected) is False


def test_target_missing_api_fails() -> None:
    """A prediction missing an expected API fails."""
    expected = ["/redfish/v1/A", "/redfish/v1/B"]
    assert evaluate_target(["/redfish/v1/A"], expected) is False


def test_target_extra_api_fails() -> None:
    """A prediction with an extra API fails."""
    expected = ["/redfish/v1/A"]
    assert evaluate_target(["/redfish/v1/A", "/redfish/v1/B"], expected) is False


# --------------------------------------------------------------------------- #
# compute_acceptance (exact judge logic)
# --------------------------------------------------------------------------- #
def test_acceptance_accepts_clean_verdict() -> None:
    """A clean verdict whose supported coverage equals the label is accepted."""
    verdict = _valid_verdict()
    selected = [c["rest_api"] for c in verdict["coverage"]]
    assert compute_acceptance(verdict, selected) is True


def test_acceptance_rejects_when_not_natural() -> None:
    """A verdict flagged not-natural is rejected."""
    verdict = _valid_verdict()
    verdict["natural"] = False
    selected = [c["rest_api"] for c in verdict["coverage"]]
    assert compute_acceptance(verdict, selected) is False


def test_acceptance_rejects_on_extra_intents() -> None:
    """A non-empty extra_intents list rejects the draft."""
    verdict = _valid_verdict()
    verdict["extra_intents"] = ["/redfish/v1/AccountService"]
    selected = [c["rest_api"] for c in verdict["coverage"]]
    assert compute_acceptance(verdict, selected) is False


def test_acceptance_rejects_on_coverage_mismatch() -> None:
    """When supported coverage != the selected set, the draft is rejected."""
    verdict = _valid_verdict()
    # One selected API is not actually supported by the text.
    verdict["coverage"][1]["supported"] = False
    selected = [c["rest_api"] for c in verdict["coverage"]]
    assert compute_acceptance(verdict, selected) is False


def test_acceptance_rejects_on_ambiguous_nonsense_duplicate() -> None:
    """Any of ambiguous / nonsense / duplicate_intent rejects the draft."""
    selected = ["/redfish/v1/A", "/redfish/v1/B"]
    for flag in ("ambiguous", "nonsense", "duplicate_intent"):
        verdict = {
            "accepted": True,
            "natural": True,
            "nonsense": False,
            "ambiguous": False,
            "duplicate_intent": False,
            "method_semantics_valid": True,
            "coverage": [
                {"rest_api": "/redfish/v1/A", "text_span": "a", "supported": True},
                {"rest_api": "/redfish/v1/B", "text_span": "b", "supported": True},
            ],
            "extra_intents": [],
            "reason": "ok",
        }
        verdict[flag] = True
        assert compute_acceptance(verdict, selected) is False, flag


def test_acceptance_rejects_bad_method_semantics() -> None:
    """method_semantics_valid=False rejects the draft."""
    verdict = _valid_verdict()
    verdict["method_semantics_valid"] = False
    selected = [c["rest_api"] for c in verdict["coverage"]]
    assert compute_acceptance(verdict, selected) is False


# --------------------------------------------------------------------------- #
# validate_judge_result + gate check
# --------------------------------------------------------------------------- #
def test_valid_judge_result_passes() -> None:
    """A well-formed judge result yields zero violations."""
    assert validate_judge_result(_valid_verdict()) == []


def test_judge_result_missing_key_fails() -> None:
    """A judge result missing ``coverage`` is reported."""
    verdict = _valid_verdict()
    del verdict["coverage"]
    violations = validate_judge_result(verdict)
    assert any("missing judge key: 'coverage'" in v for v in violations), violations


def test_judge_coverage_entry_bad_type_fails() -> None:
    """A non-bool ``supported`` in a coverage entry is a type violation."""
    verdict = _valid_verdict()
    verdict["coverage"][0]["supported"] = "yes"
    violations = validate_judge_result(verdict)
    assert any("coverage[0].supported' must be bool" in v for v in violations), violations


def test_gate_check_passes_on_committed_contract() -> None:
    """The gate check returns EXIT_OK for the committed illustrative reference."""
    assert check(CONTRACT) == EXIT_OK



def test_compute_acceptance_fail_closed_on_missing_negative_flag() -> None:
    """A verdict omitting a required flag must reject, never slip through (fail-closed)."""
    selected = [
        "/redfish/v1/Managers/1/Actions/Manager.Reset",
        "/redfish/v1/Chassis/1/Thermal",
    ]
    assert compute_acceptance(_valid_verdict(), selected) is True  # full verdict accepts
    missing_nonsense = _valid_verdict()
    del missing_nonsense["nonsense"]
    assert compute_acceptance(missing_nonsense, selected) is False
    missing_extra = _valid_verdict()
    del missing_extra["extra_intents"]
    assert compute_acceptance(missing_extra, selected) is False

# Author: Mus mbayramo@stanford.edu


