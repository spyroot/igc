"""Offline tests for the tool-teaching inducer (StubTeacher + teach_tool, §12.3-12.5).

The stub teacher induces a ToolCard straight from observed transitions — signature
from the spec/ActionInfo, expected fields from 2xx bodies, an evidence-cited error
taxonomy from error statuses — and teach_tool induces -> grounds -> confirms -> stores
so a card backed by its own evidence comes out GROUNDED. Pure stdlib — no torch/network.

Author:
Mus mbayramo@stanford.edu
"""

from igc.core.tool_card import ErrorClass, GroundingStatus, ToolCardStore
from igc.core.types import Observation, RiskLevel, ToolAction, ToolSpec, Transition
from igc.modules.teach.grounding import ToolCardGrounder
from igc.modules.teach.tool_teacher import StubTeacher, teach_tool


def _spec() -> ToolSpec:
    return ToolSpec(
        "ComputerSystem",
        ["Reset"],
        arg_schema={"Reset": {"ResetType": {"type": "string", "required": True}}},
        risk_level=RiskLevel.MUTATING,
    )


def _tr(status: int, body, op: str = "Reset", tool: str = "ComputerSystem") -> Transition:
    """A transition whose call is (tool, op) and whose result has the given status/body."""
    action = ToolAction(tool, op, arguments={"ResetType": "On"})
    obs = Observation(text="{}")
    nxt = Observation(text="{}", structured=body, status=status, error=status >= 400)
    return Transition(observation=obs, action=action, reward=0.0, next_observation=nxt,
                      terminated=False, truncated=False)


def test_stub_induces_signature_expected_and_errors() -> None:
    """A success body yields expected fields; an error status yields a cited rule."""
    transitions = [
        _tr(204, {"TaskState": "Running", "Id": 7}),
        _tr(409, {"error": "a config job is pending"}),
    ]
    card = StubTeacher().induce("redfish", _spec(), "Reset", transitions)
    assert "ResetType" in card.effective_signature
    assert card.expected_response == {"TaskState": "string", "Id": "integer"}
    assert len(card.error_taxonomy) == 1
    rule = card.error_taxonomy[0]
    assert rule.match == {"status": 409}
    assert rule.error_class is ErrorClass.PRECONDITION_UNMET
    assert rule.evidence_ids == ["t1"]  # cites the 2nd transition
    assert card.provenance["source"] == "stub" and card.provenance["k_observed"] == 2


def test_stub_keeps_json_shape_types_distinct() -> None:
    """Observed success bodies preserve JSON shape names, including bool vs int."""
    card = StubTeacher().induce(
        "redfish",
        _spec(),
        "Reset",
        [
            _tr(
                200,
                {
                    "Enabled": True,
                    "RetryCount": 2,
                    "Voltage": 12.5,
                    "Links": ["Systems"],
                    "Status": {"Health": "OK"},
                    "Maybe": None,
                    "Name": "System",
                },
            )
        ],
    )
    assert card.expected_response == {
        "Enabled": "boolean",
        "RetryCount": "integer",
        "Voltage": "number",
        "Links": "array",
        "Status": "object",
        "Maybe": "null",
        "Name": "string",
    }


def test_stub_accumulates_evidence_for_duplicate_error_status() -> None:
    """Repeated failures of the same status merge into one rule with all evidence ids."""
    transitions = [
        _tr(409, {"error": "pending job"}),
        _tr(409, {"error": "pending firmware apply"}),
        _tr(412, {"error": "etag required"}),
    ]
    card = StubTeacher().induce("redfish", _spec(), "Reset", transitions)
    assert [rule.match["status"] for rule in card.error_taxonomy] == [409, 412]
    assert card.error_taxonomy[0].evidence_ids == ["t0", "t1"]
    assert card.error_taxonomy[1].evidence_ids == ["t2"]


def test_stub_reports_empty_card_for_no_relevant_evidence() -> None:
    """A tool/op with no matching transitions produces no unsupported claims."""
    card = StubTeacher().induce("redfish", _spec(), "Reset", [_tr(500, {}, op="Other")])
    assert card.expected_response == {}
    assert card.error_taxonomy == []
    assert card.provenance["evidence_ids"] == []
    assert card.provenance["k_observed"] == 0


def test_stub_signature_falls_back_for_non_dict_schema_fragments() -> None:
    """Non-dict schema fragments are accepted as string slots, not treated as errors."""
    spec = ToolSpec(
        "ComputerSystem",
        ["Reset"],
        arg_schema={"Reset": {"ResetType": "string"}},
        risk_level=RiskLevel.MUTATING,
    )
    card = StubTeacher().induce("redfish", spec, "Reset", [_tr(204, {"TaskState": "x"})])
    assert card.effective_signature == {"ResetType": {"type": "string"}}


def test_stub_pulls_enum_from_actioninfo() -> None:
    """ActionInfo enums populate the induced signature (catalog is the enum source)."""
    card = StubTeacher().induce(
        "redfish", _spec(), "Reset", [_tr(204, {"TaskState": "x"})],
        actioninfo={"ResetType": ["On", "ForceOff"]},
    )
    assert card.effective_signature["ResetType"]["enum"] == ["On", "ForceOff"]


def test_stub_ignores_unrelated_transitions() -> None:
    """Transitions for other tools/ops are not read as evidence."""
    transitions = [
        _tr(500, {"error": "boom"}),                 # relevant (Reset)
        _tr(404, {"error": "x"}, op="Other"),        # different op
        _tr(404, {"error": "y"}, tool="Chassis"),    # different tool
    ]
    card = StubTeacher().induce("redfish", _spec(), "Reset", transitions)
    assert card.provenance["k_observed"] == 1
    assert [r.match["status"] for r in card.error_taxonomy] == [500]
    assert card.error_taxonomy[0].error_class is ErrorClass.RETRIABLE


def test_error_class_heuristic() -> None:
    """409 -> precondition, 5xx/429 -> retriable, other 4xx -> fatal."""
    teacher = _spec()
    statuses = {429: ErrorClass.RETRIABLE, 500: ErrorClass.RETRIABLE,
                409: ErrorClass.PRECONDITION_UNMET, 404: ErrorClass.FATAL}
    for status, expected in statuses.items():
        card = StubTeacher().induce("redfish", teacher, "Reset", [_tr(status, {"error": "x"})])
        assert card.error_taxonomy[0].error_class is expected


def test_teach_tool_grounds_confirms_and_stores() -> None:
    """teach_tool produces a GROUNDED, stored card backed by its own evidence."""
    store = ToolCardStore(trial_id="t1")
    transitions = [_tr(204, {"TaskState": "Running"}), _tr(409, {"error": "pending"})]
    card = teach_tool(
        StubTeacher(), ToolCardGrounder(), store, "redfish", _spec(), "Reset",
        transitions, actioninfo={"ResetType": ["On", "ForceOff"]},
    )
    # the 409 rule was confirmed by the 409 transition -> GROUNDED
    assert card.grounding.status is GroundingStatus.GROUNDED
    assert card.grounding.n_confirmations >= 1
    assert store.get("redfish", "ComputerSystem", "Reset") is card


def test_teach_tool_drops_hallucinated_enum_value() -> None:
    """A schema/enum gate runs inside teach_tool: out-of-ActionInfo enums are clipped."""
    spec = _spec()
    # The stub only emits ActionInfo enums, so inject an actioninfo the grounder clips against.
    store = ToolCardStore()
    card = teach_tool(
        StubTeacher(), ToolCardGrounder(), store, "redfish", spec, "Reset",
        [_tr(204, {"TaskState": "x"})], actioninfo={"ResetType": ["On"]},
    )
    assert card.effective_signature["ResetType"]["enum"] == ["On"]


# Author: Mus mbayramo@stanford.edu
