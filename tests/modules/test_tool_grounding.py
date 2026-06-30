"""Offline tests for ToolCardGrounder (tool-teaching grounding, ARCHITECTURE §12.5).

A teacher will fabricate; these pin the offline gates: uncited error rules are
dropped (evidence), out-of-schema arg slots are dropped and declared enums are
clipped to the catalog's allowable values (schema/enum — catalog overrides), a card
whose op schema moved is invalidated (stale), and real observations promote a card to
GROUNDED or CONTRADICTED (online falsification). Pure stdlib — no torch/network.

Author:
Mus mbayramo@stanford.edu
"""

from igc.core.tool_card import ErrorClass, ErrorRule, GroundingStatus, ToolCard
from igc.core.types import ToolSpec
from igc.modules.teach.grounding import ToolCardGrounder, confirm_against_observation


def _spec() -> ToolSpec:
    return ToolSpec(
        "ComputerSystem",
        ["Reset"],
        arg_schema={"Reset": {"ResetType": {"type": "string"}}},
    )


def test_evidence_gate_drops_uncited_error_rules() -> None:
    """An error rule citing no real transition is dropped; a cited one survives."""
    card = ToolCard(
        env_name="redfish", tool_name="ComputerSystem", op="Reset",
        error_taxonomy=[
            ErrorRule(match={"status": 409}, error_class=ErrorClass.PRECONDITION_UNMET, evidence_ids=["t1"]),
            ErrorRule(match={"status": 500}, error_class=ErrorClass.RETRIABLE, evidence_ids=[]),
        ],
    )
    grounded = ToolCardGrounder().ground(card, _spec())
    statuses = {r.match.get("status") for r in grounded.error_taxonomy}
    assert statuses == {409}


def test_evidence_gate_can_be_disabled_for_fixture_cards() -> None:
    """A local fixture can preserve uncited rules when evidence is not required."""
    card = ToolCard(
        env_name="redfish", tool_name="ComputerSystem", op="Reset",
        error_taxonomy=[
            ErrorRule(match={"status": 503}, error_class=ErrorClass.RETRIABLE, evidence_ids=[]),
        ],
    )
    grounded = ToolCardGrounder(require_evidence=False).ground(card, _spec())
    assert [r.match for r in grounded.error_taxonomy] == [{"status": 503}]


def test_schema_gate_drops_unknown_arg_slots() -> None:
    """An arg slot the catalog never declared is a hallucination and is dropped."""
    card = ToolCard(
        env_name="redfish", tool_name="ComputerSystem", op="Reset",
        effective_signature={"ResetType": {"type": "string"}, "Hallucinated": {"type": "string"}},
    )
    grounded = ToolCardGrounder().ground(card, _spec())
    assert set(grounded.effective_signature) == {"ResetType"}


def test_enum_clip_catalog_overrides_teacher() -> None:
    """A teacher enum value outside the catalog's ActionInfo set is removed."""
    card = ToolCard(
        env_name="redfish", tool_name="ComputerSystem", op="Reset",
        effective_signature={"ResetType": {"type": "string", "enum": ["On", "ForceOff", "Teleport"]}},
    )
    actioninfo = {"ResetType": ["On", "ForceOff", "GracefulShutdown"]}
    grounded = ToolCardGrounder().ground(card, _spec(), actioninfo=actioninfo)
    assert grounded.effective_signature["ResetType"]["enum"] == ["On", "ForceOff"]  # Teleport dropped


def test_observed_fields_clip_expected_response_claims() -> None:
    """Expected response fields are limited to fields observed in real bodies."""
    card = ToolCard(
        env_name="redfish", tool_name="ComputerSystem", op="Reset",
        expected_response={"TaskState": "string", "Missing": "string"},
    )
    grounded = ToolCardGrounder().ground(card, _spec(), observed_fields={"TaskState"})
    assert grounded.expected_response == {"TaskState": "string"}
    assert set(card.expected_response) == {"TaskState", "Missing"}


def test_stale_card_is_emptied_and_contradicted() -> None:
    """A card induced against a different op schema is invalidated."""
    card = ToolCard(
        env_name="redfish", tool_name="ComputerSystem", op="Reset",
        spec_fingerprint="deadbeef",  # does not match the real spec
        effective_signature={"ResetType": {"type": "string"}},
    )
    grounded = ToolCardGrounder().ground(card, _spec())
    assert grounded.effective_signature == {}
    assert grounded.grounding.status is GroundingStatus.CONTRADICTED


def test_observe_promotes_grounded_on_anticipated_error() -> None:
    """A real error the card anticipated confirms it (-> GROUNDED)."""
    card = ToolCard(
        env_name="redfish", tool_name="ComputerSystem", op="Reset",
        error_taxonomy=[ErrorRule(match={"status": 409}, error_class=ErrorClass.PRECONDITION_UNMET, evidence_ids=["t1"])],
    )
    ToolCardGrounder().observe(card, 409, {"error": "pending"})
    assert card.grounding.status is GroundingStatus.GROUNDED
    assert card.grounding.n_confirmations == 1


def test_observe_confirms_on_complete_expected_response() -> None:
    """A real 2xx body containing every declared response field confirms the card."""
    card = ToolCard(
        env_name="redfish", tool_name="ComputerSystem", op="Reset",
        expected_response={"TaskState": "string", "Id": "integer"},
    )
    ToolCardGrounder().observe(card, 200, {"TaskState": "Running", "Id": 7})
    assert card.grounding.status is GroundingStatus.GROUNDED
    assert card.grounding.n_confirmations == 1


def test_observe_contradicts_on_missing_expected_field() -> None:
    """A declared response field absent from a real 2xx body contradicts the card."""
    card = ToolCard(
        env_name="redfish", tool_name="ComputerSystem", op="Reset",
        expected_response={"TaskState": "string"},
    )
    ToolCardGrounder().observe(card, 200, {"SomethingElse": 1})
    assert card.grounding.n_contradictions == 1
    assert card.grounding.status is GroundingStatus.CONTRADICTED


def test_confirm_returns_none_for_untestable_observation() -> None:
    """An observation the card makes no prediction about leaves tallies untouched."""
    card = ToolCard(env_name="redfish", tool_name="ComputerSystem", op="Reset")
    assert confirm_against_observation(card, 404, {"error": "x"}) is None  # no error rules
    assert confirm_against_observation(card, 200, {"any": 1}) is None  # no expected_response


def test_ground_does_not_mutate_input_card() -> None:
    """ground returns a cleaned copy; the original card is left intact."""
    card = ToolCard(
        env_name="redfish", tool_name="ComputerSystem", op="Reset",
        effective_signature={"ResetType": {"type": "string"}, "Bad": {"type": "string"}},
    )
    ToolCardGrounder().ground(card, _spec())
    assert set(card.effective_signature) == {"ResetType", "Bad"}  # input unchanged


# Author: Mus mbayramo@stanford.edu
