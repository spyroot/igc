"""Offline tests for the ToolCard artifact (tool-teaching, ARCHITECTURE §12.2).

Covers JSON round-trip, content-hash stability (changes with learned content, not
with the volatile grounding tallies), spec-fingerprint stale detection, the bounded
value-independent render clause, error classification, the grounding state machine,
and the per-trial ToolCardStore. Pure stdlib — no torch/network.

Author:
Mus mbayramo@stanford.edu
"""

from igc.core.tool_card import (
    ErrorClass,
    ErrorRule,
    Grounding,
    GroundingStatus,
    ToolCard,
    ToolCardStore,
)
from igc.core.types import RiskLevel, ToolAction, ToolSpec


def _card() -> ToolCard:
    """A representative card with a signature, response, and one cited error rule."""
    return ToolCard(
        env_name="redfish",
        tool_name="ComputerSystem",
        op="Reset",
        spec_fingerprint="",
        effective_signature={"ResetType": {"type": "string", "enum": ["On", "ForceOff"], "required": True}},
        expected_response={"TaskState": "string"},
        error_taxonomy=[
            ErrorRule(
                match={"status": 409},
                error_class=ErrorClass.PRECONDITION_UNMET,
                meaning="a config job is pending",
                evidence_ids=["t3"],
            )
        ],
        provenance={"source": "stub", "evidence_ids": ["t3"], "k_observed": 4},
    )


def test_tool_card_dict_round_trip() -> None:
    """from_dict(card.to_dict()) reconstructs an equal card."""
    card = _card()
    assert ToolCard.from_dict(card.to_dict()) == card


def test_content_hash_ignores_grounding_and_version() -> None:
    """The content hash tracks learned content, not the volatile trust state."""
    card = _card()
    h0 = card.content_hash()
    card.grounding.record(True)  # counter tick
    card.version = 7
    assert card.content_hash() == h0
    card.error_taxonomy.append(
        ErrorRule(match={"status": 500}, error_class=ErrorClass.RETRIABLE, evidence_ids=["t9"])
    )
    assert card.content_hash() != h0  # learned content changed


def test_spec_fingerprint_detects_stale_card() -> None:
    """A card whose op schema moved is stale relative to the new spec."""
    spec = ToolSpec("ComputerSystem", ["Reset"], arg_schema={"Reset": {"ResetType": {"type": "string"}}})
    card = _card()
    card.spec_fingerprint = ToolCard.compute_spec_fingerprint(spec, "Reset")
    assert not card.is_stale(spec)
    moved = ToolSpec("ComputerSystem", ["Reset"], arg_schema={"Reset": {"ResetType": {"type": "string"}, "extra": {"type": "integer"}}})
    assert card.is_stale(moved)


def test_render_clause_is_bounded_deterministic_and_value_independent() -> None:
    """The clause is stable, capped, and exposes shape (not concrete values)."""
    card = _card()
    clause = card.render_clause()
    assert clause == card.render_clause()  # deterministic
    assert len(clause) <= 240
    assert clause.startswith("card=[") and clause.endswith("]")
    assert "ResetType" in clause and "409=precondition_unmet" in clause and "TaskState" in clause
    assert "ForceOff" not in clause  # enum membership is shape; not spelled out as a value


def test_render_clause_truncates_long_cards() -> None:
    """A huge signature still yields a clause within the length cap."""
    card = _card()
    card.effective_signature = {f"slot{i}": {"type": "string"} for i in range(200)}
    assert len(card.render_clause(max_len=80)) <= 80


def test_classify_error_matches_first_rule() -> None:
    """classify_error returns the matched class, or None when nothing matches."""
    card = _card()
    assert card.classify_error(409, {"error": "pending"}) is ErrorClass.PRECONDITION_UNMET
    assert card.classify_error(404, {"error": "x"}) is None


def test_error_rule_code_and_substring_match() -> None:
    """A rule keyed by code/body_substring matches the serialized body."""
    rule = ErrorRule(match={"code": "Base.1.0.PropertyMissing"}, error_class=ErrorClass.FATAL)
    assert rule.matches(400, {"error": {"code": "Base.1.0.PropertyMissing"}})
    assert not rule.matches(400, {"error": {"code": "Other"}})


def test_grounding_state_machine() -> None:
    """Confirmations promote to GROUNDED; a contradiction majority -> CONTRADICTED."""
    g = Grounding()
    assert g.status is GroundingStatus.PROVISIONAL
    g.record(True)
    assert g.status is GroundingStatus.GROUNDED
    g.record(False)
    g.record(False)
    assert g.status is GroundingStatus.CONTRADICTED


def test_store_put_get_and_action_lookup() -> None:
    """The store keys cards by (env, tool, op) and resolves them for an action."""
    store = ToolCardStore(trial_id="trial-1")
    card = _card()
    store.put(card)
    assert len(store) == 1
    assert ("redfish", "ComputerSystem", "Reset") in store
    assert store.get("redfish", "ComputerSystem", "Reset") is card
    action = ToolAction("ComputerSystem", "Reset", risk_level=RiskLevel.MUTATING)
    assert store.for_action(action, "redfish") is card
    assert store.for_action(action, "sql") is None  # env-scoped


def test_store_round_trip() -> None:
    """The store serializes and reconstructs its trial id and every card."""
    store = ToolCardStore(trial_id="trial-9")
    store.put(_card())
    back = ToolCardStore.from_dict(store.to_dict())
    assert back.trial_id == "trial-9"
    assert back.get("redfish", "ComputerSystem", "Reset") == _card()


# Author: Mus mbayramo@stanford.edu
