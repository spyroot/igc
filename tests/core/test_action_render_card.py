"""Offline tests for the ToolCard injection seam in action_render (ARCHITECTURE §12.4-A).

The opt-in guarantee: with card=None the rendering and template key are byte-identical
to the pre-card behavior (the passive pointer path is untouched), and supplying a card
changes exactly that action's rendering and key. Pure stdlib — no torch.

Author:
Mus mbayramo@stanford.edu
"""

from igc.core.action_render import action_template_key, action_to_prompt
from igc.core.tool_card import ErrorClass, ErrorRule, ToolCard
from igc.core.types import ToolAction, ToolSpec


def _action() -> ToolAction:
    return ToolAction("ComputerSystem", "Reset", arguments={"ResetType": "On"}, target="/redfish/v1/Systems/1")


def _spec() -> ToolSpec:
    return ToolSpec("ComputerSystem", ["Reset"], arg_schema={"Reset": {"ResetType": {"type": "string"}}})


def _card() -> ToolCard:
    return ToolCard(
        env_name="redfish",
        tool_name="ComputerSystem",
        op="Reset",
        effective_signature={"ResetType": {"type": "string", "required": True}},
        error_taxonomy=[ErrorRule(match={"status": 409}, error_class=ErrorClass.PRECONDITION_UNMET, evidence_ids=["t1"])],
    )


def test_card_none_renders_byte_identical() -> None:
    """card=None reproduces the original rendering exactly (parity guarantee)."""
    action, spec = _action(), _spec()
    base = action_to_prompt(action, spec)
    assert action_to_prompt(action, spec, None) == base
    assert action_to_prompt(action, spec, card=None) == base
    assert "card=[" not in base


def test_card_none_key_byte_identical() -> None:
    """card=None reproduces the original template key exactly."""
    action, spec = _action(), _spec()
    assert action_template_key(action, spec, None) == action_template_key(action, spec)


def test_card_changes_rendering_and_key() -> None:
    """A supplied card appends its clause and changes the template key."""
    action, spec, card = _action(), _spec(), _card()
    rendered = action_to_prompt(action, spec, card)
    assert rendered.startswith(action_to_prompt(action, spec))  # card clause is appended
    assert "card=[" in rendered
    assert action_template_key(action, spec, card) != action_template_key(action, spec)


def test_card_clause_is_value_independent() -> None:
    """Two actions differing only in argument VALUES render identically with the same card."""
    spec, card = _spec(), _card()
    a = ToolAction("ComputerSystem", "Reset", arguments={"ResetType": "On"})
    b = ToolAction("ComputerSystem", "Reset", arguments={"ResetType": "ForceOff"})
    assert action_to_prompt(a, spec, card) == action_to_prompt(b, spec, card)


# Author: Mus mbayramo@stanford.edu
