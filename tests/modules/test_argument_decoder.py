"""Offline tests for the second-stage argument decoder.

Covers schema-driven slot enumeration, no-explosion categorical selection (width =
the slot's own choice count), required-argument enforcement, applying values to a
template, and — where torch is present — the CategoricalArgScorer output width.

Author:
Mus mbayramo@stanford.edu
"""
import pytest

from igc.core.types import RiskLevel, ToolAction, ToolSpec
from igc.modules.policy.argument_decoder import (
    ArgumentSlot,
    apply_arguments,
    arg_slots_for,
    assemble_arguments,
    select_categorical,
)


def _reset_spec():
    """A Redfish-style reset tool: POST takes a required categorical ResetType."""
    return ToolSpec(
        tool_name="redfish",
        ops=["POST"],
        arg_schema={
            "POST": {
                "ResetType": {
                    "type": "string",
                    "enum": ["On", "ForceOff", "GracefulShutdown"],
                    "required": True,
                }
            }
        },
        risk_level=RiskLevel.MUTATING,
    )


def test_arg_slots_for_reads_enum_required_and_type():
    """arg_slots_for maps an enum to choices and carries type/required."""
    slots = arg_slots_for(_reset_spec(), "POST")
    assert len(slots) == 1
    slot = slots[0]
    assert slot.name == "ResetType" and slot.type == "string"
    assert slot.choices == ("On", "ForceOff", "GracefulShutdown")
    assert slot.required and slot.is_categorical


def test_arg_slots_for_read_op_has_no_slots():
    """An op with no declared arguments (a GET) yields an empty slot list."""
    spec = ToolSpec(tool_name="redfish", ops=["GET"], arg_schema={})
    assert arg_slots_for(spec, "GET") == []


def test_arg_slots_for_non_dict_fragment_is_freeform():
    """A non-dict schema entry degrades to a free-form (non-categorical) slot."""
    spec = ToolSpec(tool_name="t", ops=["POST"], arg_schema={"POST": {"raw": "string"}})
    slot = arg_slots_for(spec, "POST")[0]
    assert slot.name == "raw" and not slot.is_categorical


def test_select_categorical_is_argmax_over_choice_width():
    """Selection picks the argmax; the score width equals the slot's choice count."""
    slot = arg_slots_for(_reset_spec(), "POST")[0]
    scores = [0.1, 0.9, 0.2]  # width == 3 == len(choices), not a global vocab
    assert select_categorical(slot, scores) == "ForceOff"


def test_select_categorical_rejects_wrong_width_and_freeform():
    """Bad score width or a free-form slot raises rather than mis-indexing."""
    slot = arg_slots_for(_reset_spec(), "POST")[0]
    with pytest.raises(ValueError):
        select_categorical(slot, [0.1, 0.9])  # width 2 != 3 choices
    free = ArgumentSlot(name="x")
    with pytest.raises(ValueError):
        select_categorical(free, [1.0])


def test_assemble_arguments_enforces_required_and_skips_optional_none():
    """Required slots must be present; an unset optional slot is simply omitted."""
    required = arg_slots_for(_reset_spec(), "POST")
    assert assemble_arguments(required, {"ResetType": "On"}) == {"ResetType": "On"}
    with pytest.raises(ValueError):
        assemble_arguments(required, {})  # required ResetType missing
    optional = [ArgumentSlot(name="Note")]
    assert assemble_arguments(optional, {"Note": None}) == {}


def test_apply_arguments_fills_template_values_only():
    """apply_arguments sets arguments and leaves the template's other fields intact."""
    template = ToolAction(
        tool_name="redfish", op="POST",
        target="/redfish/v1/Systems/1/Actions/ComputerSystem.Reset",
        risk_level=RiskLevel.MUTATING,
    )
    action = apply_arguments(template, {"ResetType": "ForceOff"})
    assert action.arguments == {"ResetType": "ForceOff"}
    assert action.tool_name == "redfish" and action.op == "POST"
    assert action.target == template.target and action.risk_level == RiskLevel.MUTATING
    assert template.arguments == {}  # original template unchanged


def test_categorical_scorer_output_width_tracks_choices():
    """CategoricalArgScorer emits one score per candidate value (no explosion)."""
    torch = pytest.importorskip("torch")
    from igc.modules.policy.argument_decoder import CategoricalArgScorer

    scorer = CategoricalArgScorer(h_dim=8, q_dim=4)
    state_h = torch.randn(2, 8)
    value_h = torch.randn(2, 3, 8)  # K=3 candidate values
    scores = scorer(state_h, value_h)
    assert scores.shape == (2, 3)  # width == K, never a global value count


# Author: Mus mbayramo@stanford.edu
