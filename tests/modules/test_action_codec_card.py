"""Offline CPU tests: ToolCards re-key candidates in ActionCodec (ARCHITECTURE §12.4-A).

Attaching a card to one candidate re-keys and re-embeds exactly that candidate while
every other candidate stays byte-identical to the cardless path, and cards=None is the
unchanged behavior. Uses the stub encoder — no model.

Author:
Mus mbayramo@stanford.edu
"""

import torch

from igc.core.tool_card import ToolCard
from igc.core.types import ToolAction, ToolSpec
from igc.modules.llm.llm_encoder import StubLLMEncoder
from igc.modules.policy.action_codec import ActionCodec


class CountingEncoder(StubLLMEncoder):
    """Stub encoder that records how many texts it embedded."""

    def __init__(self, hidden_size: int = 16):
        super().__init__(hidden_size)
        self.encoded_count = 0

    def encode(self, texts):
        self.encoded_count += len(texts)
        return super().encode(texts)


def _actions():
    return [
        ToolAction("ComputerSystem", "Reset", target="/redfish/v1/Systems/1"),
        ToolAction("Chassis", "GET", target="/redfish/v1/Chassis"),
    ]


def _specs():
    return [
        ToolSpec("ComputerSystem", ["Reset"], arg_schema={"Reset": {"ResetType": {"type": "string"}}}),
        ToolSpec("Chassis", ["GET"]),
    ]


def _card():
    return ToolCard(
        env_name="redfish", tool_name="ComputerSystem", op="Reset",
        effective_signature={"ResetType": {"type": "string", "required": True}},
        expected_response={"TaskState": "string"},
    )


def test_cards_none_is_unchanged() -> None:
    """encode(actions) and encode(actions, cards=None) give identical keys."""
    codec = ActionCodec(StubLLMEncoder(hidden_size=16), specs=_specs())
    a = _actions()
    assert torch.equal(codec.encode(a), codec.encode(a, cards=None))


def test_card_rekeys_only_its_candidate() -> None:
    """A card for one (tool, op) re-embeds that candidate; others are byte-identical."""
    codec = ActionCodec(StubLLMEncoder(hidden_size=16), specs=_specs())
    actions = _actions()
    plain = codec.encode(actions)
    carded = codec.encode(actions, cards={("ComputerSystem", "Reset"): _card()})

    assert not torch.equal(plain[0], carded[0])  # carded candidate re-embedded
    assert torch.equal(plain[1], carded[1])  # untouched candidate identical


def test_card_triggers_exactly_one_new_embedding() -> None:
    """The carded candidate is the only cache miss the second call adds."""
    enc = CountingEncoder(hidden_size=16)
    codec = ActionCodec(enc, specs=_specs())
    actions = _actions()
    codec.encode(actions)
    assert enc.encoded_count == 2  # two base templates
    codec.encode(actions, cards={("ComputerSystem", "Reset"): _card()})
    assert enc.encoded_count == 3  # only the carded candidate is new


def test_carded_actions_dedup_values_but_preserve_target_type() -> None:
    """Carded cache keys collapse value-only changes, not distinct targets."""
    enc = CountingEncoder(hidden_size=16)
    codec = ActionCodec(enc, specs=_specs())
    actions = [
        ToolAction(
            "ComputerSystem",
            "Reset",
            target="/redfish/v1/Systems/1",
            arguments={"ResetType": "On"},
        ),
        ToolAction(
            "ComputerSystem",
            "Reset",
            target="/redfish/v1/Systems/1",
            arguments={"ResetType": "ForceOff"},
        ),
        ToolAction(
            "ComputerSystem",
            "Reset",
            target="/redfish/v1/Systems/2",
            arguments={"ResetType": "On"},
        ),
    ]

    keys = codec.encode(actions, cards={("ComputerSystem", "Reset"): _card()})

    assert enc.encoded_count == 2
    assert torch.equal(keys[0], keys[1])
    assert not torch.equal(keys[0], keys[2])


# Author: Mus mbayramo@stanford.edu
