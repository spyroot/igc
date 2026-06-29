"""Offline CPU tests for ActionCodec (renders + embeds ToolActions, cached by type).

Verifies the [N, H] key shape, that identical action TYPES (value-only differences)
embed once and identically, that the encoder is not re-called on cache hits, and the
empty-candidate edge. Uses the stub encoder — no model.

Author:
Mus mbayramo@stanford.edu
"""
import torch

from igc.core.types import ToolAction
from igc.modules.llm.llm_encoder import StubLLMEncoder
from igc.modules.policy.action_codec import ActionCodec


class CountingEncoder(StubLLMEncoder):
    """Stub encoder that records how many texts it has embedded."""

    def __init__(self, hidden_size: int = 16):
        super().__init__(hidden_size)
        self.encoded_count = 0

    def encode(self, texts):
        self.encoded_count += len(texts)
        return super().encode(texts)


def test_encode_returns_keys_for_each_candidate():
    """encode returns one [H] key per candidate, in order."""
    codec = ActionCodec(StubLLMEncoder(hidden_size=16))
    actions = [
        ToolAction("redfish", "GET", target="/redfish/v1/Systems"),
        ToolAction("redfish", "GET", target="/redfish/v1/Chassis"),
    ]
    keys = codec.encode(actions)
    assert keys.shape == (2, 16)


def test_same_type_embeds_once_and_identically():
    """Two actions of the same TYPE (value-only diff) embed once and share the key."""
    enc = CountingEncoder(hidden_size=16)
    codec = ActionCodec(enc)
    a = ToolAction("fs", "write", target="/tmp/x", arguments={"content": "hello"})
    b = ToolAction("fs", "write", target="/tmp/x", arguments={"content": "world"})
    keys = codec.encode([a, b])
    assert enc.encoded_count == 1  # one unique template
    assert torch.equal(keys[0], keys[1])  # identical key
    assert len(codec._cache) == 1


def test_cache_hit_does_not_recall_encoder():
    """Re-encoding cached candidates does not call the encoder again."""
    enc = CountingEncoder(hidden_size=8)
    codec = ActionCodec(enc)
    actions = [ToolAction("sql", "SELECT", target="users"), ToolAction("sql", "SELECT", target="orders")]
    codec.encode(actions)
    assert enc.encoded_count == 2
    codec.encode(actions)  # all cached now
    assert enc.encoded_count == 2  # unchanged


def test_distinct_types_get_distinct_keys():
    """Different action types map to different keys."""
    codec = ActionCodec(StubLLMEncoder(hidden_size=16))
    keys = codec.encode(
        [ToolAction("redfish", "GET", target="/a"), ToolAction("redfish", "HEAD", target="/a")]
    )
    assert not torch.equal(keys[0], keys[1])


def test_empty_candidates_is_well_shaped():
    """No candidates yields a [0, H] tensor."""
    codec = ActionCodec(StubLLMEncoder(hidden_size=8))
    assert codec.encode([]).shape == (0, 8)


# Author: Mus mbayramo@stanford.edu
