"""Offline CPU tests for StubLLMEncoder (no model, no download).

Verifies the stub encoder is deterministic, cached, unit-norm, and satisfies the
TextEncoder protocol — enough to drive the pointer policy + candidate cache offline.
The real LLMEncoder is download/GPU-gated and not exercised here.

Author:
Mus mbayramo@stanford.edu
"""
import torch

from igc.modules.llm.llm_encoder import StubLLMEncoder, TextEncoder


def test_encode_shape_and_determinism():
    """encode returns [B, H] and is deterministic across calls."""
    enc = StubLLMEncoder(hidden_size=32)
    texts = ["tool=redfish op=GET target=/redfish/v1", "tool=fs op=ls args=[]"]
    a = enc.encode(texts)
    b = enc.encode(texts)
    assert a.shape == (2, 32)
    assert torch.equal(a, b)


def test_cache_hit_returns_identical_vector():
    """The same text yields the identical cached vector."""
    enc = StubLLMEncoder(hidden_size=16)
    v1 = enc.encode(["same"])
    v2 = enc.encode(["same"])
    assert torch.equal(v1, v2)
    assert "same" in enc._cache


def test_distinct_texts_get_distinct_vectors():
    """Different texts map to different embeddings."""
    enc = StubLLMEncoder(hidden_size=16)
    out = enc.encode(["a", "b"])
    assert not torch.equal(out[0], out[1])


def test_vectors_are_unit_norm():
    """Stub embeddings are L2-normalized."""
    enc = StubLLMEncoder(hidden_size=64)
    out = enc.encode(["x", "y", "z"])
    assert torch.allclose(out.norm(dim=-1), torch.ones(3), atol=1e-5)


def test_empty_list_is_well_shaped():
    """An empty input yields a [0, H] tensor."""
    enc = StubLLMEncoder(hidden_size=8)
    assert enc.encode([]).shape == (0, 8)


def test_stub_satisfies_text_encoder_protocol():
    """StubLLMEncoder structurally satisfies the TextEncoder protocol."""
    assert isinstance(StubLLMEncoder(8), TextEncoder)


# Author: Mus mbayramo@stanford.edu
