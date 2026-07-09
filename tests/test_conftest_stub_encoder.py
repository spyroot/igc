"""Tests for the shared ``StubEncoder`` fixtures in ``tests/conftest.py``.

These guard the encoder contract that other offline tests rely on: a
deterministic, CPU-only encoder with the expected output shape that records the
observations it is handed.

Author:
Mus mbayramo@stanford.edu
"""

import torch


def test_stub_encoder_output_shape_and_dtype(stub_encoder):
    """encode() returns a float32 tensor of the advertised emb_shape."""
    out = stub_encoder.encode("/redfish/v1")
    assert out.shape == stub_encoder.emb_shape
    assert out.dtype == torch.float32


def test_stub_encoder_is_deterministic(stub_encoder):
    """Equal observations encode to equal tensors (no randomness/I/O)."""
    first = stub_encoder.encode("same-observation")
    second = stub_encoder.encode("same-observation")
    assert torch.equal(first, second)


def test_stub_encoder_distinguishes_inputs(stub_encoder):
    """Different observations map to different embeddings."""
    a = stub_encoder.encode("observation-a")
    b = stub_encoder.encode("observation-b")
    assert not torch.equal(a, b)


def test_stub_encoder_records_calls(stub_encoder):
    """Every encoded observation is appended to calls in order."""
    stub_encoder.encode("first")
    stub_encoder.encode("second")
    assert stub_encoder.calls == ["first", "second"]


def test_stub_encoder_cls_fixture_constructs_with_model_and_tokenizer(stub_encoder_cls):
    """The class fixture builds with (model, tokenizer) like the real encoder."""
    sentinel_model = object()
    sentinel_tok = object()
    encoder = stub_encoder_cls(model=sentinel_model, tokenizer=sentinel_tok)
    assert encoder.model is sentinel_model
    assert encoder.tokenizer is sentinel_tok
    assert encoder.calls == []


# Author: Mus mbayramo@stanford.edu
