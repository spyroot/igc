"""Offline, pure-stdlib tests for igc.modules.encoders.backbone_utils (no torch).

Verifies the backbone-agnostic accessors handle GPT-2 (n_embd / n_positions /
.transformer), modern decoders (hidden_size / max_position_embeddings / model.model),
and RoPE models (no positional table -> None), using lightweight stub objects.

Author:
Mus mbayramo@stanford.edu
"""
from igc.modules.encoders.backbone_utils import backbone_module, hidden_size, max_positions


class StubConfig:
    """A minimal config double exposing arbitrary attributes."""

    def __init__(self, **attrs):
        self.__dict__.update(attrs)


class StubModel:
    """A model double exposing a base_model_prefix and a config."""

    def __init__(self, prefix, base, config):
        self.base_model_prefix = prefix
        self.config = config
        setattr(self, prefix, base)


def test_hidden_size_modern():
    """hidden_size reads config.hidden_size for a modern decoder."""
    assert hidden_size(StubConfig(hidden_size=4096)) == 4096


def test_hidden_size_gpt2_alias():
    """hidden_size falls back to the GPT-2 n_embd alias."""
    assert hidden_size(StubConfig(n_embd=768)) == 768


def test_hidden_size_from_model():
    """hidden_size accepts a model (reads model.config)."""
    model = StubModel("model", object(), StubConfig(hidden_size=2048))
    assert hidden_size(model) == 2048


def test_hidden_size_missing_raises():
    """An unknown config raises AttributeError."""
    try:
        hidden_size(StubConfig(foo=1))
    except AttributeError:
        return
    raise AssertionError("expected AttributeError")


def test_max_positions_modern():
    """max_positions reads max_position_embeddings."""
    assert max_positions(StubConfig(max_position_embeddings=8192)) == 8192


def test_max_positions_gpt2_alias():
    """max_positions falls back to the GPT-2 n_positions alias."""
    assert max_positions(StubConfig(n_positions=1024)) == 1024


def test_max_positions_rope_is_none():
    """A RoPE model with no positional table returns None."""
    assert max_positions(StubConfig(hidden_size=4096)) is None


def test_backbone_module_via_prefix():
    """backbone_module returns the submodule named by base_model_prefix."""
    base = object()
    model = StubModel("model", base, StubConfig(hidden_size=8))  # llama-style
    assert backbone_module(model) is base


def test_backbone_module_gpt2_prefix():
    """backbone_module handles GPT-2's .transformer prefix."""
    base = object()
    model = StubModel("transformer", base, StubConfig(n_embd=768))
    assert backbone_module(model) is base


def test_backbone_module_fallback_to_model():
    """With no usable prefix, backbone_module returns the model itself."""
    sentinel = object()

    class NoPrefix:
        base_model_prefix = ""

    m = NoPrefix()
    assert backbone_module(m) is m
    assert backbone_module(sentinel) is sentinel


# Author: Mus mbayramo@stanford.edu
