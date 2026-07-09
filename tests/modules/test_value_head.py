"""Offline tests for the ``ValueHead`` module.

Pins the contract the reward-model / RL path will build on: per-token logits
of ``num_classes`` width, input width from ``word_embed_proj_dim`` (OPT-style)
else ``hidden_size``, dtype-safe forward (casts to the head's weight dtype),
and dropout controlled by ``summary_dropout_prob`` (0 disables it). CPU-only.

Author:
Mus mbayramo@stanford.edu
"""

import argparse

import pytest
import torch
import torch.nn as nn

from igc.modules.llm.value_head import ValueHead


def _config(**fields) -> argparse.Namespace:
    """A minimal config double with just the given attributes."""
    return argparse.Namespace(**fields)


def test_forward_shape_per_token_logits():
    """(B, T, H) hidden states project to (B, T, num_classes)."""
    head = ValueHead(_config(hidden_size=8, summary_dropout_prob=0.0), num_classes=3)
    out = head(torch.randn(2, 5, 8))
    assert out.shape == (2, 5, 3)


def test_scalar_value_head():
    """num_classes=1 gives the trl-style scalar value head."""
    head = ValueHead(_config(hidden_size=4, summary_dropout_prob=0.0), num_classes=1)
    assert head(torch.randn(2, 7, 4)).shape == (2, 7, 1)


def test_half_input_cast_to_head_dtype():
    """A bf16/half backbone activation is cast to the head's weight dtype."""
    head = ValueHead(_config(hidden_size=4, summary_dropout_prob=0.0), num_classes=1)
    out = head(torch.randn(1, 3, 4, dtype=torch.float16))
    assert out.dtype == head.summary.weight.dtype == torch.float32


def test_word_embed_proj_dim_preferred():
    """OPT-style word_embed_proj_dim wins over hidden_size for input width."""
    head = ValueHead(
        _config(hidden_size=8, word_embed_proj_dim=6, summary_dropout_prob=0.0),
        num_classes=2,
    )
    assert head.summary.in_features == 6


def test_zero_dropout_uses_identity():
    """summary_dropout_prob=0 installs Identity, not a Dropout(0)."""
    head = ValueHead(_config(hidden_size=4, summary_dropout_prob=0.0), num_classes=1)
    assert isinstance(head.dropout, nn.Identity)


def test_kwarg_dropout_fallback_when_config_lacks_it():
    """summary_dropout_prob kwarg applies when the config has no such field."""
    head = ValueHead(_config(hidden_size=4), num_classes=1, summary_dropout_prob=0.25)
    assert isinstance(head.dropout, nn.Dropout)
    assert head.dropout.p == 0.25


def test_config_without_width_raises():
    """A config exposing neither width attribute fails loudly at construction."""
    with pytest.raises(AttributeError):
        ValueHead(_config(summary_dropout_prob=0.0), num_classes=1)


# Author: Mus mbayramo@stanford.edu
