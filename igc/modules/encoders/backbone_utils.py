"""Backbone-agnostic accessors so igc works with any HF AutoModel, not just GPT-2.

The legacy code reaches into ``model.transformer`` and ``model.transformer.wpe.weight``
to get the base module and the hidden / positional sizes — which only works for GPT-2.
These helpers derive the same information from any HuggingFace model via
``base_model_prefix`` and ``config``, so a large decoder (Llama / Qwen / flash-class —
often RoPE, with no positional embedding table) loads and trains. Pure-logic
(attribute access only, no torch import), so it stays CPU/offline-testable.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

from typing import Any, Optional


def backbone_module(model: Any) -> Any:
    """Return the base transformer module of an HF model.

    Uses ``base_model_prefix`` (``model.transformer`` for GPT-2, ``model.model`` for
    Llama, etc.) and falls back to the model itself when the prefix is absent — e.g. a
    bare ``AutoModel`` whose top level already is the encoder.

    :param model: an HF ``PreTrainedModel`` (or anything exposing ``base_model_prefix``).
    :return: the base module.
    """
    prefix = getattr(model, "base_model_prefix", None)
    if prefix and hasattr(model, prefix):
        return getattr(model, prefix)
    return model


def _config_of(model_or_config: Any) -> Any:
    """Return the config object whether given a model or a config."""
    return getattr(model_or_config, "config", model_or_config)


def hidden_size(model_or_config: Any) -> int:
    """Hidden size ``H`` of a model/config, backbone-agnostically.

    Checks ``hidden_size`` then the GPT-2 alias ``n_embd`` then ``d_model``.

    :param model_or_config: an HF model or its config.
    :return: the hidden size.
    :raises AttributeError: if no known hidden-size attribute is present.
    """
    config = _config_of(model_or_config)
    for attr in ("hidden_size", "n_embd", "d_model"):
        value = getattr(config, attr, None)
        if value is not None:
            return int(value)
    raise AttributeError("could not determine hidden size from config")


def max_positions(model_or_config: Any) -> Optional[int]:
    """Maximum position-embedding length, or ``None`` for RoPE models.

    Checks ``max_position_embeddings`` then the GPT-2 alias ``n_positions``. Returns
    ``None`` when neither is present (RoPE backbones have no positional table) — callers
    should then fall back to a chosen sequence length rather than reading a weight shape.

    :param model_or_config: an HF model or its config.
    :return: the max position length, or ``None``.
    """
    config = _config_of(model_or_config)
    for attr in ("max_position_embeddings", "n_positions"):
        value = getattr(config, attr, None)
        if value is not None:
            return int(value)
    return None


# Author: Mus mbayramo@stanford.edu
