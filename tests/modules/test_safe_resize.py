"""Offline regressions for ``safe_resize_token_embeddings`` in ``llm_shared``.

``resize_token_embeddings`` silently SHRINKS a pretrained vocabulary (verified
on transformers 5.12: no error, no warning), which destroys a modern backbone
when paired with the smaller cached igc tokenizer. The guard must allow the
one legitimate direction (growth, adding the igc JSON special tokens), refuse
shrinks loudly, and stay a no-op on an exact match. CPU-only, fake model.

Author:
Mus mbayramo@stanford.edu
"""

import pytest

from igc.modules.shared.llm_shared import safe_resize_token_embeddings


class _Embeddings:
    def __init__(self, rows: int) -> None:
        self.num_embeddings = rows


class _FakeLM:
    """Minimal stand-in exposing the two methods the guard touches."""

    def __init__(self, vocab_rows: int) -> None:
        self._emb = _Embeddings(vocab_rows)
        self.resized_to: list[int] = []

    def get_input_embeddings(self) -> _Embeddings:
        return self._emb

    def resize_token_embeddings(self, target: int) -> None:
        self.resized_to.append(target)
        self._emb.num_embeddings = target


class _FakeTokenizer:
    def __init__(self, size: int) -> None:
        self._size = size

    def __len__(self) -> int:
        return self._size


def test_growth_resizes():
    """gpt2-style growth (50257 -> 53147 igc tokens) passes through."""
    model = _FakeLM(50257)
    safe_resize_token_embeddings(model, _FakeTokenizer(53147))
    assert model.resized_to == [53147]


def test_shrink_raises_with_clear_message():
    """Qwen-style shrink (151936 -> 53147) is refused, not silently applied."""
    model = _FakeLM(151936)
    with pytest.raises(ValueError, match="Refusing to shrink"):
        safe_resize_token_embeddings(model, _FakeTokenizer(53147))
    assert model.resized_to == []


def test_shrink_allowed_when_forced():
    """force_shrink=True is the explicit expert override."""
    model = _FakeLM(151936)
    safe_resize_token_embeddings(model, _FakeTokenizer(53147), force_shrink=True)
    assert model.resized_to == [53147]


def test_exact_match_is_a_noop():
    """Matching sizes never call resize (keeps tied weights untouched)."""
    model = _FakeLM(53147)
    safe_resize_token_embeddings(model, _FakeTokenizer(53147))
    assert model.resized_to == []


def test_returns_the_model_for_chaining():
    """The guard returns the model like resize call sites expect."""
    model = _FakeLM(10)
    assert safe_resize_token_embeddings(model, _FakeTokenizer(12)) is model


# Author: Mus mbayramo@stanford.edu
