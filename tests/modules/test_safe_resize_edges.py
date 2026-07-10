"""Edge coverage for the token-embedding safe-resize guard.

The guard lives in ``igc.modules.shared.llm_shared`` and protects modern
backbones from accidental vocabulary shrinkage while still allowing explicit
growth or expert-approved shrink operations. These tests stay CPU-only and use
small fakes because only the model/tokenizer sizing protocol matters here.
"""

import pytest

from igc.modules.shared.llm_shared import safe_resize_token_embeddings


class _Embeddings:
    def __init__(self, rows: int) -> None:
        self.num_embeddings = rows


class _ConfiglessModel:
    """Minimal fake model; deliberately has no ``config`` attribute."""

    def __init__(self, rows: int) -> None:
        self._embeddings = _Embeddings(rows)
        self.resize_calls: list[int] = []

    def get_input_embeddings(self) -> _Embeddings:
        return self._embeddings

    def resize_token_embeddings(self, rows: int) -> None:
        self.resize_calls.append(rows)
        self._embeddings.num_embeddings = rows


class _Tokenizer:
    def __init__(self, size: int) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size


def test_single_token_growth_boundary_resizes_once() -> None:
    """The smallest valid growth boundary still calls the resize hook once."""
    model = _ConfiglessModel(53147)

    result = safe_resize_token_embeddings(model, _Tokenizer(53148))

    assert result is model
    assert model.get_input_embeddings().num_embeddings == 53148
    assert model.resize_calls == [53148]


def test_force_shrink_then_grow_round_trip_updates_embedding_size() -> None:
    """A forced shrink remains explicit, and a later growth follows normal rules."""
    model = _ConfiglessModel(10)

    safe_resize_token_embeddings(model, _Tokenizer(7), force_shrink=True)
    safe_resize_token_embeddings(model, _Tokenizer(10))

    assert model.get_input_embeddings().num_embeddings == 10
    assert model.resize_calls == [7, 10]


def test_chained_safe_resize_calls_keep_latest_size() -> None:
    """Call sites can chain returned models through repeated safe growth steps."""
    model = _ConfiglessModel(3)

    same_model = safe_resize_token_embeddings(
        safe_resize_token_embeddings(model, _Tokenizer(4)),
        _Tokenizer(6),
    )

    assert same_model is model
    assert model.get_input_embeddings().num_embeddings == 6
    assert model.resize_calls == [4, 6]


def test_configless_model_supported_without_side_effects() -> None:
    """The guard needs embeddings only; a model ``config`` is not required."""
    model = _ConfiglessModel(8)

    safe_resize_token_embeddings(model, _Tokenizer(8))

    assert not hasattr(model, "config")
    assert model.get_input_embeddings().num_embeddings == 8
    assert model.resize_calls == []


def test_unforced_shrink_after_growth_is_rejected_at_current_size() -> None:
    """Shrink validation uses the model's current embedding size, not its initial size."""
    model = _ConfiglessModel(5)
    safe_resize_token_embeddings(model, _Tokenizer(9))

    with pytest.raises(ValueError, match="from 9 to 8"):
        safe_resize_token_embeddings(model, _Tokenizer(8))

    assert model.get_input_embeddings().num_embeddings == 9
    assert model.resize_calls == [9]
