"""Offline unit tests for the REST observation encoder."""

from types import SimpleNamespace

import torch

from igc.envs.rest_encoder import RestBaseEncoder


class _Embeddings:
    """Minimal embedding table shape used by the resize guard."""

    def __init__(self, rows: int) -> None:
        self.num_embeddings = rows


class _Backbone:
    """Callable fake backbone that returns deterministic hidden states."""

    def __init__(self, config: SimpleNamespace) -> None:
        self.config = config
        self.calls: list[torch.Tensor] = []

    def __call__(self, input_ids: torch.Tensor):
        self.calls.append(input_ids.clone())
        hidden = torch.stack(
            [input_ids.float() + offset for offset in range(self.config.hidden_size)],
            dim=-1,
        )
        return SimpleNamespace(last_hidden_state=hidden)


class _Model:
    """Small model double exposing the methods RestBaseEncoder touches."""

    base_model_prefix = "transformer"

    def __init__(self, *, vocab_rows: int = 6, positions: int | None = 8, hidden_size: int = 4) -> None:
        self.config = SimpleNamespace(hidden_size=hidden_size)
        if positions is not None:
            self.config.max_position_embeddings = positions
        self.transformer = _Backbone(self.config)
        self._embeddings = _Embeddings(vocab_rows)
        self.resized_to: list[int] = []

    def get_input_embeddings(self) -> _Embeddings:
        return self._embeddings

    def resize_token_embeddings(self, target: int) -> None:
        self.resized_to.append(target)
        self._embeddings.num_embeddings = target


class _Tokenizer:
    """Tokenizer double that records encode options."""

    def __init__(self, tokens: list[int], size: int = 6) -> None:
        self.tokens = tokens
        self.size = size
        self.calls: list[dict[str, object]] = []

    def __len__(self) -> int:
        return self.size

    def encode(self, text: str, **kwargs) -> list[int]:
        self.calls.append({"text": text, **kwargs})
        max_length = kwargs.get("max_length")
        if kwargs.get("truncation") and isinstance(max_length, int):
            return self.tokens[:max_length]
        return list(self.tokens)


def test_rest_encoder_derives_shape_from_backbone_config():
    """emb_shape follows backbone positions/hidden size and excludes padding."""
    model = _Model(positions=9, hidden_size=5)

    encoder = RestBaseEncoder(model=model, tokenizer=_Tokenizer([1, 2, 3]))

    assert encoder.encoder_model is model.transformer
    assert encoder.emb_shape == (8, 5)
    assert model.config.is_decoder is False
    assert model.resized_to == []


def test_rest_encoder_uses_default_position_fallback_for_rope_like_model():
    """Models without max positions use the shared 1024-token fallback."""
    encoder = RestBaseEncoder(
        model=_Model(positions=None, hidden_size=7),
        tokenizer=_Tokenizer([1, 2, 3]),
    )

    assert encoder.emb_shape == (1023, 7)


def test_encode_tokenizes_with_padding_and_returns_hidden_state_without_batch_dim():
    """encode requests fixed-length tokens and returns [seq, hidden] embeddings."""
    tokenizer = _Tokenizer([5, 6, 7, 8])
    model = _Model(hidden_size=3)
    encoder = RestBaseEncoder(model=model, tokenizer=tokenizer)

    out = encoder.encode("redfish-body", max_chunk_length=3)

    assert tokenizer.calls == [
        {
            "text": "redfish-body",
            "truncation": True,
            "add_special_tokens": True,
            "padding": "max_length",
            "max_length": 3,
        }
    ]
    assert model.transformer.calls[0].tolist() == [[5, 6, 7]]
    assert out.shape == (3, 3)
    assert out.tolist() == [
        [5.0, 6.0, 7.0],
        [6.0, 7.0, 8.0],
        [7.0, 8.0, 9.0],
    ]


def test_encode_reuses_cache_without_retokenizing_or_reencoding():
    """Repeated observations are served from cache."""
    tokenizer = _Tokenizer([1, 2, 3])
    model = _Model()
    encoder = RestBaseEncoder(model=model, tokenizer=tokenizer)

    first = encoder.encode("same-observation", max_chunk_length=3)
    second = encoder.encode("same-observation", max_chunk_length=3)

    assert second is first
    assert len(tokenizer.calls) == 1
    assert len(model.transformer.calls) == 1
    assert encoder.cache["same-observation"] is first
