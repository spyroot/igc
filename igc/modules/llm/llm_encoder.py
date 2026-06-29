"""Shared text encoder for the pointer policy: text -> [B, H] embedding.

The pointer / candidate-scoring policy needs ONE encoder for both the state+goal text
and the candidate-action text (so a query and the candidate keys live in the same
space). :class:`LLMEncoder` is the real backbone encoder — a config-driven HF
``AutoModel`` whose last hidden state is mean-pooled over the attention mask; it is
heavy (download / GPU) and ``transformers`` is imported lazily so this module stays
importable without it. :class:`StubLLMEncoder` is a deterministic, dependency-free
stand-in for the offline CPU gate: it maps each text to a stable unit vector seeded by
a hash of the text, with a per-text cache, so tests can exercise candidate scoring and
the embedding cache without loading a model.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

import hashlib
from typing import Dict, List, Optional, Protocol, runtime_checkable

import torch
import torch.nn.functional as F


@runtime_checkable
class TextEncoder(Protocol):
    """Structural contract: a text encoder exposes ``hidden_size`` and ``encode``."""

    hidden_size: int

    def encode(self, texts: List[str]) -> torch.Tensor: ...


class StubLLMEncoder:
    """Deterministic, dependency-free text encoder for the offline gate.

    Maps each text to a stable unit vector of width ``hidden_size`` (seeded by a hash of
    the text) and caches by text, so repeated encodes of the same text return the
    identical vector. No model and no download — used to drive the pointer policy and
    the candidate cache on CPU.

    :param hidden_size: embedding width ``H``.
    """

    def __init__(self, hidden_size: int = 64):
        self.hidden_size = hidden_size
        self._cache: Dict[str, torch.Tensor] = {}

    def _embed(self, text: str) -> torch.Tensor:
        """Return (and cache) the deterministic unit vector for one text.

        :param text: the text to embed.
        :return: a ``[H]`` unit vector, identical for identical text.
        """
        cached = self._cache.get(text)
        if cached is not None:
            return cached
        seed = int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "big")
        gen = torch.Generator().manual_seed(seed % (2 ** 63))
        vec = F.normalize(torch.randn(self.hidden_size, generator=gen), dim=-1)
        self._cache[text] = vec
        return vec

    def encode(self, texts: List[str]) -> torch.Tensor:
        """Encode a batch of texts.

        :param texts: list of ``B`` strings.
        :return: ``[B, H]`` embedding (``[0, H]`` for an empty list).
        """
        if not texts:
            return torch.empty(0, self.hidden_size)
        return torch.stack([self._embed(t) for t in texts], dim=0)


class LLMEncoder:
    """Mean-pooled HF ``AutoModel`` text encoder (config-driven hidden size).

    Tokenizes text, runs the backbone, and mean-pools the last hidden state over the
    attention mask. ``transformers`` is imported lazily, so importing this module (and
    using the stub) needs no model. Heavy: mark its tests ``@pytest.mark.download`` /
    ``@pytest.mark.gpu``.

    :param model_name: HF repo id or local path of the backbone.
    :param device: torch device string; defaults to ``"cpu"``.
    """

    def __init__(self, model_name: str, device: Optional[str] = None):
        from transformers import AutoModel, AutoTokenizer  # lazy: keep module import light

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModel.from_pretrained(model_name)
        self.device = device or "cpu"
        self.model.to(self.device).eval()
        self.hidden_size = self.model.config.hidden_size

    @torch.no_grad()
    def encode(self, texts: List[str]) -> torch.Tensor:
        """Encode a batch of texts to mean-pooled embeddings.

        :param texts: list of ``B`` strings.
        :return: ``[B, H]`` embedding (``[0, H]`` for an empty list).
        """
        if not texts:
            return torch.empty(0, self.hidden_size)
        enc = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        hidden = self.model(**enc).last_hidden_state  # [B, T, H]
        mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)  # [B, T, 1]
        summed = (hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        return summed / counts


# Author: Mus mbayramo@stanford.edu
