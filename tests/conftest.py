"""Shared offline pytest fixtures for the igc test suite.

This module provides a deterministic, CPU-only ``StubEncoder`` and matching
fixtures so tests that need a state/observation encoder do not have to build
one inline (and do not pull a real HuggingFace backbone). The encoder mirrors
the small surface ``igc.envs.rest_gym_env.RestApiEnv`` expects from
``RestBaseEncoder``: a ``(model, tokenizer)`` constructor, an ``emb_shape``
attribute, and an ``encode(str) -> torch.Tensor`` method whose output depends
only on the input string (deterministic, no randomness, no I/O).

Fixtures exposed:

- ``stub_encoder_cls`` — the ``StubEncoder`` class, e.g. for
  ``monkeypatch.setattr(rest_gym_env, "RestBaseEncoder", stub_encoder_cls)``.
- ``stub_encoder`` — a ready ``StubEncoder`` instance.

Author:
Mus mbayramo@stanford.edu
"""

import pytest
import torch

# Legacy test files quarantined from collection: each imports a module path that no longer
# exists (e.g. igc.shared.huggingface_utils, igc.modules.llm_module) and aborts the whole
# pytest run with a collection error. Repairing their imports (or retiring them) is queued
# as a follow-up task; remove entries here as they are fixed.
collect_ignore = [
    "hugging_face_test.py",
    "mock_server_test.py",
    "test_gym_restapi.py",
    "test_tokenizer.py",
]


class StubEncoder:
    """Deterministic, dependency-free stand-in for the state encoder.

    The encoder returns a tensor of shape :attr:`emb_shape` whose values are a
    fixed ramp offset by a hash of the observation string, so equal inputs map
    to equal outputs and different inputs (almost always) differ. It records
    every observation it was asked to encode in :attr:`calls` so tests can
    assert on what the environment fed it.

    :param model: ignored; accepted for constructor parity with the real encoder.
    :param tokenizer: ignored; accepted for constructor parity.
    """

    emb_shape = (2, 3)

    def __init__(self, model=None, tokenizer=None) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.calls: list[str] = []

    def encode(self, observation: str) -> torch.Tensor:
        """Encode an observation string into a deterministic embedding.

        :param observation: the observation text (e.g. a Redfish JSON body).
        :return: a ``float32`` tensor of shape :attr:`emb_shape`.
        """
        self.calls.append(observation)
        size = int(torch.tensor(self.emb_shape).prod().item())
        seed = sum(observation.encode("utf-8")) % 97
        return torch.arange(size, dtype=torch.float32).reshape(self.emb_shape) + seed


@pytest.fixture
def stub_encoder_cls() -> type[StubEncoder]:
    """Return the :class:`StubEncoder` class for monkeypatching a real encoder."""
    return StubEncoder


@pytest.fixture
def stub_encoder() -> StubEncoder:
    """Return a fresh :class:`StubEncoder` instance."""
    return StubEncoder()


# Author: Mus mbayramo@stanford.edu
