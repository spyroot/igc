"""Implementation-neutral boundaries around the REST simulator core."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Generic, TypeVar

from .types import RestObservationBatch, RestRequestBatch, SimulatorStepBatch


EncodingBatch = TypeVar("EncodingBatch", covariant=True)


class RestBackend(ABC):
    """Execution boundary implemented by the simulator or an external adapter."""

    @abstractmethod
    def reset(
        self,
        *,
        seeds: int | Sequence[int] | None = None,
    ) -> RestObservationBatch:
        """Reset every runtime and return the raw observation batch."""

    @abstractmethod
    def step(self, request: RestRequestBatch) -> SimulatorStepBatch:
        """Apply exactly one aligned request per runtime."""


class ObservationSpaceEncoder(ABC, Generic[EncodingBatch]):
    """External transformation from raw observations to learned features."""

    @abstractmethod
    def encode(
        self,
        observation: RestObservationBatch,
        /,
    ) -> EncodingBatch:
        """Encode one raw observation batch without mutating simulator state."""
