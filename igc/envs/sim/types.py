"""Public data contracts for the CPU-only REST simulator.

The simulator deliberately exposes raw REST graph state. Learned encoders,
Gym adapters, rewards, and policies live outside this package.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np


JsonObject = Mapping[str, Any]

METHOD_GET = 0
METHOD_HEAD = 1
METHOD_NAMES = ("GET", "HEAD")
METHOD_IDS = MappingProxyType({name: index for index, name in enumerate(METHOD_NAMES)})

ERROR_NONE = 0
ERROR_BAD_REQUEST = 400
ERROR_NOT_FOUND = 404
ERROR_METHOD_NOT_ALLOWED = 405


def freeze_json(value: Any) -> Any:
    """Return a recursively immutable copy of a JSON-compatible value."""
    if isinstance(value, MappingProxyType):
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("JSON object keys must be strings")
        return MappingProxyType(
            {
                key: freeze_json(value[key])
                for key in sorted(value)
            }
        )
    if isinstance(value, (list, tuple)):
        return tuple(freeze_json(item) for item in value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("JSON numbers must be finite")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported JSON value type: {type(value).__name__}")


def plain_json(value: Any) -> Any:
    """Return a deterministic mutable representation for JSON serialization."""
    if isinstance(value, Mapping):
        return {key: plain_json(value[key]) for key in sorted(value)}
    if isinstance(value, tuple):
        return [plain_json(item) for item in value]
    return value


def error_body(status_code: int) -> JsonObject | None:
    """Return the stable simulator error body for ``status_code``."""
    names = {
        ERROR_BAD_REQUEST: "InvalidArguments",
        ERROR_NOT_FOUND: "ResourceNotFound",
        ERROR_METHOD_NOT_ALLOWED: "MethodNotAllowed",
    }
    name = names.get(status_code)
    if name is None:
        return None
    return freeze_json({"code": name, "status": status_code})


def _immutable_array(value: Any, *, dtype: np.dtype[Any]) -> np.ndarray:
    """Copy an array into immutable byte-backed storage."""
    contiguous = np.ascontiguousarray(value, dtype=dtype)
    immutable = np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype)
    return immutable.reshape(contiguous.shape)


@dataclass(frozen=True, slots=True)
class RestResource:
    """One immutable REST resource in a captured environment."""

    node_id: int
    uri: str
    base_json: JsonObject
    allowed_methods: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "base_json", freeze_json(self.base_json))


@dataclass(frozen=True, slots=True)
class RestEdge:
    """One JSON-revealed relationship between captured resources."""

    source_id: int
    target_id: int
    relation: str


@dataclass(frozen=True, slots=True)
class ObservedRestNode:
    """Materialized, agent-visible view of one known REST node."""

    node_id: int
    uri: str
    json_body: JsonObject | None
    visited: bool
    allowed_methods: tuple[str, ...]
    last_status: int
    last_error: JsonObject | None

    def __post_init__(self) -> None:
        if self.json_body is not None:
            object.__setattr__(self, "json_body", freeze_json(self.json_body))
        if self.last_error is not None:
            object.__setattr__(self, "last_error", freeze_json(self.last_error))


@dataclass(frozen=True, slots=True)
class RestGraphObservation:
    """Readable projection containing only currently visible graph state."""

    state_version: int
    nodes: tuple[ObservedRestNode, ...]
    edges: tuple[RestEdge, ...]
    frontier_node_ids: tuple[int, ...]
    snapshot_sha256: str


@dataclass(frozen=True, slots=True)
class RestObservation:
    """One scalar simulator observation."""

    graph: RestGraphObservation
    last_response_status: int
    last_error: JsonObject | None
    step_count: int
    last_action_node_id: int

    def __post_init__(self) -> None:
        if self.last_error is not None:
            object.__setattr__(self, "last_error", freeze_json(self.last_error))


@dataclass(frozen=True, slots=True)
class RestObservationBatch:
    """Detached structure-of-arrays observation for parallel runtimes."""

    known_mask: np.ndarray
    visited_mask: np.ndarray
    frontier_mask: np.ndarray
    body_visible_mask: np.ndarray
    visible_edge_mask: np.ndarray
    last_status: np.ndarray
    error_code: np.ndarray
    body_version: np.ndarray
    state_version: np.ndarray
    step_count: np.ndarray
    last_action_node_id: np.ndarray
    last_response_status: np.ndarray
    last_response_error_code: np.ndarray

    def __post_init__(self) -> None:
        matrix_fields = {
            "known_mask": (self.known_mask, np.bool_),
            "visited_mask": (self.visited_mask, np.bool_),
            "frontier_mask": (self.frontier_mask, np.bool_),
            "body_visible_mask": (self.body_visible_mask, np.bool_),
            "last_status": (self.last_status, np.int16),
            "error_code": (self.error_code, np.int16),
            "body_version": (self.body_version, np.int32),
        }
        edge_fields = {
            "visible_edge_mask": (self.visible_edge_mask, np.bool_),
        }
        vector_fields = {
            "state_version": (self.state_version, np.int64),
            "step_count": (self.step_count, np.int64),
            "last_action_node_id": (self.last_action_node_id, np.int32),
            "last_response_status": (self.last_response_status, np.int16),
            "last_response_error_code": (
                self.last_response_error_code,
                np.int16,
            ),
        }

        batch_size: int | None = None
        node_count: int | None = None
        for name, (value, dtype) in matrix_fields.items():
            array = _immutable_array(value, dtype=np.dtype(dtype))
            if array.ndim != 2:
                raise ValueError(f"{name} must have shape [B, N]")
            if batch_size is None:
                batch_size, node_count = array.shape
            elif array.shape != (batch_size, node_count):
                raise ValueError(f"{name} must align with known_mask")
            object.__setattr__(self, name, array)

        for name, (value, dtype) in edge_fields.items():
            array = _immutable_array(value, dtype=np.dtype(dtype))
            if array.ndim != 2 or array.shape[0] != batch_size:
                raise ValueError(f"{name} must have shape [B, E]")
            object.__setattr__(self, name, array)

        for name, (value, dtype) in vector_fields.items():
            array = _immutable_array(value, dtype=np.dtype(dtype))
            if array.shape != (batch_size,):
                raise ValueError(f"{name} must have shape [B]")
            object.__setattr__(self, name, array)

        if batch_size is None or batch_size < 1:
            raise ValueError("an observation batch requires at least one runtime")

    @property
    def batch_size(self) -> int:
        """Number of independent runtime rows."""
        return int(self.known_mask.shape[0])

    @property
    def num_nodes(self) -> int:
        """Number of capture nodes represented by every row."""
        return int(self.known_mask.shape[1])

    @property
    def num_edges(self) -> int:
        """Number of capture edges represented by every row."""
        return int(self.visible_edge_mask.shape[1])


@dataclass(frozen=True, slots=True)
class RestRequest:
    """Direct REST request boundary with explicit argument bindings."""

    uri: str
    method: str
    arguments: JsonObject = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.uri, str):
            raise TypeError("uri must be a string")
        if not isinstance(self.method, str):
            raise TypeError("method must be a string")
        uri = self.uri.strip()
        method = self.method.strip().upper()
        if not uri:
            raise ValueError("uri must not be empty")
        if not method:
            raise ValueError("method must not be empty")
        if not isinstance(self.arguments, Mapping):
            raise TypeError("arguments must be a mapping")
        object.__setattr__(self, "uri", uri)
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "arguments", freeze_json(self.arguments))


@dataclass(frozen=True, slots=True)
class RestRequestBatch:
    """Internal parallel request boundary using capture-local integer IDs."""

    node_ids: np.ndarray
    method_ids: np.ndarray
    arguments: tuple[JsonObject, ...] = ()

    def __post_init__(self) -> None:
        raw_node_ids = np.asarray(self.node_ids)
        raw_method_ids = np.asarray(self.method_ids)
        if raw_node_ids.dtype.kind not in "iu":
            raise TypeError("node_ids must contain integers")
        if raw_method_ids.dtype.kind not in "iu":
            raise TypeError("method_ids must contain integers")
        int32 = np.iinfo(np.int32)
        int8 = np.iinfo(np.int8)
        if np.any(raw_node_ids < int32.min) or np.any(raw_node_ids > int32.max):
            raise ValueError("node_ids contain a value outside int32 range")
        if np.any(raw_method_ids < int8.min) or np.any(raw_method_ids > int8.max):
            raise ValueError("method_ids contain a value outside int8 range")
        node_ids = _immutable_array(raw_node_ids, dtype=np.dtype(np.int32))
        method_ids = _immutable_array(raw_method_ids, dtype=np.dtype(np.int8))
        if node_ids.ndim != 1 or node_ids.size < 1:
            raise ValueError("node_ids must have shape [B] with B >= 1")
        if method_ids.shape != node_ids.shape:
            raise ValueError("method_ids must align with node_ids")

        arguments: Sequence[JsonObject]
        if self.arguments:
            arguments = self.arguments
        else:
            arguments = tuple({} for _ in range(node_ids.size))
        if len(arguments) != node_ids.size:
            raise ValueError("one argument mapping is required per request")

        frozen_arguments: list[JsonObject] = []
        for argument in arguments:
            if not isinstance(argument, Mapping):
                raise TypeError("each argument binding must be a mapping")
            frozen_arguments.append(freeze_json(argument))

        object.__setattr__(self, "node_ids", node_ids)
        object.__setattr__(self, "method_ids", method_ids)
        object.__setattr__(self, "arguments", tuple(frozen_arguments))

    @property
    def batch_size(self) -> int:
        """Number of requests in the aligned batch."""
        return int(self.node_ids.size)


@dataclass(frozen=True, slots=True)
class RestTransition:
    """Audit evidence for one explicit simulator request."""

    request: RestRequest
    status_code: int
    json_body: JsonObject | None
    error: JsonObject | None
    first_visit: bool
    newly_discovered_node_ids: tuple[int, ...]
    newly_visible_edge_ids: tuple[int, ...]
    changed_node_ids: tuple[int, ...]
    before_version: int
    after_version: int
    before_sha256: str
    after_sha256: str

    def __post_init__(self) -> None:
        if self.json_body is not None:
            object.__setattr__(self, "json_body", freeze_json(self.json_body))
        if self.error is not None:
            object.__setattr__(self, "error", freeze_json(self.error))


@dataclass(frozen=True, slots=True)
class RestTransitionBatch:
    """Aligned transition evidence for a batch of explicit requests."""

    request: RestRequestBatch
    status_codes: np.ndarray
    json_bodies: tuple[JsonObject | None, ...]
    errors: tuple[JsonObject | None, ...]
    first_visit: np.ndarray
    newly_discovered_node_ids: tuple[tuple[int, ...], ...]
    newly_visible_edge_ids: tuple[tuple[int, ...], ...]
    changed_node_ids: tuple[tuple[int, ...], ...]
    before_versions: np.ndarray
    after_versions: np.ndarray
    before_sha256: tuple[str, ...]
    after_sha256: tuple[str, ...]

    def __post_init__(self) -> None:
        batch_size = self.request.batch_size
        object.__setattr__(
            self,
            "json_bodies",
            tuple(
                freeze_json(body) if body is not None else None
                for body in self.json_bodies
            ),
        )
        object.__setattr__(
            self,
            "errors",
            tuple(
                freeze_json(error) if error is not None else None
                for error in self.errors
            ),
        )
        tuple_fields = (
            self.json_bodies,
            self.errors,
            self.newly_discovered_node_ids,
            self.newly_visible_edge_ids,
            self.changed_node_ids,
            self.before_sha256,
            self.after_sha256,
        )
        if any(len(value) != batch_size for value in tuple_fields):
            raise ValueError("transition metadata must align with requests")

        array_fields = {
            "status_codes": (self.status_codes, np.int16),
            "first_visit": (self.first_visit, np.bool_),
            "before_versions": (self.before_versions, np.int64),
            "after_versions": (self.after_versions, np.int64),
        }
        for name, (value, dtype) in array_fields.items():
            array = _immutable_array(value, dtype=np.dtype(dtype))
            if array.shape != (batch_size,):
                raise ValueError(f"{name} must have shape [B]")
            object.__setattr__(self, name, array)

    @property
    def batch_size(self) -> int:
        """Number of aligned transitions."""
        return self.request.batch_size


@dataclass(frozen=True, slots=True)
class SimulatorStep:
    """Scalar observation and transition returned by ``RestSimulator``."""

    observation: RestObservation
    transition: RestTransition


@dataclass(frozen=True, slots=True)
class SimulatorStepBatch:
    """Batched observation and transition returned by the CPU simulator."""

    observation: RestObservationBatch
    transition: RestTransitionBatch
