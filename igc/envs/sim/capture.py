"""Immutable REST capture construction and indexing."""

from __future__ import annotations

import hashlib
import json
from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

from .links import extract_links
from .types import (
    RestEdge,
    RestResource,
    _immutable_array,
    freeze_json,
    plain_json,
)


def _normalize_methods(methods: Iterable[str], *, uri: str) -> tuple[str, ...]:
    if not isinstance(methods, Iterable) or isinstance(methods, (str, bytes)):
        raise TypeError(
            f"allowed methods for {uri!r} must be an iterable of names"
        )
    normalized: set[str] = set()
    for method in methods:
        if not isinstance(method, str):
            raise TypeError(f"allowed method for {uri!r} must be a string")
        name = method.strip().upper()
        if name:
            normalized.add(name)
    if not normalized:
        raise ValueError(f"resource {uri!r} has no allowed methods")
    return tuple(sorted(normalized))


@dataclass(frozen=True, slots=True)
class RestCapture:
    """One immutable REST graph shared by all simulator runtimes."""

    capture_id: str
    root_node_id: int
    resources: tuple[RestResource, ...]
    edges: tuple[RestEdge, ...]
    uri_to_id: Mapping[str, int]
    reachable_mask: np.ndarray
    edge_indptr: np.ndarray
    edge_indices: np.ndarray

    def __post_init__(self) -> None:
        if not self.capture_id:
            raise ValueError("capture_id must not be empty")
        if not self.resources:
            raise ValueError("a capture requires at least one resource")
        if not 0 <= self.root_node_id < len(self.resources):
            raise ValueError("root_node_id is outside the resource table")

        resources: list[RestResource] = []
        for expected_id, resource in enumerate(self.resources):
            if resource.node_id != expected_id:
                raise ValueError("resource node IDs must be contiguous and ordered")
            if not isinstance(resource.uri, str) or not resource.uri:
                raise ValueError("resource URIs must be non-empty strings")
            methods = _normalize_methods(
                resource.allowed_methods,
                uri=resource.uri,
            )
            resources.append(
                RestResource(
                    node_id=expected_id,
                    uri=resource.uri,
                    base_json=freeze_json(resource.base_json),
                    allowed_methods=methods,
                )
            )

        edges = tuple(
            sorted(
                self.edges,
                key=lambda edge: (edge.source_id, edge.target_id, edge.relation),
            )
        )
        edge_keys = {
            (edge.source_id, edge.target_id, edge.relation)
            for edge in edges
        }
        if len(edge_keys) != len(edges):
            raise ValueError("capture edges must be unique")
        for edge in edges:
            if not 0 <= edge.source_id < len(resources):
                raise ValueError("edge source is outside the resource table")
            if not 0 <= edge.target_id < len(resources):
                raise ValueError("edge target is outside the resource table")
            if not isinstance(edge.relation, str) or not edge.relation:
                raise ValueError("edge relation must be a non-empty string")

        expected_uri_to_id = {
            resource.uri: resource.node_id
            for resource in resources
        }
        if len(expected_uri_to_id) != len(resources):
            raise ValueError("resource URIs must be unique")
        if dict(self.uri_to_id) != expected_uri_to_id:
            raise ValueError("uri_to_id does not match the resource table")

        reachable = _immutable_array(
            self.reachable_mask,
            dtype=np.dtype(np.bool_),
        )
        indptr = _immutable_array(self.edge_indptr, dtype=np.dtype(np.int32))
        indices = _immutable_array(self.edge_indices, dtype=np.dtype(np.int32))
        if reachable.shape != (len(resources),):
            raise ValueError("reachable_mask must have shape [N]")
        if indptr.shape != (len(resources) + 1,):
            raise ValueError("edge_indptr must have shape [N + 1]")
        if indices.shape != (len(edges),):
            raise ValueError("edge_indices must have shape [E]")
        if indptr[0] != 0 or indptr[-1] != len(edges):
            raise ValueError("edge_indptr does not span the edge table")
        if np.any(indptr[1:] < indptr[:-1]):
            raise ValueError("edge_indptr must be monotonic")
        if not np.array_equal(indices, np.arange(len(edges), dtype=np.int32)):
            raise ValueError("edge_indices must index the ordered edge table")
        if "GET" not in resources[self.root_node_id].allowed_methods:
            raise ValueError("root resource must allow GET for reset observation")
        for source_id in range(len(resources)):
            start = int(indptr[source_id])
            stop = int(indptr[source_id + 1])
            if any(
                edges[int(edge_id)].source_id != source_id
                for edge_id in indices[start:stop]
            ):
                raise ValueError("edge CSR index does not match edge sources")

        expected_reachable = np.zeros(len(resources), dtype=np.bool_)
        expected_reachable[self.root_node_id] = True
        pending: deque[int] = deque([self.root_node_id])
        while pending:
            source_id = pending.popleft()
            start = int(indptr[source_id])
            stop = int(indptr[source_id + 1])
            for edge_id in indices[start:stop]:
                target_id = edges[int(edge_id)].target_id
                if not expected_reachable[target_id]:
                    expected_reachable[target_id] = True
                    pending.append(target_id)
        if not np.array_equal(reachable, expected_reachable):
            raise ValueError("reachable_mask does not match the rooted graph")

        object.__setattr__(self, "resources", tuple(resources))
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "uri_to_id", MappingProxyType(expected_uri_to_id))
        object.__setattr__(self, "reachable_mask", reachable)
        object.__setattr__(self, "edge_indptr", indptr)
        object.__setattr__(self, "edge_indices", indices)

    @classmethod
    def from_mappings(
        cls,
        *,
        responses: Mapping[str, Mapping[str, Any]],
        allowed_methods: Mapping[str, Iterable[str]],
        root_uri: str,
        capture_id: str | None = None,
    ) -> RestCapture:
        """Build a normalized capture from URI-to-JSON and method mappings."""
        if not responses:
            raise ValueError("responses must not be empty")
        if any(not isinstance(uri, str) or not uri for uri in responses):
            raise ValueError("captured URIs must be non-empty strings")
        if root_uri not in responses:
            raise ValueError("root_uri is absent from responses")
        if set(allowed_methods) != set(responses):
            raise ValueError("allowed_methods must cover exactly the captured URIs")

        uris = tuple(sorted(responses))
        uri_to_id = {uri: node_id for node_id, uri in enumerate(uris)}
        resources: list[RestResource] = []
        for node_id, uri in enumerate(uris):
            body = responses[uri]
            if not isinstance(body, Mapping):
                raise TypeError(f"response for {uri!r} must be a JSON object")
            methods = _normalize_methods(allowed_methods[uri], uri=uri)
            resources.append(
                RestResource(
                    node_id=node_id,
                    uri=uri,
                    base_json=freeze_json(body),
                    allowed_methods=methods,
                )
            )

        root_node_id = uri_to_id[root_uri]
        if "GET" not in resources[root_node_id].allowed_methods:
            raise ValueError("root resource must allow GET for reset observation")

        known_uris = frozenset(uris)
        edge_set: set[tuple[int, int, str]] = set()
        for resource in resources:
            for target_uri, relation in extract_links(
                resource.base_json,
                known_uris=known_uris,
                source_uri=resource.uri,
            ):
                edge_set.add(
                    (resource.node_id, uri_to_id[target_uri], relation)
                )
        edges = tuple(RestEdge(*edge) for edge in sorted(edge_set))

        edge_indptr = np.zeros(len(resources) + 1, dtype=np.int32)
        for edge in edges:
            edge_indptr[edge.source_id + 1] += 1
        np.cumsum(edge_indptr, out=edge_indptr)
        edge_indices = np.arange(len(edges), dtype=np.int32)

        reachable = np.zeros(len(resources), dtype=np.bool_)
        reachable[root_node_id] = True
        pending: deque[int] = deque([root_node_id])
        while pending:
            source_id = pending.popleft()
            start = int(edge_indptr[source_id])
            stop = int(edge_indptr[source_id + 1])
            for edge_id in edge_indices[start:stop]:
                target_id = edges[int(edge_id)].target_id
                if not reachable[target_id]:
                    reachable[target_id] = True
                    pending.append(target_id)

        if capture_id is None:
            payload = {
                "root_uri": root_uri,
                "resources": [
                    {
                        "uri": resource.uri,
                        "body": plain_json(resource.base_json),
                        "methods": list(resource.allowed_methods),
                    }
                    for resource in resources
                ],
                "edges": [
                    [edge.source_id, edge.target_id, edge.relation]
                    for edge in edges
                ],
            }
            encoded = json.dumps(
                payload,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            capture_id = hashlib.sha256(encoded).hexdigest()

        return cls(
            capture_id=capture_id,
            root_node_id=root_node_id,
            resources=tuple(resources),
            edges=edges,
            uri_to_id=uri_to_id,
            reachable_mask=reachable,
            edge_indptr=edge_indptr,
            edge_indices=edge_indices,
        )

    @property
    def num_nodes(self) -> int:
        return len(self.resources)

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    def outgoing_edge_ids(self, node_id: int) -> np.ndarray:
        """Return immutable edge IDs for one source node."""
        if not 0 <= node_id < self.num_nodes:
            raise IndexError(node_id)
        start = int(self.edge_indptr[node_id])
        stop = int(self.edge_indptr[node_id + 1])
        return self.edge_indices[start:stop]
