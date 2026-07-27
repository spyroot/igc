"""Materialization and hashing of visible REST graph observations."""

from __future__ import annotations

import hashlib

import numpy as np

from .capture import RestCapture
from .runtime import RestRuntimeBatch
from .types import (
    ObservedRestNode,
    RestGraphObservation,
    RestObservation,
    error_body,
)


def snapshot_sha256(
    capture: RestCapture,
    runtime: RestRuntimeBatch,
    row: int,
) -> str:
    """Hash one visible graph state without reserializing shared JSON bodies."""
    if not 0 <= row < runtime.num_envs:
        raise IndexError(row)
    digest = hashlib.sha256()
    digest.update(b"igc-rest-visible-graph-v1\0")
    digest.update(capture.graph_shape_bytes)
    digest.update(runtime.known[row].tobytes())
    digest.update(runtime.visited[row].tobytes())
    digest.update(runtime.body_visible[row].tobytes())
    digest.update(runtime.visible_edges[row].tobytes())
    for node_id_value in np.flatnonzero(runtime.known[row]):
        node_id = int(node_id_value)
        digest.update(capture.resource_metadata_fingerprints[node_id])
        if runtime.body_visible[row, node_id]:
            digest.update(capture.resource_body_fingerprints[node_id])
    for edge_id_value in np.flatnonzero(runtime.visible_edges[row]):
        digest.update(capture.edge_fingerprints[int(edge_id_value)])
    digest.update(
        np.ascontiguousarray(runtime.last_status[row], dtype="<i2").tobytes()
    )
    digest.update(
        np.ascontiguousarray(runtime.error_code[row], dtype="<i2").tobytes()
    )
    digest.update(
        np.ascontiguousarray(runtime.body_version[row], dtype="<i4").tobytes()
    )
    return digest.hexdigest()


def materialize_observation(
    capture: RestCapture,
    runtime: RestRuntimeBatch,
    *,
    row: int,
) -> RestObservation:
    """Project one runtime row into its known-only readable graph."""
    if not 0 <= row < runtime.num_envs:
        raise IndexError(row)

    known_ids = np.flatnonzero(runtime.known[row])
    nodes = tuple(
        ObservedRestNode(
            node_id=int(node_id),
            uri=capture.resources[int(node_id)].uri,
            json_body=(
                capture.resources[int(node_id)].base_json
                if runtime.body_visible[row, node_id]
                else None
            ),
            visited=bool(runtime.visited[row, node_id]),
            allowed_methods=capture.resources[int(node_id)].allowed_methods,
            last_status=int(runtime.last_status[row, node_id]),
            last_error=error_body(int(runtime.error_code[row, node_id])),
        )
        for node_id in known_ids
    )
    visible_edge_ids = np.flatnonzero(runtime.visible_edges[row])
    edges = tuple(capture.edges[int(edge_id)] for edge_id in visible_edge_ids)
    frontier = tuple(
        int(node_id)
        for node_id in np.flatnonzero(
            runtime.known[row] & ~runtime.visited[row]
        )
    )
    graph = RestGraphObservation(
        state_version=int(runtime.state_version[row]),
        nodes=nodes,
        edges=edges,
        frontier_node_ids=frontier,
        snapshot_sha256=snapshot_sha256(capture, runtime, row),
    )
    return RestObservation(
        graph=graph,
        last_response_status=int(runtime.last_response_status[row]),
        last_error=error_body(int(runtime.last_response_error_code[row])),
        step_count=int(runtime.step_count[row]),
        last_action_node_id=int(runtime.last_action_node_id[row]),
    )
