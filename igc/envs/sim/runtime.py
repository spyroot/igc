"""Mutable structure-of-arrays state for independent REST runtimes."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .capture import RestCapture
from .types import ERROR_NONE, RestObservationBatch


@dataclass(slots=True)
class RestRuntimeBatch:
    """Small mutable state for a fixed number of parallel simulations."""

    known: np.ndarray
    visited: np.ndarray
    body_visible: np.ndarray
    visible_edges: np.ndarray
    last_status: np.ndarray
    error_code: np.ndarray
    body_version: np.ndarray
    state_version: np.ndarray
    step_count: np.ndarray
    last_action_node_id: np.ndarray
    last_response_status: np.ndarray
    last_response_error_code: np.ndarray
    seeds: np.ndarray
    initialized: bool = False

    @classmethod
    def create(cls, capture: RestCapture, *, num_envs: int) -> RestRuntimeBatch:
        if num_envs < 1:
            raise ValueError("num_envs must be at least one")
        matrix_shape = (num_envs, capture.num_nodes)
        return cls(
            known=np.zeros(matrix_shape, dtype=np.bool_),
            visited=np.zeros(matrix_shape, dtype=np.bool_),
            body_visible=np.zeros(matrix_shape, dtype=np.bool_),
            visible_edges=np.zeros(
                (num_envs, capture.num_edges),
                dtype=np.bool_,
            ),
            last_status=np.zeros(matrix_shape, dtype=np.int16),
            error_code=np.zeros(matrix_shape, dtype=np.int16),
            body_version=np.full(matrix_shape, -1, dtype=np.int32),
            state_version=np.zeros(num_envs, dtype=np.int64),
            step_count=np.zeros(num_envs, dtype=np.int64),
            last_action_node_id=np.full(num_envs, -1, dtype=np.int32),
            last_response_status=np.zeros(num_envs, dtype=np.int16),
            last_response_error_code=np.zeros(num_envs, dtype=np.int16),
            seeds=np.zeros(num_envs, dtype=np.int64),
            initialized=False,
        )

    @property
    def num_envs(self) -> int:
        return int(self.known.shape[0])

    def reset(
        self,
        capture: RestCapture,
        *,
        seeds: int | Sequence[int] | None = None,
    ) -> RestObservationBatch:
        """Restore paper-bound free-root initial observations."""
        if self.known.shape != (self.num_envs, capture.num_nodes):
            raise ValueError("runtime node shape does not match capture")
        if self.visible_edges.shape != (self.num_envs, capture.num_edges):
            raise ValueError("runtime edge shape does not match capture")

        int64 = np.iinfo(np.int64)
        if seeds is None:
            seed_values = np.zeros(self.num_envs, dtype=np.int64)
        elif isinstance(seeds, (bool, np.bool_)):
            raise TypeError("seeds must contain integers")
        elif isinstance(seeds, (int, np.integer)):
            seed = int(seeds)
            if not int64.min <= seed <= int64.max:
                raise ValueError("seed is outside int64 range")
            seed_values = np.full(self.num_envs, seed, dtype=np.int64)
        else:
            seed_items = tuple(seeds)
            if len(seed_items) != self.num_envs:
                raise ValueError("one seed is required per runtime")
            if any(
                not isinstance(seed, (int, np.integer))
                or isinstance(seed, (bool, np.bool_))
                for seed in seed_items
            ):
                raise TypeError("seeds must contain integers")
            if any(
                not int64.min <= int(seed) <= int64.max
                for seed in seed_items
            ):
                raise ValueError("seeds contain a value outside int64 range")
            seed_values = np.asarray(seed_items, dtype=np.int64)

        self.known.fill(False)
        self.visited.fill(False)
        self.body_visible.fill(False)
        self.visible_edges.fill(False)
        self.last_status.fill(0)
        self.error_code.fill(ERROR_NONE)
        self.body_version.fill(-1)
        self.state_version.fill(0)
        self.step_count.fill(0)
        self.last_action_node_id.fill(-1)
        self.last_response_status.fill(200)
        self.last_response_error_code.fill(ERROR_NONE)
        self.seeds[:] = seed_values
        self.initialized = True

        root = capture.root_node_id
        self.known[:, root] = True
        self.visited[:, root] = True
        self.body_visible[:, root] = True
        self.last_status[:, root] = 200
        self.body_version[:, root] = 0

        root_edge_ids = capture.outgoing_edge_ids(root)
        if root_edge_ids.size:
            self.visible_edges[:, root_edge_ids] = True
            targets = np.fromiter(
                (
                    capture.edges[int(edge_id)].target_id
                    for edge_id in root_edge_ids
                ),
                dtype=np.int32,
            )
            self.known[:, targets] = True

        return self.observe()

    def observe(self) -> RestObservationBatch:
        """Return a detached immutable snapshot of all runtime rows."""
        if not self.initialized:
            raise RuntimeError("reset must be called before observe")
        return RestObservationBatch(
            known_mask=self.known,
            visited_mask=self.visited,
            frontier_mask=self.known & ~self.visited,
            body_visible_mask=self.body_visible,
            visible_edge_mask=self.visible_edges,
            last_status=self.last_status,
            error_code=self.error_code,
            body_version=self.body_version,
            state_version=self.state_version,
            step_count=self.step_count,
            last_action_node_id=self.last_action_node_id,
            last_response_status=self.last_response_status,
            last_response_error_code=self.last_response_error_code,
        )
