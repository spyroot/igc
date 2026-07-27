"""Pure CPU GET/HEAD REST simulator."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .capture import RestCapture
from .interfaces import RestBackend
from .observation import materialize_observation, snapshot_sha256
from .runtime import RestRuntimeBatch
from .types import (
    ERROR_BAD_REQUEST,
    ERROR_METHOD_NOT_ALLOWED,
    ERROR_NONE,
    ERROR_NOT_FOUND,
    METHOD_GET,
    METHOD_HEAD,
    METHOD_IDS,
    METHOD_NAMES,
    JsonObject,
    RestObservation,
    RestObservationBatch,
    RestRequest,
    RestRequestBatch,
    RestTransition,
    RestTransitionBatch,
    SimulatorStep,
    SimulatorStepBatch,
    error_body,
)


class BatchedRestSimulator(RestBackend):
    """Advance aligned requests over independent rows sharing one capture."""

    def __init__(
        self,
        *,
        capture: RestCapture,
        num_envs: int | None = None,
        runtime: RestRuntimeBatch | None = None,
    ) -> None:
        if runtime is None:
            if num_envs is None:
                raise ValueError("num_envs is required when runtime is omitted")
            runtime = RestRuntimeBatch.create(capture, num_envs=num_envs)
        elif num_envs is not None and num_envs != runtime.num_envs:
            raise ValueError("num_envs does not match runtime")
        if runtime.known.shape[1] != capture.num_nodes:
            raise ValueError("runtime does not match capture")
        self.capture = capture
        self.runtime = runtime

    @property
    def num_envs(self) -> int:
        return self.runtime.num_envs

    def reset(
        self,
        *,
        seeds: int | Sequence[int] | None = None,
    ) -> RestObservationBatch:
        """Return free-root initial observations for all runtime rows."""
        return self.runtime.reset(self.capture, seeds=seeds)

    def observe(self) -> RestObservationBatch:
        """Return a detached immutable batch snapshot."""
        return self.runtime.observe()

    def step(self, request: RestRequestBatch) -> SimulatorStepBatch:
        if not self.runtime.initialized:
            raise RuntimeError("reset must be called before step")
        if request.batch_size != self.num_envs:
            raise ValueError("one request is required per runtime")

        status_codes = np.empty(self.num_envs, dtype=np.int16)
        first_visit = np.zeros(self.num_envs, dtype=np.bool_)
        before_versions = self.runtime.state_version.copy()
        after_versions = np.empty(self.num_envs, dtype=np.int64)
        json_bodies: list[JsonObject | None] = []
        errors: list[JsonObject | None] = []
        newly_discovered: list[tuple[int, ...]] = []
        newly_visible: list[tuple[int, ...]] = []
        changed_nodes: list[tuple[int, ...]] = []
        before_hashes = tuple(
            snapshot_sha256(self.capture, self.runtime, row)
            for row in range(self.num_envs)
        )

        for row in range(self.num_envs):
            node_id = int(request.node_ids[row])
            method_id = int(request.method_ids[row])
            arguments = request.arguments[row]
            status = 200
            graph_changed = False
            response_body = None
            discovered: list[int] = []
            visible: list[int] = []

            self.runtime.step_count[row] += 1
            known_node = (
                0 <= node_id < self.capture.num_nodes
                and bool(self.runtime.known[row, node_id])
            )
            if not known_node:
                status = ERROR_NOT_FOUND
                self.runtime.last_action_node_id[row] = -1
            else:
                self.runtime.last_action_node_id[row] = node_id
                resource = self.capture.resources[node_id]
                method = (
                    METHOD_NAMES[method_id]
                    if 0 <= method_id < len(METHOD_NAMES)
                    else None
                )
                if arguments:
                    status = ERROR_BAD_REQUEST
                elif method is None or method not in resource.allowed_methods:
                    status = ERROR_METHOD_NOT_ALLOWED
                elif method_id == METHOD_GET:
                    first_visit[row] = not self.runtime.visited[row, node_id]
                    if not self.runtime.visited[row, node_id]:
                        self.runtime.visited[row, node_id] = True
                        graph_changed = True
                    if not self.runtime.body_visible[row, node_id]:
                        self.runtime.body_visible[row, node_id] = True
                        graph_changed = True
                    if self.runtime.body_version[row, node_id] != 0:
                        self.runtime.body_version[row, node_id] = 0
                        graph_changed = True
                    response_body = resource.base_json
                    for edge_id_value in self.capture.outgoing_edge_ids(node_id):
                        edge_id = int(edge_id_value)
                        edge = self.capture.edges[edge_id]
                        if not self.runtime.visible_edges[row, edge_id]:
                            self.runtime.visible_edges[row, edge_id] = True
                            visible.append(edge_id)
                            graph_changed = True
                        if not self.runtime.known[row, edge.target_id]:
                            self.runtime.known[row, edge.target_id] = True
                            discovered.append(edge.target_id)
                            graph_changed = True
                elif method_id == METHOD_HEAD:
                    response_body = None

                next_error = ERROR_NONE if status == 200 else status
                if self.runtime.last_status[row, node_id] != status:
                    self.runtime.last_status[row, node_id] = status
                    graph_changed = True
                if self.runtime.error_code[row, node_id] != next_error:
                    self.runtime.error_code[row, node_id] = next_error
                    graph_changed = True

            response_error = error_body(status)
            self.runtime.last_response_status[row] = status
            self.runtime.last_response_error_code[row] = (
                ERROR_NONE if status == 200 else status
            )
            if graph_changed:
                self.runtime.state_version[row] += 1

            status_codes[row] = status
            after_versions[row] = self.runtime.state_version[row]
            json_bodies.append(response_body)
            errors.append(response_error)
            newly_discovered.append(tuple(discovered))
            newly_visible.append(tuple(visible))
            changed_nodes.append(())

        after_hashes = tuple(
            snapshot_sha256(self.capture, self.runtime, row)
            for row in range(self.num_envs)
        )
        transition = RestTransitionBatch(
            request=request,
            status_codes=status_codes,
            json_bodies=tuple(json_bodies),
            errors=tuple(errors),
            first_visit=first_visit,
            newly_discovered_node_ids=tuple(newly_discovered),
            newly_visible_edge_ids=tuple(newly_visible),
            changed_node_ids=tuple(changed_nodes),
            before_versions=before_versions,
            after_versions=after_versions,
            before_sha256=before_hashes,
            after_sha256=after_hashes,
        )
        return SimulatorStepBatch(
            observation=self.runtime.observe(),
            transition=transition,
        )

    def step_many(self, request: RestRequestBatch) -> SimulatorStepBatch:
        """Alias exposing the batch execution contract explicitly."""
        return self.step(request)


class RestSimulator:
    """Readable scalar wrapper over one row of the batched simulator."""

    def __init__(
        self,
        *,
        capture: RestCapture,
        runtime: RestRuntimeBatch | None = None,
    ) -> None:
        if runtime is not None and runtime.num_envs != 1:
            raise ValueError("a scalar simulator requires exactly one runtime row")
        self.capture = capture
        self._batch = BatchedRestSimulator(
            capture=capture,
            num_envs=1,
            runtime=runtime,
        )
        self.runtime = self._batch.runtime

    def reset(self, *, seed: int) -> RestObservation:
        self._batch.reset(seeds=seed)
        return materialize_observation(self.capture, self.runtime, row=0)

    def observe(self) -> RestObservation:
        return materialize_observation(self.capture, self.runtime, row=0)

    def step(self, request: RestRequest) -> SimulatorStep:
        node_id = self.capture.uri_to_id.get(request.uri, -1)
        method_id = METHOD_IDS.get(request.method, -1)
        batch_request = RestRequestBatch(
            node_ids=np.asarray([node_id], dtype=np.int32),
            method_ids=np.asarray([method_id], dtype=np.int8),
            arguments=(request.arguments,),
        )
        batch_step = self._batch.step(batch_request)
        batch_transition = batch_step.transition
        transition = RestTransition(
            request=request,
            status_code=int(batch_transition.status_codes[0]),
            json_body=batch_transition.json_bodies[0],
            error=batch_transition.errors[0],
            first_visit=bool(batch_transition.first_visit[0]),
            newly_discovered_node_ids=(
                batch_transition.newly_discovered_node_ids[0]
            ),
            newly_visible_edge_ids=batch_transition.newly_visible_edge_ids[0],
            changed_node_ids=batch_transition.changed_node_ids[0],
            before_version=int(batch_transition.before_versions[0]),
            after_version=int(batch_transition.after_versions[0]),
            before_sha256=batch_transition.before_sha256[0],
            after_sha256=batch_transition.after_sha256[0],
        )
        return SimulatorStep(
            observation=materialize_observation(
                self.capture,
                self.runtime,
                row=0,
            ),
            transition=transition,
        )
