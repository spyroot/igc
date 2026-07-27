from __future__ import annotations

import ast
from collections.abc import Callable
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

from igc.envs.sim import (
    METHOD_GET,
    METHOD_HEAD,
    BatchedRestSimulator,
    ObservationSpaceEncoder,
    RestBackend,
    RestCapture,
    RestRequest,
    RestRequestBatch,
    RestSimulator,
)


ROOT = "/redfish/v1"
SYSTEMS = "/redfish/v1/Systems"
SYSTEM_1 = "/redfish/v1/Systems/1"
BIOS = "/redfish/v1/Systems/1/Bios"
BIOS_SETTINGS = "/redfish/v1/Systems/1/Bios/Settings"
MANAGERS = "/redfish/v1/Managers"
MANAGER_1 = "/redfish/v1/Managers/1"
CHASSIS = "/redfish/v1/Chassis"
UNREACHABLE = "/redfish/v1/Unreachable"

DIRECT_ROOT_CHILDREN = frozenset({SYSTEMS, MANAGERS, CHASSIS})
REACHABLE_URIS = frozenset(
    {
        ROOT,
        SYSTEMS,
        SYSTEM_1,
        BIOS,
        BIOS_SETTINGS,
        MANAGERS,
        MANAGER_1,
        CHASSIS,
    }
)


@pytest.fixture
def capture() -> RestCapture:
    responses = {
        ROOT: {
            "@odata.id": ROOT,
            "Systems": {"@odata.id": SYSTEMS},
            "Managers": {"@odata.id": MANAGERS},
            "Chassis": {"@odata.id": CHASSIS},
            "Escaped/Key~": {"@odata.id": CHASSIS},
            "External": {"@odata.id": "/not/captured"},
        },
        SYSTEMS: {
            "@odata.id": SYSTEMS,
            "Members": [{"@odata.id": SYSTEM_1}],
        },
        SYSTEM_1: {
            "@odata.id": SYSTEM_1,
            "Bios": {"@odata.id": BIOS},
            "Links": {"Manager": {"@odata.id": MANAGER_1}},
        },
        BIOS: {
            "@odata.id": BIOS,
            "Settings": {"@odata.id": BIOS_SETTINGS},
        },
        BIOS_SETTINGS: {
            "@odata.id": BIOS_SETTINGS,
            "Values": {"BootMode": "Uefi"},
        },
        MANAGERS: {
            "@odata.id": MANAGERS,
            "Members": [{"@odata.id": MANAGER_1}],
        },
        MANAGER_1: {
            "@odata.id": MANAGER_1,
            "Links": {"ServiceRoot": {"@odata.id": ROOT}},
        },
        CHASSIS: {
            "@odata.id": CHASSIS,
            "Name": "Chassis",
        },
        UNREACHABLE: {
            "@odata.id": UNREACHABLE,
            "Name": "Not reachable from root",
        },
    }
    allowed_methods = {
        ROOT: {"GET", "HEAD"},
        SYSTEMS: {"GET", "HEAD"},
        SYSTEM_1: {"GET"},
        BIOS: {"GET"},
        BIOS_SETTINGS: {"GET"},
        MANAGERS: {"GET", "HEAD"},
        MANAGER_1: {"GET"},
        CHASSIS: {"GET"},
        UNREACHABLE: {"GET"},
    }
    return RestCapture.from_mappings(
        responses=responses,
        allowed_methods=allowed_methods,
        root_uri=ROOT,
        capture_id="unit-capture",
    )


def node_id(capture: RestCapture, uri: str) -> int:
    return capture.uri_to_id[uri]


def uri_ids(capture: RestCapture, uris: set[str] | frozenset[str]) -> set[int]:
    return {node_id(capture, uri) for uri in uris}


def true_ids(mask: np.ndarray) -> set[int]:
    return set(np.flatnonzero(mask).astype(int).tolist())


def plain_error(error) -> dict[str, object] | None:
    return dict(error) if error is not None else None


def get_node(observation, uri: str):
    for node in observation.graph.nodes:
        if node.uri == uri:
            return node
    raise AssertionError(f"{uri} was not visible in the observation")


def batch_get(node: int, count: int) -> RestRequestBatch:
    return RestRequestBatch(
        node_ids=np.full(count, node, dtype=np.int32),
        method_ids=np.full(count, METHOD_GET, dtype=np.int8),
    )


class MockObservationEncoder(ObservationSpaceEncoder[dict[str, int]]):
    """Synthetic external encoder with an injected latency mechanism."""

    def __init__(
        self,
        *,
        latency_seconds: float,
        sleep: Callable[[float], None],
    ) -> None:
        self.latency_seconds = latency_seconds
        self.sleep = sleep

    def encode(self, observation):
        self.sleep(self.latency_seconds)
        return {
            "batch_size": observation.batch_size,
            "num_nodes": observation.num_nodes,
        }


class MockLatencyBackend(RestBackend):
    """External REST backend stand-in with one response delay per batch."""

    def __init__(
        self,
        *,
        simulator: BatchedRestSimulator,
        latency_seconds: float,
        sleep: Callable[[float], None],
    ) -> None:
        self.simulator = simulator
        self.latency_seconds = latency_seconds
        self.sleep = sleep

    def reset(self, *, seeds=None):
        return self.simulator.reset(seeds=seeds)

    def step(self, request: RestRequestBatch):
        result = self.simulator.step(request)
        self.sleep(self.latency_seconds)
        return result


def run_encoder(
    encoder: ObservationSpaceEncoder[dict[str, int]],
    observation,
) -> dict[str, int]:
    return encoder.encode(observation)


def run_backend(backend: RestBackend, request: RestRequestBatch):
    return backend.step(request)


def test_reset_observes_root_for_free_and_reveals_only_direct_children(
    capture: RestCapture,
) -> None:
    simulator = BatchedRestSimulator(capture=capture, num_envs=2)

    observation = simulator.reset(seeds=[7, 7])

    root = node_id(capture, ROOT)
    direct_children = uri_ids(capture, DIRECT_ROOT_CHILDREN)
    expected_known = {root, *direct_children}
    root_edges = set(capture.outgoing_edge_ids(root).astype(int).tolist())

    for row in range(observation.batch_size):
        assert int(observation.step_count[row]) == 0
        assert int(observation.state_version[row]) == 0
        assert int(observation.last_action_node_id[row]) == -1
        assert int(observation.last_response_status[row]) == 200
        assert true_ids(observation.known_mask[row]) == expected_known
        assert true_ids(observation.visited_mask[row]) == {root}
        assert true_ids(observation.body_visible_mask[row]) == {root}
        assert true_ids(observation.frontier_mask[row]) == direct_children
        assert true_ids(observation.visible_edge_mask[row]) == root_edges


def test_external_profile_applies_api_and_encoder_batch_latency(
    capture: RestCapture,
) -> None:
    requested_sleeps: list[float] = []
    simulator = BatchedRestSimulator(capture=capture, num_envs=32)
    backend = MockLatencyBackend(
        simulator=simulator,
        latency_seconds=0.002,
        sleep=requested_sleeps.append,
    )
    backend.reset(seeds=101)
    encoder = MockObservationEncoder(
        latency_seconds=0.012,
        sleep=requested_sleeps.append,
    )

    step = run_backend(
        backend,
        batch_get(node_id(capture, SYSTEMS), 32),
    )
    encoding = run_encoder(encoder, step.observation)

    assert requested_sleeps == [0.002, 0.012]
    assert encoding == {
        "batch_size": 32,
        "num_nodes": capture.num_nodes,
    }


def test_materialized_reset_masks_child_bodies(capture: RestCapture) -> None:
    simulator = RestSimulator(capture=capture)

    observation = simulator.reset(seed=11)

    root_node = get_node(observation, ROOT)
    assert root_node.visited is True
    assert root_node.json_body is not None
    assert {node.uri for node in observation.graph.nodes} == {
        ROOT,
        *DIRECT_ROOT_CHILDREN,
    }
    for child in DIRECT_ROOT_CHILDREN:
        child_node = get_node(observation, child)
        assert child_node.visited is False
        assert child_node.json_body is None
    assert {
        capture.edges[edge_id].target_id
        for edge_id in capture.outgoing_edge_ids(capture.root_node_id)
    } == uri_ids(capture, DIRECT_ROOT_CHILDREN)
    assert any(
        edge.relation == "/Escaped~1Key~0/@odata.id"
        for edge in observation.graph.edges
    )


def test_first_explicit_get_root_is_repeat_and_preserves_graph_hash(
    capture: RestCapture,
) -> None:
    simulator = RestSimulator(capture=capture)
    initial = simulator.reset(seed=17)

    step = simulator.step(RestRequest(uri=ROOT, method="GET", arguments={}))

    assert step.transition.status_code == 200
    assert step.transition.first_visit is False
    assert step.transition.newly_discovered_node_ids == ()
    assert step.transition.newly_visible_edge_ids == ()
    assert step.transition.before_version == 0
    assert step.transition.after_version == 0
    assert step.transition.before_sha256 == initial.graph.snapshot_sha256
    assert step.transition.after_sha256 == initial.graph.snapshot_sha256
    assert step.observation.step_count == 1


def test_recursive_discovery_reveals_links_only_after_successful_get(
    capture: RestCapture,
) -> None:
    simulator = RestSimulator(capture=capture)
    simulator.reset(seed=23)

    hidden = simulator.step(
        RestRequest(uri=SYSTEM_1, method="GET", arguments={}),
    )
    assert hidden.transition.status_code == 404
    assert SYSTEM_1 not in {node.uri for node in hidden.observation.graph.nodes}

    systems = simulator.step(
        RestRequest(uri=SYSTEMS, method="GET", arguments={}),
    )
    assert systems.transition.status_code == 200
    assert systems.transition.first_visit is True
    assert systems.transition.newly_discovered_node_ids == (
        node_id(capture, SYSTEM_1),
    )
    assert set(systems.transition.newly_visible_edge_ids) == set(
        capture.outgoing_edge_ids(node_id(capture, SYSTEMS))
        .astype(int)
        .tolist()
    )
    assert SYSTEM_1 in {node.uri for node in systems.observation.graph.nodes}
    assert get_node(systems.observation, SYSTEMS).visited is True
    assert get_node(systems.observation, SYSTEMS).json_body is not None
    assert get_node(systems.observation, SYSTEM_1).json_body is None

    system = simulator.step(
        RestRequest(uri=SYSTEM_1, method="GET", arguments={}),
    )
    assert system.transition.status_code == 200
    assert get_node(system.observation, SYSTEM_1).visited is True
    assert get_node(system.observation, SYSTEM_1).json_body is not None
    assert set(system.transition.newly_visible_edge_ids) == set(
        capture.outgoing_edge_ids(node_id(capture, SYSTEM_1))
        .astype(int)
        .tolist()
    )
    assert set(system.transition.newly_discovered_node_ids) == uri_ids(
        capture,
        {BIOS, MANAGER_1},
    )

    manager = simulator.step(
        RestRequest(uri=MANAGER_1, method="GET", arguments={}),
    )
    assert manager.transition.status_code == 200
    assert manager.transition.newly_discovered_node_ids == ()
    assert ROOT in {node.uri for node in manager.observation.graph.nodes}


def test_reachable_mask_is_exact_and_cycle_safe(capture: RestCapture) -> None:
    reachable = {
        capture.resources[int(node_id_value)].uri
        for node_id_value in np.flatnonzero(capture.reachable_mask)
    }

    assert reachable == REACHABLE_URIS
    assert UNREACHABLE not in reachable


def test_hidden_uri_and_unknown_uri_are_indistinguishable_404(
    capture: RestCapture,
) -> None:
    hidden_simulator = RestSimulator(capture=capture)
    unknown_simulator = RestSimulator(capture=capture)
    hidden_initial = hidden_simulator.reset(seed=29)
    unknown_initial = unknown_simulator.reset(seed=29)

    hidden = hidden_simulator.step(
        RestRequest(uri=SYSTEM_1, method="GET", arguments={}),
    )
    unknown = unknown_simulator.step(
        RestRequest(uri="/redfish/v1/NoSuchResource", method="GET", arguments={}),
    )

    assert hidden.transition.status_code == 404
    assert unknown.transition.status_code == 404
    assert plain_error(hidden.transition.error) == plain_error(
        unknown.transition.error,
    )
    assert hidden.transition.json_body is None
    assert unknown.transition.json_body is None
    assert hidden.transition.after_version == hidden.transition.before_version == 0
    assert unknown.transition.after_version == unknown.transition.before_version == 0
    assert hidden.transition.after_sha256 == hidden_initial.graph.snapshot_sha256
    assert unknown.transition.after_sha256 == unknown_initial.graph.snapshot_sha256
    assert {node.uri for node in hidden.observation.graph.nodes} == {
        ROOT,
        *DIRECT_ROOT_CHILDREN,
    }


@pytest.mark.parametrize("method", ["HEAD", "PATCH"])
def test_hidden_uri_does_not_leak_through_method_validation(
    capture: RestCapture,
    method: str,
) -> None:
    hidden_simulator = RestSimulator(capture=capture)
    unknown_simulator = RestSimulator(capture=capture)
    hidden_initial = hidden_simulator.reset(seed=30)
    unknown_initial = unknown_simulator.reset(seed=30)

    hidden = hidden_simulator.step(
        RestRequest(uri=SYSTEM_1, method=method, arguments={}),
    )
    unknown = unknown_simulator.step(
        RestRequest(
            uri="/redfish/v1/NoSuchResource",
            method=method,
            arguments={},
        ),
    )

    assert hidden.transition.status_code == 404
    assert unknown.transition.status_code == 404
    assert plain_error(hidden.transition.error) == plain_error(
        unknown.transition.error,
    )
    assert hidden.transition.after_version == 0
    assert unknown.transition.after_version == 0
    assert hidden.transition.after_sha256 == hidden_initial.graph.snapshot_sha256
    assert unknown.transition.after_sha256 == unknown_initial.graph.snapshot_sha256
    assert hidden.transition.newly_discovered_node_ids == ()
    assert hidden.transition.newly_visible_edge_ids == ()


def test_unsupported_method_returns_405_for_known_uri(capture: RestCapture) -> None:
    simulator = RestSimulator(capture=capture)
    initial = simulator.reset(seed=31)

    step = simulator.step(RestRequest(uri=ROOT, method="PATCH", arguments={}))

    assert step.transition.status_code == 405
    assert plain_error(step.transition.error) == {
        "code": "MethodNotAllowed",
        "status": 405,
    }
    assert step.transition.before_version == 0
    assert step.transition.after_version == 1
    assert step.transition.before_sha256 == initial.graph.snapshot_sha256
    assert step.transition.after_sha256 != initial.graph.snapshot_sha256
    assert get_node(step.observation, ROOT).last_status == 405
    assert plain_error(get_node(step.observation, ROOT).last_error) == {
        "code": "MethodNotAllowed",
        "status": 405,
    }


def test_head_does_not_return_body_visit_or_discover(capture: RestCapture) -> None:
    simulator = RestSimulator(capture=capture)
    initial = simulator.reset(seed=37)

    step = simulator.step(RestRequest(uri=MANAGERS, method="HEAD", arguments={}))

    assert step.transition.status_code == 200
    assert step.transition.json_body is None
    assert step.transition.error is None
    assert step.transition.first_visit is False
    assert step.transition.newly_discovered_node_ids == ()
    assert step.transition.newly_visible_edge_ids == ()
    assert get_node(step.observation, MANAGERS).visited is False
    assert get_node(step.observation, MANAGERS).json_body is None
    assert MANAGER_1 not in {node.uri for node in step.observation.graph.nodes}
    assert step.transition.before_sha256 == initial.graph.snapshot_sha256
    assert step.transition.after_sha256 != initial.graph.snapshot_sha256


def test_two_runtimes_share_capture_but_keep_independent_masks(
    capture: RestCapture,
) -> None:
    first = RestSimulator(capture=capture)
    second = RestSimulator(capture=capture)
    first.reset(seed=41)
    second.reset(seed=41)

    assert first.capture is second.capture

    first_step = first.step(RestRequest(uri=SYSTEMS, method="GET", arguments={}))
    second_observation = second.observe()

    assert SYSTEM_1 in {node.uri for node in first_step.observation.graph.nodes}
    assert SYSTEM_1 not in {node.uri for node in second_observation.graph.nodes}
    assert get_node(second_observation, SYSTEMS).visited is False


@pytest.mark.parametrize("num_envs", [1, 32, 256, 1024])
def test_batch_result_length_always_equals_runtime_count(
    capture: RestCapture,
    num_envs: int,
) -> None:
    simulator = BatchedRestSimulator(capture=capture, num_envs=num_envs)
    simulator.reset(seeds=123)

    step = simulator.step_many(batch_get(node_id(capture, SYSTEMS), num_envs))

    assert step.observation.batch_size == num_envs
    assert step.transition.batch_size == num_envs
    assert step.transition.status_codes.shape == (num_envs,)
    assert step.transition.first_visit.shape == (num_envs,)
    assert len(step.transition.newly_discovered_node_ids) == num_envs
    assert step.observation.visible_edge_mask.shape == (
        num_envs,
        capture.num_edges,
    )
    assert np.all(step.transition.status_codes == 200)


def test_batch_rejects_misaligned_request_count(capture: RestCapture) -> None:
    simulator = BatchedRestSimulator(capture=capture, num_envs=2)
    simulator.reset(seeds=[1, 2])

    request = RestRequestBatch(
        node_ids=np.asarray([node_id(capture, SYSTEMS)], dtype=np.int32),
        method_ids=np.asarray([METHOD_GET], dtype=np.int8),
    )

    with pytest.raises(ValueError, match="one request is required per runtime"):
        simulator.step_many(request)


def test_step_requires_reset(capture: RestCapture) -> None:
    simulator = BatchedRestSimulator(capture=capture, num_envs=1)

    with pytest.raises(RuntimeError, match="reset must be called before step"):
        simulator.step_many(batch_get(capture.root_node_id, 1))


@pytest.mark.parametrize("seeds", [True, [1.5], [False]])
def test_reset_rejects_non_integer_seeds(
    capture: RestCapture,
    seeds,
) -> None:
    simulator = BatchedRestSimulator(capture=capture, num_envs=1)

    with pytest.raises(TypeError, match="seeds must contain integers"):
        simulator.reset(seeds=seeds)


def test_same_seed_replays_same_trace_and_hashes(capture: RestCapture) -> None:
    def run_trace() -> tuple[object, ...]:
        simulator = RestSimulator(capture=capture)
        reset = simulator.reset(seed=43)
        first = simulator.step(RestRequest(uri=SYSTEMS, method="GET", arguments={}))
        second = simulator.step(
            RestRequest(uri=SYSTEM_1, method="GET", arguments={}),
        )
        return (
            reset.graph.snapshot_sha256,
            first.transition.status_code,
            first.transition.first_visit,
            first.transition.newly_discovered_node_ids,
            first.transition.after_sha256,
            second.transition.status_code,
            second.transition.first_visit,
            second.transition.newly_discovered_node_ids,
            second.transition.after_sha256,
        )

    assert run_trace() == run_trace()


def test_capture_json_and_observation_batches_are_immutable_and_detached(
    capture: RestCapture,
) -> None:
    simulator = BatchedRestSimulator(capture=capture, num_envs=1)
    observation = simulator.reset(seeds=47)
    original_known = true_ids(observation.known_mask[0])

    with pytest.raises(TypeError):
        capture.resources[node_id(capture, ROOT)].base_json["New"] = "value"
    with pytest.raises(TypeError):
        capture.resources[node_id(capture, ROOT)].base_json["Systems"][
            "@odata.id"
        ] = "/mutated"
    with pytest.raises(ValueError, match="read-only"):
        observation.known_mask[0, node_id(capture, UNREACHABLE)] = True

    simulator.runtime.known[:, :] = True

    assert true_ids(observation.known_mask[0]) == original_known
    assert true_ids(simulator.observe().known_mask[0]) != original_known


def test_mapping_proxy_input_is_deeply_copied_and_frozen() -> None:
    caller_owned = ["original"]
    request = RestRequest(
        uri=ROOT,
        method="GET",
        arguments=MappingProxyType({"nested": caller_owned}),
    )

    caller_owned.append("mutated")

    assert request.arguments["nested"] == ("original",)
    with pytest.raises(TypeError):
        request.arguments["new"] = "value"


def test_malformed_request_types_fail_before_state_changes(
    capture: RestCapture,
) -> None:
    simulator = RestSimulator(capture=capture)
    initial = simulator.reset(seed=49)

    with pytest.raises(TypeError, match="uri must be a string"):
        RestRequest(uri=1, method="GET", arguments={})
    with pytest.raises(TypeError, match="method must be a string"):
        RestRequest(uri=ROOT, method=1, arguments={})
    with pytest.raises(TypeError, match="node_ids must contain integers"):
        RestRequestBatch(
            node_ids=np.asarray([1.5]),
            method_ids=np.asarray([METHOD_GET]),
        )

    assert simulator.observe() == initial


def test_snapshot_hash_changes_only_when_visible_graph_state_changes(
    capture: RestCapture,
) -> None:
    simulator = RestSimulator(capture=capture)
    initial = simulator.reset(seed=53)

    repeated_get = simulator.step(
        RestRequest(uri=ROOT, method="GET", arguments={}),
    )
    assert repeated_get.transition.after_version == 0
    assert repeated_get.transition.after_sha256 == initial.graph.snapshot_sha256

    first_bad_method = simulator.step(
        RestRequest(uri=ROOT, method="PATCH", arguments={}),
    )
    assert first_bad_method.transition.after_version == 1
    assert first_bad_method.transition.after_sha256 != initial.graph.snapshot_sha256

    second_bad_method = simulator.step(
        RestRequest(uri=ROOT, method="PATCH", arguments={}),
    )
    assert second_bad_method.transition.after_version == 1
    assert (
        second_bad_method.transition.after_sha256
        == first_bad_method.transition.after_sha256
    )


def test_snapshot_hash_binds_capture_content_not_only_capture_label() -> None:
    def capture_with_name(name: str) -> RestCapture:
        return RestCapture.from_mappings(
            responses={ROOT: {"Name": name}},
            allowed_methods={ROOT: {"GET"}},
            root_uri=ROOT,
            capture_id="reused-caller-label",
        )

    first_capture = capture_with_name("First")
    second_capture = capture_with_name("Second")
    first = RestSimulator(capture=first_capture).reset(seed=89)
    second = RestSimulator(capture=second_capture).reset(seed=89)

    assert first_capture.capture_id == second_capture.capture_id
    assert first_capture.content_sha256 != second_capture.content_sha256
    assert first.graph.snapshot_sha256 != second.graph.snapshot_sha256


def test_snapshot_hash_does_not_expose_hidden_body_content() -> None:
    def capture_with_hidden_name(name: str) -> RestCapture:
        return RestCapture.from_mappings(
            responses={
                ROOT: {"Systems": {"@odata.id": SYSTEMS}},
                SYSTEMS: {"Name": name},
            },
            allowed_methods={ROOT: {"GET"}, SYSTEMS: {"GET"}},
            root_uri=ROOT,
            capture_id="reused-caller-label",
        )

    first_capture = capture_with_hidden_name("First")
    second_capture = capture_with_hidden_name("Second")
    first_simulator = RestSimulator(capture=first_capture)
    second_simulator = RestSimulator(capture=second_capture)

    first_reset = first_simulator.reset(seed=101)
    second_reset = second_simulator.reset(seed=101)
    first_step = first_simulator.step(
        RestRequest(uri=SYSTEMS, method="GET", arguments={}),
    )
    second_step = second_simulator.step(
        RestRequest(uri=SYSTEMS, method="GET", arguments={}),
    )

    assert first_capture.content_sha256 != second_capture.content_sha256
    assert (
        first_reset.graph.snapshot_sha256
        == second_reset.graph.snapshot_sha256
    )
    assert (
        first_step.transition.after_sha256
        != second_step.transition.after_sha256
    )


def test_snapshot_hash_identity_ignores_caller_capture_label() -> None:
    def capture_with_label(label: str) -> RestCapture:
        return RestCapture.from_mappings(
            responses={
                SYSTEMS: {"Members": []},
                ROOT: {
                    "Name": "Root",
                    "Systems": {"@odata.id": SYSTEMS},
                },
            },
            allowed_methods={
                ROOT: {"GET", "HEAD"},
                SYSTEMS: {"GET"},
            },
            root_uri=ROOT,
            capture_id=label,
        )

    first_capture = capture_with_label("caller-label-a")
    second_capture = capture_with_label("caller-label-b")
    first_simulator = RestSimulator(capture=first_capture)
    second_simulator = RestSimulator(capture=second_capture)

    first_reset = first_simulator.reset(seed=97)
    second_reset = second_simulator.reset(seed=97)
    first_step = first_simulator.step(
        RestRequest(uri=SYSTEMS, method="GET", arguments={}),
    )
    second_step = second_simulator.step(
        RestRequest(uri=SYSTEMS, method="GET", arguments={}),
    )

    assert first_capture.capture_id != second_capture.capture_id
    assert first_capture.content_sha256 == second_capture.content_sha256
    assert (
        first_reset.graph.snapshot_sha256
        == second_reset.graph.snapshot_sha256
    )
    assert (
        first_step.transition.after_sha256
        == second_step.transition.after_sha256
    )


def test_head_method_id_has_no_body_or_discovery_in_batch(
    capture: RestCapture,
) -> None:
    simulator = BatchedRestSimulator(capture=capture, num_envs=2)
    simulator.reset(seeds=[59, 61])
    managers = node_id(capture, MANAGERS)
    request = RestRequestBatch(
        node_ids=np.full(2, managers, dtype=np.int32),
        method_ids=np.full(2, METHOD_HEAD, dtype=np.int8),
    )

    step = simulator.step_many(request)

    assert np.all(step.transition.status_codes == 200)
    assert step.transition.json_bodies == (None, None)
    assert step.transition.newly_discovered_node_ids == ((), ())
    assert not np.any(step.observation.visited_mask[:, managers])
    assert not np.any(step.observation.body_visible_mask[:, managers])


def test_batch_preserves_heterogeneous_row_semantics(capture: RestCapture) -> None:
    simulator = BatchedRestSimulator(capture=capture, num_envs=3)
    simulator.reset(seeds=[67, 71, 73])
    request = RestRequestBatch(
        node_ids=np.asarray(
            [
                capture.root_node_id,
                node_id(capture, MANAGERS),
                -1,
            ],
            dtype=np.int32,
        ),
        method_ids=np.asarray(
            [METHOD_GET, METHOD_HEAD, METHOD_GET],
            dtype=np.int8,
        ),
    )

    step = simulator.step(request)

    assert step.transition.status_codes.tolist() == [200, 200, 404]
    assert step.transition.first_visit.tolist() == [False, False, False]
    assert step.transition.after_versions.tolist() == [0, 1, 0]
    assert step.transition.newly_discovered_node_ids == ((), (), ())
    assert step.transition.json_bodies[1:] == (None, None)
    assert not step.observation.visited_mask[1, node_id(capture, MANAGERS)]


def test_batch_invalid_method_id_returns_405_for_known_node(
    capture: RestCapture,
) -> None:
    simulator = BatchedRestSimulator(capture=capture, num_envs=2)
    simulator.reset(seeds=[79, 83])
    request = RestRequestBatch(
        node_ids=np.full(2, capture.root_node_id, dtype=np.int32),
        method_ids=np.full(2, 99, dtype=np.int8),
    )

    step = simulator.step(request)

    assert step.transition.status_codes.tolist() == [405, 405]
    assert step.transition.first_visit.tolist() == [False, False]
    assert step.transition.after_versions.tolist() == [1, 1]
    assert all(
        plain_error(error) == {"code": "MethodNotAllowed", "status": 405}
        for error in step.transition.errors
    )


def test_sim_core_has_no_forbidden_runtime_imports() -> None:
    forbidden = {"torch", "transformers", "requests", "gym", "gymnasium"}
    sim_dir = Path(__file__).parents[3] / "igc" / "envs" / "sim"

    offenders: list[tuple[str, str]] = []
    for path in sorted(sim_dir.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root_name = alias.name.split(".", maxsplit=1)[0]
                    if root_name in forbidden:
                        offenders.append((path.name, alias.name))
            elif isinstance(node, ast.ImportFrom) and node.module:
                root_name = node.module.split(".", maxsplit=1)[0]
                if root_name in forbidden:
                    offenders.append((path.name, node.module))

    assert offenders == []
