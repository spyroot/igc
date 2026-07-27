"""Machine-readable GET-only REST simulator microbenchmark.

Audience: agent and human. This tool is non-interactive and writes one bounded
JSON document to stdout. Diagnostics are written to stderr.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from igc.envs.sim import (
    METHOD_GET,
    BatchedRestSimulator,
    ObservationSpaceEncoder,
    RestBackend,
    RestCapture,
    RestRequestBatch,
)


BENCHMARK_SCHEMA_VERSION = 2
LATENCY_PROFILE_ID = "gym-ci-v1"


@dataclass(frozen=True, slots=True)
class SyntheticLatencyProfile:
    """Configured batch-boundary delays; values are not measured claims."""

    api_response_seconds: float
    encoder_pass_seconds: float


class SyntheticEncoder(ObservationSpaceEncoder[tuple[int, int]]):
    """External encoder stand-in that models one batched encode pass."""

    def __init__(self, profile: SyntheticLatencyProfile) -> None:
        self.profile = profile

    def encode(self, observation) -> tuple[int, int]:
        if self.profile.encoder_pass_seconds:
            time.sleep(self.profile.encoder_pass_seconds)
        return observation.batch_size, observation.num_nodes


class SyntheticLatencyBackend(RestBackend):
    """External REST stand-in with fixed explicit-step latency and SIM timing.

    Reset retains the paper's free-root convention and is not an RL transition,
    so the synthetic HTTP delay applies once per explicit request batch only.
    """

    def __init__(
        self,
        *,
        simulator: BatchedRestSimulator,
        profile: SyntheticLatencyProfile,
    ) -> None:
        self.simulator = simulator
        self.profile = profile
        self.last_simulator_seconds = 0.0

    def reset(self, *, seeds=None):
        self.last_simulator_seconds = 0.0
        return self.simulator.reset(seeds=seeds)

    def step(self, request: RestRequestBatch):
        started = time.perf_counter()
        result = self.simulator.step(request)
        self.last_simulator_seconds = time.perf_counter() - started
        if self.profile.api_response_seconds:
            time.sleep(self.profile.api_response_seconds)
        return result


def _capture() -> RestCapture:
    root = "/redfish/v1"
    systems = "/redfish/v1/Systems"
    return RestCapture.from_mappings(
        responses={
            root: {"Systems": {"@odata.id": systems}},
            systems: {"Members": []},
        },
        allowed_methods={root: {"GET", "HEAD"}, systems: {"GET"}},
        root_uri=root,
        capture_id="get-benchmark-v1",
    )


def _runtime_bytes(simulator: BatchedRestSimulator) -> int:
    runtime = simulator.runtime
    arrays = (
        runtime.known,
        runtime.visited,
        runtime.body_visible,
        runtime.visible_edges,
        runtime.last_status,
        runtime.error_code,
        runtime.body_version,
        runtime.state_version,
        runtime.step_count,
        runtime.last_action_node_id,
        runtime.last_response_status,
        runtime.last_response_error_code,
        runtime.seeds,
    )
    return sum(array.nbytes for array in arrays)


def _same_replay(first, second) -> bool:
    """Compare all observable and transition evidence from two seeded steps."""
    first_observation = first.observation
    second_observation = second.observation
    observation_arrays = (
        "known_mask",
        "visited_mask",
        "frontier_mask",
        "body_visible_mask",
        "visible_edge_mask",
        "last_status",
        "error_code",
        "body_version",
        "state_version",
        "step_count",
        "last_action_node_id",
        "last_response_status",
        "last_response_error_code",
    )
    if any(
        not np.array_equal(
            getattr(first_observation, name),
            getattr(second_observation, name),
        )
        for name in observation_arrays
    ):
        return False

    first_transition = first.transition
    second_transition = second.transition
    transition_arrays = (
        "status_codes",
        "first_visit",
        "before_versions",
        "after_versions",
    )
    if any(
        not np.array_equal(
            getattr(first_transition, name),
            getattr(second_transition, name),
        )
        for name in transition_arrays
    ):
        return False
    return all(
        getattr(first_transition, name) == getattr(second_transition, name)
        for name in (
            "json_bodies",
            "errors",
            "newly_discovered_node_ids",
            "newly_visible_edge_ids",
            "changed_node_ids",
            "before_sha256",
            "after_sha256",
        )
    )


def _run_case(
    *,
    capture: RestCapture,
    runtimes: int,
    iterations: int,
    warmup: int,
    rounds: int,
    latency: SyntheticLatencyProfile,
) -> dict[str, object]:
    simulator = BatchedRestSimulator(capture=capture, num_envs=runtimes)
    systems = capture.uri_to_id["/redfish/v1/Systems"]
    request = RestRequestBatch(
        node_ids=np.full(runtimes, systems, dtype=np.int32),
        method_ids=np.full(runtimes, METHOD_GET, dtype=np.int8),
    )
    backend = SyntheticLatencyBackend(
        simulator=simulator,
        profile=latency,
    )
    backend.reset(seeds=20260727)
    encoder = SyntheticEncoder(latency)
    for _ in range(warmup):
        warmup_step = backend.step(request)
        encoder.encode(warmup_step.observation)

    elapsed_samples: list[float] = []
    simulator_elapsed_samples: list[float] = []
    last_step = None
    for _ in range(rounds):
        started = time.perf_counter()
        simulator_elapsed = 0.0
        for _ in range(iterations):
            last_step = backend.step(request)
            simulator_elapsed += backend.last_simulator_seconds
            encoder.encode(last_step.observation)
        elapsed_samples.append(time.perf_counter() - started)
        simulator_elapsed_samples.append(simulator_elapsed)

    if last_step is None:
        raise RuntimeError("benchmark executed no simulator steps")
    elapsed = statistics.median(elapsed_samples)
    simulator_elapsed = statistics.median(simulator_elapsed_samples)
    get_count = runtimes * iterations
    simulator_ms_per_batch = simulator_elapsed * 1_000.0 / iterations
    end_to_end_ms_per_batch = elapsed * 1_000.0 / iterations
    configured_external_ms_per_batch = (
        latency.api_response_seconds + latency.encoder_pass_seconds
    ) * 1_000.0
    batch_aligned = (
        last_step.observation.batch_size == runtimes
        and last_step.transition.batch_size == runtimes
        and last_step.transition.status_codes.shape == (runtimes,)
    )

    backend.reset(seeds=20260727)
    first = backend.step(request)
    backend.reset(seeds=20260727)
    second = backend.step(request)
    deterministic_replay = _same_replay(first, second)

    return {
        "runtimes": runtimes,
        "iterations_per_round": iterations,
        "warmup_iterations": warmup,
        "rounds": rounds,
        "median_seconds": elapsed,
        "simulator_gets_per_second": get_count / simulator_elapsed,
        "simulator_microseconds_per_get": (
            simulator_elapsed * 1_000_000.0 / get_count
        ),
        "simulator_ms_per_batch": simulator_ms_per_batch,
        "end_to_end_gets_per_second": get_count / elapsed,
        "end_to_end_microseconds_per_get": elapsed * 1_000_000.0 / get_count,
        "end_to_end_batches_per_second": iterations / elapsed,
        "end_to_end_ms_per_batch": end_to_end_ms_per_batch,
        "configured_external_ms_per_batch": configured_external_ms_per_batch,
        "simulator_fraction_of_end_to_end": simulator_elapsed / elapsed,
        "runtime_array_bytes_per_runtime": _runtime_bytes(simulator) / runtimes,
        "synthetic_api_response_ms_per_batch": (
            latency.api_response_seconds * 1_000.0
        ),
        "synthetic_encoder_ms_per_batch": (
            latency.encoder_pass_seconds * 1_000.0
        ),
        "batch_aligned": batch_aligned,
        "deterministic_replay": deterministic_replay,
    }


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least one")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError(
            "value must be finite and nonnegative",
        )
    return parsed


def _batch_sizes(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(_positive_int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("batch sizes must be integers") from error
    if not parsed:
        raise argparse.ArgumentTypeError("at least one batch size is required")
    return parsed


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark aligned GET-only REST simulator batches.",
    )
    parser.add_argument(
        "--batch-sizes",
        type=_batch_sizes,
        default=(1, 32, 256, 1024),
    )
    parser.add_argument("--iterations", type=_positive_int, default=50)
    parser.add_argument("--warmup", type=_positive_int, default=5)
    parser.add_argument("--rounds", type=_positive_int, default=5)
    parser.add_argument(
        "--api-latency-ms",
        type=_nonnegative_float,
        default=2.0,
        help="Synthetic average API response latency per batch.",
    )
    parser.add_argument(
        "--encoder-latency-ms",
        type=_nonnegative_float,
        default=12.0,
        help="Synthetic average observation encode latency per batch.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for the complete JSON benchmark artifact.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    try:
        capture = _capture()
        latency = SyntheticLatencyProfile(
            api_response_seconds=args.api_latency_ms / 1_000.0,
            encoder_pass_seconds=args.encoder_latency_ms / 1_000.0,
        )
        results = [
            _run_case(
                capture=capture,
                runtimes=runtimes,
                iterations=args.iterations,
                warmup=args.warmup,
                rounds=args.rounds,
                latency=latency,
            )
            for runtimes in args.batch_sizes
        ]
    except Exception as error:  # noqa: BLE001 - stable CLI failure contract
        diagnostic = {
            "status": "BLOCKED",
            "error": f"{type(error).__name__}: {error}",
            "safe_next_step": "run the focused SIM contract tests before retrying",
        }
        print(json.dumps(diagnostic, sort_keys=True), file=sys.stderr)
        return 1

    output = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "benchmark": "rest_sim_get_only",
        "capture_id": capture.capture_id,
        "capture_sha256": capture.content_sha256,
        "capture_nodes": capture.num_nodes,
        "capture_edges": capture.num_edges,
        "latency_source": "synthetic_configured",
        "latency_profile": {
            "id": LATENCY_PROFILE_ID,
            "api_response_ms_per_batch": args.api_latency_ms,
            "encoder_pass_ms_per_batch": args.encoder_latency_ms,
        },
        "measurement_contract": {
            "performance_decision_source": "simulator_metrics_only",
            "end_to_end_metrics": "context_only",
        },
        "results": results,
    }
    encoded = json.dumps(output, sort_keys=True)
    if args.output is not None:
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded + "\n", encoding="utf-8")
        except OSError as error:
            print(
                f"BLOCKED: unable to write benchmark artifact: {error}",
                file=sys.stderr,
            )
            return 1
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
