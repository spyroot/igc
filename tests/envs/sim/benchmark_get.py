"""Machine-readable GET-only REST simulator microbenchmark.

Audience: agent and human. This tool is non-interactive and writes one bounded
JSON document to stdout. Diagnostics are written to stderr.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections.abc import Sequence

import numpy as np

from igc.envs.sim import (
    METHOD_GET,
    BatchedRestSimulator,
    RestCapture,
    RestRequestBatch,
)


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


def _run_case(
    *,
    capture: RestCapture,
    runtimes: int,
    iterations: int,
    warmup: int,
    rounds: int,
) -> dict[str, object]:
    simulator = BatchedRestSimulator(capture=capture, num_envs=runtimes)
    systems = capture.uri_to_id["/redfish/v1/Systems"]
    request = RestRequestBatch(
        node_ids=np.full(runtimes, systems, dtype=np.int32),
        method_ids=np.full(runtimes, METHOD_GET, dtype=np.int8),
    )
    simulator.reset(seeds=20260727)
    for _ in range(warmup):
        simulator.step_many(request)

    elapsed_samples: list[float] = []
    last_step = None
    for _ in range(rounds):
        started = time.perf_counter()
        for _ in range(iterations):
            last_step = simulator.step_many(request)
        elapsed_samples.append(time.perf_counter() - started)

    if last_step is None:
        raise RuntimeError("benchmark executed no simulator steps")
    elapsed = statistics.median(elapsed_samples)
    get_count = runtimes * iterations
    batch_aligned = (
        last_step.observation.batch_size == runtimes
        and last_step.transition.batch_size == runtimes
        and last_step.transition.status_codes.shape == (runtimes,)
    )

    simulator.reset(seeds=20260727)
    first = simulator.step_many(request)
    simulator.reset(seeds=20260727)
    second = simulator.step_many(request)
    deterministic_replay = (
        np.array_equal(
            first.transition.status_codes,
            second.transition.status_codes,
        )
        and first.transition.after_sha256 == second.transition.after_sha256
    )

    return {
        "runtimes": runtimes,
        "iterations_per_round": iterations,
        "rounds": rounds,
        "median_seconds": elapsed,
        "gets_per_second": get_count / elapsed,
        "microseconds_per_get": elapsed * 1_000_000.0 / get_count,
        "runtime_array_bytes_per_runtime": _runtime_bytes(simulator) / runtimes,
        "batch_aligned": batch_aligned,
        "deterministic_replay": deterministic_replay,
    }


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least one")
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
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    try:
        capture = _capture()
        results = [
            _run_case(
                capture=capture,
                runtimes=runtimes,
                iterations=args.iterations,
                warmup=args.warmup,
                rounds=args.rounds,
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
        "schema_version": 1,
        "benchmark": "rest_sim_get_only",
        "capture_nodes": capture.num_nodes,
        "capture_edges": capture.num_edges,
        "results": results,
    }
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
