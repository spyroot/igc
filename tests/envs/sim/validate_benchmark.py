"""Validate one machine-readable REST SIM benchmark for the GYM CI gate.

This validator deliberately freezes the synthetic external-latency profile but
does not use end-to-end timing as a performance verdict. HTTP and observation
encoding are replaceable boundaries; only ``simulator_*`` metrics describe the
component owned by this gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


EXPECTED_SCHEMA_VERSION = 2
EXPECTED_BATCH_SIZES = (1, 32, 256, 1024)
EXPECTED_PROFILE_ID = "gym-ci-v1"
EXPECTED_CAPTURE_ID = "get-benchmark-v1"
EXPECTED_CAPTURE_SHA256 = (
    "0b62487bfa1c464363075e467cfb1eefe786b859a0fbbdf1aea8eb763ef0b238"
)
EXPECTED_CAPTURE_NODES = 2
EXPECTED_CAPTURE_EDGES = 1
EXPECTED_ITERATIONS_PER_ROUND = 50
EXPECTED_WARMUP_ITERATIONS = 5
EXPECTED_ROUNDS = 5
EXPECTED_API_MS_PER_BATCH = 2.0
EXPECTED_ENCODER_MS_PER_BATCH = 12.0
EXPECTED_RUNTIME_ARRAY_BYTES_PER_RUNTIME = 55.0
MAX_INPUT_BYTES = 2_000_000

_POSITIVE_METRICS = (
    "median_seconds",
    "simulator_gets_per_second",
    "simulator_microseconds_per_get",
    "simulator_ms_per_batch",
    "end_to_end_gets_per_second",
    "end_to_end_microseconds_per_get",
    "end_to_end_batches_per_second",
    "end_to_end_ms_per_batch",
)


def _finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _positive_number(value: object) -> bool:
    return _finite_number(value) and float(value) > 0.0


def _exact_float(value: object, expected: float) -> bool:
    return _finite_number(value) and float(value) == expected


def _reject_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _consistent(value: object, expected: float) -> bool:
    return _finite_number(value) and math.isclose(
        float(value),
        expected,
        rel_tol=1e-9,
        abs_tol=1e-12,
    )


def validate_benchmark(document: object) -> list[str]:
    """Return deterministic contract violations for one benchmark document."""
    if not isinstance(document, Mapping):
        return ["benchmark document must be a JSON object"]

    violations: list[str] = []
    if document.get("schema_version") != EXPECTED_SCHEMA_VERSION:
        violations.append(
            f"schema_version must be {EXPECTED_SCHEMA_VERSION}",
        )
    if document.get("benchmark") != "rest_sim_get_only":
        violations.append("benchmark must be 'rest_sim_get_only'")
    if document.get("latency_source") != "synthetic_configured":
        violations.append("latency_source must be 'synthetic_configured'")
    if document.get("capture_id") != EXPECTED_CAPTURE_ID:
        violations.append(f"capture_id must be {EXPECTED_CAPTURE_ID!r}")
    if document.get("capture_sha256") != EXPECTED_CAPTURE_SHA256:
        violations.append(
            f"capture_sha256 must be {EXPECTED_CAPTURE_SHA256!r}",
        )
    if document.get("capture_nodes") != EXPECTED_CAPTURE_NODES:
        violations.append(f"capture_nodes must be {EXPECTED_CAPTURE_NODES}")
    if document.get("capture_edges") != EXPECTED_CAPTURE_EDGES:
        violations.append(f"capture_edges must be {EXPECTED_CAPTURE_EDGES}")

    latency_profile = document.get("latency_profile")
    if not isinstance(latency_profile, Mapping):
        violations.append("latency_profile must be an object")
    else:
        if latency_profile.get("id") != EXPECTED_PROFILE_ID:
            violations.append(
                f"latency_profile.id must be {EXPECTED_PROFILE_ID!r}",
            )
        if not _exact_float(
            latency_profile.get("api_response_ms_per_batch"),
            EXPECTED_API_MS_PER_BATCH,
        ):
            violations.append(
                "API latency must remain pinned at "
                f"{EXPECTED_API_MS_PER_BATCH} ms per batch",
            )
        if not _exact_float(
            latency_profile.get("encoder_pass_ms_per_batch"),
            EXPECTED_ENCODER_MS_PER_BATCH,
        ):
            violations.append(
                "encoder latency must remain pinned at "
                f"{EXPECTED_ENCODER_MS_PER_BATCH} ms per batch",
            )

    measurement = document.get("measurement_contract")
    if not isinstance(measurement, Mapping):
        violations.append("measurement_contract must be an object")
    else:
        if measurement.get("performance_decision_source") != (
            "simulator_metrics_only"
        ):
            violations.append(
                "performance_decision_source must be 'simulator_metrics_only'",
            )
        if measurement.get("end_to_end_metrics") != "context_only":
            violations.append("end_to_end_metrics must be 'context_only'")

    results = document.get("results")
    if not isinstance(results, list):
        violations.append("results must be a list")
        return violations

    batches = [
        result.get("runtimes") if isinstance(result, Mapping) else None
        for result in results
    ]
    if batches != list(EXPECTED_BATCH_SIZES):
        violations.append(
            "results runtimes must be exactly "
            f"{list(EXPECTED_BATCH_SIZES)!r}",
        )

    for index, result in enumerate(results):
        label = f"results[{index}]"
        if not isinstance(result, Mapping):
            violations.append(f"{label} must be an object")
            continue
        runtimes = result.get("runtimes")
        label = f"batch {runtimes!r}"
        if type(runtimes) is not int or runtimes < 1:
            violations.append(f"{label}: runtimes must be a positive integer")
        if result.get("iterations_per_round") != EXPECTED_ITERATIONS_PER_ROUND:
            violations.append(
                f"{label}: iterations_per_round must be "
                f"{EXPECTED_ITERATIONS_PER_ROUND}",
            )
        if result.get("warmup_iterations") != EXPECTED_WARMUP_ITERATIONS:
            violations.append(
                f"{label}: warmup_iterations must be "
                f"{EXPECTED_WARMUP_ITERATIONS}",
            )
        if result.get("rounds") != EXPECTED_ROUNDS:
            violations.append(f"{label}: rounds must be {EXPECTED_ROUNDS}")
        if result.get("batch_aligned") is not True:
            violations.append(f"{label}: batch_aligned must be true")
        if result.get("deterministic_replay") is not True:
            violations.append(f"{label}: deterministic_replay must be true")
        if not _exact_float(
            result.get("synthetic_api_response_ms_per_batch"),
            EXPECTED_API_MS_PER_BATCH,
        ):
            violations.append(f"{label}: API latency pin changed")
        if not _exact_float(
            result.get("synthetic_encoder_ms_per_batch"),
            EXPECTED_ENCODER_MS_PER_BATCH,
        ):
            violations.append(f"{label}: encoder latency pin changed")
        if not _exact_float(
            result.get("configured_external_ms_per_batch"),
            EXPECTED_API_MS_PER_BATCH + EXPECTED_ENCODER_MS_PER_BATCH,
        ):
            violations.append(f"{label}: combined external latency is invalid")
        for metric in _POSITIVE_METRICS:
            value = result.get(metric)
            if not _finite_number(value) or float(value) <= 0.0:
                violations.append(f"{label}: {metric} must be finite and positive")
        runtime_bytes = result.get("runtime_array_bytes_per_runtime")
        if not _exact_float(
            runtime_bytes,
            EXPECTED_RUNTIME_ARRAY_BYTES_PER_RUNTIME,
        ):
            violations.append(
                f"{label}: runtime_array_bytes_per_runtime must be "
                f"{EXPECTED_RUNTIME_ARRAY_BYTES_PER_RUNTIME}",
            )
        simulator_fraction = result.get("simulator_fraction_of_end_to_end")
        if (
            not _finite_number(simulator_fraction)
            or not 0.0 <= float(simulator_fraction) <= 1.0
        ):
            violations.append(
                f"{label}: simulator_fraction_of_end_to_end must be in [0, 1]",
            )
        if type(runtimes) is int and runtimes > 0:
            simulator_us = result.get("simulator_microseconds_per_get")
            simulator_ms = result.get("simulator_ms_per_batch")
            simulator_rate = result.get("simulator_gets_per_second")
            end_to_end_us = result.get("end_to_end_microseconds_per_get")
            end_to_end_ms = result.get("end_to_end_ms_per_batch")
            end_to_end_rate = result.get("end_to_end_gets_per_second")
            batch_rate = result.get("end_to_end_batches_per_second")
            median_seconds = result.get("median_seconds")
            if _positive_number(simulator_us) and not _consistent(
                simulator_ms,
                float(simulator_us) * runtimes / 1_000.0,
            ):
                violations.append(f"{label}: simulator timing fields disagree")
            if _positive_number(simulator_us) and not _consistent(
                simulator_rate,
                1_000_000.0 / float(simulator_us),
            ):
                violations.append(f"{label}: simulator rate fields disagree")
            if _positive_number(end_to_end_us) and not _consistent(
                end_to_end_ms,
                float(end_to_end_us) * runtimes / 1_000.0,
            ):
                violations.append(f"{label}: end-to-end timing fields disagree")
            if _positive_number(end_to_end_us) and not _consistent(
                end_to_end_rate,
                1_000_000.0 / float(end_to_end_us),
            ):
                violations.append(f"{label}: end-to-end rate fields disagree")
            if _positive_number(end_to_end_ms) and not _consistent(
                batch_rate,
                1_000.0 / float(end_to_end_ms),
            ):
                violations.append(f"{label}: batch rate fields disagree")
            if _positive_number(end_to_end_ms) and not _consistent(
                median_seconds,
                (
                    float(end_to_end_ms)
                    * EXPECTED_ITERATIONS_PER_ROUND
                    / 1_000.0
                ),
            ):
                violations.append(f"{label}: median timing field disagrees")
            if _positive_number(end_to_end_ms) and float(end_to_end_ms) < (
                EXPECTED_API_MS_PER_BATCH + EXPECTED_ENCODER_MS_PER_BATCH
            ):
                violations.append(
                    f"{label}: configured external latency was not observed",
                )
            if (
                _positive_number(simulator_ms)
                and _positive_number(end_to_end_ms)
                and float(end_to_end_ms) < float(simulator_ms)
            ):
                violations.append(
                    f"{label}: end-to-end time is below simulator time",
                )
            if (
                _finite_number(simulator_ms)
                and _finite_number(end_to_end_ms)
                and float(end_to_end_ms) > 0.0
                and not _consistent(
                    simulator_fraction,
                    float(simulator_ms) / float(end_to_end_ms),
                )
            ):
                violations.append(f"{label}: simulator fraction disagrees")
    return violations


def build_gate_report(
    *,
    document: object,
    input_sha256: str,
    violations: Sequence[str],
) -> dict[str, Any]:
    """Build the bounded, sanitized gate artifact."""
    results = document.get("results", []) if isinstance(document, Mapping) else []
    simulator_metrics = []
    if isinstance(results, list):
        for result in results:
            if not isinstance(result, Mapping):
                continue
            simulator_metrics.append(
                {
                    "runtimes": result.get("runtimes"),
                    "simulator_gets_per_second": result.get(
                        "simulator_gets_per_second",
                    ),
                    "simulator_microseconds_per_get": result.get(
                        "simulator_microseconds_per_get",
                    ),
                    "runtime_array_bytes_per_runtime": result.get(
                        "runtime_array_bytes_per_runtime",
                    ),
                },
            )
    return {
        "schema_version": 1,
        "gate": "GYM",
        "status": "FAIL" if violations else "PASS",
        "benchmark_sha256": input_sha256,
        "capture_sha256": EXPECTED_CAPTURE_SHA256,
        "latency_profile": {
            "id": EXPECTED_PROFILE_ID,
            "api_response_ms_per_batch": EXPECTED_API_MS_PER_BATCH,
            "encoder_pass_ms_per_batch": EXPECTED_ENCODER_MS_PER_BATCH,
        },
        "performance_decision_source": "simulator_metrics_only",
        "simulator_metrics": simulator_metrics,
        "violations": list(violations),
    }


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the fixed-profile REST SIM benchmark artifact.",
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    try:
        raw = args.input.read_bytes()
        if len(raw) > MAX_INPUT_BYTES:
            raise ValueError(
                f"input exceeds {MAX_INPUT_BYTES} bytes",
            )
        document = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
        violations = validate_benchmark(document)
        report = build_gate_report(
            document=document,
            input_sha256=hashlib.sha256(raw).hexdigest(),
            violations=violations,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (OSError, TypeError, ValueError, RecursionError) as error:
        print(
            f"BLOCKED: unable to validate REST SIM benchmark: {error}",
            file=sys.stderr,
        )
        return 2

    print(json.dumps(report, sort_keys=True))
    for violation in violations:
        print(f"FAIL: {violation}", file=sys.stderr)
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
