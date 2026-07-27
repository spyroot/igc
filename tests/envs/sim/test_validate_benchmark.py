from __future__ import annotations

import copy
import importlib.util
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest


VALIDATOR_PATH = Path(__file__).with_name("validate_benchmark.py")
SPEC = importlib.util.spec_from_file_location(
    "rest_sim_validate_benchmark",
    VALIDATOR_PATH,
)
assert SPEC is not None
assert SPEC.loader is not None
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)


EXPECTED_BATCH_SIZES = (1, 32, 256, 1024)


def _result(runtimes: int) -> dict[str, Any]:
    simulator_us_per_get = 10.0
    simulator_ms_per_batch = simulator_us_per_get * runtimes / 1_000.0
    end_to_end_ms_per_batch = 20.0
    end_to_end_us_per_get = end_to_end_ms_per_batch * 1_000.0 / runtimes
    return {
        "runtimes": runtimes,
        "iterations_per_round": 50,
        "warmup_iterations": 5,
        "rounds": 5,
        "median_seconds": 50 * end_to_end_ms_per_batch / 1_000.0,
        "simulator_gets_per_second": 1_000_000.0 / simulator_us_per_get,
        "simulator_microseconds_per_get": simulator_us_per_get,
        "simulator_ms_per_batch": simulator_ms_per_batch,
        "end_to_end_gets_per_second": 1_000_000.0 / end_to_end_us_per_get,
        "end_to_end_microseconds_per_get": end_to_end_us_per_get,
        "end_to_end_batches_per_second": 1_000.0 / end_to_end_ms_per_batch,
        "end_to_end_ms_per_batch": end_to_end_ms_per_batch,
        "configured_external_ms_per_batch": 14.0,
        "simulator_fraction_of_end_to_end": (
            simulator_ms_per_batch / end_to_end_ms_per_batch
        ),
        "runtime_array_bytes_per_runtime": 55.0,
        "synthetic_api_response_ms_per_batch": 2.0,
        "synthetic_encoder_ms_per_batch": 12.0,
        "batch_aligned": True,
        "deterministic_replay": True,
    }


def _valid_document() -> dict[str, Any]:
    return {
        "schema_version": 2,
        "benchmark": "rest_sim_get_only",
        "capture_id": "get-benchmark-v1",
        "capture_sha256": (
            "0b62487bfa1c464363075e467cfb1eefe"
            "786b859a0fbbdf1aea8eb763ef0b238"
        ),
        "capture_nodes": 2,
        "capture_edges": 1,
        "latency_source": "synthetic_configured",
        "latency_profile": {
            "id": "gym-ci-v1",
            "api_response_ms_per_batch": 2.0,
            "encoder_pass_ms_per_batch": 12.0,
        },
        "measurement_contract": {
            "performance_decision_source": "simulator_metrics_only",
            "end_to_end_metrics": "context_only",
        },
        "results": [_result(runtimes) for runtimes in EXPECTED_BATCH_SIZES],
    }


def _set_end_to_end_ms(result: dict[str, Any], ms_per_batch: float) -> None:
    runtimes = result["runtimes"]
    us_per_get = ms_per_batch * 1_000.0 / runtimes
    result["end_to_end_gets_per_second"] = 1_000_000.0 / us_per_get
    result["end_to_end_microseconds_per_get"] = us_per_get
    result["end_to_end_batches_per_second"] = 1_000.0 / ms_per_batch
    result["end_to_end_ms_per_batch"] = ms_per_batch
    result["median_seconds"] = (
        result["iterations_per_round"] * ms_per_batch / 1_000.0
    )
    result["simulator_fraction_of_end_to_end"] = (
        result["simulator_ms_per_batch"] / ms_per_batch
    )


def _set_simulator_us(result: dict[str, Any], us_per_get: float) -> None:
    runtimes = result["runtimes"]
    result["simulator_gets_per_second"] = 1_000_000.0 / us_per_get
    result["simulator_microseconds_per_get"] = us_per_get
    result["simulator_ms_per_batch"] = us_per_get * runtimes / 1_000.0
    result["simulator_fraction_of_end_to_end"] = (
        result["simulator_ms_per_batch"] / result["end_to_end_ms_per_batch"]
    )


def _make_simulator_slower_than_end_to_end(
    result: dict[str, Any],
) -> None:
    _set_simulator_us(result, 500.0)
    _set_end_to_end_ms(result, 15.0)


def _violations(document: dict[str, Any]) -> list[str]:
    return validator.validate_benchmark(document)


def test_valid_fixed_profile_benchmark_document_passes() -> None:
    assert _violations(_valid_document()) == []


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (
            lambda document: document["latency_profile"].__setitem__(
                "api_response_ms_per_batch",
                2.1,
            ),
            "API latency must remain pinned at 2.0 ms per batch",
        ),
        (
            lambda document: document["latency_profile"].__setitem__(
                "encoder_pass_ms_per_batch",
                12.1,
            ),
            "encoder latency must remain pinned at 12.0 ms per batch",
        ),
        (
            lambda document: document["latency_profile"].__setitem__(
                "id",
                "custom-profile",
            ),
            "latency_profile.id must be 'gym-ci-v1'",
        ),
        (
            lambda document: document["results"][0].__setitem__(
                "synthetic_api_response_ms_per_batch",
                2.1,
            ),
            "batch 1: API latency pin changed",
        ),
        (
            lambda document: document["results"][0].__setitem__(
                "synthetic_encoder_ms_per_batch",
                12.1,
            ),
            "batch 1: encoder latency pin changed",
        ),
        (
            lambda document: document["results"][0].__setitem__(
                "configured_external_ms_per_batch",
                14.1,
            ),
            "batch 1: combined external latency is invalid",
        ),
    ],
)
def test_altered_or_inflated_latency_profile_fails(
    mutate: Callable[[dict[str, Any]], None],
    expected: str,
) -> None:
    document = _valid_document()

    mutate(document)

    assert expected in _violations(document)


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (
            lambda document: document.__setitem__(
                "capture_id",
                "larger-capture",
            ),
            "capture_id must be 'get-benchmark-v1'",
        ),
        (
            lambda document: document.__setitem__(
                "capture_sha256",
                "f" * 64,
            ),
            "capture_sha256 must be "
            "'0b62487bfa1c464363075e467cfb1eefe786b859a0fbbdf1aea8eb763ef0b238'",
        ),
        (
            lambda document: document.__setitem__("capture_nodes", 3),
            "capture_nodes must be 2",
        ),
        (
            lambda document: document.__setitem__("capture_edges", 2),
            "capture_edges must be 1",
        ),
        (
            lambda document: document["results"][0].__setitem__(
                "iterations_per_round",
                49,
            ),
            "batch 1: iterations_per_round must be 50",
        ),
        (
            lambda document: document["results"][0].__setitem__(
                "warmup_iterations",
                4,
            ),
            "batch 1: warmup_iterations must be 5",
        ),
        (
            lambda document: document["results"][0].__setitem__(
                "rounds",
                4,
            ),
            "batch 1: rounds must be 5",
        ),
    ],
)
def test_pinned_capture_and_workload_changes_fail(
    mutate: Callable[[dict[str, Any]], None],
    expected: str,
) -> None:
    document = _valid_document()

    mutate(document)

    assert expected in _violations(document)


@pytest.mark.parametrize(
    "runtimes",
    [
        [1, 32, 1024],
        [1, 32, 256],
        [1, 32, 32, 1024],
        [1, 256, 32, 1024],
        [32, 256, 1024, 1],
    ],
)
def test_missing_or_unaligned_required_batch_set_fails(
    runtimes: list[int],
) -> None:
    document = _valid_document()
    document["results"] = [_result(runtime) for runtime in runtimes]

    assert (
        "results runtimes must be exactly [1, 32, 256, 1024]"
        in _violations(document)
    )


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        ("batch_aligned", "batch 256: batch_aligned must be true"),
        (
            "deterministic_replay",
            "batch 256: deterministic_replay must be true",
        ),
    ],
)
def test_nondeterministic_or_misaligned_result_fails(
    field: str,
    expected: str,
) -> None:
    document = _valid_document()
    document["results"][2][field] = False

    assert expected in _violations(document)


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (
            lambda result: result.__setitem__(
                "simulator_ms_per_batch",
                result["simulator_ms_per_batch"] + 0.001,
            ),
            "batch 32: simulator timing fields disagree",
        ),
        (
            lambda result: result.__setitem__(
                "simulator_gets_per_second",
                result["simulator_gets_per_second"] + 1.0,
            ),
            "batch 32: simulator rate fields disagree",
        ),
        (
            lambda result: result.__setitem__(
                "simulator_fraction_of_end_to_end",
                result["simulator_fraction_of_end_to_end"] + 0.001,
            ),
            "batch 32: simulator fraction disagrees",
        ),
    ],
)
def test_inconsistent_simulator_metrics_fail(
    mutate: Callable[[dict[str, Any]], None],
    expected: str,
) -> None:
    document = _valid_document()

    mutate(document["results"][1])

    assert expected in _violations(document)


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        (
            "median_seconds",
            1.1,
            "batch 32: median timing field disagrees",
        ),
        (
            "runtime_array_bytes_per_runtime",
            56.0,
            "batch 32: runtime_array_bytes_per_runtime must be 55.0",
        ),
    ],
)
def test_pinned_memory_and_median_timing_changes_fail(
    field: str,
    value: float,
    expected: str,
) -> None:
    document = _valid_document()

    document["results"][1][field] = value

    assert expected in _violations(document)


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (
            lambda result: result.__setitem__(
                "end_to_end_ms_per_batch",
                result["end_to_end_ms_per_batch"] + 0.001,
            ),
            "batch 32: end-to-end timing fields disagree",
        ),
        (
            lambda result: result.__setitem__(
                "end_to_end_gets_per_second",
                result["end_to_end_gets_per_second"] + 1.0,
            ),
            "batch 32: end-to-end rate fields disagree",
        ),
        (
            lambda result: result.__setitem__(
                "end_to_end_batches_per_second",
                result["end_to_end_batches_per_second"] + 1.0,
            ),
            "batch 32: batch rate fields disagree",
        ),
    ],
)
def test_inconsistent_end_to_end_formula_metrics_fail(
    mutate: Callable[[dict[str, Any]], None],
    expected: str,
) -> None:
    document = _valid_document()

    mutate(document["results"][1])

    assert expected in _violations(document)


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (
            lambda result: _set_end_to_end_ms(result, 13.0),
            "batch 1: configured external latency was not observed",
        ),
        (
            _make_simulator_slower_than_end_to_end,
            "batch 32: end-to-end time is below simulator time",
        ),
    ],
)
def test_end_to_end_floor_invariants_fail(
    mutate: Callable[[dict[str, Any]], None],
    expected: str,
) -> None:
    document = _valid_document()
    target = document["results"][0 if "batch 1:" in expected else 1]

    mutate(target)

    assert expected in _violations(document)


def test_consistent_end_to_end_metric_changes_do_not_affect_valid_result() -> None:
    document = _valid_document()
    for result in document["results"]:
        _set_end_to_end_ms(result, 28.0)

    assert _violations(document) == []


def test_duplicate_json_keys_are_rejected_through_main(tmp_path: Path) -> None:
    input_path = tmp_path / "benchmark.json"
    output_path = tmp_path / "gate-report.json"
    input_path.write_text(
        '{"schema_version": 2, "schema_version": 3, "results": []}',
        encoding="utf-8",
    )

    exit_code = validator.main(
        ["--input", str(input_path), "--output", str(output_path)],
    )

    assert exit_code == 2
    assert not output_path.exists()


def test_gate_report_exposes_simulator_metrics_only() -> None:
    document = _valid_document()
    document["private_context"] = "must not appear in the report"
    for result in document["results"]:
        result["end_to_end_gets_per_second"] = 3.14
        result["external_debug"] = "must not appear in the report"

    report = validator.build_gate_report(
        document=document,
        input_sha256="0" * 64,
        violations=[],
    )

    assert report["status"] == "PASS"
    assert report["performance_decision_source"] == "simulator_metrics_only"
    assert set(report) == {
        "schema_version",
        "gate",
        "status",
        "benchmark_sha256",
        "capture_sha256",
        "latency_profile",
        "performance_decision_source",
        "simulator_metrics",
        "violations",
    }
    assert set(report["simulator_metrics"][0]) == {
        "runtimes",
        "simulator_gets_per_second",
        "simulator_microseconds_per_get",
        "runtime_array_bytes_per_runtime",
    }
    encoded_report = json.dumps(report, sort_keys=True)
    assert "end_to_end" not in encoded_report
    assert "private_context" not in encoded_report
    assert "external_debug" not in encoded_report


def test_gate_report_reports_fail_status_with_violations() -> None:
    document = _valid_document()
    broken = copy.deepcopy(document)
    broken["results"][0]["deterministic_replay"] = False
    violations = _violations(broken)

    report = validator.build_gate_report(
        document=broken,
        input_sha256="1" * 64,
        violations=violations,
    )

    assert report["status"] == "FAIL"
    assert "batch 1: deterministic_replay must be true" in report["violations"]
