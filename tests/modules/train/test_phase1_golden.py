"""Offline tests for Phase 1 held-out prediction metrics.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from igc.modules.base.metric_keys import PHASE1_FINETUNE, phase_metric
from igc.modules.train.phase1_golden import (
    Phase1GoldenError,
    build_phase1_golden_payload,
    evaluate_prediction_jsonl,
    expected_calibration_error,
    parse_json_value,
)


def _target(rest_api: str, name: str) -> dict:
    return {"@odata.id": rest_api, "Name": name}


def _row(rest_api: str, prediction, **overrides) -> dict:
    target = _target(rest_api, overrides.pop("target_name", rest_api.rsplit("/", 1)[-1]))
    row = {
        "id": rest_api,
        "phase": 1,
        "task": "redfish_json_reconstruction",
        "x": {"rest_api": rest_api, "allowed_methods": ["GET"], "json": target},
        "y_true": {"json": target},
        "y_pred": {"json": prediction},
        "generated_tokens": overrides.pop("generated_tokens", 10),
        "sequence_length": overrides.pop("sequence_length", 12),
        "padding_tokens": overrides.pop("padding_tokens", 1),
        "latency_sec": overrides.pop("latency_sec", 1.0),
        "memory_peak_mb": overrides.pop("memory_peak_mb", 128.0),
        "confidence": overrides.pop("confidence", 0.9),
        "log_prob": overrides.pop("log_prob", -5.0),
    }
    row.update(overrides)
    return row


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")
    return path


def test_build_phase1_golden_payload_compares_baseline_and_model(tmp_path: Path) -> None:
    """model_x metrics are emitted with deltas against the baseline artifact."""
    rest_a = "/redfish/v1/Systems/1"
    rest_b = "/redfish/v1/Managers/1"
    baseline = _write_jsonl(
        tmp_path / "baseline.jsonl",
        [
            _row(rest_a, _target(rest_a, "1"), generated_tokens=10, latency_sec=1.0),
            _row(
                rest_b,
                {"@odata.id": rest_b, "Name": "wrong"},
                generated_tokens=20,
                latency_sec=2.0,
                sequence_length=20,
                padding_tokens=2,
                confidence=0.7,
            ),
        ],
    )
    model = _write_jsonl(
        tmp_path / "model_x.jsonl",
        [
            _row(
                rest_a,
                json.dumps(_target(rest_a, "1")),
                generated_tokens=10,
                latency_sec=1.0,
                sequence_length=10,
                padding_tokens=1,
            ),
            _row(
                rest_b,
                _target(rest_b, "1"),
                target_name="1",
                generated_tokens=20,
                latency_sec=2.0,
                sequence_length=20,
                padding_tokens=2,
                confidence=0.8,
            ),
        ],
    )

    payload = build_phase1_golden_payload(baseline_jsonl=baseline, model_jsonl=model)
    model_metrics = payload["metrics"]["model_x"]
    baseline_metrics = payload["metrics"]["baseline"]
    exact_key = phase_metric(PHASE1_FINETUNE, "eval", "json_exact_match_rate")
    throughput_key = phase_metric(PHASE1_FINETUNE, "throughput", "eval_tokens_per_sec")

    assert payload["metric_namespace"] == PHASE1_FINETUNE
    assert baseline_metrics[exact_key] == 0.5
    assert model_metrics[exact_key] == 1.0
    assert payload["comparison"]["delta"][exact_key] == 0.5
    assert model_metrics[throughput_key] == pytest.approx(10.0)
    assert model_metrics[phase_metric(PHASE1_FINETUNE, "data", "padding_ratio")] == 0.1
    assert model_metrics[phase_metric(PHASE1_FINETUNE, "test", "latency_sec_p95")] == 1.95
    assert payload["evidence"]["model_x"]["artifact"]["name"] == "model_x.jsonl"
    assert "sha256" in payload["evidence"]["model_x"]["artifact"]


def test_prediction_parser_accepts_fenced_json_fragment() -> None:
    """Model outputs wrapped in text still parse when they contain one JSON object."""
    parsed, ok, error = parse_json_value('prefix ```json\n{"a": 1}\n``` suffix')
    assert ok, error
    assert parsed == {"a": 1}


def test_prediction_parser_skips_braces_inside_quoted_text() -> None:
    """JSON fragment extraction starts outside surrounding quoted prose."""
    parsed, ok, error = parse_json_value('"not {json}" then {"ok": true}')
    assert ok, error
    assert parsed == {"ok": True}


def test_top_k_accuracy_is_reported_when_candidates_exist(tmp_path: Path) -> None:
    """Top-k accuracy is optional and only uses rows with candidate lists."""
    rest_api = "/redfish/v1/Systems/1"
    target = _target(rest_api, "1")
    path = _write_jsonl(
        tmp_path / "topk.jsonl",
        [
            _row(
                rest_api,
                {"not": "correct"},
                top_k_predictions=[{"json": {"not": "correct"}}, {"json": target}],
            )
        ],
    )

    metrics = evaluate_prediction_jsonl(path, role="model_x").to_metric_dict()
    assert metrics[phase_metric(PHASE1_FINETUNE, "eval", "top_k_accuracy")] == 1.0


def test_bad_jsonl_row_raises_clear_error(tmp_path: Path) -> None:
    """Malformed JSONL rows fail closed."""
    path = tmp_path / "bad.jsonl"
    path.write_text("{not json}\n")
    with pytest.raises(Phase1GoldenError, match="invalid JSONL row"):
        evaluate_prediction_jsonl(path, role="model_x")


def test_missing_prediction_counts_as_missing_prediction_row(tmp_path: Path) -> None:
    """Rows with a target but no prediction are counted for fail-closed gates."""
    rest_api = "/redfish/v1/Systems/1"
    target = _target(rest_api, "1")
    path = _write_jsonl(
        tmp_path / "missing-prediction.jsonl",
        [
            {
                "id": "missing",
                "x": {"rest_api": rest_api, "json": target},
                "y_true": {"json": target},
            }
        ],
    )

    metrics = evaluate_prediction_jsonl(path, role="model_x").to_metric_dict()
    assert metrics["missing_prediction_rows"] == 1
    assert metrics[phase_metric(PHASE1_FINETUNE, "eval", "json_parse_rate")] == 0.0


def test_missing_target_is_not_filled_from_input_json(tmp_path: Path) -> None:
    """Input JSON is context, not an implicit reconstruction target."""
    rest_api = "/redfish/v1/Systems/1"
    target = _target(rest_api, "1")
    path = _write_jsonl(
        tmp_path / "missing-target.jsonl",
        [
            {
                "id": "missing-target",
                "x": {"rest_api": rest_api, "json": target},
                "y_pred": {"json": target},
            }
        ],
    )

    metrics = evaluate_prediction_jsonl(path, role="model_x").to_metric_dict()
    assert metrics["missing_target_rows"] == 1
    assert metrics[phase_metric(PHASE1_FINETUNE, "eval", "json_exact_match_rate")] is None


def test_expected_calibration_error_uses_equal_width_bins() -> None:
    """ECE uses absolute confidence/accuracy gaps weighted by bin population."""
    assert expected_calibration_error([(0.9, True), (0.2, False)], bins=10) == pytest.approx(
        0.15
    )


# Author: Mus mbayramo@stanford.edu
