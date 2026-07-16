"""Offline tests for Phase 1 golden acceptance checks.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

from igc.modules.base.metric_keys import PHASE1_FINETUNE, phase_metric
from igc.modules.train.phase1_acceptance import AcceptanceSpec, evaluate_acceptance


def _payload() -> dict:
    exact_key = phase_metric(PHASE1_FINETUNE, "eval", "json_exact_match_rate")
    parse_key = phase_metric(PHASE1_FINETUNE, "eval", "json_parse_rate")
    odata_key = phase_metric(PHASE1_FINETUNE, "eval", "odata_id_match_rate")
    ece_key = phase_metric(PHASE1_FINETUNE, "calibration", "ece")
    latency_key = phase_metric(PHASE1_FINETUNE, "test", "latency_sec_p95")
    samples_key = phase_metric(PHASE1_FINETUNE, "throughput", "eval_samples_per_sec")
    tokens_key = phase_metric(PHASE1_FINETUNE, "throughput", "eval_tokens_per_sec")
    return {
        "metrics": {
            "model_x": {
                "missing_target_rows": 0,
                "missing_prediction_rows": 0,
                exact_key: 0.9,
                parse_key: 1.0,
                odata_key: 1.0,
                ece_key: 0.05,
                latency_key: 1.2,
                samples_key: 4.0,
                tokens_key: 100.0,
            }
        },
        "comparison": {
            "delta": {
                exact_key: 0.1,
                parse_key: 0.0,
                odata_key: 0.0,
            }
        },
        "evidence": {"model_x": {"counts": {"rows": 10}}},
    }


def test_acceptance_passes_configured_thresholds() -> None:
    """Configured thresholds pass when all observed values satisfy them."""
    spec = AcceptanceSpec.from_mapping(
        {
            "metrics": {"ece_bins": 10},
            "thresholds": {
                "min_rows": 10,
                "max_missing_target_rows": 0,
                "max_missing_prediction_rows": 0,
                "min_model_json_parse_rate": 1.0,
                "min_model_json_exact_match_rate": 0.8,
                "min_model_odata_id_match_rate": 1.0,
                "max_model_ece": 0.1,
                "max_model_latency_sec_p95": 2.0,
                "min_model_eval_samples_per_sec": 1.0,
                "min_model_eval_tokens_per_sec": 50.0,
                "min_exact_match_delta_vs_baseline": 0.0,
            },
        }
    )

    result = evaluate_acceptance(_payload(), spec)
    assert result["status"] == "pass"
    assert result["checks"]
    assert not result["failures"]


def test_acceptance_fails_when_delta_threshold_is_not_met() -> None:
    """A configured model-vs-baseline improvement threshold is enforced."""
    spec = AcceptanceSpec.from_mapping(
        {"thresholds": {"min_exact_match_delta_vs_baseline": 0.2}}
    )
    result = evaluate_acceptance(_payload(), spec)
    assert result["status"] == "fail"
    assert result["failures"][0]["name"] == "min_exact_match_delta_vs_baseline"


def test_acceptance_fails_when_required_metric_is_absent() -> None:
    """A configured threshold over a missing metric fails closed."""
    payload = _payload()
    exact_key = phase_metric(PHASE1_FINETUNE, "eval", "json_exact_match_rate")
    del payload["metrics"]["model_x"][exact_key]
    spec = AcceptanceSpec.from_mapping(
        {"thresholds": {"min_model_json_exact_match_rate": 0.1}}
    )

    result = evaluate_acceptance(payload, spec)
    assert result["status"] == "fail"
    assert result["failures"][0]["observed"] is None


def test_none_thresholds_are_report_only() -> None:
    """Unset thresholds do not create checks."""
    spec = AcceptanceSpec.from_mapping(
        {
            "thresholds": {
                "min_rows": None,
                "max_model_ece": None,
            }
        }
    )
    result = evaluate_acceptance(_payload(), spec)
    assert result["status"] == "pass"
    assert result["checks"] == []


# Author: Mus mbayramo@stanford.edu
