"""Spec-driven Phase 1 golden acceptance checks.

The acceptance layer consumes the compact payload from
``igc.modules.train.phase1_golden`` and evaluates only configured thresholds.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from igc.modules.base.metric_keys import PHASE1_FINETUNE, phase_metric


class Phase1AcceptanceError(ValueError):
    """Raised when an acceptance spec is malformed."""


@dataclass(frozen=True)
class AcceptanceSpec:
    """Thresholds for one Phase 1 held-out acceptance run."""

    thresholds: Mapping[str, Any]
    ece_bins: int = 10

    @classmethod
    def from_mapping(cls, spec: Mapping[str, Any] | None) -> "AcceptanceSpec":
        """Build an acceptance spec from YAML/JSON mapping data."""
        root = spec or {}
        if not isinstance(root, Mapping):
            raise Phase1AcceptanceError("acceptance spec must be a mapping")
        metrics = root.get("metrics") or {}
        if not isinstance(metrics, Mapping):
            raise Phase1AcceptanceError("metrics spec must be a mapping")
        thresholds = root.get("thresholds") or {}
        if not isinstance(thresholds, Mapping):
            raise Phase1AcceptanceError("thresholds spec must be a mapping")
        ece_bins = int(metrics.get("ece_bins", 10))
        if ece_bins <= 0:
            raise Phase1AcceptanceError("metrics.ece_bins must be positive")
        return cls(thresholds=thresholds, ece_bins=ece_bins)


def evaluate_acceptance(
    payload: Mapping[str, Any],
    spec: AcceptanceSpec,
) -> dict[str, Any]:
    """Evaluate configured checks against a Phase 1 metrics payload."""
    checks: list[dict[str, Any]] = []
    metrics = _mapping(payload.get("metrics"), "metrics")
    evidence = _mapping(payload.get("evidence"), "evidence")
    comparison = _mapping(payload.get("comparison"), "comparison")
    model_metrics = _mapping(metrics.get("model_x"), "metrics.model_x")
    delta = _mapping(comparison.get("delta"), "comparison.delta")
    model_evidence = _mapping(evidence.get("model_x"), "evidence.model_x")
    model_counts = _mapping(model_evidence.get("counts"), "evidence.model_x.counts")
    thresholds = spec.thresholds

    _add_min_check(checks, "min_rows", model_counts.get("rows"), thresholds.get("min_rows"))
    _add_max_check(
        checks,
        "max_missing_target_rows",
        model_metrics.get("missing_target_rows"),
        thresholds.get("max_missing_target_rows"),
    )
    _add_max_check(
        checks,
        "max_missing_prediction_rows",
        model_metrics.get("missing_prediction_rows"),
        thresholds.get("max_missing_prediction_rows"),
    )
    _add_min_check(
        checks,
        "min_model_json_parse_rate",
        model_metrics.get(phase_metric(PHASE1_FINETUNE, "eval", "json_parse_rate")),
        thresholds.get("min_model_json_parse_rate"),
    )
    _add_min_check(
        checks,
        "min_model_json_exact_match_rate",
        model_metrics.get(phase_metric(PHASE1_FINETUNE, "eval", "json_exact_match_rate")),
        thresholds.get("min_model_json_exact_match_rate"),
    )
    _add_min_check(
        checks,
        "min_model_odata_id_match_rate",
        model_metrics.get(phase_metric(PHASE1_FINETUNE, "eval", "odata_id_match_rate")),
        thresholds.get("min_model_odata_id_match_rate"),
    )
    _add_min_check(
        checks,
        "min_model_eval_tokens_per_sec",
        model_metrics.get(phase_metric(PHASE1_FINETUNE, "throughput", "eval_tokens_per_sec")),
        thresholds.get("min_model_eval_tokens_per_sec"),
    )
    _add_min_check(
        checks,
        "min_model_eval_samples_per_sec",
        model_metrics.get(phase_metric(PHASE1_FINETUNE, "throughput", "eval_samples_per_sec")),
        thresholds.get("min_model_eval_samples_per_sec"),
    )
    _add_max_check(
        checks,
        "max_model_ece",
        model_metrics.get(phase_metric(PHASE1_FINETUNE, "calibration", "ece")),
        thresholds.get("max_model_ece"),
    )
    _add_max_check(
        checks,
        "max_model_latency_sec_p95",
        model_metrics.get(phase_metric(PHASE1_FINETUNE, "test", "latency_sec_p95")),
        thresholds.get("max_model_latency_sec_p95"),
    )
    _add_min_check(
        checks,
        "min_exact_match_delta_vs_baseline",
        delta.get(phase_metric(PHASE1_FINETUNE, "eval", "json_exact_match_rate")),
        thresholds.get("min_exact_match_delta_vs_baseline"),
    )
    _add_min_check(
        checks,
        "min_parse_rate_delta_vs_baseline",
        delta.get(phase_metric(PHASE1_FINETUNE, "eval", "json_parse_rate")),
        thresholds.get("min_parse_rate_delta_vs_baseline"),
    )
    _add_min_check(
        checks,
        "min_odata_id_delta_vs_baseline",
        delta.get(phase_metric(PHASE1_FINETUNE, "eval", "odata_id_match_rate")),
        thresholds.get("min_odata_id_delta_vs_baseline"),
    )

    failures = [check for check in checks if not check["passed"]]
    return {
        "schema_version": "phase1_golden_acceptance.v1",
        "status": "pass" if not failures else "fail",
        "checks": checks,
        "failures": failures,
    }


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise Phase1AcceptanceError(f"{name} must be a mapping")
    return value


def _add_min_check(
    checks: list[dict[str, Any]],
    name: str,
    observed: Any,
    threshold: Any,
) -> None:
    if threshold is None:
        return
    checks.append(
        _check(name, observed, threshold, ">=", _number(observed) >= _number(threshold))
    )


def _add_max_check(
    checks: list[dict[str, Any]],
    name: str,
    observed: Any,
    threshold: Any,
) -> None:
    if threshold is None:
        return
    checks.append(
        _check(name, observed, threshold, "<=", _number(observed) <= _number(threshold))
    )


def _check(
    name: str,
    observed: Any,
    threshold: Any,
    operator: str,
    passed: bool,
) -> dict[str, Any]:
    return {
        "name": name,
        "operator": operator,
        "observed": observed,
        "threshold": threshold,
        "passed": bool(passed),
    }


def _number(value: Any) -> float:
    if value is None or isinstance(value, bool):
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


# Author: Mus mbayramo@stanford.edu
