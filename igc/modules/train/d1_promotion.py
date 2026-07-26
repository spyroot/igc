"""Promotion checks for a real released D1 dataset."""

from __future__ import annotations

import math
from typing import Any, Mapping

from igc.modules.base.metric_keys import PHASE2_LABELLED_REQUESTS, phase_metric
from igc.modules.train.promotion_evidence import (
    is_sha256,
    validate_promotion_evidence,
)


class D1PromotionError(ValueError):
    """Raised when D1 promotion evidence is incomplete or malformed."""


def evaluate_d1_promotion(
    *,
    release_manifest: Mapping[str, Any],
    observed_artifact_sha: str,
    build_metrics: Mapping[str, Any],
    calibration_metrics: Mapping[str, Any],
    build_thresholds: Mapping[str, Any],
    calibration_thresholds: Mapping[str, Any],
    artifact_evidence: Mapping[str, Any],
    observed_full_manifest_sha: str,
    observed_heldout_manifest_sha: str,
    observed_heldout_sha: str,
    observed_heldout_rows: int,
) -> dict[str, Any]:
    """Require an immutable real release, calibrated judge, and parent smoke."""
    build_metric_map = {
        "min_pro_accept_rate": phase_metric(PHASE2_LABELLED_REQUESTS, "pro_accept_rate"),
        "min_rest_api_set_match_rate": phase_metric(
            PHASE2_LABELLED_REQUESTS,
            "rest_api_set_match_rate",
        ),
        "max_nonsense_rate": phase_metric(PHASE2_LABELLED_REQUESTS, "nonsense_rate"),
        "max_invalid_json_rate": phase_metric(PHASE2_LABELLED_REQUESTS, "invalid_json_rate"),
    }
    required_build = set(build_metric_map)
    required_release_thresholds = {"min_empty_set_accepted_rows"}
    required_calibration = {"min_precision", "min_recall", "max_false_accept_rate"}
    missing = sorted(
        {name for name in required_build if build_thresholds.get(name) is None}
        | {
            name
            for name in required_release_thresholds
            if build_thresholds.get(name) is None
        }
        | {name for name in required_calibration if calibration_thresholds.get(name) is None}
    )
    if missing:
        raise D1PromotionError(f"missing non-null quality thresholds: {missing}")

    checks: list[dict[str, Any]] = []
    _required(checks, "release_manifest_is_d1", release_manifest.get("dataset") == "D1")
    _required(checks, "release_is_immutable", release_manifest.get("immutable") is True)
    _required(checks, "release_is_complete", release_manifest.get("complete") is True)
    _required(checks, "release_has_rows", int(release_manifest.get("rows", 0) or 0) > 0)
    _required(checks, "release_balance_valid", release_manifest.get("balance_valid"))
    width_counts = release_manifest.get("sample_width_counts")
    _required(
        checks,
        "release_contains_balanced_k1_k2_k3",
        isinstance(width_counts, Mapping)
        and {"1", "2", "3"} <= set(width_counts)
        and set(width_counts) <= {"0", "1", "2", "3"}
        and all(
            isinstance(count, int) and not isinstance(count, bool) and count > 0
            for count in width_counts.values()
        )
        and max(width_counts[key] for key in ("1", "2", "3"))
        - min(width_counts[key] for key in ("1", "2", "3"))
        <= 1,
    )
    _comparison(
        checks,
        "release_contains_bounded_empty_set_negatives",
        width_counts.get("0") if isinstance(width_counts, Mapping) else None,
        build_thresholds.get("min_empty_set_accepted_rows"),
        ">=",
    )
    _required(
        checks,
        "release_judge_evidence_valid",
        release_manifest.get("judge_evidence_valid"),
    )
    _required(
        checks,
        "release_used_real_model_x_provider",
        release_manifest.get("draft_provider_adapter") == "openai-compatible",
    )
    _required(
        checks,
        "release_used_real_judge_provider",
        release_manifest.get("judge_provider_adapter") == "openai-compatible",
    )
    _required(
        checks,
        "release_has_resolved_judge_identity",
        _resolved_identifier(release_manifest.get("judge_route"))
        and _resolved_identifier(release_manifest.get("judge_model"))
        and _resolved_identifier(release_manifest.get("judge_profile")),
    )
    manifest_sha = release_manifest.get("artifact_sha256")
    _required(
        checks,
        "artifact_sha_matches_release",
        _is_digest(manifest_sha) and manifest_sha == observed_artifact_sha,
    )
    checks.extend(validate_promotion_evidence(
        artifact_evidence,
        expected_parent_role="model_x",
        reload_target="parent",
        observed_source_full_manifest_sha=observed_full_manifest_sha,
        observed_source_full_sha=observed_artifact_sha,
        observed_heldout_manifest_sha=observed_heldout_manifest_sha,
        observed_heldout_sha=observed_heldout_sha,
        observed_heldout_rows=observed_heldout_rows,
    ))
    parent_checkpoint = artifact_evidence.get("real_promoted_parent_checkpoint")
    parent_artifact_sha = (
        parent_checkpoint.get("artifact_sha")
        if isinstance(parent_checkpoint, Mapping)
        else None
    )
    _required(
        checks,
        "release_model_x_matches_promoted_parent",
        _is_digest(release_manifest.get("model_x_artifact_sha"))
        and release_manifest.get("model_x_artifact_sha") == parent_artifact_sha,
    )
    _required(
        checks,
        "released_artifact_matches_promotion_evidence",
        artifact_evidence.get("artifact_sha") == observed_artifact_sha,
    )

    by_width = build_metrics.get("by_sample_width")
    if not isinstance(by_width, Mapping) or set(by_width) != {"0", "1", "2", "3"}:
        raise D1PromotionError(
            "canonical D1 build metrics require by_sample_width for k=0,1,2,3"
        )
    for width in ("0", "1", "2", "3"):
        width_metrics = by_width[width]
        if not isinstance(width_metrics, Mapping):
            raise D1PromotionError(f"D1 width {width} metrics must be an object")
        for threshold_name, metric_name in build_metric_map.items():
            operator = "<=" if threshold_name.startswith("max_") else ">="
            _comparison(
                checks,
                f"{threshold_name}_k{width}",
                width_metrics.get(metric_name),
                build_thresholds[threshold_name],
                operator,
            )
    calibration_map = {
        "min_precision": "precision",
        "min_recall": "recall",
        "max_false_accept_rate": "false_accept_rate",
    }
    _required(
        checks,
        "calibration_has_positive_examples",
        _positive_int(calibration_metrics.get("positive_examples")),
    )
    _required(
        checks,
        "calibration_has_negative_examples",
        _positive_int(calibration_metrics.get("negative_examples")),
    )
    for threshold_name, metric_name in calibration_map.items():
        operator = "<=" if threshold_name.startswith("max_") else ">="
        _comparison(
            checks,
            threshold_name,
            calibration_metrics.get(metric_name),
            calibration_thresholds[threshold_name],
            operator,
        )
    failures = [check for check in checks if not check["passed"]]
    return {
        "schema_version": "d1_promotion.v1",
        "status": "pass" if not failures else "fail",
        "checks": checks,
        "failures": failures,
    }


def _is_digest(value: Any) -> bool:
    return is_sha256(value)


def _resolved_identifier(value: Any) -> bool:
    return (
        isinstance(value, str)
        and bool(value.strip())
        and not value.strip().startswith("${")
    )


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _required(checks: list[dict[str, Any]], name: str, observed: Any) -> None:
    checks.append({
        "name": name,
        "operator": "required",
        "observed": observed,
        "passed": bool(observed),
    })


def _comparison(
    checks: list[dict[str, Any]],
    name: str,
    observed: Any,
    threshold: Any,
    operator: str,
) -> None:
    finite = all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        for value in (observed, threshold)
    )
    passed = finite and (
        float(observed) >= float(threshold)
        if operator == ">="
        else float(observed) <= float(threshold)
    )
    checks.append({
        "name": name,
        "operator": operator,
        "observed": observed,
        "threshold": threshold,
        "passed": passed,
    })
