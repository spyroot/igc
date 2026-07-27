"""Set-based evaluation and hard promotion checks for the Phase 2 checkpoint."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from typing import Any, Mapping, Sequence

from igc.ds.rest_goal_contract import CONTEXT_FIELDS, HTTP_METHODS, evaluate_rest_api_set
from igc.modules.train.promotion_evidence import (
    validate_promotion_evidence,
    validate_training_run_evidence,
)


REQUIRED_ROBUSTNESS_VARIANTS = frozenset({
    "base",
    "api_context_shuffled",
    "json_key_order_shuffled",
    "target_serialization_reversed",
    "irrelevant_distractors_added",
})


class Phase2PromotionError(ValueError):
    """Raised when Phase 2 promotion evidence is incomplete or malformed."""


def evaluate_phase2_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Evaluate strict Phase 2 predictions as unordered API sets."""
    if not rows:
        raise Phase2PromotionError("Phase 2 promotion requires held-out rows")
    totals: dict[str, float] = defaultdict(float)
    by_width: dict[int, list[float]] = defaultdict(list)
    by_group: dict[str, list[float]] = defaultdict(list)
    group_counts: dict[str, int] = defaultdict(int)
    empty_exact: list[float] = []
    robustness: dict[str, dict[str, bool]] = defaultdict(dict)

    for index, row in enumerate(rows):
        if (
            row.get("phase") != 2
            or row.get("dataset") != "D1"
            or row.get("source_dataset") != "D0"
            or row.get("task") != "text_to_rest_api_list"
            or row.get("target_semantics") != "unordered_unique_rest_api_set"
        ):
            raise Phase2PromotionError(f"row {index}: Phase 2 D1 identity is invalid")
        x = _mapping(row, "x")
        if set(x) != {"text", "api_context"}:
            raise Phase2PromotionError(
                f"row {index}: x must contain exactly text and api_context"
            )
        expected = _target_list(_mapping(row, "y_true"), label=f"row {index} y_true")
        catalog = _catalog_apis(x.get("api_context"), row_index=index)
        if not set(expected) <= catalog:
            raise Phase2PromotionError(f"row {index}: target API absent from api_context")
        distractors = catalog - set(expected)
        if len(distractors) < 4:
            raise Phase2PromotionError(
                f"row {index}: api_context requires at least 4 distractors"
            )
        parsed, predicted, duplicate = _prediction(row.get("y_pred"))
        invalid = len(set(predicted) - catalog) if parsed else 0
        scored_prediction = list(dict.fromkeys(predicted))
        metrics = (
            evaluate_rest_api_set(expected, scored_prediction)
            if parsed
            else _zero_set_metrics(expected)
        )
        exact = float(metrics["set_match_rate"]) if not duplicate and not invalid else 0.0
        totals["json_parse"] += float(parsed)
        totals["set_exact"] += exact
        totals["precision"] += float(metrics["precision"])
        totals["recall"] += float(metrics["recall"])
        totals["f1"] += float(metrics["f1"])
        totals["cardinality"] += float(
            parsed and not duplicate and len(predicted) == len(expected)
        )
        totals["invalid_api"] += float(invalid > 0)
        totals["duplicate_api"] += float(duplicate)

        metadata = _mapping(row, "metadata")
        width = metadata.get("sample_width_k", len(expected))
        if width not in (0, 1, 2, 3):
            raise Phase2PromotionError(f"row {index}: sample_width_k must be 0, 1, 2, or 3")
        if int(width) != len(expected):
            raise Phase2PromotionError(
                f"row {index}: sample_width_k must equal target cardinality"
            )
        if width:
            by_width[int(width)].append(exact)
        raw_groups = metadata.get("heldout_vendor_or_model")
        if isinstance(raw_groups, str):
            groups = [raw_groups]
        elif isinstance(raw_groups, list):
            groups = raw_groups
        else:
            groups = []
        if (
            not groups
            or not all(isinstance(group, str) and group.strip() for group in groups)
            or len(groups) != len(set(groups))
        ):
            raise Phase2PromotionError(
                f"row {index}: metadata.heldout_vendor_or_model must be "
                "a unique non-empty string list"
            )
        for group in groups:
            by_group[group].append(exact)
            group_counts[group] += 1
        if not expected:
            empty_exact.append(exact)

        case_id = metadata.get("semantic_case_id")
        variant = metadata.get("robustness_variant")
        if case_id is not None or variant is not None:
            if not isinstance(case_id, str) or not case_id:
                raise Phase2PromotionError(f"row {index}: semantic_case_id is invalid")
            if variant not in REQUIRED_ROBUSTNESS_VARIANTS:
                raise Phase2PromotionError(f"row {index}: robustness_variant is invalid")
            if variant in robustness[case_id]:
                raise Phase2PromotionError(
                    f"row {index}: duplicate robustness variant {variant!r}"
                )
            robustness[case_id][variant] = bool(exact)

    count = len(rows)
    return {
        "rows": count,
        "json_parse_rate": totals["json_parse"] / count,
        "set_exact_match_rate": totals["set_exact"] / count,
        "precision": totals["precision"] / count,
        "recall": totals["recall"] / count,
        "f1": totals["f1"] / count,
        "cardinality_accuracy": totals["cardinality"] / count,
        "invalid_api_rate": totals["invalid_api"] / count,
        "duplicate_api_rate": totals["duplicate_api"] / count,
        "empty_set_exact_match_rate": _mean(empty_exact),
        "empty_set_rows": len(empty_exact),
        "set_exact_match_by_width": {
            str(width): _mean(values) for width, values in sorted(by_width.items())
        },
        "set_exact_match_by_vendor_or_model": {
            group: _mean(values) for group, values in sorted(by_group.items())
        },
        "rows_by_vendor_or_model": dict(sorted(group_counts.items())),
        "robustness": _robustness_result(robustness),
    }


def evaluate_phase2_promotion(
    *,
    rows: Sequence[Mapping[str, Any]],
    thresholds: Mapping[str, Any],
    artifact_evidence: Mapping[str, Any],
    run_report: Mapping[str, Any],
    observed_source_full_manifest_sha: str,
    observed_source_full_sha: str,
    observed_train_manifest_sha: str,
    observed_train_sha: str,
    observed_heldout_manifest_sha: str,
    observed_heldout_sha: str,
    observed_split_release_sha: str,
) -> dict[str, Any]:
    """Combine held-out metrics, robustness, lineage, and load evidence."""
    required_thresholds = {
        "min_json_parse_rate",
        "min_set_exact_match_rate",
        "min_set_exact_match_by_width",
        "max_duplicate_api_rate",
        "max_invalid_api_rate",
        "min_heldout_vendor_or_model_set_exact",
        "min_heldout_rows_per_vendor_or_model",
        "min_empty_set_exact_match_rate",
    }
    missing = sorted(required_thresholds - set(thresholds))
    if missing or any(thresholds.get(key) is None for key in required_thresholds):
        raise Phase2PromotionError(f"missing non-null thresholds: {missing}")

    metrics = evaluate_phase2_rows(rows)
    checks = validate_promotion_evidence(
        artifact_evidence,
        expected_parent_role="model_x",
        reload_target="artifact",
        observed_source_full_manifest_sha=observed_source_full_manifest_sha,
        observed_source_full_sha=observed_source_full_sha,
        observed_train_manifest_sha=observed_train_manifest_sha,
        observed_train_sha=observed_train_sha,
        observed_heldout_manifest_sha=observed_heldout_manifest_sha,
        observed_heldout_sha=observed_heldout_sha,
        observed_split_release_sha=observed_split_release_sha,
        observed_heldout_rows=len(rows),
    )
    checks.extend(validate_training_run_evidence(
        run_report,
        expected_phase="phase2_goal_extraction",
        expected_task="text_to_rest_api_list",
        expected_parent_role="model_x",
        expected_parent_artifact_sha=_parent_artifact_sha(artifact_evidence),
        expected_output_role="goal_extractor",
        observed_source_full_manifest_sha=observed_source_full_manifest_sha,
        observed_train_manifest_sha=observed_train_manifest_sha,
        observed_train_sha=observed_train_sha,
        observed_heldout_manifest_sha=observed_heldout_manifest_sha,
        observed_heldout_sha=observed_heldout_sha,
        reload_evidence=_mapping(artifact_evidence, "checkpoint_reload"),
    ))
    _required(checks, "finite_promotion_metrics", _all_finite_metrics(metrics))
    _minimum(checks, "min_json_parse_rate", metrics["json_parse_rate"], thresholds)
    _minimum(
        checks,
        "min_set_exact_match_rate",
        metrics["set_exact_match_rate"],
        thresholds,
    )
    _maximum(
        checks,
        "max_duplicate_api_rate",
        metrics["duplicate_api_rate"],
        thresholds,
    )
    _maximum(checks, "max_invalid_api_rate", metrics["invalid_api_rate"], thresholds)
    _minimum(
        checks,
        "min_empty_set_exact_match_rate",
        metrics["empty_set_exact_match_rate"],
        thresholds,
    )
    _required(checks, "empty_set_heldout_present", metrics["empty_set_rows"] > 0)

    width_thresholds = thresholds["min_set_exact_match_by_width"]
    if not isinstance(width_thresholds, Mapping):
        raise Phase2PromotionError("min_set_exact_match_by_width must be an object")
    for width in ("1", "2", "3"):
        observed = metrics["set_exact_match_by_width"].get(width)
        threshold = width_thresholds.get(width)
        if threshold is None:
            raise Phase2PromotionError(f"missing width threshold {width}")
        _check_minimum(checks, f"min_set_exact_match_k{width}", observed, threshold)

    group_rates = metrics["set_exact_match_by_vendor_or_model"]
    minimum_group_rate = min(group_rates.values()) if group_rates else float("nan")
    _minimum(
        checks,
        "min_heldout_vendor_or_model_set_exact",
        minimum_group_rate,
        thresholds,
    )
    minimum_group_rows = min(metrics["rows_by_vendor_or_model"].values(), default=0)
    _minimum(
        checks,
        "min_heldout_rows_per_vendor_or_model",
        minimum_group_rows,
        thresholds,
    )
    robustness = metrics["robustness"]
    for variant in sorted(REQUIRED_ROBUSTNESS_VARIANTS - {"base"}):
        _required(checks, f"robustness_{variant}", robustness.get(variant))

    failures = [check for check in checks if not check["passed"]]
    return {
        "schema_version": "phase2_goal_extractor_promotion.v1",
        "status": "pass" if not failures else "fail",
        "role": "goal_extractor",
        "artifact_sha": artifact_evidence.get("artifact_sha"),
        "metrics": metrics,
        "checks": checks,
        "failures": failures,
    }


def _prediction(value: Any) -> tuple[bool, list[str], bool]:
    try:
        if isinstance(value, str):
            value = json.loads(value)
        if not isinstance(value, Mapping) or set(value) != {"rest_api_list"}:
            return False, [], False
        raw = value["rest_api_list"]
        if not isinstance(raw, list) or not all(
            isinstance(item, str) and item.strip() for item in raw
        ):
            return False, [], False
        normalized = [item.strip() for item in raw]
        return True, normalized, len(normalized) != len(set(normalized))
    except (json.JSONDecodeError, TypeError, ValueError):
        return False, [], False


def _target_list(value: Mapping[str, Any], *, label: str) -> list[str]:
    if set(value) != {"rest_api_list"}:
        raise Phase2PromotionError(f"{label} must contain exactly rest_api_list")
    raw = value["rest_api_list"]
    if not isinstance(raw, list) or not all(isinstance(item, str) and item for item in raw):
        raise Phase2PromotionError(f"{label}.rest_api_list must be list[str]")
    if len(raw) != len(set(raw)):
        raise Phase2PromotionError(f"{label}.rest_api_list must be unique")
    return list(raw)


def _catalog_apis(value: Any, *, row_index: int) -> set[str]:
    if not isinstance(value, list):
        raise Phase2PromotionError(f"row {row_index}: api_context must be a list")
    apis: list[str] = []
    for context in value:
        if not isinstance(context, Mapping):
            raise Phase2PromotionError(f"row {row_index}: api_context item is not an object")
        if set(context) != set(CONTEXT_FIELDS):
            raise Phase2PromotionError(
                f"row {row_index}: api_context fields must match the public context contract"
            )
        api = context.get("rest_api")
        if not isinstance(api, str) or not api:
            raise Phase2PromotionError(f"row {row_index}: context rest_api is invalid")
        methods = context.get("allowed_methods")
        if (
            not isinstance(methods, list)
            or not all(
                isinstance(method, str)
                and method
                and method == method.upper()
                and method in HTTP_METHODS
                for method in methods
            )
            or len(methods) != len(set(methods))
        ):
            raise Phase2PromotionError(
                f"row {row_index}: context allowed_methods is invalid"
            )
        operation_names = context.get("operation_names")
        if (
            not isinstance(operation_names, list)
            or not all(isinstance(name, str) and name.strip() for name in operation_names)
            or len(operation_names) != len(set(operation_names))
        ):
            raise Phase2PromotionError(
                f"row {row_index}: context operation_names is invalid"
            )
        if not isinstance(context.get("argument_schema"), Mapping):
            raise Phase2PromotionError(
                f"row {row_index}: context argument_schema must be an object"
            )
        if not isinstance(context.get("json"), Mapping):
            raise Phase2PromotionError(
                f"row {row_index}: context json must be an object"
            )
        apis.append(api)
    if len(apis) != len(set(apis)):
        raise Phase2PromotionError(f"row {row_index}: duplicate API in api_context")
    return set(apis)


def _robustness_result(cases: Mapping[str, Mapping[str, bool]]) -> dict[str, bool]:
    if not cases:
        return {variant: False for variant in REQUIRED_ROBUSTNESS_VARIANTS}
    result: dict[str, bool] = {}
    for variant in REQUIRED_ROBUSTNESS_VARIANTS:
        result[variant] = all(
            set(case) == REQUIRED_ROBUSTNESS_VARIANTS and bool(case.get(variant))
            for case in cases.values()
        )
    return result


def _zero_set_metrics(expected: Sequence[str]) -> dict[str, float | int | bool]:
    return {
        "set_match_rate": 0.0,
        "precision": 0.0,
        "recall": 0.0 if expected else 1.0,
        "f1": 0.0,
    }


def _mapping(row: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = row.get(key)
    if not isinstance(value, Mapping):
        raise Phase2PromotionError(f"{key} must be an object")
    return value


def _mean(values: Sequence[float]) -> float:
    return float("nan") if not values else sum(values) / len(values)


def _required(checks: list[dict[str, Any]], name: str, observed: Any) -> None:
    checks.append({"name": name, "observed": observed, "operator": "required", "passed": bool(observed)})


def _minimum(
    checks: list[dict[str, Any]],
    name: str,
    observed: Any,
    thresholds: Mapping[str, Any],
) -> None:
    _check_minimum(checks, name, observed, thresholds[name])


def _check_minimum(
    checks: list[dict[str, Any]],
    name: str,
    observed: Any,
    threshold: Any,
) -> None:
    passed = _finite(observed) and _finite(threshold) and float(observed) >= float(threshold)
    checks.append({"name": name, "observed": observed, "threshold": threshold, "operator": ">=", "passed": passed})


def _maximum(
    checks: list[dict[str, Any]],
    name: str,
    observed: Any,
    thresholds: Mapping[str, Any],
) -> None:
    threshold = thresholds[name]
    passed = _finite(observed) and _finite(threshold) and float(observed) <= float(threshold)
    checks.append({"name": name, "observed": observed, "threshold": threshold, "operator": "<=", "passed": passed})


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _parent_artifact_sha(evidence: Mapping[str, Any]) -> str:
    parent = _mapping(evidence, "real_promoted_parent_checkpoint")
    return str(parent.get("artifact_sha", ""))


def _all_finite_metrics(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_all_finite_metrics(item) for item in value.values())
    if isinstance(value, bool):
        return True
    return _finite(value)
