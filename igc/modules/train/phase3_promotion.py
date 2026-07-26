"""REST-API-keyed evaluation and promotion checks for Phase 3."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from typing import Any, Mapping, Sequence

from igc.ds.rest_goal_contract import (
    ARGUMENT_GROUNDING_SOURCES,
    CALL_FIELDS,
    CONTEXT_FIELDS,
    HTTP_METHODS,
    arguments_match_schema,
)
from igc.modules.train.promotion_evidence import (
    validate_promotion_evidence,
    validate_training_run_evidence,
)


REQUIRED_ARGUMENT_CLASSES = frozenset({
    "read_only_empty",
    "patch_scalar",
    "patch_nested",
    "post_no_arguments",
    "post_one_argument",
    "post_multiple_arguments",
    "delete_no_body",
})
REQUIRED_HTTP_METHODS = frozenset({"GET", "HEAD", "PATCH", "POST", "DELETE"})


class Phase3PromotionError(ValueError):
    """Raised when Phase 3 promotion evidence is incomplete or malformed."""


def validate_phase_views(
    phase2_row: Mapping[str, Any],
    phase3_row: Mapping[str, Any],
) -> None:
    """Require Phase 2 and Phase 3 views of one master row to select the same APIs."""
    phase2_x = _mapping(phase2_row, "x")
    phase3_x = _mapping(phase3_row, "x")
    if phase2_x.get("text") != phase3_x.get("text"):
        raise Phase3PromotionError("D1 Phase 2/3 views have different text labels")
    if phase2_x.get("api_context") != phase3_x.get("api_context"):
        raise Phase3PromotionError("D1 Phase 2/3 views have different API context")
    phase2_target = _mapping(phase2_row, "y_true").get("rest_api_list")
    if (
        not isinstance(phase2_target, list)
        or not all(isinstance(api, str) and api for api in phase2_target)
        or len(phase2_target) != len(set(phase2_target))
    ):
        raise Phase3PromotionError("Phase 2 target must be a unique rest_api_list")
    parsed, calls, duplicates = _parse_calls(_mapping(phase3_row, "y_true"))
    if not parsed or duplicates:
        raise Phase3PromotionError("Phase 3 target must be a valid unique call set")
    if set(phase2_target) != {call["rest_api"] for call in calls}:
        raise Phase3PromotionError("D1 Phase 2/3 views select different REST API sets")
    phase3_selected = phase3_x.get("rest_api_list")
    if (
        not isinstance(phase3_selected, list)
        or not all(isinstance(api, str) and api for api in phase3_selected)
        or len(phase3_selected) != len(set(phase3_selected))
        or set(phase3_selected) != set(phase2_target)
    ):
        raise Phase3PromotionError(
            "D1 Phase 3 input REST API set disagrees with the Phase 2 target"
        )
    phase2_metadata = _mapping(phase2_row, "metadata")
    phase3_metadata = _mapping(phase3_row, "metadata")
    row_id = phase2_metadata.get("row_id")
    if not isinstance(row_id, str) or phase3_metadata.get("row_id") != row_id:
        raise Phase3PromotionError("D1 Phase 2/3 views have different row identities")


def evaluate_phase3_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Evaluate Phase 3 predictions by REST API identity, never list position."""
    if not rows:
        raise Phase3PromotionError("Phase 3 promotion requires held-out rows")
    row_totals: dict[str, float] = defaultdict(float)
    call_totals: dict[str, float] = defaultdict(float)
    by_width: dict[int, list[float]] = defaultdict(list)
    by_argument_class: dict[str, list[float]] = defaultdict(list)
    expected_by_method: dict[str, int] = defaultdict(int)

    for index, row in enumerate(rows):
        if row.get("phase") != 3 or row.get("source_dataset") != "D1":
            raise Phase3PromotionError(f"row {index}: Phase 3 lineage is invalid")
        if row.get("task") != "text_and_rest_api_list_to_calls":
            raise Phase3PromotionError(f"row {index}: Phase 3 task is invalid")
        if row.get("target_semantics") != "unordered_unique_call_set":
            raise Phase3PromotionError(f"row {index}: Phase 3 target semantics are invalid")
        x = _mapping(row, "x")
        if set(x) != {"text", "rest_api_list", "api_context"}:
            raise Phase3PromotionError(
                f"row {index}: x must contain text, rest_api_list, and api_context"
            )
        selected = _unique_strings(x.get("rest_api_list"), f"row {index} rest_api_list")
        if len(selected) not in (0, 1, 2, 3):
            raise Phase3PromotionError(
                f"row {index}: Phase 3 width must be 0, 1, 2, or 3"
            )
        contexts = _context_by_api(x.get("api_context"), row_index=index)
        if set(selected) - set(contexts):
            raise Phase3PromotionError(f"row {index}: selected API lacks api_context")

        expected_ok, expected_calls, expected_duplicates = _parse_calls(
            _mapping(row, "y_true")
        )
        if not expected_ok or expected_duplicates:
            raise Phase3PromotionError(f"row {index}: invalid y_true.calls")
        expected_by_api = {call["rest_api"]: call for call in expected_calls}
        if set(expected_by_api) != set(selected):
            raise Phase3PromotionError(
                f"row {index}: exactly one y_true call is required per Phase 2 API"
            )
        grounding = _grounding_by_api(row, expected_by_api, row_index=index)
        _validate_expected_labels(expected_by_api, contexts, grounding, row_index=index)

        prediction_value = row.get("y_pred")
        json_parsed = _json_object_parse_ok(prediction_value)
        invalid_method_count, raw_call_count = _invalid_method_count(
            prediction_value,
            contexts,
        )
        parsed, predicted_calls, duplicate_count = _parse_calls(prediction_value)
        predicted_by_api: dict[str, Mapping[str, Any]] = {}
        for call in predicted_calls:
            predicted_by_api.setdefault(call["rest_api"], call)
        expected_apis = set(expected_by_api)
        predicted_apis = set(predicted_by_api)
        coverage_exact = parsed and not duplicate_count and predicted_apis == expected_apis
        call_set_exact = coverage_exact and all(
            predicted_by_api[api] == expected_by_api[api] for api in expected_apis
        )
        row_totals["json_parse"] += float(json_parsed)
        row_totals["coverage_exact"] += float(coverage_exact)
        row_totals["call_set_exact"] += float(call_set_exact)
        row_totals["duplicate_rows"] += float(duplicate_count > 0)
        by_width[len(selected)].append(float(call_set_exact))

        call_totals["expected"] += len(expected_apis)
        call_totals["predicted"] += len(predicted_calls)
        call_totals["missing"] += len(expected_apis - predicted_apis)
        call_totals["extra"] += len(predicted_apis - expected_apis)
        call_totals["duplicates"] += duplicate_count
        call_totals["invalid_method"] += invalid_method_count
        call_totals["raw_calls"] += raw_call_count
        for api, expected in expected_by_api.items():
            predicted = predicted_by_api.get(api)
            context = contexts[api]
            argument_class = _argument_class(expected)
            expected_by_method[expected["http_method"]] += 1
            argument_exact = bool(predicted and predicted["arguments"] == expected["arguments"])
            by_argument_class[argument_class].append(float(argument_exact))
            call_totals["method"] += float(
                bool(predicted and predicted["http_method"] == expected["http_method"])
            )
            call_totals["operation"] += float(
                bool(predicted and predicted["operation_name"] == expected["operation_name"])
            )
            call_totals["arguments"] += float(argument_exact)
            schema_valid = bool(
                predicted
                and arguments_match_schema(
                    predicted["arguments"],
                    context.get("argument_schema"),
                )
            )
            call_totals["schema_valid"] += float(schema_valid)
            grounded = bool(
                predicted
                and argument_exact
                and (not expected["arguments"] or grounding[api])
            )
            call_totals["grounded"] += float(grounded)
            if expected["http_method"] in {"GET", "HEAD"}:
                call_totals["readonly"] += 1
                call_totals["readonly_empty"] += float(
                    bool(predicted and predicted["arguments"] == {})
                )

    row_count = len(rows)
    expected_count = int(call_totals["expected"])
    predicted_count = int(call_totals["predicted"])
    return {
        "rows": row_count,
        "json_parse_rate": row_totals["json_parse"] / row_count,
        "call_set_exact_match_rate": row_totals["call_set_exact"] / row_count,
        "rest_api_coverage_exact_rate": row_totals["coverage_exact"] / row_count,
        "http_method_exact_match_rate": _rate(call_totals["method"], expected_count),
        "operation_name_exact_match_rate": _rate(call_totals["operation"], expected_count),
        "arguments_exact_match_rate": _rate(call_totals["arguments"], expected_count),
        "argument_schema_valid_rate": _rate(call_totals["schema_valid"], expected_count),
        "argument_value_grounding_rate": _rate(call_totals["grounded"], expected_count),
        "invalid_method_rate": _error_rate(
            call_totals["invalid_method"],
            int(call_totals["raw_calls"]),
        ),
        "missing_call_rate": _error_rate(call_totals["missing"], expected_count),
        "extra_call_rate": _error_rate(call_totals["extra"], expected_count),
        "duplicate_call_rate": _error_rate(call_totals["duplicates"], predicted_count),
        "readonly_empty_arguments_rate": _rate(
            call_totals["readonly_empty"],
            int(call_totals["readonly"]),
        ),
        "call_set_exact_match_by_width": {
            str(width): _mean(values) for width, values in sorted(by_width.items())
        },
        "arguments_exact_match_by_class": {
            name: _mean(values) for name, values in sorted(by_argument_class.items())
        },
        "expected_calls_by_http_method": dict(sorted(expected_by_method.items())),
    }


def evaluate_phase3_promotion(
    *,
    rows: Sequence[Mapping[str, Any]],
    thresholds: Mapping[str, Any],
    artifact_evidence: Mapping[str, Any],
    run_report: Mapping[str, Any],
    view_pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    observed_source_full_manifest_sha: str,
    observed_source_full_sha: str,
    observed_train_manifest_sha: str,
    observed_train_sha: str,
    observed_heldout_manifest_sha: str,
    observed_heldout_sha: str,
    observed_split_release_sha: str,
) -> dict[str, Any]:
    """Apply non-null quality floors and hard real-artifact gates."""
    minimums = (
        "min_json_parse_rate",
        "min_call_set_exact_match_rate",
        "min_rest_api_coverage_exact",
        "min_method_exact_match_rate",
        "min_operation_name_exact_match_rate",
        "min_argument_schema_valid_rate",
        "min_argument_value_grounding_rate",
        "min_arguments_exact_match_rate",
        "min_readonly_empty_arguments_rate",
    )
    maximums = (
        "max_invalid_method_rate",
        "max_duplicate_call_rate",
        "max_missing_call_rate",
        "max_extra_call_rate",
    )
    missing = [name for name in (*minimums, *maximums) if thresholds.get(name) is None]
    if missing:
        raise Phase3PromotionError(f"missing non-null thresholds: {missing}")
    metrics = evaluate_phase3_rows(rows)
    checks = validate_promotion_evidence(
        artifact_evidence,
        expected_parent_role="goal_extractor",
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
        expected_phase="phase3_argument_extraction",
        expected_task="text_and_rest_api_list_to_calls",
        expected_parent_role="goal_extractor",
        expected_parent_artifact_sha=_parent_artifact_sha(artifact_evidence),
        expected_output_role="argument_extractor",
        observed_source_full_manifest_sha=observed_source_full_manifest_sha,
        observed_train_manifest_sha=observed_train_manifest_sha,
        observed_train_sha=observed_train_sha,
        observed_heldout_manifest_sha=observed_heldout_manifest_sha,
        observed_heldout_sha=observed_heldout_sha,
        reload_evidence=_mapping(artifact_evidence, "checkpoint_reload"),
    ))
    _required(checks, "finite_promotion_metrics", _all_finite_metrics(metrics))
    views_consistent = bool(view_pairs)
    for phase2_row, phase3_row in view_pairs:
        try:
            validate_phase_views(phase2_row, phase3_row)
        except Phase3PromotionError:
            views_consistent = False
    _required(checks, "d1_phase2_phase3_view_consistency", views_consistent)
    for width in ("1", "2", "3"):
        _required(
            checks,
            f"heldout_width_k{width}_present",
            width in metrics["call_set_exact_match_by_width"],
        )
    for name in sorted(REQUIRED_ARGUMENT_CLASSES):
        _required(
            checks,
            f"argument_class_{name}_present",
            name in metrics["arguments_exact_match_by_class"],
        )
    for method in sorted(REQUIRED_HTTP_METHODS):
        _required(
            checks,
            f"http_method_{method.lower()}_present",
            metrics["expected_calls_by_http_method"].get(method, 0) > 0,
        )
    metric_for_minimum = {
        "min_json_parse_rate": "json_parse_rate",
        "min_call_set_exact_match_rate": "call_set_exact_match_rate",
        "min_rest_api_coverage_exact": "rest_api_coverage_exact_rate",
        "min_method_exact_match_rate": "http_method_exact_match_rate",
        "min_operation_name_exact_match_rate": "operation_name_exact_match_rate",
        "min_argument_schema_valid_rate": "argument_schema_valid_rate",
        "min_argument_value_grounding_rate": "argument_value_grounding_rate",
        "min_arguments_exact_match_rate": "arguments_exact_match_rate",
        "min_readonly_empty_arguments_rate": "readonly_empty_arguments_rate",
    }
    metric_for_maximum = {
        "max_invalid_method_rate": "invalid_method_rate",
        "max_duplicate_call_rate": "duplicate_call_rate",
        "max_missing_call_rate": "missing_call_rate",
        "max_extra_call_rate": "extra_call_rate",
    }
    for threshold_name, metric_name in metric_for_minimum.items():
        _comparison(checks, threshold_name, metrics[metric_name], thresholds[threshold_name], ">=")
    for threshold_name, metric_name in metric_for_maximum.items():
        _comparison(checks, threshold_name, metrics[metric_name], thresholds[threshold_name], "<=")
    failures = [check for check in checks if not check["passed"]]
    return {
        "schema_version": "phase3_argument_extractor_promotion.v1",
        "status": "pass" if not failures else "fail",
        "role": "argument_extractor",
        "artifact_sha": artifact_evidence.get("artifact_sha"),
        "metrics": metrics,
        "checks": checks,
        "failures": failures,
    }


def _parse_calls(value: Any) -> tuple[bool, list[dict[str, Any]], int]:
    try:
        if isinstance(value, str):
            value = json.loads(value)
        if not isinstance(value, Mapping) or set(value) != {"calls"}:
            return False, [], 0
        raw_calls = value["calls"]
        if not isinstance(raw_calls, list):
            return False, [], 0
        calls: list[dict[str, Any]] = []
        seen: set[str] = set()
        duplicates = 0
        for raw in raw_calls:
            if not isinstance(raw, Mapping) or set(raw) != CALL_FIELDS:
                return False, [], 0
            api = raw["rest_api"]
            method = raw["http_method"]
            operation = raw["operation_name"]
            arguments = raw["arguments"]
            if not isinstance(api, str) or not api.strip():
                return False, [], 0
            if (
                not isinstance(method, str)
                or method != method.upper()
                or method not in HTTP_METHODS
            ):
                return False, [], 0
            if operation is not None and (not isinstance(operation, str) or not operation.strip()):
                return False, [], 0
            if not isinstance(arguments, Mapping):
                return False, [], 0
            if method in {"GET", "HEAD"} and arguments:
                return False, [], 0
            if api in seen:
                duplicates += 1
            seen.add(api)
            calls.append({
                "rest_api": api,
                "http_method": method,
                "operation_name": operation.strip() if isinstance(operation, str) else None,
                "arguments": dict(arguments),
            })
        return True, calls, duplicates
    except (json.JSONDecodeError, TypeError, ValueError):
        return False, [], 0


def _validate_expected_labels(
    expected: Mapping[str, Mapping[str, Any]],
    contexts: Mapping[str, Mapping[str, Any]],
    grounding: Mapping[str, bool],
    *,
    row_index: int,
) -> None:
    for api, call in expected.items():
        allowed = _unique_methods(contexts[api].get("allowed_methods"), f"row {row_index} {api}")
        if call["http_method"] not in allowed:
            raise Phase3PromotionError(f"row {row_index}: y_true method is not legal for {api}")
        operation_names = _unique_strings(
            contexts[api].get("operation_names"),
            f"row {row_index} {api} operation_names",
        )
        if call["operation_name"] is not None and call["operation_name"] not in operation_names:
            raise Phase3PromotionError(
                f"row {row_index}: y_true operation_name is not declared for {api}"
            )
        if not arguments_match_schema(
            call["arguments"],
            contexts[api].get("argument_schema"),
        ):
            raise Phase3PromotionError(f"row {row_index}: y_true arguments violate schema for {api}")
        if call["arguments"] and not grounding[api]:
            raise Phase3PromotionError(f"row {row_index}: y_true values are not grounded for {api}")


def _grounding_by_api(
    row: Mapping[str, Any],
    expected: Mapping[str, Mapping[str, Any]],
    *,
    row_index: int,
) -> dict[str, bool]:
    evidence = _mapping(row, "label_evidence")
    raw = evidence.get("argument_value_grounding_by_api")
    if not isinstance(raw, Mapping) or set(raw) != set(expected):
        raise Phase3PromotionError(
            f"row {row_index}: grounding evidence keys must match y_true calls"
        )
    result: dict[str, bool] = {}
    for api, item in raw.items():
        if not isinstance(item, Mapping) or set(item) != {"grounded", "sources"}:
            raise Phase3PromotionError(f"row {row_index}: grounding evidence is invalid")
        sources = item["sources"]
        if not isinstance(sources, list) or not all(isinstance(source, str) for source in sources):
            raise Phase3PromotionError(f"row {row_index}: grounding sources are invalid")
        if set(sources) - ARGUMENT_GROUNDING_SOURCES:
            raise Phase3PromotionError(
                f"row {row_index}: unsupported argument-grounding source"
            )
        if expected[api]["arguments"] and (
            "operator_text" not in sources
            or not ({"argument_schema", "operation_definition"} & set(sources))
        ):
            raise Phase3PromotionError(
                f"row {row_index}: mutation arguments lack text and schema/action grounding"
            )
        if not isinstance(item["grounded"], bool):
            raise Phase3PromotionError(f"row {row_index}: grounded must be boolean")
        result[str(api)] = item["grounded"]
    return result


def _argument_class(call: Mapping[str, Any]) -> str:
    method = call["http_method"]
    arguments = call["arguments"]
    if method in {"GET", "HEAD"}:
        return "read_only_empty"
    if method == "DELETE" and not arguments:
        return "delete_no_body"
    if method == "POST":
        if not arguments:
            return "post_no_arguments"
        return "post_one_argument" if len(arguments) == 1 else "post_multiple_arguments"
    if method == "PATCH":
        return (
            "patch_nested"
            if any(isinstance(value, (Mapping, list)) for value in arguments.values())
            else "patch_scalar"
        )
    return f"{method.lower()}_{'arguments' if arguments else 'no_body'}"


def _context_by_api(value: Any, *, row_index: int) -> dict[str, Mapping[str, Any]]:
    if not isinstance(value, list) or not value:
        raise Phase3PromotionError(
            f"row {row_index}: api_context must be a non-empty list"
        )
    result: dict[str, Mapping[str, Any]] = {}
    for item in value:
        if not isinstance(item, Mapping):
            raise Phase3PromotionError(f"row {row_index}: api_context item is invalid")
        if set(item) != set(CONTEXT_FIELDS):
            raise Phase3PromotionError(
                f"row {row_index}: api_context fields must match the public contract"
            )
        api = item.get("rest_api")
        if not isinstance(api, str) or not api or api in result:
            raise Phase3PromotionError(f"row {row_index}: api_context rest_api is invalid")
        _unique_methods(item.get("allowed_methods"), f"row {row_index} {api}")
        _unique_strings(item.get("operation_names"), f"row {row_index} operation_names")
        if not isinstance(item.get("argument_schema"), Mapping):
            raise Phase3PromotionError(
                f"row {row_index}: context argument_schema must be an object"
            )
        if not isinstance(item.get("json"), Mapping):
            raise Phase3PromotionError(f"row {row_index}: context json must be an object")
        result[api] = item
    return result


def _json_object_parse_ok(value: Any) -> bool:
    try:
        if isinstance(value, str):
            value = json.loads(value)
        return isinstance(value, Mapping)
    except (json.JSONDecodeError, TypeError, ValueError):
        return False


def _invalid_method_count(
    value: Any,
    contexts: Mapping[str, Mapping[str, Any]],
) -> tuple[int, int]:
    try:
        if isinstance(value, str):
            value = json.loads(value)
        if not isinstance(value, Mapping) or not isinstance(value.get("calls"), list):
            return 0, 0
        invalid = 0
        raw_calls = value["calls"]
        for raw in raw_calls:
            if not isinstance(raw, Mapping):
                continue
            method = raw.get("http_method")
            api = raw.get("rest_api")
            supported = (
                isinstance(method, str)
                and method == method.upper()
                and method in HTTP_METHODS
            )
            context = contexts.get(api) if isinstance(api, str) else None
            allowed = (
                _unique_methods(context.get("allowed_methods"), f"prediction {api}")
                if context is not None
                else set()
            )
            invalid += int(not supported or context is None or method not in allowed)
        return invalid, len(raw_calls)
    except (json.JSONDecodeError, TypeError, ValueError, Phase3PromotionError):
        return 0, 0


def _unique_strings(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise Phase3PromotionError(f"{label} must be list[str]")
    if len(value) != len(set(value)):
        raise Phase3PromotionError(f"{label} must be unique")
    return list(value)


def _unique_methods(value: Any, label: str) -> set[str]:
    methods = _unique_strings(value, f"{label} allowed_methods")
    if any(method != method.upper() for method in methods):
        raise Phase3PromotionError(f"{label} allowed_methods must be uppercase")
    normalized = {method.upper() for method in methods}
    if not normalized <= HTTP_METHODS:
        raise Phase3PromotionError(f"{label} contains unsupported methods")
    return normalized


def _mapping(row: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = row.get(key)
    if not isinstance(value, Mapping):
        raise Phase3PromotionError(f"{key} must be an object")
    return value


def _rate(numerator: float, denominator: int) -> float:
    return 1.0 if denominator == 0 else float(numerator) / denominator


def _error_rate(numerator: float, denominator: int) -> float:
    return 0.0 if denominator == 0 else float(numerator) / denominator


def _parent_artifact_sha(evidence: Mapping[str, Any]) -> str:
    parent = _mapping(evidence, "real_promoted_parent_checkpoint")
    return str(parent.get("artifact_sha", ""))


def _all_finite_metrics(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_all_finite_metrics(item) for item in value.values())
    if isinstance(value, bool):
        return True
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _mean(values: Sequence[float]) -> float:
    return float("nan") if not values else sum(values) / len(values)


def _required(checks: list[dict[str, Any]], name: str, observed: Any) -> None:
    checks.append({"name": name, "operator": "required", "observed": observed, "passed": bool(observed)})


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
    checks.append({"name": name, "operator": operator, "observed": observed, "threshold": threshold, "passed": passed})
