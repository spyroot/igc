"""Canonical unordered Phase 2 REST-goal and Phase 3 call contracts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from igc.modules.base.metric_keys import (
    PHASE2_WANDB_METRIC_KEYS,
    PHASE3_WANDB_METRIC_KEYS,
)
from igc.modules.train.sft_tasks import resolve_sft_task


MODEL_X = "model_x"
D0 = "D0"
D1 = "D1"

PHASE2_GOAL_EXTRACT_METRIC_KEYS = PHASE2_WANDB_METRIC_KEYS
PHASE3_ARGUMENT_EXTRACT_METRIC_KEYS = PHASE3_WANDB_METRIC_KEYS

CALL_FIELDS = frozenset({
    "rest_api",
    "http_method",
    "operation_name",
    "arguments",
})
CONTEXT_FIELDS = frozenset({
    "rest_api",
    "allowed_methods",
    "operation_names",
    "argument_schema",
    "json",
})
HTTP_METHODS = frozenset({"GET", "HEAD", "POST", "PATCH", "PUT", "DELETE"})
ARGUMENT_GROUNDING_SOURCES = frozenset({
    "operator_text",
    "argument_schema",
    "operation_definition",
})


@dataclass(frozen=True)
class RedfishContext:
    """One candidate resource from a discovered Redfish environment.

    ``rest_api`` and ``json`` come from the discovered corpus.
    ``allowed_methods``, operation names, and argument schemas come from the
    matching ``rest_api_map.npy`` and HTTP-semantics evidence.
    """

    rest_api: str
    allowed_methods: Sequence[str]
    json: Mapping[str, Any]
    operation_names: Sequence[str] = ()
    argument_schema: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the context object shown to Phase 2 and Phase 3."""
        return {
            "rest_api": self.rest_api,
            "allowed_methods": _normalize_methods(self.allowed_methods),
            "operation_names": _normalize_operation_names(self.operation_names),
            "argument_schema": dict(self.argument_schema),
            "json": dict(self.json),
        }


@dataclass(frozen=True)
class RenderedContractExample:
    """Prompt/completion split consumed by the shared SFT dataset."""

    prompt: str
    target_json: str
    target_char_start: int

    @property
    def full_text(self) -> str:
        """Return prompt plus completion."""
        return self.prompt + self.target_json


def canonical_json(value: Any) -> str:
    """Serialize deterministic JSON for SFT targets and conformance tests."""
    return json.dumps(value, indent=2, sort_keys=True)


def d1_row_id(row: Mapping[str, Any]) -> str:
    """Return a stable identity for one Phase 2 D1 text/API label pair."""
    payload = {
        "phase": row.get("phase"),
        "dataset": row.get("dataset"),
        "source_dataset": row.get("source_dataset"),
        "task": row.get("task"),
        "contract_version": row.get("contract_version"),
        "target_semantics": row.get("target_semantics"),
        "x": _mapping(row, "x"),
        "y_true": _mapping(row, "y_true"),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def build_d1_rest_api_list_row(
    *,
    text: str,
    contexts: Sequence[RedfishContext],
    rest_api_list: Sequence[str],
    validation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one Phase 2 labelled-request row.

    Contexts may include distractors. The target API set is deliberately absent
    from ``x`` and is canonically sorted only to provide one stable token target.
    """
    task = resolve_sft_task("text_to_rest_api_list")
    by_api = _contexts_by_api(contexts)
    targets = _canonical_unique_strings(rest_api_list, "rest_api_list")
    _require_context(targets, by_api)
    row = {
        "phase": 2,
        "dataset": D1,
        "source_dataset": D0,
        "task": task.name,
        "contract_version": task.contract_version,
        "target_semantics": "unordered_unique_rest_api_set",
        "x": {
            "text": _required_text(text),
            "api_context": [context.to_dict() for context in contexts],
        },
        "y_true": {"rest_api_list": targets},
        "validation": dict(validation or {}),
    }
    _validate_phase2_context_policy(row, task.input_policy)
    return row


def build_call_row(
    *,
    text: str,
    contexts: Sequence[RedfishContext],
    rest_api_list: Sequence[str],
    method_by_api: Mapping[str, str],
    operation_name_by_api: Mapping[str, str | None],
    arguments_by_api: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Build one Phase 3 row from explicit method/operation/argument labels."""
    task = resolve_sft_task("text_and_rest_api_list_to_calls")
    by_api = _contexts_by_api(contexts)
    if not by_api:
        raise ValueError("Phase 3 requires non-empty API context evidence")
    targets = _canonical_unique_strings(rest_api_list, "rest_api_list")
    _require_context(targets, by_api)
    _require_label_keys("method_by_api", method_by_api, targets)
    _require_label_keys("operation_name_by_api", operation_name_by_api, targets)
    _require_label_keys("arguments_by_api", arguments_by_api, targets)

    calls: list[dict[str, Any]] = []
    for rest_api in targets:
        context = by_api.get(rest_api)
        allowed_methods = (
            _normalize_methods(context.allowed_methods) if context is not None else []
        )
        raw_http_method = method_by_api[rest_api]
        if not isinstance(raw_http_method, str) or raw_http_method != raw_http_method.upper():
            raise ValueError(f"http_method must be an uppercase string for {rest_api!r}")
        http_method = raw_http_method
        if http_method not in HTTP_METHODS:
            raise ValueError(f"unsupported http_method {http_method!r}")
        if allowed_methods and http_method not in allowed_methods:
            raise ValueError(
                f"http_method {http_method!r} is not allowed for {rest_api!r}"
            )
        raw_operation_name = operation_name_by_api[rest_api]
        if raw_operation_name is None:
            operation_name = None
        else:
            operation_name = str(raw_operation_name).strip()
            if not operation_name:
                raise ValueError(f"operation_name is empty for {rest_api!r}")
        if (
            context is not None
            and operation_name is not None
            and operation_name not in _normalize_operation_names(context.operation_names)
        ):
            raise ValueError(
                f"operation_name {operation_name!r} is not declared for {rest_api!r}"
            )
        arguments = dict(arguments_by_api[rest_api])
        if http_method in {"GET", "HEAD"} and arguments:
            raise ValueError(
                f"{http_method} call arguments must be empty for {rest_api!r}"
            )
        if context is not None and not arguments_match_schema(
            arguments,
            context.argument_schema,
        ):
            raise ValueError(f"arguments do not match schema for {rest_api!r}")
        calls.append({
            "rest_api": rest_api,
            "http_method": http_method,
            "operation_name": operation_name,
            "arguments": arguments,
        })

    return {
        "phase": 3,
        "source_dataset": D1,
        "task": task.name,
        "contract_version": task.contract_version,
        "target_semantics": "unordered_unique_call_set",
        "x": {
            "text": _required_text(text),
            "rest_api_list": targets,
            "api_context": [context.to_dict() for context in contexts],
        },
        "y_true": {"calls": calls},
    }


def build_d1_master_record(
    *,
    text: str,
    contexts: Sequence[RedfishContext],
    method_by_api: Mapping[str, str],
    operation_name_by_api: Mapping[str, str | None],
    arguments_by_api: Mapping[str, Mapping[str, Any]],
    argument_value_grounding_by_api: Mapping[str, Mapping[str, Any]],
    validation: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one private master label record for deterministic Phase 2/3 views.

    The selected API set exists once, as the keys of the explicit call labels.
    Mutation values require positive grounding evidence and are never inferred
    from the current JSON observation.
    """
    selected = _canonical_unique_strings(
        list(method_by_api),
        "method_by_api keys",
    )
    phase3 = build_call_row(
        text=text,
        contexts=contexts,
        rest_api_list=selected,
        method_by_api=method_by_api,
        operation_name_by_api=operation_name_by_api,
        arguments_by_api=arguments_by_api,
    )
    calls = phase3["y_true"]["calls"]
    grounding = _normalize_argument_grounding(
        argument_value_grounding_by_api,
        calls=calls,
    )
    phase2 = build_d1_rest_api_list_row(
        text=text,
        contexts=contexts,
        rest_api_list=selected,
        validation=validation,
    )
    metadata_value = dict(metadata)
    if metadata_value.get("row_id") != d1_row_id(phase2):
        raise ValueError("D1 master metadata.row_id must match its Phase 2 view")
    return {
        "schema_version": "d1_master.v1",
        "dataset": D1,
        "source_dataset": D0,
        "text": phase2["x"]["text"],
        "api_context": phase2["x"]["api_context"],
        "calls": calls,
        "validation": dict(validation),
        "label_evidence": {
            "argument_value_grounding_by_api": grounding,
        },
        "metadata": metadata_value,
    }


def render_d1_master_views(
    master: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Render strict Phase 2 and Phase 3 rows from one master label record."""
    _require_exact_fields(
        master,
        {
            "schema_version",
            "dataset",
            "source_dataset",
            "text",
            "api_context",
            "calls",
            "validation",
            "label_evidence",
            "metadata",
        },
        "D1 master record",
    )
    if (
        master.get("schema_version") != "d1_master.v1"
        or master.get("dataset") != D1
        or master.get("source_dataset") != D0
    ):
        raise ValueError("D1 master identity is invalid")
    contexts = _contexts_from_raw(master.get("api_context"))
    calls = parse_calls_y_pred(
        {"calls": master.get("calls")},
        contexts=contexts,
    )
    selected = [call["rest_api"] for call in calls]
    phase2 = build_d1_rest_api_list_row(
        text=_required_text(master.get("text")),
        contexts=contexts,
        rest_api_list=selected,
        validation=_mapping(master, "validation"),
    )
    phase2["metadata"] = dict(_mapping(master, "metadata"))
    if phase2["metadata"].get("row_id") != d1_row_id(phase2):
        raise ValueError("D1 master metadata.row_id does not match its Phase 2 view")
    grounding = _normalize_argument_grounding(
        _mapping(master, "label_evidence").get(
            "argument_value_grounding_by_api"
        ),
        calls=calls,
    )
    phase3 = {
        "phase": 3,
        "source_dataset": D1,
        "task": "text_and_rest_api_list_to_calls",
        "contract_version": "phase3-call-set/v1",
        "target_semantics": "unordered_unique_call_set",
        "x": {
            "text": phase2["x"]["text"],
            "rest_api_list": selected,
            "api_context": phase2["x"]["api_context"],
        },
        "y_true": {"calls": calls},
        "label_evidence": {
            "argument_value_grounding_by_api": grounding,
        },
        "metadata": dict(_mapping(master, "metadata")),
    }
    return phase2, phase3


def render_rest_api_list_example(
    row: Mapping[str, Any],
) -> RenderedContractExample:
    """Render one Phase 2 row from its YAML-owned prompt template."""
    task = resolve_sft_task("text_to_rest_api_list")
    _require_task(row, task.name)
    x = _mapping(row, "x")
    _require_exact_fields(x, {"text", "api_context"}, "Phase 2 x")
    y_true = _mapping(row, "y_true")
    target = {"rest_api_list": parse_rest_api_list_y_pred(y_true)}
    _validate_phase2_context_policy(row, task.input_policy)
    prompt = task.render_prompt({
        "text": _required_text(x.get("text")),
        "api_context": canonical_json(x.get("api_context", [])),
    })
    completion = f"{canonical_json(target)}\n"
    return RenderedContractExample(prompt, completion, len(prompt))


def render_call_example(row: Mapping[str, Any]) -> RenderedContractExample:
    """Render one Phase 3 row from its YAML-owned prompt template."""
    task = resolve_sft_task("text_and_rest_api_list_to_calls")
    _require_task(row, task.name)
    x = _mapping(row, "x")
    _require_exact_fields(
        x,
        {"text", "rest_api_list", "api_context"},
        "Phase 3 x",
    )
    y_true = _mapping(row, "y_true")
    contexts = _contexts_from_raw(x.get("api_context", []))
    if not contexts:
        raise ValueError("Phase 3 requires non-empty API context evidence")
    calls = parse_calls_y_pred(y_true, contexts=contexts)
    rest_api_list = _canonical_unique_strings(
        x.get("rest_api_list", []),
        "rest_api_list",
    )
    if {call["rest_api"] for call in calls} != set(rest_api_list):
        raise ValueError("Phase 3 calls must map exactly to x.rest_api_list")
    prompt = task.render_prompt({
        "text": _required_text(x.get("text")),
        "rest_api_list": canonical_json(
            rest_api_list
        ),
        "api_context": canonical_json([context.to_dict() for context in contexts]),
    })
    completion = f"{canonical_json({'calls': calls})}\n"
    return RenderedContractExample(prompt, completion, len(prompt))


def render_phase2_sft(row: Mapping[str, Any]) -> tuple[str, str]:
    """Return Phase 2 ``(prompt, completion)`` for the shared dataset."""
    rendered = render_rest_api_list_example(row)
    return rendered.prompt, rendered.target_json


def render_phase3_sft(row: Mapping[str, Any]) -> tuple[str, str]:
    """Return Phase 3 ``(prompt, completion)`` for the shared dataset."""
    rendered = render_call_example(row)
    return rendered.prompt, rendered.target_json


def parse_rest_api_list_y_pred(y_pred: Mapping[str, Any] | str) -> list[str]:
    """Parse a strict Phase 2 prediction and return its canonical API set."""
    value = _prediction_mapping(y_pred)
    if set(value) != {"rest_api_list"}:
        raise ValueError("Phase 2 output must contain exactly rest_api_list")
    return _canonical_unique_strings(value["rest_api_list"], "rest_api_list")


def parse_calls_y_pred(
    y_pred: Mapping[str, Any] | str,
    *,
    contexts: Sequence[RedfishContext] | None = None,
) -> list[dict[str, Any]]:
    """Parse a strict Phase 3 call set and validate context method legality."""
    value = _prediction_mapping(y_pred)
    if set(value) != {"calls"} or not isinstance(value["calls"], list):
        raise ValueError("Phase 3 output must contain exactly calls:list")
    by_api = _contexts_by_api(contexts or ())
    parsed: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_call in value["calls"]:
        if not isinstance(raw_call, Mapping) or set(raw_call) != CALL_FIELDS:
            raise ValueError(f"each call must contain exactly {sorted(CALL_FIELDS)}")
        rest_api = _required_text(raw_call["rest_api"])
        if rest_api in seen:
            raise ValueError(f"duplicate Phase 3 rest_api: {rest_api}")
        seen.add(rest_api)
        http_method = _required_text(raw_call["http_method"])
        if http_method != http_method.upper():
            raise ValueError("http_method must be uppercase")
        if http_method not in HTTP_METHODS:
            raise ValueError(f"unsupported http_method {http_method!r}")
        raw_operation_name = raw_call["operation_name"]
        if raw_operation_name is None:
            operation_name = None
        else:
            operation_name = _required_text(raw_operation_name)
        arguments = raw_call["arguments"]
        if not isinstance(arguments, Mapping):
            raise ValueError("call arguments must be an object")
        arguments = dict(arguments)
        if http_method in {"GET", "HEAD"} and arguments:
            raise ValueError(f"{http_method} call arguments must be empty")
        if contexts is not None:
            if rest_api not in by_api:
                raise ValueError(f"call rest_api is absent from context: {rest_api}")
            allowed = _normalize_methods(by_api[rest_api].allowed_methods)
            if http_method not in allowed:
                raise ValueError(
                    f"http_method {http_method!r} is not allowed for {rest_api!r}"
                )
            declared_operations = _normalize_operation_names(
                by_api[rest_api].operation_names
            )
            if operation_name is not None and operation_name not in declared_operations:
                raise ValueError(
                    f"operation_name {operation_name!r} is not declared for {rest_api!r}"
                )
            if not arguments_match_schema(
                arguments,
                by_api[rest_api].argument_schema,
            ):
                raise ValueError(f"arguments do not match schema for {rest_api!r}")
        parsed.append({
            "rest_api": rest_api,
            "http_method": http_method,
            "operation_name": operation_name,
            "arguments": arguments,
        })
    return sorted(parsed, key=lambda call: call["rest_api"])


def evaluate_rest_api_set(
    expected: Sequence[str],
    predicted: Sequence[str],
) -> dict[str, Any]:
    """Evaluate Phase 2 without giving list order any semantic meaning."""
    expected_set = set(_canonical_unique_strings(expected, "expected"))
    predicted_set = set(_canonical_unique_strings(predicted, "predicted"))
    true_positive = len(expected_set & predicted_set)
    both_empty = not expected_set and not predicted_set
    precision = (
        1.0 if both_empty else
        0.0 if not predicted_set else
        true_positive / len(predicted_set)
    )
    recall = (
        1.0 if both_empty else
        0.0 if not expected_set else
        true_positive / len(expected_set)
    )
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    exact = expected_set == predicted_set
    return {
        "set_exact_match": exact,
        "set_match_rate": 1.0 if exact else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "expected_count": len(expected_set),
        "predicted_count": len(predicted_set),
    }


def evaluate_calls(
    expected_calls: Sequence[Mapping[str, Any]],
    predicted_calls: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Evaluate Phase 3 calls by REST API key, independent of list order."""
    expected = _calls_by_api(expected_calls, "expected")
    predicted = _calls_by_api(predicted_calls, "predicted")
    expected_apis = set(expected)
    predicted_apis = set(predicted)
    common = expected_apis & predicted_apis
    method_matches = sum(
        expected[api]["http_method"] == predicted[api]["http_method"]
        for api in common
    )
    operation_matches = sum(
        expected[api]["operation_name"] == predicted[api]["operation_name"]
        for api in common
    )
    argument_matches = sum(
        expected[api]["arguments"] == predicted[api]["arguments"]
        for api in common
    )
    readonly_apis = {
        api
        for api, call in expected.items()
        if call["http_method"] in {"GET", "HEAD"}
    }
    readonly_empty = sum(
        api in predicted and not predicted[api]["arguments"]
        for api in readonly_apis
    )
    call_exact = expected == predicted
    denominator = max(len(expected), len(predicted))
    return {
        "call_set_exact_match": call_exact,
        "call_set_exact_match_rate": 1.0 if call_exact else 0.0,
        "rest_api_set_match_rate": 1.0 if expected_apis == predicted_apis else 0.0,
        "http_method_exact_match_rate": _rate(method_matches, denominator),
        "operation_name_exact_match_rate": _rate(operation_matches, denominator),
        "arguments_exact_match_rate": _rate(argument_matches, denominator),
        "readonly_empty_arguments_rate": _rate(readonly_empty, len(readonly_apis)),
        "expected_call_count": len(expected),
        "predicted_call_count": len(predicted),
    }


def evaluate_calls_y_pred(
    row: Mapping[str, Any],
    y_pred: Mapping[str, Any] | str,
) -> dict[str, Any]:
    """Parse and score one Phase 3 prediction against its unordered call set."""
    x = _mapping(row, "x")
    contexts = _contexts_from_raw(x.get("api_context", []))
    expected = parse_calls_y_pred(
        _mapping(row, "y_true"),
        contexts=contexts or None,
    )
    try:
        predicted = parse_calls_y_pred(y_pred, contexts=contexts or None)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        return {
            "parsed": False,
            "accepted": False,
            "reason": str(exc),
            "json_parse_rate": 0.0,
            "call_set_exact_match_rate": 0.0,
            "rest_api_set_match_rate": 0.0,
            "http_method_exact_match_rate": 0.0,
            "operation_name_exact_match_rate": 0.0,
            "arguments_exact_match_rate": 0.0,
            "readonly_empty_arguments_rate": 0.0,
            "invalid_method_rate": (
                1.0 if "http_method" in str(exc) and "not allowed" in str(exc) else 0.0
            ),
        }
    metrics = evaluate_calls(expected, predicted)
    return {
        "parsed": True,
        "accepted": bool(metrics["call_set_exact_match"]),
        "reason": "",
        "json_parse_rate": 1.0,
        "invalid_method_rate": 0.0,
        **metrics,
    }


def inference_calls_json(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return the Phase 3 inference handoff without execution-order claims."""
    calls = parse_calls_y_pred(_mapping(row, "y_true"))
    return {"calls": calls}


def _prediction_mapping(value: Mapping[str, Any] | str) -> Mapping[str, Any]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, Mapping):
        raise ValueError("prediction must be an object")
    nested = value.get("y_pred", value)
    if not isinstance(nested, Mapping):
        raise ValueError("y_pred must be an object")
    return nested


def _contexts_by_api(
    contexts: Sequence[RedfishContext],
) -> dict[str, RedfishContext]:
    by_api: dict[str, RedfishContext] = {}
    for context in contexts:
        if not context.rest_api:
            raise ValueError("context rest_api must be non-empty")
        if context.rest_api in by_api:
            raise ValueError(f"duplicate rest_api in context: {context.rest_api}")
        by_api[context.rest_api] = context
    return by_api


def _contexts_from_raw(value: Any) -> list[RedfishContext]:
    if not isinstance(value, list):
        raise ValueError("api_context must be a list")
    contexts: list[RedfishContext] = []
    for raw in value:
        if not isinstance(raw, Mapping):
            raise ValueError("each context must be an object")
        _require_exact_fields(raw, set(CONTEXT_FIELDS), "Redfish context")
        contexts.append(RedfishContext(
            rest_api=_required_text(raw.get("rest_api")),
            allowed_methods=raw.get("allowed_methods", []),
            operation_names=raw.get("operation_names", []),
            argument_schema=_mapping_or_empty(raw.get("argument_schema")),
            json=_mapping_or_empty(raw.get("json")),
        ))
    _contexts_by_api(contexts)
    return contexts


def _canonical_unique_strings(value: Any, label: str) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be list[str]")
    items = [_required_text(item) for item in value]
    if len(items) != len(set(items)):
        raise ValueError(f"{label} must not contain duplicates")
    return sorted(items)


def _normalize_methods(methods: Sequence[str]) -> list[str]:
    if isinstance(methods, (str, bytes)):
        raise ValueError("allowed_methods must be list[str]")
    if not all(
        isinstance(method, str) and method and method == method.upper()
        for method in methods
    ):
        raise ValueError("allowed_methods must contain uppercase non-empty strings")
    normalized = list(methods)
    if len(normalized) != len(set(normalized)):
        raise ValueError("allowed_methods must be unique")
    if not set(normalized) <= HTTP_METHODS:
        raise ValueError("allowed_methods contains an unsupported HTTP method")
    return sorted(set(normalized))


def _normalize_operation_names(names: Sequence[str]) -> list[str]:
    if isinstance(names, (str, bytes)):
        raise ValueError("operation_names must be list[str]")
    if not all(isinstance(name, str) and name.strip() for name in names):
        raise ValueError("operation_names must contain non-empty strings")
    normalized = [name.strip() for name in names]
    if len(normalized) != len(set(normalized)):
        raise ValueError("operation_names must be unique")
    return sorted(normalized)


def arguments_match_schema(arguments: Mapping[str, Any], schema: Any) -> bool:
    """Validate an argument object against the bounded context schema subset."""
    if not isinstance(arguments, Mapping):
        return False
    if not schema:
        return not arguments
    if not isinstance(schema, Mapping):
        return False
    properties = schema.get("properties", schema)
    if not isinstance(properties, Mapping):
        return False
    required = schema.get("required", []) if "properties" in schema else list(properties)
    if not isinstance(required, list) or not all(isinstance(name, str) for name in required):
        return False
    if not set(required) <= set(arguments) or not set(arguments) <= set(properties):
        return False
    return all(
        _value_matches_schema(arguments[name], properties[name])
        for name in arguments
    )


def _value_matches_schema(value: Any, spec: Any) -> bool:
    if isinstance(spec, Mapping) and "enum" in spec:
        enum = spec["enum"]
        if not isinstance(enum, list) or value not in enum:
            return False
    if isinstance(spec, Mapping) and "type" not in spec and "properties" not in spec:
        if not isinstance(value, Mapping) or set(value) != set(spec):
            return False
        return all(_value_matches_schema(value[name], spec[name]) for name in value)
    expected = spec.get("type") if isinstance(spec, Mapping) else spec
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "object":
        if not isinstance(value, Mapping):
            return False
        properties = spec.get("properties", {}) if isinstance(spec, Mapping) else {}
        required = spec.get("required", []) if isinstance(spec, Mapping) else []
        if not isinstance(properties, Mapping) or not isinstance(required, list):
            return False
        if not set(required) <= set(value) or not set(value) <= set(properties):
            return False
        return all(_value_matches_schema(value[name], properties[name]) for name in value)
    if expected == "array":
        if not isinstance(value, list):
            return False
        item_spec = spec.get("items") if isinstance(spec, Mapping) else None
        return item_spec is None or all(_value_matches_schema(item, item_spec) for item in value)
    if expected == "null":
        return value is None
    return False


def _require_context(
    rest_api_list: Sequence[str],
    by_api: Mapping[str, RedfishContext],
) -> None:
    missing = sorted(set(rest_api_list) - set(by_api))
    if missing:
        raise ValueError(f"rest_api not present in current context: {missing}")


def _require_label_keys(
    label: str,
    mapping: Mapping[str, Any],
    rest_api_list: Sequence[str],
) -> None:
    if set(mapping) != set(rest_api_list):
        raise ValueError(f"{label} keys must exactly match rest_api_list")


def _normalize_argument_grounding(
    value: Any,
    *,
    calls: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Validate explicit argument-value grounding keyed by REST API."""
    if not isinstance(value, Mapping):
        raise ValueError("argument value grounding must be an object")
    expected = {call["rest_api"]: call for call in calls}
    if set(value) != set(expected):
        raise ValueError("argument grounding keys must exactly match calls")
    normalized: dict[str, dict[str, Any]] = {}
    for rest_api in sorted(expected):
        item = value[rest_api]
        if not isinstance(item, Mapping) or set(item) != {"grounded", "sources"}:
            raise ValueError(f"argument grounding is invalid for {rest_api!r}")
        grounded = item["grounded"]
        sources = item["sources"]
        if not isinstance(grounded, bool):
            raise ValueError(f"argument grounded must be boolean for {rest_api!r}")
        if (
            not isinstance(sources, list)
            or not all(isinstance(source, str) and source.strip() for source in sources)
            or len(sources) != len(set(sources))
        ):
            raise ValueError(f"argument grounding sources are invalid for {rest_api!r}")
        sources = sorted(source.strip() for source in sources)
        unknown_sources = set(sources) - ARGUMENT_GROUNDING_SOURCES
        if unknown_sources:
            raise ValueError(
                "argument grounding contains unsupported sources: "
                f"{sorted(unknown_sources)}"
            )
        has_arguments = bool(expected[rest_api]["arguments"])
        has_shape_evidence = bool(
            {"argument_schema", "operation_definition"} & set(sources)
        )
        if has_arguments and (
            not grounded
            or "operator_text" not in sources
            or not has_shape_evidence
        ):
            raise ValueError(
                "mutation arguments require operator_text plus argument_schema "
                f"or operation_definition grounding for {rest_api!r}"
            )
        normalized[rest_api] = {
            "grounded": grounded,
            "sources": sources,
        }
    return normalized


def _require_task(row: Mapping[str, Any], task_name: str) -> None:
    if row.get("task") != task_name:
        raise ValueError(f"row task must be {task_name!r}")


def _require_exact_fields(
    value: Mapping[str, Any],
    expected: set[str],
    label: str,
) -> None:
    fields = set(value)
    if fields != expected:
        raise ValueError(
            f"{label} must contain exactly {sorted(expected)}; got {sorted(fields)}"
        )


def _validate_phase2_context_policy(
    row: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> None:
    x = _mapping(row, "x")
    y_true = _mapping(row, "y_true")
    _require_exact_fields(x, {"text", "api_context"}, "Phase 2 x")
    targets = set(parse_rest_api_list_y_pred(y_true))
    contexts = _contexts_from_raw(x.get("api_context", []))
    context_apis = {context.rest_api for context in contexts}
    if bool(policy.get("all_target_apis_present", False)) and not targets <= context_apis:
        raise ValueError("Phase 2 contexts must include every target API")
    min_distractors = int(policy.get("min_distractors_per_row", 0) or 0)
    distractors = context_apis - targets
    if len(distractors) < min_distractors:
        raise ValueError(
            f"Phase 2 contexts require at least {min_distractors} distractors; "
            f"got {len(distractors)}"
        )


def _mapping(row: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = row.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be an object")
    return value


def _mapping_or_empty(value: Any) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("value must be an object")
    return value


def _required_text(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("value must be a non-empty string")
    return value.strip()


def _calls_by_api(
    calls: Sequence[Mapping[str, Any]],
    label: str,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for call in calls:
        if not isinstance(call, Mapping) or set(call) != CALL_FIELDS:
            raise ValueError(f"{label} call has invalid fields")
        api = _required_text(call["rest_api"])
        if api in result:
            raise ValueError(f"{label} calls contain duplicate rest_api {api!r}")
        result[api] = dict(call)
    return result


def _rate(matches: int, total: int) -> float:
    return 1.0 if total == 0 else matches / total


__all__ = (
    "ARGUMENT_GROUNDING_SOURCES",
    "CALL_FIELDS",
    "CONTEXT_FIELDS",
    "D0",
    "D1",
    "HTTP_METHODS",
    "MODEL_X",
    "PHASE2_GOAL_EXTRACT_METRIC_KEYS",
    "PHASE3_ARGUMENT_EXTRACT_METRIC_KEYS",
    "RedfishContext",
    "RenderedContractExample",
    "arguments_match_schema",
    "build_call_row",
    "build_d1_master_record",
    "build_d1_rest_api_list_row",
    "canonical_json",
    "d1_row_id",
    "evaluate_calls",
    "evaluate_calls_y_pred",
    "evaluate_rest_api_set",
    "inference_calls_json",
    "parse_calls_y_pred",
    "parse_rest_api_list_y_pred",
    "render_call_example",
    "render_d1_master_views",
    "render_phase2_sft",
    "render_phase3_sft",
    "render_rest_api_list_example",
)
