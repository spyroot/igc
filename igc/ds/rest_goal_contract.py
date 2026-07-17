"""Mock REST-goal schema fixtures for Phase 2/3 compatibility tests.

Used by tests and future dataset builders as the narrow mock-plumbing seam for
the Phase 2/3 Redfish instruction contracts: text/context to REST APIs, then
text/API list/context to method/argument calls. The production Phase 2
``phase2_labelled_requests`` builder owns prompt/model/judge config; this module
owns tiny schema examples, canonical renderers, and shared metric-key re-exports
only. It does not train, decode, crawl, judge, or infer labels from text.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from igc.modules.base.metric_keys import (
    PHASE2_LABELLED_REQUESTS,
    PHASE2_WANDB_METRIC_KEYS,
    PHASE3_WANDB_METRIC_KEYS,
)


MODEL_X = "model_x"
D0 = "D0"

PHASE2_GOAL_EXTRACT_METRIC_KEYS = PHASE2_WANDB_METRIC_KEYS
PHASE3_ARGUMENT_EXTRACT_METRIC_KEYS = PHASE3_WANDB_METRIC_KEYS


@dataclass(frozen=True)
class RedfishContext:
    """Current Redfish resource/method context for one REST API.

    :param rest_api: concrete Redfish URI supplied by the current discovery context.
    :param allowed_methods: HTTP methods legal for ``rest_api`` from the same method map.
    :param json: full Redfish JSON resource body observed for ``rest_api``.
    """

    rest_api: str  # Concrete Redfish URI used as a legal REST target.
    allowed_methods: Sequence[str]  # HTTP methods legal on this URI in the current context.
    json: Mapping[str, Any]  # Full Redfish JSON body paired with this URI.

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the locked field names used by the Phase 2/3 rows."""
        return {
            "rest_api": self.rest_api,                 # One concrete Redfish URI.
            "allowed_methods": list(self.allowed_methods),  # Legal methods for this URI.
            "json": dict(self.json),                   # Resource body supplied as context.
        }


@dataclass(frozen=True)
class RenderedContractExample:
    """Prompt/target split for causal-LM fine-tuning examples.

    :param prompt: rendered ``x`` context, excluding the target completion.
    :param target_json: canonical JSON string rendered from ``y_true``.
    :param target_char_start: character offset where ``y_true`` begins in ``full_text``.
    """

    prompt: str  # Rendered input context only.
    target_json: str  # Canonical serialized target completion.
    target_char_start: int  # Boundary used by tokenizers to mask prompt tokens later.

    @property
    def full_text(self) -> str:
        """Return the text shown to a causal LM before tokenization."""
        return self.prompt + self.target_json


def _canonical_json(value: Mapping[str, Any] | Sequence[Any]) -> str:
    """Render deterministic JSON for training labels and golden tests."""
    return json.dumps(value, indent=2, sort_keys=True)


def _contexts_by_api(contexts: Sequence[RedfishContext]) -> dict[str, RedfishContext]:
    """Index contexts by ``rest_api`` and reject ambiguous duplicates."""
    by_api: dict[str, RedfishContext] = {}
    for context in contexts:
        if context.rest_api in by_api:
            raise ValueError(f"duplicate rest_api in context: {context.rest_api}")
        by_api[context.rest_api] = context
    return by_api


def _require_context(rest_api_list: Sequence[str], by_api: Mapping[str, RedfishContext]) -> None:
    """Reject labels that name APIs absent from the current context."""
    missing = [rest_api for rest_api in rest_api_list if rest_api not in by_api]
    if missing:
        raise ValueError(f"rest_api not present in current context: {missing}")


def _allowed_methods_map(contexts: Sequence[RedfishContext]) -> dict[str, list[str]]:
    """Build the locked ``allowed_methods`` map from current context rows."""
    return {
        context.rest_api: [method.upper() for method in context.allowed_methods]
        for context in contexts
    }


def build_phase2_labelled_request_row(
    *,
    text: str,
    contexts: Sequence[RedfishContext],
    rest_api_list: Sequence[str],
    order_evidence: str = "explicit_then",
) -> dict[str, Any]:
    """Build one mock ``phase2_labelled_requests`` schema row.

    :param text: operator sentence.
    :param contexts: current Redfish JSON/method context.
    :param rest_api_list: target REST APIs, ordered only for explicit-order fixtures.
    :param order_evidence: label describing why order should be evaluated strictly.
    :return: JSON-compatible Phase 2 row with locked field names.
    """
    by_api = _contexts_by_api(contexts)
    _require_context(rest_api_list, by_api)
    return {
        "phase": 2,                         # Phase 2: text -> rest_api_list.
        "dataset": PHASE2_LABELLED_REQUESTS,  # Canonical Phase 2 dataset name.
        "source_dataset": D0,               # D0 is the Phase 1 JSON reconstruction source.
        "model_x": MODEL_X,                 # model_x creates/reviews labels after Phase 1.
        "task": "text_to_rest_api_list",    # Contract name from the phase workflow.
        "x": {
            "text": text,                   # Operator sentence shown to the model.
            "json": [dict(context.json) for context in contexts],  # Current resource bodies.
            "allowed_methods": _allowed_methods_map(contexts),  # Method legality context.
        },
        "y_true": {
            "rest_api_list": list(rest_api_list),  # Ordered API label, never sorted.
            "order_evidence": order_evidence,     # Whether strict order evidence is explicit.
        },
        "validation": {
            "text_source": "mock_fixture",         # Tiny offline fixture, not real generation.
            "review_judged": False,                # Real review waits for model_x checkpoint.
            "all_rest_api_present": True,          # All labels are present in current context.
            "extra_rest_api_present": False,       # The mock row carries only requested APIs.
            "order_preserved": True,               # The label keeps the caller-provided order.
        },
    }


def _default_method(allowed_methods: Sequence[str]) -> str:
    """Pick the safe default method from the context method set."""
    normalized = [method.upper() for method in allowed_methods]
    if "GET" in normalized:
        return "GET"
    if normalized:
        return normalized[0]
    return "GET"


def build_ordered_call_row(
    *,
    text: str,
    contexts: Sequence[RedfishContext],
    rest_api_list: Sequence[str],
    method_by_api: Mapping[str, str] | None = None,
    arguments_by_api: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build one mock Phase 3 row for ordered method/argument extraction.

    :param text: operator sentence.
    :param contexts: current Redfish JSON/method context.
    :param rest_api_list: ordered REST APIs emitted by Phase 2.
    :param method_by_api: optional explicit method labels by API.
    :param arguments_by_api: optional explicit argument labels by API.
    :return: JSON-compatible Phase 3 row with ordered calls.
    """
    method_by_api = method_by_api or {}
    arguments_by_api = arguments_by_api or {}
    by_api = _contexts_by_api(contexts)
    _require_context(rest_api_list, by_api)

    calls: list[dict[str, Any]] = []
    for rest_api in rest_api_list:
        context = by_api[rest_api]
        allowed_methods = [method.upper() for method in context.allowed_methods]
        method = method_by_api.get(rest_api, _default_method(allowed_methods)).upper()
        if method not in allowed_methods:
            raise ValueError(f"method {method} is not in allowed_methods for {rest_api}")
        explicit_arguments = dict(arguments_by_api.get(rest_api) or {})
        arguments = {} if method in ("GET", "HEAD") else explicit_arguments
        calls.append({
            "rest_api": rest_api,             # Ordered REST API copied from rest_api_list.
            "allowed_methods": allowed_methods,  # Legal methods for this API.
            "method": method,                 # Selected method label for the call.
            "arguments": arguments,           # Explicit body/action args; never inferred.
        })

    return {
        "phase": 3,                           # Phase 3: ordered APIs -> ordered calls.
        "source_dataset": PHASE2_LABELLED_REQUESTS,  # Phase 3 starts from accepted labels.
        "model_x": MODEL_X,                   # model_x is the Phase 1 checkpoint lineage.
        "task": "text_and_rest_api_list_to_calls",  # Contract name from the workflow.
        "x": {
            "text": text,                     # Operator sentence shown to Phase 3.
            "rest_api_list": list(rest_api_list),  # Ordered API input from Phase 2.
            "json": [dict(context.json) for context in contexts],  # Current resource bodies.
            "allowed_methods": _allowed_methods_map(contexts),  # Method legality context.
        },
        "y_true": {
            "calls": calls,                   # Ordered call labels with methods and args.
        },
    }


def render_rest_api_list_example(row: Mapping[str, Any]) -> RenderedContractExample:
    """Render a Phase 2 row into prompt and target JSON text.

    :param row: row from :func:`build_phase2_labelled_request_row`.
    :return: rendered prompt/target split.
    """
    x = row["x"]
    target = {"rest_api_list": list(row["y_true"]["rest_api_list"])}
    prompt = (
        "### Operator Text\n"
        f"{x['text']}\n\n"
        "### Current Redfish JSON\n"
        f"{_canonical_json(x['json'])}\n\n"
        "### Allowed Methods\n"
        f"{_canonical_json(x['allowed_methods'])}\n\n"
        "### Ordered REST API List\n"
    )
    return RenderedContractExample(
        prompt=prompt,
        target_json=_canonical_json(target),
        target_char_start=len(prompt),
    )


def render_ordered_call_example(row: Mapping[str, Any]) -> RenderedContractExample:
    """Render a Phase 3 row into prompt and target JSON text.

    :param row: row from :func:`build_ordered_call_row`.
    :return: rendered prompt/target split.
    """
    x = row["x"]
    target = {"calls": list(row["y_true"]["calls"])}
    prompt = (
        "### Operator Text\n"
        f"{x['text']}\n\n"
        "### Ordered REST API List\n"
        f"{_canonical_json(x['rest_api_list'])}\n\n"
        "### Current Redfish JSON\n"
        f"{_canonical_json(x['json'])}\n\n"
        "### Allowed Methods\n"
        f"{_canonical_json(x['allowed_methods'])}\n\n"
        "### Ordered REST Calls\n"
    )
    return RenderedContractExample(
        prompt=prompt,
        target_json=_canonical_json(target),
        target_char_start=len(prompt),
    )


def parse_rest_api_list_y_pred(y_pred: Mapping[str, Any] | str) -> list[str]:
    """Parse Phase 2 model output into an ordered ``rest_api_list``.

    :param y_pred: model output as a mapping or JSON string.
    :return: ordered REST API list.
    """
    if isinstance(y_pred, str):
        y_pred = json.loads(y_pred)
    if not isinstance(y_pred, Mapping):
        raise ValueError("y_pred must be an object")
    value = y_pred.get("y_pred", y_pred)
    if not isinstance(value, Mapping):
        raise ValueError("y_pred.y_pred must be an object")
    rest_api_list = value.get("rest_api_list")
    if not isinstance(rest_api_list, list):
        raise ValueError("y_pred.rest_api_list must be a list")
    if not all(isinstance(rest_api, str) for rest_api in rest_api_list):
        raise ValueError("each y_pred.rest_api_list item must be a string")
    return list(rest_api_list)


def parse_ordered_calls_y_pred(y_pred: Mapping[str, Any] | str) -> list[dict[str, Any]]:
    """Parse Phase 3 model output into ordered call dictionaries.

    :param y_pred: model output as a mapping or JSON string.
    :return: ordered calls with ``rest_api``, ``allowed_methods``, ``method``, and ``arguments``.
    """
    if isinstance(y_pred, str):
        y_pred = json.loads(y_pred)
    if not isinstance(y_pred, Mapping):
        raise ValueError("y_pred must be an object")
    value = y_pred.get("y_pred", y_pred)
    if not isinstance(value, Mapping):
        raise ValueError("y_pred.y_pred must be an object")
    calls = value.get("calls")
    if not isinstance(calls, list):
        raise ValueError("y_pred.calls must be a list")
    parsed: list[dict[str, Any]] = []
    for call in calls:
        if not isinstance(call, Mapping):
            raise ValueError("each y_pred.calls item must be an object")
        missing = [
            field
            for field in ("rest_api", "allowed_methods", "method", "arguments")
            if field not in call
        ]
        if missing:
            raise ValueError(f"y_pred.calls item missing required field(s): {missing}")
        if not isinstance(call["rest_api"], str):
            raise ValueError("y_pred.calls.rest_api must be a string")
        if not isinstance(call["allowed_methods"], list):
            raise ValueError("y_pred.calls.allowed_methods must be a list")
        if not all(isinstance(method, str) for method in call["allowed_methods"]):
            raise ValueError("each y_pred.calls.allowed_methods item must be a string")
        if not isinstance(call["method"], str):
            raise ValueError("y_pred.calls.method must be a string")
        allowed_methods = [method.upper() for method in call["allowed_methods"]]
        method = call["method"].upper()
        if method not in allowed_methods:
            raise ValueError(
                f"y_pred.calls.method {method} is not in allowed_methods "
                f"for {call['rest_api']}"
            )
        if not isinstance(call["arguments"], Mapping):
            raise ValueError("y_pred.calls.arguments must be an object")
        arguments = dict(call["arguments"])
        if method in ("GET", "HEAD") and arguments:
            raise ValueError("read-only y_pred.calls.arguments must be empty")
        parsed.append({
            "rest_api": call["rest_api"],
            "allowed_methods": allowed_methods,
            "method": method,
            "arguments": arguments,
        })
    return parsed


def evaluate_ordered_calls_y_pred(
    row: Mapping[str, Any],
    y_pred: Mapping[str, Any] | str,
) -> dict[str, Any]:
    """Evaluate one Phase 3 prediction against a row's ordered calls.

    :param row: row from :func:`build_ordered_call_row`.
    :param y_pred: model output as a mapping or JSON string.
    :return: parse status plus strict ordered-call comparison metrics.
    """
    expected_calls = list(row["y_true"]["calls"])
    try:
        predicted_calls = parse_ordered_calls_y_pred(y_pred)
    except json.JSONDecodeError as exc:
        return _failed_ordered_call_evaluation(expected_calls, f"invalid_json: {exc.msg}")
    except ValueError as exc:
        return _failed_ordered_call_evaluation(expected_calls, str(exc))
    return evaluate_ordered_calls(expected_calls, predicted_calls)


def evaluate_ordered_calls(
    expected_calls: Sequence[Mapping[str, Any]],
    predicted_calls: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare expected and predicted Phase 3 calls without truncating extras.

    :param expected_calls: target call sequence.
    :param predicted_calls: parsed prediction call sequence.
    :return: strict ordered-call comparison metrics.
    """
    expected = [dict(call) for call in expected_calls]
    predicted = [dict(call) for call in predicted_calls]
    comparison_count = max(len(expected), len(predicted))
    paired_count = min(len(expected), len(predicted))

    rest_api_matches = 0
    allowed_methods_matches = 0
    method_matches = 0
    argument_matches = 0
    for index in range(paired_count):
        expected_call = expected[index]
        predicted_call = predicted[index]
        if predicted_call.get("rest_api") == expected_call.get("rest_api"):
            rest_api_matches += 1
        if predicted_call.get("allowed_methods") == expected_call.get("allowed_methods"):
            allowed_methods_matches += 1
        if predicted_call.get("method") == expected_call.get("method"):
            method_matches += 1
        if predicted_call.get("arguments") == expected_call.get("arguments"):
            argument_matches += 1

    readonly_expected = [
        call
        for call in expected
        if str(call.get("method", "")).upper() in ("GET", "HEAD")
    ]
    readonly_empty_matches = 0
    for index, expected_call in enumerate(expected[:paired_count]):
        expected_method = str(expected_call.get("method", "")).upper()
        predicted_call = predicted[index]
        predicted_method = str(predicted_call.get("method", "")).upper()
        if (
            expected_method in ("GET", "HEAD")
            and predicted_method in ("GET", "HEAD")
            and not predicted_call.get("arguments")
        ):
            readonly_empty_matches += 1

    ordered_exact = expected == predicted
    return {
        "parsed": True,
        "parse_error": "",
        "expected_call_count": len(expected),
        "predicted_call_count": len(predicted),
        "call_count_match": len(expected) == len(predicted),
        "call_ordered_exact_match": ordered_exact,
        "call_ordered_exact_match_rate": 1.0 if ordered_exact else 0.0,
        "call_order_correct_rate": _comparison_rate(rest_api_matches, comparison_count),
        "rest_api_exact_match_rate": _comparison_rate(rest_api_matches, comparison_count),
        "allowed_methods_exact_match_rate": _comparison_rate(
            allowed_methods_matches,
            comparison_count,
        ),
        "method_exact_match_rate": _comparison_rate(method_matches, comparison_count),
        "arguments_exact_match_rate": _comparison_rate(argument_matches, comparison_count),
        "arguments_json_parse_rate": 1.0,
        "invalid_method_rate": 0.0,
        "readonly_empty_arguments_rate": _comparison_rate(
            readonly_empty_matches,
            len(readonly_expected),
        ),
    }


def _failed_ordered_call_evaluation(
    expected_calls: Sequence[Mapping[str, Any]],
    parse_error: str,
) -> dict[str, Any]:
    """Return a zeroed comparison result for an unparseable Phase 3 prediction."""
    invalid_method = "not in allowed_methods" in parse_error
    return {
        "parsed": False,
        "parse_error": parse_error,
        "expected_call_count": len(expected_calls),
        "predicted_call_count": 0,
        "call_count_match": False,
        "call_ordered_exact_match": False,
        "call_ordered_exact_match_rate": 0.0,
        "call_order_correct_rate": 0.0,
        "rest_api_exact_match_rate": 0.0,
        "allowed_methods_exact_match_rate": 0.0,
        "method_exact_match_rate": 0.0,
        "arguments_exact_match_rate": 0.0,
        "arguments_json_parse_rate": 0.0,
        "invalid_method_rate": 1.0 if invalid_method else 0.0,
        "readonly_empty_arguments_rate": 0.0,
    }


def _comparison_rate(matches: int, total: int) -> float:
    """Return a comparison rate, treating two empty sequences as a perfect match."""
    if total == 0:
        return 1.0
    return matches / total


def inference_ordered_goals_json(row: Mapping[str, Any]) -> dict[str, Any]:
    """Build the combined inference JSON handoff from a Phase 3 row.

    :param row: row from :func:`build_ordered_call_row`.
    :return: ``{"text": ..., "ordered_goals": [...]}``.
    """
    return {
        "text": str(row["x"]["text"]),        # Operator sentence tied to the call sequence.
        "ordered_goals": list(row["y_true"]["calls"]),  # Ordered calls for the RL handoff.
    }


# Author: Mus mbayramo@stanford.edu
