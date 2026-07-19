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

from igc.modules.base.metric_keys import PHASE2_WANDB_METRIC_KEYS, PHASE3_WANDB_METRIC_KEYS


MODEL_X = "model_x"
D0 = "D0"
D1 = "D1"

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


def _unique_rest_api_list(rest_api_list: Sequence[str]) -> list[str]:
    """Return the REST API list unchanged, raising on any duplicate entry.

    The D1 label is an unordered unique set; a repeated API is a contract
    violation, not something to silently dedupe.
    """
    result: list[str] = []
    seen: set[str] = set()
    for rest_api in rest_api_list:
        rest_api = str(rest_api)
        if rest_api in seen:
            raise ValueError("D1 y_true.rest_api_list must be an unordered unique set")
        seen.add(rest_api)
        result.append(rest_api)
    return result


def _locked_d1_validation(validation: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return the exact one-string/eight-bool D1 validation block."""
    result: dict[str, Any] = {
        "text_source": "mock_fixture",
        "review_judged": False,
        "natural": True,
        "exact_api_coverage": True,
        "extra_intent": False,
        "duplicate_intent": False,
        "ambiguous": False,
        "nonsense": False,
        "method_semantics_valid": True,
    }
    if validation:
        result.update(dict(validation))
    required_keys = {
        "text_source",
        "review_judged",
        "natural",
        "exact_api_coverage",
        "extra_intent",
        "duplicate_intent",
        "ambiguous",
        "nonsense",
        "method_semantics_valid",
    }
    keys = set(result)
    if keys != required_keys:
        missing = sorted(required_keys - keys)
        extra = sorted(keys - required_keys)
        raise ValueError(f"D1 validation keys mismatch missing={missing} extra={extra}")
    if not isinstance(result["text_source"], str):
        raise ValueError("D1 validation.text_source must be a string")
    for key in sorted(required_keys - {"text_source"}):
        if not isinstance(result[key], bool):
            raise ValueError(f"D1 validation.{key} must be a bool")
    return result


def build_d1_rest_api_list_row(
    *,
    text: str,
    contexts: Sequence[RedfishContext],
    rest_api_list: Sequence[str],
    validation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one locked ``D1`` row for text-to-REST-API training.

    :param text: operator sentence.
    :param contexts: current Redfish JSON/method context.
    :param rest_api_list: target REST APIs, stored as an unordered unique set.
    :param validation: optional judge/provenance flags for accepted rows.
    :return: JSON-compatible Phase 2 row with locked field names.
    """
    by_api = _contexts_by_api(contexts)
    _require_context(rest_api_list, by_api)
    api_set = _unique_rest_api_list(rest_api_list)
    return {
        "phase": 2,                       # Phase 2: text -> REST API set.
        "dataset": D1,                    # D1 is the accepted Phase 2 dataset.
        "source_dataset": D0,             # D1 is drafted from D0 context.
        "model_x": MODEL_X,               # model_x creates draft text before judging.
        "task": "text_to_rest_api_list",  # Contract name from the phase workflow.
        "target_semantics": "unordered_unique_set",
        "x": {"text": text},              # Phase 2 training input is text-only.
        "y_true": {"rest_api_list": api_set},
        "validation": _locked_d1_validation(validation),
    }


def build_phase2_labelled_request_row(
    *,
    text: str,
    contexts: Sequence[RedfishContext],
    rest_api_list: Sequence[str],
    validation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compatibility wrapper for the canonical D1 row builder.

    New code should call :func:`build_d1_rest_api_list_row`. This wrapper keeps
    older imports from creating a second D1 shape while callers migrate.
    """
    return build_d1_rest_api_list_row(
        text=text,
        contexts=contexts,
        rest_api_list=rest_api_list,
        validation=validation,
    )


def build_call_row(
    *,
    text: str,
    contexts: Sequence[RedfishContext],
    rest_api_list: Sequence[str],
    method_by_api: Mapping[str, str],
    arguments_by_api: Mapping[str, Mapping[str, Any]] | None = None,
    operation_name_by_api: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build one mock Phase 3 row: an UNORDERED set of bound calls.

    Phase 3 binds each Phase 2 API to an explicit HTTP method and explicit
    arguments. The emitted calls form an unordered unique set (canonical sort is
    dedup identity only, never execution order — order belongs to the RL oracle).
    Methods are explicit per API; there is no inferred default. A Call is
    exactly ``{rest_api, http_method, operation_name, arguments}`` —
    ``operation_name`` names the action/function when one exists and is null
    for plain REST verbs.

    :param text: operator sentence.
    :param contexts: current Redfish JSON/method context.
    :param rest_api_list: REST API set emitted by Phase 2 (unordered, unique).
    :param method_by_api: explicit HTTP method label per API; every selected API
        must have one — a missing method raises instead of defaulting.
    :param arguments_by_api: explicit argument bindings by API. Every mutation
        (non-GET/HEAD) call must have an explicit binding — a no-argument
        action binds ``{}`` explicitly; a missing binding raises instead of
        silently becoming ``{}``.
    :param operation_name_by_api: optional action/function name per API
        (e.g. a Redfish action name); absent APIs carry ``None``.
    :return: JSON-compatible Phase 3 row with the unordered call set.
    """
    arguments_by_api = arguments_by_api or {}
    operation_name_by_api = operation_name_by_api or {}
    by_api = _contexts_by_api(contexts)
    _require_context(rest_api_list, by_api)
    api_set = sorted(_unique_rest_api_list(rest_api_list))

    calls: list[dict[str, Any]] = []
    for rest_api in api_set:
        context = by_api[rest_api]
        allowed_methods = [method.upper() for method in context.allowed_methods]
        if rest_api not in method_by_api:
            raise ValueError(
                f"explicit method required for {rest_api}: methods are never inferred"
            )
        method = method_by_api[rest_api].upper()
        if method not in allowed_methods:
            raise ValueError(f"method {method} is not in allowed_methods for {rest_api}")
        if method in ("GET", "HEAD"):
            if arguments_by_api.get(rest_api):
                raise ValueError(
                    f"read-only {method} arguments must be empty for {rest_api}"
                )
            explicit_arguments: dict[str, Any] = {}
        else:
            if rest_api not in arguments_by_api:
                raise ValueError(
                    f"explicit arguments required for {method} {rest_api}: "
                    "a no-argument action binds {} explicitly"
                )
            explicit_arguments = dict(arguments_by_api[rest_api])
        operation_name = operation_name_by_api.get(rest_api)
        calls.append({
            "rest_api": rest_api,             # One selected REST API from the Phase 2 set.
            "http_method": method,            # Explicit HTTP method label; never inferred.
            "operation_name": operation_name,  # Action/function name, or None for plain verbs.
            "arguments": explicit_arguments,  # Explicit body/action args; {} for reads.
        })

    return {
        "phase": 3,                           # Phase 3: API set -> bound call set.
        "source_dataset": D1,                 # Phase 3 starts from accepted D1 labels.
        "model_x": MODEL_X,                   # model_x is the Phase 1 checkpoint lineage.
        "task": "text_and_rest_api_list_to_calls",  # Contract name from the workflow.
        "target_semantics": "unordered_call_set",   # Calls are a set, not a plan.
        "x": {
            "text": text,                     # Operator sentence shown to Phase 3.
            "rest_api_list": api_set,         # Canonical unique API set from Phase 2.
            "json": [dict(context.json) for context in contexts],  # Current resource bodies.
            "allowed_methods": _allowed_methods_map(contexts),  # Method legality evidence.
        },
        "y_true": {
            "calls": calls,                   # Unordered bound calls: rest_api/http_method/operation_name/arguments.
        },
    }


def render_rest_api_list_example(row: Mapping[str, Any]) -> RenderedContractExample:
    """Render a Phase 2 row into prompt and target JSON text.

    :param row: row from :func:`build_d1_rest_api_list_row`.
    :return: rendered prompt/target split.
    """
    x = row["x"]
    target = {"rest_api_list": list(row["y_true"]["rest_api_list"])}
    prompt = (
        "### Operator Text\n"
        f"{x['text']}\n\n"
        "### REST API Set\n"
    )
    return RenderedContractExample(
        prompt=prompt,
        target_json=_canonical_json(target),
        target_char_start=len(prompt),
    )


def render_call_example(row: Mapping[str, Any]) -> RenderedContractExample:
    """Render a Phase 3 row into prompt and target JSON text.

    :param row: row from :func:`build_call_row`.
    :return: rendered prompt/target split.
    """
    x = row["x"]
    target = {"calls": list(row["y_true"]["calls"])}
    prompt = (
        "### Operator Text\n"
        f"{x['text']}\n\n"
        "### REST API Set\n"
        f"{_canonical_json(x['rest_api_list'])}\n\n"
        "### Current Redfish JSON\n"
        f"{_canonical_json(x['json'])}\n\n"
        "### Allowed Methods\n"
        f"{_canonical_json(x['allowed_methods'])}\n\n"
        "### REST Calls\n"
    )
    return RenderedContractExample(
        prompt=prompt,
        target_json=_canonical_json(target),
        target_char_start=len(prompt),
    )


def parse_rest_api_list_y_pred(y_pred: Mapping[str, Any] | str) -> list[str]:
    """Parse Phase 2 model output into a ``rest_api_list`` (evaluated as a set).

    :param y_pred: model output as a mapping or JSON string.
    :return: REST API list exactly as predicted; set/duplicate checks happen in
        :func:`evaluate_rest_api_list_y_pred`.
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


def evaluate_rest_api_list_y_pred(
    row: Mapping[str, Any],
    y_pred: Mapping[str, Any] | str,
) -> dict[str, Any]:
    """Evaluate a Phase 2 prediction as an unordered unique REST API set.

    :param row: row from :func:`build_d1_rest_api_list_row`.
    :param y_pred: model output as a mapping or JSON string.
    :return: parse status plus duplicate-aware unordered-set metrics.
    """
    expected = list(row["y_true"]["rest_api_list"])
    try:
        predicted = parse_rest_api_list_y_pred(y_pred)
    except json.JSONDecodeError as exc:
        return _failed_rest_api_list_evaluation(expected, f"invalid_json: {exc.msg}")
    except ValueError as exc:
        return _failed_rest_api_list_evaluation(expected, str(exc))

    expected_set = set(expected)
    predicted_set = set(predicted)
    duplicate_prediction = len(predicted) != len(predicted_set)
    set_match = not duplicate_prediction and predicted_set == expected_set
    return {
        "parse_ok": True,
        "error": "",
        "set_match": set_match,
        "duplicate_prediction": duplicate_prediction,
        "missing_rest_api": sorted(expected_set - predicted_set),
        "extra_rest_api": sorted(predicted_set - expected_set),
        "expected_count": len(expected),
        "predicted_count": len(predicted),
    }


def parse_calls_y_pred(y_pred: Mapping[str, Any] | str) -> list[dict[str, Any]]:
    """Parse Phase 3 model output into bound-call dictionaries.

    A Call is exactly ``{rest_api, http_method, operation_name, arguments}`` —
    ``allowed_methods`` is row context evidence, never part of the emitted call.
    ``operation_name`` is optional in the raw output ("when available") and is
    normalized to ``None`` when absent. Method legality against the row evidence
    is checked by :func:`evaluate_calls_y_pred`, which holds the row.

    :param y_pred: model output as a mapping or JSON string.
    :return: calls with ``rest_api``, ``http_method``, ``operation_name``, and
        ``arguments``.
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
            for field in ("rest_api", "http_method", "arguments")
            if field not in call
        ]
        if missing:
            raise ValueError(f"y_pred.calls item missing required field(s): {missing}")
        if not isinstance(call["rest_api"], str):
            raise ValueError("y_pred.calls.rest_api must be a string")
        if not isinstance(call["http_method"], str):
            raise ValueError("y_pred.calls.http_method must be a string")
        method = call["http_method"].upper()
        operation_name = call.get("operation_name")
        if operation_name is not None and not isinstance(operation_name, str):
            raise ValueError("y_pred.calls.operation_name must be a string or null")
        if not isinstance(call["arguments"], Mapping):
            raise ValueError("y_pred.calls.arguments must be an object")
        arguments = dict(call["arguments"])
        if method in ("GET", "HEAD") and arguments:
            raise ValueError("read-only y_pred.calls.arguments must be empty")
        parsed.append({
            "rest_api": call["rest_api"],     # The API this call binds.
            "http_method": method,            # Explicit HTTP method label.
            "operation_name": operation_name,  # Action/function name, or None.
            "arguments": arguments,           # Explicit args; {} for reads.
        })
    return parsed


def evaluate_calls_y_pred(
    row: Mapping[str, Any],
    y_pred: Mapping[str, Any] | str,
) -> dict[str, Any]:
    """Evaluate one Phase 3 prediction against a row's unordered call set.

    :param row: row from :func:`build_call_row`.
    :param y_pred: model output as a mapping or JSON string.
    :return: parse status plus duplicate-aware call-set comparison metrics.
    """
    expected_calls = list(row["y_true"]["calls"])
    allowed_methods_by_api = {
        str(rest_api): [str(method).upper() for method in methods]
        for rest_api, methods in dict(row["x"].get("allowed_methods", {})).items()
    }
    try:
        predicted_calls = parse_calls_y_pred(y_pred)
    except json.JSONDecodeError as exc:
        return _failed_call_evaluation(expected_calls, f"invalid_json: {exc.msg}")
    except ValueError as exc:
        return _failed_call_evaluation(expected_calls, str(exc))
    return evaluate_calls(
        expected_calls,
        predicted_calls,
        allowed_methods_by_api=allowed_methods_by_api,
    )


def evaluate_calls(
    expected_calls: Sequence[Mapping[str, Any]],
    predicted_calls: Sequence[Mapping[str, Any]],
    *,
    allowed_methods_by_api: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    """Compare expected and predicted Phase 3 calls as UNORDERED sets.

    Calls are matched by ``rest_api`` (contract v1: exactly one call per API);
    a duplicated predicted API, a missing API, or an extra API all fail the set
    match. Serialization order never matters — order is RL-oracle evidence.

    :param expected_calls: target call set.
    :param predicted_calls: parsed prediction call set.
    :param allowed_methods_by_api: row method-legality evidence used for
        ``invalid_method_rate``; omitted -> no legality signal (rate 0.0).
    :return: duplicate-aware call-set comparison metrics.
    """
    expected = [dict(call) for call in expected_calls]
    predicted = [dict(call) for call in predicted_calls]
    legality = {
        str(api): [str(method).upper() for method in methods]
        for api, methods in dict(allowed_methods_by_api or {}).items()
    }

    expected_by_api = {str(call.get("rest_api", "")): call for call in expected}
    predicted_apis = [str(call.get("rest_api", "")) for call in predicted]
    predicted_api_set = set(predicted_apis)
    duplicate_prediction = len(predicted_apis) != len(predicted_api_set)
    expected_api_set = set(expected_by_api)
    api_set_match = not duplicate_prediction and predicted_api_set == expected_api_set
    predicted_by_api = {str(call.get("rest_api", "")): call for call in predicted}

    # Per-field agreement over the API intersection (set semantics, not index pairs).
    shared_apis = sorted(expected_api_set & predicted_api_set)
    method_matches = 0
    argument_matches = 0
    call_matches = 0
    for rest_api in shared_apis:
        expected_call = expected_by_api[rest_api]
        predicted_call = predicted_by_api[rest_api]
        method_ok = predicted_call.get("http_method") == expected_call.get("http_method")
        arguments_ok = predicted_call.get("arguments") == expected_call.get("arguments")
        method_matches += 1 if method_ok else 0
        argument_matches += 1 if arguments_ok else 0
        call_matches += 1 if (method_ok and arguments_ok) else 0
    comparison_count = max(len(expected_api_set), len(predicted_api_set))

    # Full set equality: same APIs (no dup/extra/missing) and every shared call binds
    # the same method and the same arguments.
    call_set_exact = api_set_match and call_matches == len(expected_api_set)

    # No-argument accuracy: expected {}-argument calls whose prediction also binds {}.
    no_argument_expected = [
        call for call in expected if not call.get("arguments")
    ]
    no_argument_hits = sum(
        1
        for call in no_argument_expected
        if str(call.get("rest_api", "")) in predicted_by_api
        and not predicted_by_api[str(call.get("rest_api", ""))].get("arguments")
    )

    # Required-argument coverage: expected mutation calls (non-empty arguments) whose
    # prediction carries every expected argument key.
    required_expected = [call for call in expected if call.get("arguments")]
    required_hits = 0
    for call in required_expected:
        rest_api = str(call.get("rest_api", ""))
        predicted_call = predicted_by_api.get(rest_api)
        if predicted_call is None:
            continue
        expected_keys = set(dict(call.get("arguments", {})))
        predicted_keys = set(dict(predicted_call.get("arguments", {})))
        if expected_keys <= predicted_keys:
            required_hits += 1

    # Unsafe-argument rejection: matched predictions that inject NO argument key
    # beyond the expected binding (unsupported/unsafe args must be rejected).
    matched_predictions = [
        (expected_by_api[rest_api], predicted_by_api[rest_api]) for rest_api in shared_apis
    ]
    safe_hits = sum(
        1
        for expected_call, predicted_call in matched_predictions
        if set(dict(predicted_call.get("arguments", {})))
        <= set(dict(expected_call.get("arguments", {})))
    )

    # Method legality against the row's allowed_methods evidence.
    legality_checked = [
        call for call in predicted if str(call.get("rest_api", "")) in legality
    ]
    invalid_methods = sum(
        1
        for call in legality_checked
        if str(call.get("http_method", "")).upper()
        not in legality[str(call.get("rest_api", ""))]
    )

    return {
        "parsed": True,
        "parse_error": "",
        "expected_call_count": len(expected),
        "predicted_call_count": len(predicted),
        "call_count_match": len(expected) == len(predicted),
        "duplicate_prediction": duplicate_prediction,
        "call_set_exact_match": call_set_exact,
        "call_set_exact_match_rate": 1.0 if call_set_exact else 0.0,
        "rest_api_set_match_rate": 1.0 if api_set_match else 0.0,
        "method_exact_match_rate": _comparison_rate(method_matches, comparison_count),
        "arguments_exact_match_rate": _comparison_rate(argument_matches, comparison_count),
        "arguments_json_validity_rate": 1.0,
        "required_argument_coverage_rate": _comparison_rate(
            required_hits,
            len(required_expected),
        ),
        "no_argument_accuracy_rate": _comparison_rate(
            no_argument_hits,
            len(no_argument_expected),
        ),
        "unsafe_argument_rejection_rate": _comparison_rate(
            safe_hits,
            len(matched_predictions),
        ),
        "invalid_method_rate": _comparison_rate(invalid_methods, len(legality_checked))
        if legality_checked
        else 0.0,
    }


def _failed_call_evaluation(
    expected_calls: Sequence[Mapping[str, Any]],
    parse_error: str,
) -> dict[str, Any]:
    """Return a zeroed comparison result for an unparseable Phase 3 prediction."""
    return {
        "parsed": False,
        "parse_error": parse_error,
        "expected_call_count": len(expected_calls),
        "predicted_call_count": 0,
        "call_count_match": False,
        "duplicate_prediction": False,
        "call_set_exact_match": False,
        "call_set_exact_match_rate": 0.0,
        "rest_api_set_match_rate": 0.0,
        "method_exact_match_rate": 0.0,
        "arguments_exact_match_rate": 0.0,
        "arguments_json_validity_rate": 0.0,
        "required_argument_coverage_rate": 0.0,
        "no_argument_accuracy_rate": 0.0,
        "unsafe_argument_rejection_rate": 0.0,
        "invalid_method_rate": 0.0,
    }


def _failed_rest_api_list_evaluation(expected: Sequence[str], error: str) -> dict[str, Any]:
    """Return a stable Phase 2 evaluation result for parse failures."""
    return {
        "parse_ok": False,
        "error": error,
        "set_match": False,
        "duplicate_prediction": False,
        "missing_rest_api": sorted(set(expected)),
        "extra_rest_api": [],
        "expected_count": len(expected),
        "predicted_count": 0,
    }


def _comparison_rate(matches: int, total: int) -> float:
    """Return a comparison rate, treating two empty sequences as a perfect match."""
    if total == 0:
        return 1.0
    return matches / total


# NOTE: there is deliberately NO call-handoff helper here. Phase 3 output is the
# unordered call set above; execution order is RL-oracle training evidence with
# its own shape ({compiled_goal_id, expert_call_order, success_predicate} — see
# configs/contracts/goal_latent.yaml), produced outside the Phase 2/3 contract.


# Author: Mus mbayramo@stanford.edu
