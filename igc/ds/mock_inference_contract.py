"""Phase 2/3 mock rows that reuse the Phase 1 ``x``/``y_true`` contract.

Used by ``tests/ds/test_mock_inference_contract.py`` to pin the offline-only row
shape while real Phase 2/3 examples wait for a Phase 1 ``model_x`` checkpoint. The
module keeps the shared renderer split: ``render_model_x`` renders only ``x`` and
``render_y_true`` renders the completion that a tokenizer label mask should train on.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

import json as json_lib
from collections.abc import Iterable, Mapping
from typing import Any, TypedDict

_MODEL_X = "model_x"
_ODATA_ID = "@odata.id"
_ACTION_ALLOWABLE_SUFFIX = "@Redfish.AllowableValues"
_URI_CHARS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789/_-")


class Phase2Target(TypedDict):
    """Phase 2 target/prediction payload."""

    rest_api_list: list[str]  # ordered REST endpoints chosen from the current JSON context


class Phase3Call(TypedDict):
    """One selected Phase 3 REST call."""

    rest_api: str  # endpoint to call, copied from the ordered rest_api_list input
    allowed_methods: list[str]  # HTTP methods legal on this endpoint from discovery
    method: str  # selected HTTP method for this call
    arguments: dict[str, Any]  # selected method arguments; read-only GET calls use {}


class Phase3Target(TypedDict):
    """Phase 3 target/prediction payload."""

    calls: list[Phase3Call]  # ordered calls, one call object per REST endpoint


class Phase2X(TypedDict):
    """Phase 2 input context, rendered with the same split as Phase 1."""

    text: str  # operator sentence shown to the model
    rest_api: str  # current REST endpoint whose JSON response supplies context
    method: str  # current HTTP method for the context row, usually GET in mock data
    json: list[dict[str, Any]]  # current Redfish JSON bodies used to discover candidates
    allowed_methods: dict[str, list[str]]  # endpoint -> legal HTTP methods from discovery


class Phase3X(Phase2X):
    """Phase 3 input context, extending Phase 2 with ordered endpoints."""

    rest_api_list: list[str]  # ordered REST endpoints emitted by Phase 2 and consumed here


class Phase2MockRow(TypedDict):
    """Serialized Phase 2 mock row."""

    model_x: str  # locked placeholder for the Phase 1 Redfish-tuned model
    x: Phase2X  # rendered input context; labels should mask these prompt tokens
    y_true: Phase2Target  # expected ordered rest_api_list completion
    y_pred: Phase2Target  # prediction slot; empty in committed mock fixtures


class Phase3MockRow(TypedDict):
    """Serialized Phase 3 mock row."""

    model_x: str  # locked placeholder for the Phase 1 Redfish-tuned model
    x: Phase3X  # rendered input context; labels should mask these prompt tokens
    y_true: Phase3Target  # expected ordered calls as the completion
    y_pred: Phase3Target  # prediction slot; empty in committed mock fixtures


_D0_JSON = {
    "@odata.id": "/redfish/v1",
    "Links": [
        {"@odata.id": "/redfish/v1/Systems/1"},
        {"@odata.id": "/redfish/v1/Managers/1"},
    ],
}

D0: tuple[Phase2MockRow, ...] = (
    {
        "model_x": _MODEL_X,
        "x": {
            "text": "Read /redfish/v1/Managers/1 first, then /redfish/v1/Systems/1.",
            "rest_api": "/redfish/v1",
            "method": "GET",
            "json": [_D0_JSON],
            "allowed_methods": {
                "/redfish/v1": ["GET", "HEAD"],
                "/redfish/v1/Managers/1": ["GET", "HEAD"],
                "/redfish/v1/Systems/1": ["GET", "HEAD"],
            },
        },
        "y_true": {
            "rest_api_list": [
                "/redfish/v1/Managers/1",
                "/redfish/v1/Systems/1",
            ],
        },
        "y_pred": {"rest_api_list": []},
    },
)

_D1_JSON = {
    "@odata.id": "/redfish/v1/Systems/1",
    "Actions": {
        "#ComputerSystem.Reset": {
            "target": "/redfish/v1/Systems/1/Actions/ComputerSystem.Reset",
            "ResetType@Redfish.AllowableValues": [
                "On",
                "GracefulRestart",
            ],
        },
    },
}

D1: tuple[Phase3MockRow, ...] = (
    {
        "model_x": _MODEL_X,
        "x": {
            "text": "Read /redfish/v1/Systems/1, then reset with GracefulRestart at "
                    "/redfish/v1/Systems/1/Actions/ComputerSystem.Reset.",
            "rest_api": "/redfish/v1/Systems/1",
            "method": "GET",
            "json": [_D1_JSON],
            "rest_api_list": [
                "/redfish/v1/Systems/1",
                "/redfish/v1/Systems/1/Actions/ComputerSystem.Reset",
            ],
            "allowed_methods": {
                "/redfish/v1/Systems/1": ["GET", "HEAD"],
                "/redfish/v1/Systems/1/Actions/ComputerSystem.Reset": ["POST"],
            },
        },
        "y_true": {
            "calls": [
                {
                    "rest_api": "/redfish/v1/Systems/1",
                    "allowed_methods": ["GET", "HEAD"],
                    "method": "GET",
                    "arguments": {},
                },
                {
                    "rest_api": "/redfish/v1/Systems/1/Actions/ComputerSystem.Reset",
                    "allowed_methods": ["POST"],
                    "method": "POST",
                    "arguments": {"ResetType": "GracefulRestart"},
                },
            ],
        },
        "y_pred": {"calls": []},
    },
)


def render_model_x(row: Mapping[str, Any]) -> str:
    """Render the shared prompt side of a Phase 2/3 mock row.

    :param row: mock row with top-level ``x`` context.
    :return: prompt text whose tokens should be masked in causal-LM labels.
    """
    x = _context(row)
    blocks = [
        ("REST API", str(x.get("rest_api", ""))),
        ("Method", str(x.get("method", ""))),
        ("Allowed Methods", _render_allowed_methods(x)),
        ("Redfish JSON Input", _json_dumps(x.get("json", []))),
    ]
    rest_api_list = x.get("rest_api_list")
    if isinstance(rest_api_list, list):
        blocks.append(("REST API List", _json_dumps(rest_api_list)))
    blocks.append(("Operator Sentence", str(x.get("text", ""))))
    return "\n\n".join(f"### {title}\n{body}" for title, body in blocks)


def render_y_true(row: Mapping[str, Any]) -> str:
    """Render the completion side of a Phase 2/3 mock row.

    :param row: mock row with a ``y_true`` target.
    :return: canonical JSON completion that should remain unmasked in labels.
    """
    return _json_dumps(row.get("y_true", {}))


def predict_phase2_rest_api_list(row: Mapping[str, Any]) -> list[str]:
    """Predict the ordered Phase 2 ``rest_api_list`` using exact URI mentions.

    This deterministic mock helper validates row plumbing only. It discovers candidate
    URIs from the current ``x.json`` context and returns those whose exact URI appears in
    the operator sentence, sorted by first mention.

    :param row: Phase 2 row using shared ``x`` context.
    :return: ordered REST endpoint list.
    """
    x = _context(row)
    text = str(x.get("text", ""))
    candidates = _ordered_unique(_iter_odata_ids(x.get("json")))
    current = x.get("rest_api")
    if isinstance(current, str):
        candidates = _ordered_unique([current, *candidates])

    mentioned = [
        (position, candidate)
        for candidate in candidates
        for position in [_first_exact_uri_mention(text, candidate)]
        if position >= 0
    ]
    mentioned.sort(key=lambda item: item[0])
    return [candidate for _, candidate in mentioned]


def build_phase3_calls(row: Mapping[str, Any]) -> list[Phase3Call]:
    """Build ordered Phase 3 calls from a mock row.

    ``GET`` is preferred for read-only rows and always emits ``arguments={}``. For
    mutating rows, arguments are chosen from ``@Redfish.AllowableValues`` in the current
    ``x.json`` context when the operator sentence names one of the allowable values.

    :param row: Phase 3 row using ordered ``x.rest_api_list`` and ``x.allowed_methods``.
    :return: ordered call objects.
    """
    x = _context(row)
    text = str(x.get("text", ""))
    body = x.get("json", [])
    allowed_by_rest_api = _as_mapping(x.get("allowed_methods"))
    calls: list[Phase3Call] = []

    for rest_api in x.get("rest_api_list", []):
        if not isinstance(rest_api, str):
            continue
        allowed_methods = _normal_methods(allowed_by_rest_api.get(rest_api))
        method = _select_method(allowed_methods)
        arguments = {} if method == "GET" else _arguments_for_action(text, rest_api, body)
        calls.append({
            "rest_api": rest_api,
            "allowed_methods": allowed_methods,
            "method": method,
            "arguments": arguments,
        })
    return calls


def validate_phase2_row(row: Mapping[str, Any]) -> list[str]:
    """Return contract errors for a Phase 2 mock row.

    :param row: candidate Phase 2 row.
    :return: list of human-readable errors, empty when valid.
    """
    errors = _missing_or_extra_errors(row, {"model_x", "x", "y_true", "y_pred"})
    errors.extend(_validate_model_x(row))
    errors.extend(_validate_context(row, {"text", "rest_api", "method", "json", "allowed_methods"}))
    if not _target_list(row.get("y_true")):
        errors.append("y_true.rest_api_list must be a non-empty list[str]")
    if _as_mapping(row.get("y_pred")).get("rest_api_list") != []:
        errors.append("y_pred.rest_api_list must be empty for committed mock rows")
    return errors


def validate_phase3_row(row: Mapping[str, Any]) -> list[str]:
    """Return contract errors for a Phase 3 mock row.

    :param row: candidate Phase 3 row.
    :return: list of human-readable errors, empty when valid.
    """
    errors = _missing_or_extra_errors(row, {"model_x", "x", "y_true", "y_pred"})
    errors.extend(_validate_model_x(row))
    errors.extend(_validate_context(
        row,
        {"text", "rest_api", "method", "json", "allowed_methods", "rest_api_list"},
    ))
    if _as_mapping(row.get("y_pred")).get("calls") != []:
        errors.append("y_pred.calls must be empty for committed mock rows")
    errors.extend(_validate_phase3_calls(row))
    return errors


def _context(row: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return top-level ``x`` as a mapping."""
    return _as_mapping(row.get("x"))


def _as_mapping(value: object) -> Mapping[str, Any]:
    """Return *value* as a mapping or an empty mapping."""
    return value if isinstance(value, Mapping) else {}


def _json_dumps(value: object) -> str:
    """Return canonical, human-readable JSON text."""
    return json_lib.dumps(value, indent=2, sort_keys=True)


def _render_allowed_methods(x: Mapping[str, Any]) -> str:
    """Render allowed methods for the current endpoint, falling back to the full map."""
    allowed_methods = _as_mapping(x.get("allowed_methods"))
    rest_api = x.get("rest_api")
    methods = allowed_methods.get(rest_api)
    if isinstance(methods, list):
        return ", ".join(str(method) for method in methods)
    return _json_dumps(allowed_methods)


def _iter_odata_ids(value: object) -> Iterable[str]:
    """Yield ``@odata.id`` strings from nested JSON while preserving encounter order."""
    if isinstance(value, Mapping):
        odata_id = value.get(_ODATA_ID)
        if isinstance(odata_id, str):
            yield odata_id
        for child in value.values():
            yield from _iter_odata_ids(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_odata_ids(child)


def _ordered_unique(values: Iterable[str]) -> list[str]:
    """Return values once, preserving first occurrence order."""
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def _first_exact_uri_mention(text: str, uri: str) -> int:
    """Return first exact URI-token mention, excluding prefix-only matches."""
    start = text.find(uri)
    while start >= 0:
        before = text[start - 1] if start > 0 else ""
        after_index = start + len(uri)
        after = text[after_index] if after_index < len(text) else ""
        before_ok = not before or before not in _URI_CHARS
        after_ok = not after or after not in _URI_CHARS
        if before_ok and after_ok:
            return start
        start = text.find(uri, start + 1)
    return -1


def _target_list(target: object) -> list[str]:
    """Read ``rest_api_list`` from a Phase 2 target-like mapping."""
    data = _as_mapping(target)
    values = data.get("rest_api_list")
    if not isinstance(values, list):
        return []
    return [value for value in values if isinstance(value, str)]


def _missing_or_extra_errors(row: Mapping[str, Any], required: set[str]) -> list[str]:
    """Return schema-key errors for a row."""
    keys = set(row)
    errors: list[str] = []
    missing = sorted(required - keys)
    extra = sorted(keys - required)
    if missing:
        errors.append(f"missing keys: {missing}")
    if extra:
        errors.append(f"extra keys: {extra}")
    return errors


def _validate_model_x(row: Mapping[str, Any]) -> list[str]:
    """Return ``model_x`` errors for a row."""
    return [] if row.get("model_x") == _MODEL_X else ["model_x must be the locked model_x value"]


def _validate_context(row: Mapping[str, Any], required: set[str]) -> list[str]:
    """Return schema errors for a row's ``x`` context."""
    x = row.get("x")
    if not isinstance(x, Mapping):
        return ["x must be a dict context"]
    errors = _missing_or_extra_errors(x, required)
    if not isinstance(x.get("text"), str):
        errors.append("x.text must be the operator sentence string")
    if not isinstance(x.get("json"), list):
        errors.append("x.json must be a list of JSON dicts")
    if not isinstance(x.get("allowed_methods"), dict):
        errors.append("x.allowed_methods must be a dict[str, list[str]]")
    return errors


def _normal_methods(methods: object) -> list[str]:
    """Normalize a method list to uppercase strings."""
    if not isinstance(methods, list):
        return []
    return [str(method).upper() for method in methods]


def _select_method(methods: list[str]) -> str:
    """Choose the deterministic mock method for one endpoint."""
    if "GET" in methods:
        return "GET"
    if "POST" in methods:
        return "POST"
    return methods[0] if methods else ""


def _iter_action_objects(value: object) -> Iterable[Mapping[str, Any]]:
    """Yield Redfish action objects from nested JSON."""
    if isinstance(value, Mapping):
        actions = value.get("Actions")
        if isinstance(actions, Mapping):
            for action in actions.values():
                if isinstance(action, Mapping):
                    yield action
        for child in value.values():
            yield from _iter_action_objects(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_action_objects(child)


def _arguments_for_action(x: str, rest_api: str, body: object) -> dict[str, Any]:
    """Extract deterministic mock arguments for a selected mutating action."""
    for action in _iter_action_objects(body):
        if action.get("target") != rest_api:
            continue
        for key, values in action.items():
            if not key.endswith(_ACTION_ALLOWABLE_SUFFIX) or not isinstance(values, list):
                continue
            argument = key.removesuffix(_ACTION_ALLOWABLE_SUFFIX)
            for value in values:
                if str(value) in x:
                    return {argument: value}
            if values:
                return {argument: values[0]}
    return {}


def _validate_phase3_calls(row: Mapping[str, Any]) -> list[str]:
    """Return call-level errors for a Phase 3 row."""
    errors: list[str] = []
    x = _context(row)
    allowed_by_rest_api = _as_mapping(x.get("allowed_methods"))
    rest_api_list = x.get("rest_api_list", [])
    calls = _as_mapping(row.get("y_true")).get("calls")
    if not isinstance(calls, list) or not calls:
        return ["y_true.calls must be a non-empty list[call]"]

    for index, call in enumerate(calls):
        if not isinstance(call, Mapping):
            errors.append(f"call {index} must be a dict")
            continue
        rest_api = call.get("rest_api")
        method = call.get("method")
        allowed_methods = _normal_methods(call.get("allowed_methods"))
        if rest_api not in rest_api_list:
            errors.append(f"call {index} rest_api must come from x.rest_api_list")
        if allowed_methods != _normal_methods(allowed_by_rest_api.get(rest_api)):
            errors.append(f"call {index} allowed_methods must match x.allowed_methods")
        if method not in allowed_methods:
            errors.append(f"call {index} method must be allowed")
        if method == "GET" and call.get("arguments") != {}:
            errors.append(f"call {index} GET arguments must be {{}}")
    return errors


# Author: Mus mbayramo@stanford.edu
