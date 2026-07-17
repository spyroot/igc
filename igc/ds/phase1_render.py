"""Canonical Phase 1 prompt and target rendering.

Phase 1 trains on a prompt followed by the whole target Redfish JSON document.
Producer jobs, offline gates, and the tokenizer bridge should share this module
so they stay on-distribution when prompt wording evolves.
"""
from __future__ import annotations

import json
from typing import Any, Mapping, Sequence


def render_phase1_prompt(example: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    """Render a Phase 1 prompt and return the target JSON object.

    Accepted rows are the explicit ``x``/``y_true`` schema used by the Phase 1
    corpus builder and the legacy normalized corpus rows that carry
    ``request_or_action`` plus ``response``.
    """

    rest_api, allowed_methods, input_json, target_json = _phase1_fields(example)
    allowed = ", ".join(allowed_methods) if allowed_methods else "UNKNOWN"
    prompt = (
        "### REST API\n"
        f"{rest_api}\n\n"
        "### Allowed Methods\n"
        f"{allowed}\n\n"
        "### Redfish JSON Input\n"
        f"{phase1_json_dumps(input_json)}\n\n"
        "### Complete Redfish JSON\n"
    )
    return prompt, target_json


def render_phase1_completion(target_json: Mapping[str, Any]) -> str:
    """Render the Phase 1 completion exactly as the trainer labels it."""

    return f"{phase1_json_dumps(target_json)}\n"


def phase1_json_dumps(value: Any) -> str:
    """Stable pretty JSON rendering for Phase 1 documents."""

    return json.dumps(value or {}, indent=2, sort_keys=True)


def _phase1_fields(
        example: Mapping[str, Any]) -> tuple[str, list[str], dict[str, Any], dict[str, Any]]:
    """Extract Phase 1 fields from explicit or normalized corpus rows."""

    x = example.get("x")
    y_true = example.get("y_true")
    if isinstance(x, Mapping) and isinstance(y_true, Mapping):
        input_json = x.get("json", {})
        target_json = y_true.get("json", input_json)
        rest_api = str(x.get("rest_api") or _odata_id(input_json))
        methods = _methods(x.get("allowed_methods", []))
        return rest_api, methods, dict(input_json), dict(target_json)

    action = example.get("request_or_action", {}) or {}
    response = example.get("response", {}) or {}
    rest_api = str(action.get("url") or _odata_id(response))
    methods = _methods(example.get("allowed_methods", []))
    return rest_api, methods, dict(response), dict(response)


def _methods(value: Any) -> list[str]:
    """Normalize an allowed-method field to uppercase strings."""

    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, Sequence):
        return []
    return [str(method).upper() for method in value]


def _odata_id(body: Any) -> str:
    """Best-effort ``@odata.id`` extraction from a JSON body."""

    return str(body.get("@odata.id", "")) if isinstance(body, Mapping) else ""


__all__ = (
    "phase1_json_dumps",
    "render_phase1_completion",
    "render_phase1_prompt",
)
