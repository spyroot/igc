"""Canonical Phase 1 prompt and target rendering.

Phase 1 renders a prompt followed by the whole target Redfish JSON document.
The selected training profile decides which completion spans receive loss.
Producer jobs, gates, and the tokenizer bridge share this module so target
bytes remain identical.
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from igc.modules.train.sft_tasks import resolve_sft_task


PHASE1_DATASET = "D0"
PHASE1_TASK = "redfish_json_reconstruction"


@dataclass(frozen=True)
class Phase1JSONSpan:
    """Character span in the canonical Phase 1 JSON rendering."""

    kind: str
    path: tuple[str | int, ...]
    key: str | None
    start: int
    end: int


def build_phase1_row(
    *,
    rest_api: str,
    allowed_methods: Sequence[str],
    input_json: Mapping[str, Any],
    target_json: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one strict D0 row for Phase 1 JSON reconstruction."""
    row = {
        "phase": 1,
        "dataset": PHASE1_DATASET,
        "task": PHASE1_TASK,
        "x": {
            "rest_api": rest_api,
            "allowed_methods": list(allowed_methods),
            "json": copy.deepcopy(dict(input_json)),
        },
        "y_true": {"json": copy.deepcopy(dict(target_json))},
    }
    if metadata is not None:
        row["metadata"] = dict(metadata)
    validate_phase1_row(row)
    return row


def build_phase1_source_row(record: Any) -> dict[str, Any]:
    """Convert one provenance-tagged source record into a canonical D0 row."""
    source = getattr(record, "source", None)
    rest_api = getattr(record, "url", None)
    response = getattr(record, "response", None)
    trust_level = getattr(record, "trust_level", None)
    if not isinstance(source, str) or not source.strip():
        raise ValueError("Phase 1 source record must have a non-empty source")
    if not isinstance(rest_api, str) or not rest_api.strip():
        raise ValueError("Phase 1 source record must have a non-empty URL")
    if not isinstance(response, Mapping):
        raise ValueError("Phase 1 source record response must be an object")
    trust_name = getattr(trust_level, "name", None)
    if not isinstance(trust_name, str) or not trust_name:
        raise ValueError("Phase 1 source record must have a named trust level")
    row_id = "sha256:" + hashlib.sha256(
        f"{source}\0{rest_api}".encode("utf-8")
    ).hexdigest()
    return build_phase1_row(
        rest_api=rest_api,
        allowed_methods=getattr(record, "allowed_methods", None) or [],
        input_json=response,
        target_json=response,
        metadata={
            "row_id": row_id,
            "source_corpus": source,
            "trust_level": trust_name,
            "vendor": getattr(record, "vendor", None),
        },
    )


def validate_phase1_row(example: Mapping[str, Any]) -> None:
    """Reject any Phase 1 row that violates the D0 training contract."""
    if example.get("phase") != 1:
        raise ValueError("Phase 1 row phase must equal 1")
    if example.get("dataset") != PHASE1_DATASET:
        raise ValueError("Phase 1 row dataset must equal D0")
    if example.get("task") != PHASE1_TASK:
        raise ValueError("Phase 1 row task must equal redfish_json_reconstruction")
    if "y_pred" in example:
        raise ValueError("Phase 1 training rows must not contain y_pred")
    required_fields = {"phase", "dataset", "task", "x", "y_true"}
    if set(example) not in (required_fields, required_fields | {"metadata"}):
        raise ValueError("Phase 1 row contains unknown or missing top-level fields")

    x = example.get("x")
    y_true = example.get("y_true")
    if not isinstance(x, Mapping) or set(x) != {"rest_api", "allowed_methods", "json"}:
        raise ValueError("Phase 1 x must contain rest_api, allowed_methods, and json")
    if not isinstance(y_true, Mapping) or set(y_true) != {"json"}:
        raise ValueError("Phase 1 y_true must contain exactly json")
    rest_api = x.get("rest_api")
    if not isinstance(rest_api, str) or not rest_api.strip():
        raise ValueError("Phase 1 x.rest_api must be a non-empty string")
    methods = x.get("allowed_methods")
    if not isinstance(methods, list) or any(
        not isinstance(method, str) or not method or method != method.upper()
        for method in methods
    ):
        raise ValueError("Phase 1 x.allowed_methods must be uppercase list[str]")
    if len(methods) != len(set(methods)):
        raise ValueError("Phase 1 x.allowed_methods must be unique")
    if not isinstance(x.get("json"), Mapping):
        raise ValueError("Phase 1 x.json must be an object")
    if not isinstance(y_true.get("json"), Mapping):
        raise ValueError("Phase 1 y_true.json must be an object")
    if "metadata" in example:
        _validate_phase1_metadata(example["metadata"])


def _validate_phase1_metadata(value: Any) -> None:
    """Validate optional source lineage retained beside the D0 model fields."""
    required = {"row_id", "source_corpus", "trust_level", "vendor"}
    if not isinstance(value, Mapping) or set(value) not in (required, required | {"chunk"}):
        raise ValueError(
            "Phase 1 metadata must contain source lineage and optional chunk metadata"
        )
    row_id = value.get("row_id")
    digest = row_id.removeprefix("sha256:") if isinstance(row_id, str) else ""
    if (
        not isinstance(row_id, str)
        or not row_id.startswith("sha256:")
        or len(digest) != 64
        or any(char not in "0123456789abcdef" for char in digest.lower())
    ):
        raise ValueError("Phase 1 metadata.row_id must be a SHA-256 id")
    if not isinstance(value.get("source_corpus"), str) or not value["source_corpus"].strip():
        raise ValueError("Phase 1 metadata.source_corpus must be non-empty")
    if not isinstance(value.get("trust_level"), str) or not value["trust_level"].strip():
        raise ValueError("Phase 1 metadata.trust_level must be non-empty")
    if value.get("vendor") is not None and not isinstance(value["vendor"], str):
        raise ValueError("Phase 1 metadata.vendor must be null or a string")
    if "chunk" in value:
        _validate_phase1_chunk_metadata(value["chunk"])


def _validate_phase1_chunk_metadata(value: Any) -> None:
    """Validate exact lossless-chunk lineage without interpreting JSON values."""

    required = {
        "version",
        "original_row_id",
        "original_json_sha256",
        "tokenizer_sha",
        "max_tokens",
        "index",
        "count",
        "json_path",
        "kind",
        "range_start",
        "range_stop",
        "container_length",
        "chunk_json_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("Phase 1 metadata.chunk has missing or unknown fields")
    if not isinstance(value["version"], str) or not value["version"]:
        raise ValueError("Phase 1 metadata.chunk.version must be non-empty")
    for name in (
        "original_row_id",
        "original_json_sha256",
        "tokenizer_sha",
        "chunk_json_sha256",
    ):
        digest = value[name].removeprefix("sha256:") if isinstance(value[name], str) else ""
        if (
            not isinstance(value[name], str)
            or not value[name].startswith("sha256:")
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest.lower())
        ):
            raise ValueError(f"Phase 1 metadata.chunk.{name} must be a SHA-256 id")
    max_tokens = value["max_tokens"]
    index = value["index"]
    count = value["count"]
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 2:
        raise ValueError("Phase 1 metadata.chunk.max_tokens must be >= 2")
    if not isinstance(index, int) or isinstance(index, bool) or index < 0:
        raise ValueError("Phase 1 metadata.chunk.index must be non-negative")
    if not isinstance(count, int) or isinstance(count, bool) or count < 1 or index >= count:
        raise ValueError("Phase 1 metadata.chunk.count must contain index")
    path = value["json_path"]
    if not isinstance(path, list) or any(
        not isinstance(component, (str, int))
        or isinstance(component, bool)
        or isinstance(component, str) and not component
        or isinstance(component, int) and component < 0
        for component in path
    ):
        raise ValueError("Phase 1 metadata.chunk.json_path must be string/int components")
    kind = value["kind"]
    if kind not in {
        "whole_document",
        "object_fields",
        "array_slice",
        "string_slice",
        "scalar_value",
    }:
        raise ValueError("Phase 1 metadata.chunk.kind is unknown")
    ranges = (
        value["range_start"],
        value["range_stop"],
        value["container_length"],
    )
    if kind in {"array_slice", "string_slice"}:
        if not all(isinstance(item, int) and not isinstance(item, bool) for item in ranges):
            raise ValueError("Phase 1 slice chunks require integer range metadata")
        start, stop, length = ranges
        if not 0 <= start <= stop <= length:
            raise ValueError("Phase 1 slice chunk range is invalid")
    elif any(item is not None for item in ranges):
        raise ValueError("Phase 1 non-slice chunks require null range metadata")


def render_phase1_prompt(
    example: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Render a Phase 1 prompt and return the target JSON object.

    Phase 1 training accepts only explicit D0 rows. Raw source records must be
    converted with :func:`build_phase1_source_row` before persistence.

    """

    rest_api, allowed_methods, input_json, target_json = _phase1_fields(example)
    allowed = ", ".join(allowed_methods) if allowed_methods else "UNKNOWN"
    task = resolve_sft_task(PHASE1_TASK)
    values = {
        "rest_api": rest_api,
        "allowed_methods": allowed,
        "json_context": phase1_json_dumps(input_json),
    }
    return task.render_prompt(values), target_json


def render_phase1_completion(target_json: Mapping[str, Any]) -> str:
    """Render the Phase 1 completion exactly as the trainer labels it."""

    return f"{phase1_json_dumps(target_json)}\n"


def phase1_json_dumps(value: Any) -> str:
    """Render canonical pretty JSON while sharing the structural span index."""

    return phase1_json_with_spans(value)[0]


def phase1_json_with_spans(value: Any) -> tuple[str, tuple[Phase1JSONSpan, ...]]:
    """Render canonical pretty JSON and index keys, values, objects, and arrays."""

    pieces: list[str] = []
    spans: list[Phase1JSONSpan] = []
    length = 0

    def append(text: str) -> None:
        nonlocal length
        pieces.append(text)
        length += len(text)

    def render(current: Any, level: int, path: tuple[str | int, ...]) -> None:
        start = length
        if isinstance(current, Mapping):
            append("{")
            if current:
                append("\n")
                items = sorted(current.items(), key=lambda item: str(item[0]))
                for index, (raw_key, child) in enumerate(items):
                    key = str(raw_key)
                    append("  " * (level + 1))
                    key_start = length
                    append(json.dumps(key))
                    key_end = length
                    append(": ")
                    render(child, level + 1, path + (key,))
                    value_end = length
                    spans.append(
                        Phase1JSONSpan("key", path + (key,), key, key_start, key_end)
                    )
                    spans.append(
                        Phase1JSONSpan(
                            "key_value",
                            path + (key,),
                            key,
                            key_start,
                            value_end,
                        )
                    )
                    if index + 1 < len(items):
                        append(",")
                    append("\n")
                append("  " * level)
            append("}")
            spans.append(Phase1JSONSpan("object", path, None, start, length))
            return
        if isinstance(current, list):
            append("[")
            if current:
                append("\n")
                for index, child in enumerate(current):
                    append("  " * (level + 1))
                    render(child, level + 1, path + (index,))
                    if index + 1 < len(current):
                        append(",")
                    append("\n")
                append("  " * level)
            append("]")
            spans.append(Phase1JSONSpan("array", path, None, start, length))
            return
        append(json.dumps(current))
        spans.append(Phase1JSONSpan("scalar", path, None, start, length))

    render({} if value is None else value, 0, ())
    return "".join(pieces), tuple(spans)


def _phase1_fields(
        example: Mapping[str, Any]) -> tuple[str, list[str], dict[str, Any], dict[str, Any]]:
    """Extract Phase 1 fields from one validated canonical D0 row."""

    validate_phase1_row(example)
    x = example.get("x")
    y_true = example.get("y_true")
    assert isinstance(x, Mapping) and isinstance(y_true, Mapping)
    return (
        str(x["rest_api"]),
        list(x["allowed_methods"]),
        dict(x["json"]),
        dict(y_true["json"]),
    )


__all__ = (
    "PHASE1_DATASET",
    "PHASE1_TASK",
    "build_phase1_row",
    "build_phase1_source_row",
    "Phase1JSONSpan",
    "phase1_json_dumps",
    "phase1_json_with_spans",
    "render_phase1_completion",
    "render_phase1_prompt",
    "validate_phase1_row",
)
