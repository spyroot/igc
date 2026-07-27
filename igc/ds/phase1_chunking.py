"""Lossless, tokenizer-aware chunking for Phase 1 Redfish JSON rows."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from igc.ds.phase1_render import (
    build_phase1_row,
    render_phase1_completion,
    render_phase1_prompt,
    validate_phase1_row,
)
from igc.ds.sft_dataset import token_ids


PHASE1_CONTRACT_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "contracts" / "phase1.yaml"
)
_MISSING = object()


@dataclass(frozen=True)
class Phase1ChunkingPolicy:
    """Immutable inputs that determine one Phase 1 chunk transform."""

    transform: str
    tokenizer_sha: str
    max_tokens: int
    telemetry_rest_api_markers: tuple[str, ...]


@dataclass(frozen=True)
class _Fragment:
    path: tuple[str | int, ...]
    kind: str
    value: Any
    range_start: int | None = None
    range_stop: int | None = None
    container_length: int | None = None


def load_phase1_chunking_policy(
    *,
    tokenizer_sha: str,
    max_tokens: int,
    contract_path: str | Path = PHASE1_CONTRACT_PATH,
) -> Phase1ChunkingPolicy:
    """Resolve the lossless transform contract against one training profile."""

    raw = yaml.safe_load(Path(contract_path).read_text(encoding="utf-8"))
    chunking = raw.get("chunking") if isinstance(raw, Mapping) else None
    if not isinstance(chunking, Mapping):
        raise ValueError("Phase 1 contract must define chunking")
    required = {
        "transform",
        "required_for_registry_materialization",
        "split_before_chunk",
        "overflow_policy",
        "padding",
        "max_tokens_source",
        "tokenizer_sha_source",
        "telemetry_reporting",
    }
    if set(chunking) != required:
        raise ValueError("Phase 1 chunking contract has missing or unknown fields")
    if chunking["required_for_registry_materialization"] is not True:
        raise ValueError("Phase 1 registry materialization must require chunking")
    if chunking["split_before_chunk"] is not True:
        raise ValueError("Phase 1 must split resources before chunking")
    if chunking["overflow_policy"] != "lossless_json_chunk":
        raise ValueError("Phase 1 overflow policy must be lossless_json_chunk")
    if chunking["padding"] != "max_length":
        raise ValueError("Phase 1 chunk padding must be max_length")
    if chunking["max_tokens_source"] != "training_profile.seq_len":
        raise ValueError("Phase 1 max_tokens must come from training_profile.seq_len")
    if chunking["tokenizer_sha_source"] != "training_profile.tokenizer_sha":
        raise ValueError("Phase 1 tokenizer SHA must come from the training profile")
    telemetry = chunking["telemetry_reporting"]
    if not isinstance(telemetry, Mapping) or set(telemetry) != {
        "classification_effect",
        "rest_api_markers",
    }:
        raise ValueError("Phase 1 telemetry_reporting contract is invalid")
    if telemetry["classification_effect"] != "report_only_never_drop":
        raise ValueError("Phase 1 telemetry classification must never filter data")
    markers = telemetry["rest_api_markers"]
    if not isinstance(markers, list) or not all(
        isinstance(marker, str) and marker for marker in markers
    ):
        raise ValueError("Phase 1 telemetry markers must be non-empty strings")
    _validate_sha(tokenizer_sha, "tokenizer_sha")
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 2:
        raise ValueError("Phase 1 max_tokens must be an integer >= 2")
    transform = chunking["transform"]
    if not isinstance(transform, str) or not transform:
        raise ValueError("Phase 1 chunking transform must be non-empty")
    return Phase1ChunkingPolicy(
        transform=transform,
        tokenizer_sha=tokenizer_sha,
        max_tokens=max_tokens,
        telemetry_rest_api_markers=tuple(markers),
    )


def chunk_phase1_row(
    row: Mapping[str, Any],
    *,
    tokenizer: Any,
    policy: Phase1ChunkingPolicy,
) -> list[dict[str, Any]]:
    """Partition one canonical source row without losing JSON information."""

    validate_phase1_row(row)
    x = row["x"]
    target = row["y_true"]["json"]
    if x["json"] != target:
        raise ValueError("Phase 1 source chunking requires x.json == y_true.json")
    metadata = row.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("Phase 1 registry chunking requires source metadata")
    if "chunk" in metadata:
        raise ValueError("Phase 1 source row is already chunked")

    original = copy.deepcopy(dict(target))
    original_sha = _json_sha(original)

    def fits(fragment: Mapping[str, Any]) -> bool:
        candidate = build_phase1_row(
            rest_api=x["rest_api"],
            allowed_methods=x["allowed_methods"],
            input_json=fragment,
            target_json=fragment,
        )
        return phase1_row_token_count(candidate, tokenizer=tokenizer) <= policy.max_tokens

    fragments = list(_partition_value((), original, fits))
    if not fragments:
        raise ValueError("Phase 1 chunking produced no fragments")
    chunk_count = len(fragments)
    rows: list[dict[str, Any]] = []
    original_row_id = str(metadata["row_id"])
    for index, fragment in enumerate(fragments):
        rendered = _wrap_fragment(fragment.path, copy.deepcopy(fragment.value))
        chunk_sha = _json_sha(rendered)
        chunk_record = {
            "version": policy.transform,
            "original_row_id": original_row_id,
            "original_json_sha256": original_sha,
            "tokenizer_sha": policy.tokenizer_sha,
            "max_tokens": policy.max_tokens,
            "index": index,
            "count": chunk_count,
            "json_path": list(fragment.path),
            "kind": fragment.kind,
            "range_start": fragment.range_start,
            "range_stop": fragment.range_stop,
            "container_length": fragment.container_length,
            "chunk_json_sha256": chunk_sha,
        }
        chunk_row_id = _chunk_row_id(chunk_record)
        chunk_metadata = dict(metadata)
        chunk_metadata["row_id"] = chunk_row_id
        chunk_metadata["chunk"] = chunk_record
        chunk_row = build_phase1_row(
            rest_api=x["rest_api"],
            allowed_methods=x["allowed_methods"],
            input_json=rendered,
            target_json=rendered,
            metadata=chunk_metadata,
        )
        observed_tokens = phase1_row_token_count(chunk_row, tokenizer=tokenizer)
        if observed_tokens > policy.max_tokens:
            raise AssertionError(
                "Phase 1 chunk exceeds its validated token budget: "
                f"tokens={observed_tokens} max_tokens={policy.max_tokens}"
            )
        rows.append(chunk_row)

    if reassemble_phase1_rows(rows) != original:
        raise AssertionError("Phase 1 chunks failed exact JSON reassembly")
    return rows


def reassemble_phase1_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Reconstruct and verify the original JSON object from shuffled chunks."""

    if not rows:
        raise ValueError("Phase 1 reassembly requires at least one chunk")
    chunks: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    common: tuple[Any, ...] | None = None
    for row in rows:
        validate_phase1_row(row)
        metadata = row.get("metadata")
        chunk = metadata.get("chunk") if isinstance(metadata, Mapping) else None
        if not isinstance(chunk, Mapping):
            raise ValueError("Phase 1 reassembly requires chunk metadata")
        identity = (
            chunk["version"],
            chunk["original_row_id"],
            chunk["original_json_sha256"],
            chunk["tokenizer_sha"],
            chunk["max_tokens"],
            chunk["count"],
        )
        if common is None:
            common = identity
        elif identity != common:
            raise ValueError("Phase 1 chunks do not share one immutable identity")
        fragment = row["x"]["json"]
        if fragment != row["y_true"]["json"]:
            raise ValueError("Phase 1 chunk input and target differ")
        if _json_sha(fragment) != chunk["chunk_json_sha256"]:
            raise ValueError("Phase 1 chunk JSON SHA does not match its fragment")
        chunks.append((chunk, fragment))

    assert common is not None
    expected_count = common[-1]
    indexes = [int(chunk["index"]) for chunk, _ in chunks]
    if len(chunks) != expected_count or sorted(indexes) != list(range(expected_count)):
        raise ValueError("Phase 1 chunk indexes are missing, duplicated, or out of range")

    root: dict[str, Any] = {}
    string_parts: dict[tuple[str | int, ...], tuple[int, list[Any]]] = {}
    whole_seen = False
    for chunk, rendered in sorted(chunks, key=lambda item: int(item[0]["index"])):
        path = tuple(chunk["json_path"])
        value = _unwrap_fragment(rendered, path)
        kind = str(chunk["kind"])
        if kind == "whole_document":
            if whole_seen or len(chunks) != 1 or path:
                raise ValueError("whole_document must be the only root chunk")
            if not isinstance(value, Mapping):
                raise ValueError("whole_document chunk must contain an object")
            root = copy.deepcopy(dict(value))
            whole_seen = True
        elif kind == "object_fields":
            if not isinstance(value, Mapping):
                raise ValueError("object_fields chunk must contain an object")
            target = _ensure_path(root, path, dict)
            for key, child in value.items():
                if key in target:
                    raise ValueError("Phase 1 object chunks overlap")
                target[key] = copy.deepcopy(child)
        elif kind == "array_slice":
            if not isinstance(value, list):
                raise ValueError("array_slice chunk must contain a list")
            start, stop, length = _range_fields(chunk)
            if stop - start != len(value):
                raise ValueError("Phase 1 array slice length is inconsistent")
            target = _ensure_list_path(root, path, length)
            for offset, child in enumerate(value, start=start):
                if target[offset] is not _MISSING:
                    raise ValueError("Phase 1 array chunks overlap")
                target[offset] = copy.deepcopy(child)
        elif kind == "string_slice":
            if not isinstance(value, str):
                raise ValueError("string_slice chunk must contain a string")
            start, stop, length = _range_fields(chunk)
            if stop - start != len(value):
                raise ValueError("Phase 1 string slice length is inconsistent")
            known_length, parts = string_parts.setdefault(
                path, (length, [_MISSING] * length)
            )
            if known_length != length:
                raise ValueError("Phase 1 string chunks disagree on length")
            for offset, character in enumerate(value, start=start):
                if parts[offset] is not _MISSING:
                    raise ValueError("Phase 1 string chunks overlap")
                parts[offset] = character
        elif kind == "scalar_value":
            _set_path_once(root, path, copy.deepcopy(value))
        else:
            raise ValueError(f"unknown Phase 1 chunk kind {kind!r}")

    for path, (_length, parts) in string_parts.items():
        if any(value is _MISSING for value in parts):
            raise ValueError("Phase 1 string chunks contain a gap")
        _set_path_once(root, path, "".join(parts))
    _reject_missing(root)
    if _json_sha(root) != common[2]:
        raise ValueError("Phase 1 reassembled JSON SHA does not match the original")
    return root


def phase1_row_token_count(row: Mapping[str, Any], *, tokenizer: Any) -> int:
    """Return the exact prompt-plus-completion token count used by the trainer."""

    prompt, target = render_phase1_prompt(row)
    completion = render_phase1_completion(target)
    return int(
        token_ids(tokenizer, prompt).numel()
        + token_ids(tokenizer, completion).numel()
    )


def phase1_token_distribution(
    rows: Iterable[Mapping[str, Any]],
    *,
    tokenizer: Any,
    telemetry_rest_api_markers: Sequence[str],
) -> dict[str, Any]:
    """Summarize original row lengths; telemetry classification is report-only."""

    all_lengths: list[int] = []
    telemetry_lengths: list[int] = []
    non_telemetry_lengths: list[int] = []
    folded_markers = tuple(marker.casefold() for marker in telemetry_rest_api_markers)
    for row in rows:
        length = phase1_row_token_count(row, tokenizer=tokenizer)
        all_lengths.append(length)
        rest_api = str(row["x"]["rest_api"]).casefold()
        destination = (
            telemetry_lengths
            if any(marker in rest_api for marker in folded_markers)
            else non_telemetry_lengths
        )
        destination.append(length)
    return {
        "all_resources": _distribution(all_lengths),
        "non_telemetry_resources": _distribution(non_telemetry_lengths),
        "telemetry_resources": _distribution(telemetry_lengths),
    }


def _partition_value(
    path: tuple[str | int, ...],
    value: Any,
    fits: Any,
) -> Iterable[_Fragment]:
    rendered = _wrap_fragment(path, value)
    if fits(rendered):
        yield _whole_fragment(path, value)
        return
    if isinstance(value, Mapping):
        if not value:
            raise ValueError("empty Phase 1 object cannot fit the token budget")
        keys = sorted(value)
        if not all(isinstance(key, str) for key in keys):
            raise ValueError("Phase 1 JSON object keys must be strings")
        start = 0
        while start < len(keys):
            best = _largest_fitting_stop(
                start=start,
                length=len(keys),
                fits=lambda stop: fits(
                    _wrap_fragment(
                        path,
                        {key: value[key] for key in keys[start:stop]},
                    )
                ),
            )
            if best == start:
                key = keys[start]
                yield from _partition_value((*path, key), value[key], fits)
                start += 1
            else:
                yield _Fragment(
                    path,
                    "object_fields",
                    {key: value[key] for key in keys[start:best]},
                )
                start = best
        return
    if isinstance(value, list):
        if not value:
            raise ValueError("empty Phase 1 array cannot fit the token budget")
        start = 0
        while start < len(value):
            best = _largest_fitting_stop(
                start=start,
                length=len(value),
                fits=lambda stop: fits(_wrap_fragment(path, value[start:stop])),
            )
            if best == start:
                yield from _partition_value((*path, start), value[start], fits)
                start += 1
            else:
                yield _Fragment(
                    path,
                    "array_slice",
                    value[start:best],
                    start,
                    best,
                    len(value),
                )
                start = best
        return
    if isinstance(value, str):
        if not value:
            raise ValueError("empty Phase 1 string cannot fit the token budget")
        start = 0
        while start < len(value):
            best = _largest_fitting_stop(
                start=start,
                length=len(value),
                fits=lambda stop: fits(_wrap_fragment(path, value[start:stop])),
            )
            if best == start:
                raise ValueError(
                    "Phase 1 token budget cannot fit one string code point with its JSON path"
                )
            yield _Fragment(
                path,
                "string_slice",
                value[start:best],
                start,
                best,
                len(value),
            )
            start = best
        return
    raise ValueError(
        "Phase 1 token budget cannot fit one scalar with its JSON path; "
        "increase training_profile.seq_len"
    )


def _largest_fitting_stop(*, start: int, length: int, fits: Any) -> int:
    """Find a bounded fitting range without probing the whole remainder first."""

    first = start + 1
    if first > length or not fits(first):
        return start
    best = first
    step = 2
    failed_stop: int | None = None
    while best < length:
        candidate = min(length, start + step)
        if fits(candidate):
            best = candidate
            if best == length:
                return best
            step *= 2
        else:
            failed_stop = candidate
            break
    if failed_stop is None:
        return best
    low, high = best + 1, failed_stop - 1
    while low <= high:
        middle = (low + high) // 2
        if fits(middle):
            best = middle
            low = middle + 1
        else:
            high = middle - 1
    return best


def _whole_fragment(path: tuple[str | int, ...], value: Any) -> _Fragment:
    if not path:
        return _Fragment(path, "whole_document", value)
    if isinstance(value, Mapping):
        return _Fragment(path, "object_fields", value)
    if isinstance(value, list):
        return _Fragment(path, "array_slice", value, 0, len(value), len(value))
    if isinstance(value, str):
        return _Fragment(path, "string_slice", value, 0, len(value), len(value))
    return _Fragment(path, "scalar_value", value)


def _wrap_fragment(path: Sequence[str | int], value: Any) -> dict[str, Any]:
    wrapped = value
    for component in reversed(path):
        wrapped = {component: wrapped} if isinstance(component, str) else [wrapped]
    if not isinstance(wrapped, Mapping):
        raise ValueError("Phase 1 root fragments must remain JSON objects")
    return copy.deepcopy(dict(wrapped))


def _unwrap_fragment(rendered: Mapping[str, Any], path: Sequence[str | int]) -> Any:
    value: Any = rendered
    for component in path:
        if isinstance(component, str):
            if not isinstance(value, Mapping) or set(value) != {component}:
                raise ValueError("Phase 1 chunk does not match its object path")
            value = value[component]
        else:
            if not isinstance(value, list) or len(value) != 1:
                raise ValueError("Phase 1 chunk does not match its array path")
            value = value[0]
    return value


def _ensure_path(
    root: dict[str, Any],
    path: Sequence[str | int],
    leaf_type: type,
) -> Any:
    if not path:
        if not isinstance(root, leaf_type):
            raise ValueError("Phase 1 root container type mismatch")
        return root
    current: Any = root
    for offset, component in enumerate(path):
        last = offset == len(path) - 1
        expected_type = leaf_type if last else (list if isinstance(path[offset + 1], int) else dict)
        if isinstance(component, str):
            if not isinstance(current, dict):
                raise ValueError("Phase 1 object path crosses a non-object")
            if component not in current:
                current[component] = expected_type()
            elif not isinstance(current[component], expected_type):
                raise ValueError("Phase 1 path container type mismatch")
            current = current[component]
        else:
            if not isinstance(current, list) or component < 0:
                raise ValueError("Phase 1 array path is invalid")
            while len(current) <= component:
                current.append(_MISSING)
            if current[component] is _MISSING:
                current[component] = expected_type()
            elif not isinstance(current[component], expected_type):
                raise ValueError("Phase 1 path container type mismatch")
            current = current[component]
    return current


def _ensure_list_path(
    root: dict[str, Any], path: Sequence[str | int], length: int
) -> list[Any]:
    target = _ensure_path(root, path, list)
    if not target:
        target.extend([_MISSING] * length)
    elif len(target) != length:
        raise ValueError("Phase 1 array chunks disagree on container length")
    return target


def _set_path_once(root: dict[str, Any], path: Sequence[str | int], value: Any) -> None:
    if not path:
        raise ValueError("Phase 1 scalar/string chunk cannot replace the root object")
    parent_path = path[:-1]
    component = path[-1]
    parent_type = list if isinstance(component, int) else dict
    parent = _ensure_path(root, parent_path, parent_type)
    if isinstance(component, str):
        if component in parent:
            raise ValueError("Phase 1 chunks overlap at one object field")
        parent[component] = value
    else:
        while len(parent) <= component:
            parent.append(_MISSING)
        if parent[component] is not _MISSING:
            raise ValueError("Phase 1 chunks overlap at one array element")
        parent[component] = value


def _range_fields(chunk: Mapping[str, Any]) -> tuple[int, int, int]:
    values = (
        chunk.get("range_start"),
        chunk.get("range_stop"),
        chunk.get("container_length"),
    )
    if not all(isinstance(value, int) and not isinstance(value, bool) for value in values):
        raise ValueError("Phase 1 slice metadata requires integer range fields")
    start, stop, length = values
    if not 0 <= start <= stop <= length:
        raise ValueError("Phase 1 slice range is invalid")
    return start, stop, length


def _reject_missing(value: Any) -> None:
    if value is _MISSING:
        raise ValueError("Phase 1 reassembled JSON contains a gap")
    if isinstance(value, Mapping):
        for child in value.values():
            _reject_missing(child)
    elif isinstance(value, list):
        for child in value:
            _reject_missing(child)


def _json_sha(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _chunk_row_id(chunk: Mapping[str, Any]) -> str:
    encoded = json.dumps(chunk, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _distribution(values: Sequence[int]) -> dict[str, int | float | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "p50": None,
            "p95": None,
            "p99": None,
        }
    ordered = sorted(values)

    def percentile(fraction: float) -> int:
        return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]

    return {
        "count": len(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        "mean": sum(ordered) / len(ordered),
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
    }


def _validate_sha(value: Any, label: str) -> None:
    digest = value.removeprefix("sha256:") if isinstance(value, str) else ""
    if (
        not isinstance(value, str)
        or not value.startswith("sha256:")
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest.lower())
    ):
        raise ValueError(f"Phase 1 {label} must be sha256:<64 hex>")


__all__ = (
    "PHASE1_CONTRACT_PATH",
    "Phase1ChunkingPolicy",
    "chunk_phase1_row",
    "load_phase1_chunking_policy",
    "phase1_row_token_count",
    "phase1_token_distribution",
    "reassemble_phase1_rows",
)
