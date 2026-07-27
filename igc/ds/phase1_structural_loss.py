"""Historical Phase 1 structural masking, adapted to completion-only SFT."""
from __future__ import annotations

import copy
import hashlib
import json
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import yaml

from igc.ds.phase1_render import (
    Phase1JSONSpan,
    phase1_json_with_spans,
    validate_phase1_row,
)


STRUCTURAL_LOSS_SPEC_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "training"
    / "phase1_structural_loss.yaml"
)
_PROFILE_FIELDS = {"enabled", "mask_token", "selection", "families"}
_FAMILY_FIELDS = {"name", "selector", "patterns", "max_spans"}
_SELECTORS = {
    "key_value",
    "key",
    "object",
    "array",
    "key_value_suffix",
    "substring",
}
_SELECTIONS = {"row_epoch_cycle"}


@dataclass(frozen=True)
class StructuralLossFamily:
    """One historical mask family and its target-span selector."""

    name: str
    selector: str
    patterns: tuple[str, ...]
    max_spans: int | None


@dataclass(frozen=True)
class Phase1StructuralLossProfile:
    """Resolved structural-loss curriculum with exact YAML identity."""

    name: str
    enabled: bool
    mask_token: str
    selection: str
    families: tuple[StructuralLossFamily, ...]
    spec_sha256: str


@dataclass(frozen=True)
class Phase1StructuralLossResult:
    """One prompt view and the completion character spans that receive loss."""

    row: dict[str, Any]
    family: str
    completion_spans: tuple[tuple[int, int], ...]
    operations: tuple[str, ...]


@dataclass(frozen=True)
class Phase1Repair:
    """One exact change required to restore a corrupted Phase 1 input."""

    kind: str
    path: tuple[str | int, ...]
    value: Any = None


def load_phase1_structural_loss_profile(
    name: str,
    path: str | Path = STRUCTURAL_LOSS_SPEC_PATH,
) -> Phase1StructuralLossProfile:
    """Load one strictly validated YAML-owned structural-loss profile."""

    return _load_phase1_structural_loss_profile_cached(
        name,
        str(Path(path).expanduser().resolve()),
    )


@lru_cache(maxsize=32)
def _load_phase1_structural_loss_profile_cached(
    name: str,
    path: str,
) -> Phase1StructuralLossProfile:
    spec_path = Path(path)
    raw_bytes = spec_path.read_bytes()
    try:
        payload = yaml.safe_load(raw_bytes)
    except yaml.YAMLError as exc:
        raise ValueError(
            f"cannot parse Phase 1 structural-loss spec {spec_path}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping) or payload.get("version") != 1:
        raise ValueError("Phase 1 structural-loss spec version must be 1")
    profiles = payload.get("profiles")
    if not isinstance(profiles, Mapping) or name not in profiles:
        choices = sorted(profiles) if isinstance(profiles, Mapping) else []
        raise KeyError(
            f"unknown Phase 1 structural-loss profile {name!r}; choose from {choices}"
        )
    raw = profiles[name]
    if not isinstance(raw, Mapping) or set(raw) != _PROFILE_FIELDS:
        raise ValueError(
            f"Phase 1 structural-loss profile {name!r} must contain exactly "
            f"{sorted(_PROFILE_FIELDS)}"
        )
    enabled = _require_bool(raw["enabled"], f"{name}.enabled")
    mask_token = _require_string(raw["mask_token"], f"{name}.mask_token")
    selection = _require_string(raw["selection"], f"{name}.selection")
    if selection not in _SELECTIONS:
        raise ValueError(f"{name}.selection must be one of {sorted(_SELECTIONS)}")
    raw_families = raw["families"]
    if not isinstance(raw_families, list):
        raise ValueError(f"{name}.families must be a list")
    families = tuple(
        _family_from_raw(name, index, family)
        for index, family in enumerate(raw_families)
    )
    if enabled and not families:
        raise ValueError(f"{name}.families must not be empty when enabled")
    names = [family.name for family in families]
    if len(names) != len(set(names)):
        raise ValueError(f"{name}.families must have unique names")
    return Phase1StructuralLossProfile(
        name=name,
        enabled=enabled,
        mask_token=mask_token,
        selection=selection,
        families=families,
        spec_sha256="sha256:" + hashlib.sha256(raw_bytes).hexdigest(),
    )


def build_phase1_structural_loss_view(
    example: Mapping[str, Any],
    *,
    profile: Phase1StructuralLossProfile,
    mode: str,
    run_seed: int,
    epoch: int,
    row_index: int,
) -> Phase1StructuralLossResult:
    """Select and hide one available historical structural-loss family."""

    validate_phase1_row(example)
    if mode not in {"train", "evaluation"}:
        raise ValueError("Phase 1 structural-loss mode must be train or evaluation")
    if not profile.enabled:
        return Phase1StructuralLossResult(
            row=copy.deepcopy(dict(example)),
            family="full_completion",
            completion_spans=(),
            operations=(),
        )

    target_json = example["y_true"]["json"]
    if example["x"]["json"] != target_json:
        raise ValueError(
            "Phase 1 structural loss requires x.json == y_true.json before masking"
        )
    completion, indexed_spans = phase1_json_with_spans(target_json)
    effective_epoch = epoch if mode == "train" else 0
    start_index = (row_index + effective_epoch) % len(profile.families)
    rng = random.Random(
        _sample_seed(
            example,
            profile=profile,
            mode=mode,
            run_seed=run_seed,
            epoch=effective_epoch,
            row_index=row_index,
        )
    )

    selected_family: StructuralLossFamily | None = None
    selected_spans: tuple[Phase1JSONSpan | tuple[int, int], ...] = ()
    for offset in range(len(profile.families)):
        family = profile.families[(start_index + offset) % len(profile.families)]
        candidates = _family_spans(family, completion, indexed_spans)
        if not candidates:
            continue
        candidates = list(candidates)
        rng.shuffle(candidates)
        selected_family = family
        selected_spans = tuple(
            candidates
            if family.max_spans is None
            else candidates[:family.max_spans]
        )
        break
    if selected_family is None:
        raise ValueError("Phase 1 row has no span for any structural-loss family")

    row = copy.deepcopy(dict(example))
    operations = _mask_input_family(
        row,
        family=selected_family,
        selected_spans=selected_spans,
        mask_token=profile.mask_token,
    )
    validate_phase1_row(row)
    char_spans = tuple(
        (span.start, span.end) if isinstance(span, Phase1JSONSpan) else span
        for span in selected_spans
    )
    if not char_spans or any(start >= end for start, end in char_spans):
        raise ValueError("Phase 1 structural-loss selection produced an empty span")
    return Phase1StructuralLossResult(
        row=row,
        family=selected_family.name,
        completion_spans=char_spans,
        operations=tuple(operations),
    )


def detect_phase1_repairs(
    observed: Any,
    expected: Any,
) -> tuple[Phase1Repair, ...]:
    """Return the deterministic minimal repair plan from observed to expected."""

    repairs: list[Phase1Repair] = []

    def visit(current: Any, target: Any, path: tuple[str | int, ...]) -> None:
        if isinstance(current, Mapping) and isinstance(target, Mapping):
            current_keys = set(current)
            target_keys = set(target)
            for key in sorted(current_keys - target_keys, key=str):
                repairs.append(Phase1Repair("delete", path + (key,)))
            for key in sorted(target_keys - current_keys, key=str):
                repairs.append(
                    Phase1Repair("set", path + (key,), copy.deepcopy(target[key]))
                )
            for key in sorted(current_keys & target_keys, key=str):
                visit(current[key], target[key], path + (key,))
            return
        if isinstance(current, list) and isinstance(target, list):
            if current != target:
                repairs.append(Phase1Repair("set", path, copy.deepcopy(target)))
            return
        if current != target:
            repairs.append(Phase1Repair("set", path, copy.deepcopy(target)))

    visit(observed, expected, ())
    return tuple(repairs)


def apply_phase1_repairs(
    observed: Any,
    repairs: Sequence[Phase1Repair],
) -> Any:
    """Apply a repair plan without mutating the observed Phase 1 input."""

    repaired = copy.deepcopy(observed)
    for repair in repairs:
        if repair.kind not in {"set", "delete"}:
            raise ValueError(f"unknown Phase 1 repair kind {repair.kind!r}")
        if not repair.path:
            if repair.kind == "delete":
                raise ValueError("cannot delete the Phase 1 repair root")
            repaired = copy.deepcopy(repair.value)
            continue
        parent, key = _resolve_parent(repaired, repair.path)
        if repair.kind == "delete":
            if isinstance(parent, MutableMapping):
                if key not in parent:
                    raise ValueError(
                        f"Phase 1 repair path does not exist: {repair.path!r}"
                    )
                del parent[key]
            elif isinstance(parent, list) and isinstance(key, int):
                del parent[key]
            else:
                raise ValueError(
                    f"Phase 1 repair path is not deletable: {repair.path!r}"
                )
            continue
        if isinstance(parent, MutableMapping):
            parent[key] = copy.deepcopy(repair.value)
        elif isinstance(parent, list) and isinstance(key, int):
            parent[key] = copy.deepcopy(repair.value)
        else:
            raise ValueError(f"Phase 1 repair path is not assignable: {repair.path!r}")
    return repaired


def _family_from_raw(
    profile_name: str,
    index: int,
    raw: Any,
) -> StructuralLossFamily:
    field = f"{profile_name}.families[{index}]"
    if not isinstance(raw, Mapping) or set(raw) != _FAMILY_FIELDS:
        raise ValueError(f"{field} must contain exactly {sorted(_FAMILY_FIELDS)}")
    name = _require_string(raw["name"], f"{field}.name")
    selector = _require_string(raw["selector"], f"{field}.selector")
    if selector not in _SELECTORS:
        raise ValueError(f"{field}.selector must be one of {sorted(_SELECTORS)}")
    patterns = raw["patterns"]
    if not isinstance(patterns, list) or any(
        not isinstance(pattern, str) or not pattern for pattern in patterns
    ):
        raise ValueError(f"{field}.patterns must be list[str]")
    if selector in {"key_value", "key", "key_value_suffix", "substring"} and not patterns:
        raise ValueError(f"{field}.patterns must not be empty for {selector}")
    if selector in {"object", "array"} and patterns:
        raise ValueError(f"{field}.patterns must be empty for {selector}")
    raw_max_spans = raw["max_spans"]
    if raw_max_spans == "all":
        if selector != "substring":
            raise ValueError(f"{field}.max_spans='all' is limited to substring selectors")
        max_spans = None
    else:
        max_spans = raw_max_spans
        if (
            isinstance(max_spans, bool)
            or not isinstance(max_spans, int)
            or max_spans < 1
        ):
            raise ValueError(
                f"{field}.max_spans must be a positive integer or 'all'"
            )
    return StructuralLossFamily(name, selector, tuple(patterns), max_spans)


def _family_spans(
    family: StructuralLossFamily,
    completion: str,
    spans: Sequence[Phase1JSONSpan],
) -> list[Phase1JSONSpan | tuple[int, int]]:
    if family.selector == "key_value":
        return [
            span
            for span in spans
            if span.kind == "key_value" and span.key in family.patterns
        ]
    if family.selector == "key":
        return [
            span
            for span in spans
            if span.kind == "key" and span.key in family.patterns
        ]
    if family.selector == "key_value_suffix":
        return [
            span
            for span in spans
            if span.kind == "key_value"
            and span.key is not None
            and any(span.key.endswith(pattern) for pattern in family.patterns)
        ]
    if family.selector == "object":
        nested_objects = [
            span for span in spans if span.kind == "object" and span.path
        ]
        if nested_objects:
            return nested_objects
        # A root object is never a valid mask target: replacing it would
        # remove the entire observation and turn the task into URL memorization.
        # Flat resources instead supervise one bounded root field.
        return [
            span
            for span in spans
            if span.kind == "key_value" and len(span.path) == 1
        ]
    if family.selector == "array":
        return [span for span in spans if span.kind == "array"]
    matches: list[tuple[int, int]] = []
    for pattern in family.patterns:
        start = 0
        while True:
            index = completion.find(pattern, start)
            if index < 0:
                break
            matches.append((index, index + len(pattern)))
            start = index + len(pattern)
    return matches


def _mask_input_family(
    row: dict[str, Any],
    *,
    family: StructuralLossFamily,
    selected_spans: Sequence[Phase1JSONSpan | tuple[int, int]],
    mask_token: str,
) -> list[str]:
    input_json = row["x"]["json"]
    operations: list[str] = []
    if family.selector == "substring":
        row["x"]["rest_api"] = _replace_patterns(
            row["x"]["rest_api"],
            family.patterns,
            mask_token,
        )
        _replace_substrings(input_json, family.patterns, mask_token)
        return [f"mask:{family.name}:substring:{len(selected_spans)}"]

    for selected in selected_spans:
        if not isinstance(selected, Phase1JSONSpan):
            continue
        path = selected.path
        if family.selector == "object" and not path:
            row["x"]["json"] = {mask_token: True}
            operations.append(f"mask:{family.name}:/")
            continue
        parent, key = _resolve_parent(input_json, path)
        if family.selector == "key":
            if isinstance(parent, MutableMapping) and key in parent:
                del parent[key]
        else:
            parent[key] = mask_token
        operations.append(
            f"mask:{family.name}:/" + "/".join(str(component) for component in path)
        )
    return operations


def _replace_substrings(value: Any, patterns: Sequence[str], mask_token: str) -> None:
    if isinstance(value, MutableMapping):
        for key, child in list(value.items()):
            if isinstance(child, str):
                value[key] = _replace_patterns(child, patterns, mask_token)
            else:
                _replace_substrings(child, patterns, mask_token)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            if isinstance(child, str):
                value[index] = _replace_patterns(child, patterns, mask_token)
            else:
                _replace_substrings(child, patterns, mask_token)


def _replace_patterns(value: str, patterns: Sequence[str], mask_token: str) -> str:
    for pattern in patterns:
        value = value.replace(pattern, mask_token)
    return value


def _resolve_parent(root: Any, path: Sequence[str | int]) -> tuple[Any, str | int]:
    if not path:
        raise ValueError("root path has no parent")
    parent = root
    for component in path[:-1]:
        parent = parent[component]
    return parent, path[-1]


def _sample_seed(
    example: Mapping[str, Any],
    *,
    profile: Phase1StructuralLossProfile,
    mode: str,
    run_seed: int,
    epoch: int,
    row_index: int,
) -> int:
    metadata = example.get("metadata")
    row_id = metadata.get("row_id") if isinstance(metadata, Mapping) else None
    if not isinstance(row_id, str) or not row_id:
        row_id = hashlib.sha256(
            json.dumps(example, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
    material = "\0".join(
        (
            profile.spec_sha256,
            profile.name,
            mode,
            str(run_seed),
            str(epoch),
            str(row_index),
            row_id,
        )
    )
    return int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:8], "big")


def _require_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _require_bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be boolean")
    return value


__all__ = (
    "STRUCTURAL_LOSS_SPEC_PATH",
    "Phase1StructuralLossProfile",
    "Phase1StructuralLossResult",
    "Phase1Repair",
    "StructuralLossFamily",
    "apply_phase1_repairs",
    "build_phase1_structural_loss_view",
    "detect_phase1_repairs",
    "load_phase1_structural_loss_profile",
)
