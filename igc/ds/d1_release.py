"""Atomic publication for fully validated D1 release directories."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from igc.ds.phase2_labelled_requests import ProJudgeResult, judge_result_is_accepted
from igc.ds.rest_goal_contract import d1_row_id, render_rest_api_list_example


class D1ReleaseError(ValueError):
    """Raised when a pending D1 batch cannot be released."""


def release_d1_jsonl(
    *,
    output_dir: str | Path,
    rows: Sequence[Mapping[str, Any]],
    expected_widths: Sequence[int],
    release_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one pending release directory, then publish it with one rename.

    The canonical path is a directory containing ``data.jsonl`` and
    ``manifest.json``. Publishing the directory, rather than two sibling files,
    prevents a process crash from exposing canonical data without its manifest.
    """
    metadata = dict(release_metadata)
    reserved_manifest_fields = {
        "schema_version",
        "dataset",
        "artifact_sha256",
        "rows",
        "sample_width_counts",
        "judge_evidence_valid",
        "balance_valid",
        "immutable",
        "complete",
    }
    conflicting_fields = sorted(reserved_manifest_fields & set(metadata))
    if conflicting_fields:
        raise D1ReleaseError(
            "release_metadata cannot override canonical manifest fields: "
            + ", ".join(conflicting_fields)
        )
    released = Path(output_dir)
    pending = Path(f"{released}.pending")
    release_lock = Path(f"{released}.release.lock")
    pending_data = pending / "data.jsonl"
    pending_manifest = pending / "manifest.json"
    released.parent.mkdir(parents=True, exist_ok=True)
    try:
        with release_lock.open("xb"):
            pass
    except FileExistsError as exc:
        raise D1ReleaseError(
            "D1 release is locked; an active or stale publisher requires inspection"
        ) from exc
    try:
        if released.exists():
            raise D1ReleaseError(
                "immutable canonical D1 release already exists; publish a new versioned path"
            )
        if pending.exists():
            raise D1ReleaseError(
                f"stale D1 pending release requires operator inspection: {pending}"
            )
        try:
            pending.mkdir()
            with pending_data.open("xb") as handle:
                for row in rows:
                    handle.write(json.dumps(dict(row), sort_keys=True).encode("utf-8"))
                    handle.write(b"\n")
            validated = validate_pending_d1_jsonl(
                pending_data,
                expected_widths=expected_widths,
            )
            digest = _sha256(pending_data)
            manifest = {
                "schema_version": "d1_release.v1",
                "dataset": "D1",
                "artifact_sha256": digest,
                "rows": validated["rows"],
                "sample_width_counts": validated["sample_width_counts"],
                "judge_evidence_valid": True,
                "balance_valid": True,
                "immutable": True,
                "complete": True,
                **metadata,
            }
            pending_manifest.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            if _sha256(pending_data) != digest:
                raise D1ReleaseError("pending D1 changed after validation")
            os.replace(pending, released)
            return manifest
        except Exception:
            if pending.exists():
                shutil.rmtree(pending)
            raise
    finally:
        try:
            release_lock.unlink()
        except FileNotFoundError:
            pass


def validate_pending_d1_jsonl(
    path: str | Path,
    *,
    expected_widths: Sequence[int],
) -> dict[str, Any]:
    """Read back and validate every pending D1 row, evidence record, and width count."""
    pending = Path(path)
    expected = tuple(sorted(set(expected_widths)))
    if not expected or any(width not in (0, 1, 2, 3) for width in expected):
        raise D1ReleaseError("expected_widths must be a non-empty subset of [0, 1, 2, 3]")
    counts: Counter[int] = Counter()
    row_count = 0
    with pending.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise D1ReleaseError(f"{pending}:{line_number}: blank JSONL row")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise D1ReleaseError(
                    f"{pending}:{line_number}: invalid JSON: {exc.msg}"
                ) from exc
            if not isinstance(row, Mapping):
                raise D1ReleaseError(f"{pending}:{line_number}: row must be an object")
            _validate_row(row, line_number=line_number)
            width = _mapping(row, "metadata").get("sample_width_k")
            if width not in expected:
                raise D1ReleaseError(
                    f"{pending}:{line_number}: sample_width_k {width!r} was not requested"
                )
            counts[int(width)] += 1
            row_count += 1
    if row_count == 0:
        raise D1ReleaseError("pending D1 is empty")
    missing_widths = sorted(set(expected) - set(counts))
    if missing_widths:
        raise D1ReleaseError(
            f"pending D1 is missing required sample widths {missing_widths}"
        )
    positive_widths = [width for width in expected if width > 0]
    width_counts = [counts[width] for width in positive_widths]
    if width_counts and max(width_counts) - min(width_counts) > 1:
        raise D1ReleaseError(
            "pending D1 sample width balance differs by more than one row: "
            f"{dict(counts)}"
        )
    return {
        "rows": row_count,
        "sample_width_counts": {str(width): counts[width] for width in expected},
    }


def _validate_row(row: Mapping[str, Any], *, line_number: int) -> None:
    if row.get("phase") != 2 or row.get("dataset") != "D1":
        raise D1ReleaseError(f"row {line_number}: Phase 2 D1 identity is invalid")
    if row.get("source_dataset") != "D0" or row.get("task") != "text_to_rest_api_list":
        raise D1ReleaseError(f"row {line_number}: D1 lineage/task is invalid")
    if row.get("target_semantics") != "unordered_unique_rest_api_set":
        raise D1ReleaseError(f"row {line_number}: D1 target semantics are invalid")
    if "y_pred" in row:
        raise D1ReleaseError(f"row {line_number}: committed y_pred is forbidden")
    render_rest_api_list_example(row)
    selected = _mapping(row, "y_true").get("rest_api_list")
    if (
        not isinstance(selected, list)
        or not all(isinstance(api, str) and api.strip() for api in selected)
        or len(selected) != len(set(selected))
    ):
        raise D1ReleaseError(
            f"row {line_number}: rest_api_list must be a unique list[str]"
        )
    sample_width = _mapping(row, "metadata").get("sample_width_k")
    row_id = _mapping(row, "metadata").get("row_id")
    if row_id != d1_row_id(row):
        raise D1ReleaseError(f"row {line_number}: metadata.row_id does not match row")
    if (
        not isinstance(sample_width, int)
        or isinstance(sample_width, bool)
        or sample_width != len(selected)
    ):
        raise D1ReleaseError(
            f"row {line_number}: sample_width_k must equal target cardinality"
        )
    validation = _mapping(row, "validation")
    required = {
        "valid_json",
        "accepted",
        "natural",
        "nonsense",
        "ambiguous",
        "duplicate_intent",
        "extra_intents",
        "method_semantics_valid",
        "covered_api_set",
    }
    if not required <= set(validation):
        raise D1ReleaseError(f"row {line_number}: judge evidence is incomplete")
    boolean_fields = required - {"covered_api_set"}
    if any(not isinstance(validation[field], bool) for field in boolean_fields):
        raise D1ReleaseError(
            f"row {line_number}: judge predicate fields must be booleans"
        )
    covered = validation["covered_api_set"]
    if (
        not isinstance(covered, list)
        or not all(isinstance(api, str) and api.strip() for api in covered)
        or len(covered) != len(set(covered))
    ):
        raise D1ReleaseError(
            f"row {line_number}: covered_api_set must be a unique list[str]"
        )
    verdict = ProJudgeResult(
        valid_json=validation["valid_json"],
        accepted=validation["accepted"],
        natural=validation["natural"],
        nonsense=validation["nonsense"],
        ambiguous=validation["ambiguous"],
        duplicate_intent=validation["duplicate_intent"],
        extra_intents=validation["extra_intents"],
        method_semantics_valid=validation["method_semantics_valid"],
        covered_api_set=tuple(covered),
        reason="released evidence",
    )
    if not judge_result_is_accepted(verdict, selected_api_set=selected):
        raise D1ReleaseError(f"row {line_number}: judge acceptance predicate failed")


def _mapping(row: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = row.get(key)
    if not isinstance(value, Mapping):
        raise D1ReleaseError(f"{key} must be an object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"
