#!/usr/bin/env python3
"""Build atomic Phase 2/3 views from D1 plus explicit call labels. Audience: agent."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from igc.ds.rest_goal_contract import (  # noqa: E402
    RedfishContext,
    build_d1_master_record,
    d1_row_id,
    render_call_example,
    render_d1_master_views,
    render_rest_api_list_example,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse immutable inputs and one new release directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d1-jsonl", required=True)
    parser.add_argument("--d1-manifest", required=True)
    parser.add_argument("--call-labels-jsonl", required=True)
    parser.add_argument("--call-labels-manifest", required=True)
    parser.add_argument(
        "--output-dir",
        required=True,
        help="New atomic release directory; it must not already exist.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Validate every input/label/view and atomically publish one release."""
    args = parse_args(argv)
    try:
        result = build_grounded_views_release(
            d1_path=Path(args.d1_jsonl),
            d1_manifest_path=Path(args.d1_manifest),
            labels_path=Path(args.call_labels_jsonl),
            labels_manifest_path=Path(args.call_labels_manifest),
            output_dir=Path(args.output_dir),
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    except (OSError, TypeError, ValueError) as exc:
        print(
            json.dumps({
                "status": "blocked",
                "error": str(exc),
                "safe_next_step": (
                    "fix the immutable D1/call-label inputs or choose a new output directory"
                ),
            }, sort_keys=True),
            file=sys.stderr,
        )
        return 2


def build_grounded_views_release(
    *,
    d1_path: Path,
    d1_manifest_path: Path,
    labels_path: Path,
    labels_manifest_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Build and atomically release master, Phase 2, and Phase 3 JSONL views."""
    d1_rows = _read_jsonl(d1_path)
    label_rows = _read_jsonl(labels_path)
    d1_manifest_sha = _verify_input_manifest(
        d1_manifest_path,
        artifact_path=d1_path,
        rows=len(d1_rows),
    )
    labels_manifest_sha = _verify_input_manifest(
        labels_manifest_path,
        artifact_path=labels_path,
        rows=len(label_rows),
    )
    d1_by_id: dict[str, Mapping[str, Any]] = {}
    for row in d1_rows:
        render_rest_api_list_example(row)
        row_id = _row_id(row)
        if row_id != d1_row_id(row):
            raise ValueError(f"D1 row identity does not match content: {row_id}")
        if row_id in d1_by_id:
            raise ValueError(f"duplicate D1 row_id: {row_id}")
        d1_by_id[row_id] = row

    labels_by_id: dict[str, Mapping[str, Any]] = {}
    for row in label_rows:
        _require_exact_fields(
            row,
            {
                "row_id",
                "method_by_api",
                "operation_name_by_api",
                "arguments_by_api",
                "argument_value_grounding_by_api",
            },
            "call label",
        )
        row_id = _required_string(row.get("row_id"), "call-label row_id")
        if row_id in labels_by_id:
            raise ValueError(f"duplicate call-label row_id: {row_id}")
        labels_by_id[row_id] = row
    if set(labels_by_id) != set(d1_by_id):
        missing = sorted(set(d1_by_id) - set(labels_by_id))
        extra = sorted(set(labels_by_id) - set(d1_by_id))
        raise ValueError(
            f"call-label coverage must exactly match D1; missing={missing} extra={extra}"
        )

    masters: list[dict[str, Any]] = []
    phase2_rows: list[dict[str, Any]] = []
    phase3_rows: list[dict[str, Any]] = []
    for row_id in sorted(d1_by_id):
        d1 = d1_by_id[row_id]
        labels = labels_by_id[row_id]
        expected_apis = set(_mapping(d1, "y_true").get("rest_api_list", []))
        method_by_api = _mapping(labels, "method_by_api")
        if set(method_by_api) != expected_apis:
            raise ValueError(f"call-label API set differs from D1 for {row_id}")
        master = build_d1_master_record(
            text=_required_string(_mapping(d1, "x").get("text"), "D1 text"),
            contexts=_contexts(_mapping(d1, "x").get("api_context")),
            method_by_api=method_by_api,
            operation_name_by_api=_mapping(labels, "operation_name_by_api"),
            arguments_by_api=_mapping(labels, "arguments_by_api"),
            argument_value_grounding_by_api=_mapping(
                labels,
                "argument_value_grounding_by_api",
            ),
            validation=_mapping(d1, "validation"),
            metadata=_mapping(d1, "metadata"),
        )
        phase2, phase3 = render_d1_master_views(master)
        render_rest_api_list_example(phase2)
        render_call_example(phase3)
        masters.append(master)
        phase2_rows.append(phase2)
        phase3_rows.append(phase3)

    return _publish_release(
        output_dir=output_dir,
        masters=masters,
        phase2_rows=phase2_rows,
        phase3_rows=phase3_rows,
        d1_manifest_sha=d1_manifest_sha,
        labels_manifest_sha=labels_manifest_sha,
    )


def _publish_release(
    *,
    output_dir: Path,
    masters: list[dict[str, Any]],
    phase2_rows: list[dict[str, Any]],
    phase3_rows: list[dict[str, Any]],
    d1_manifest_sha: str,
    labels_manifest_sha: str,
) -> dict[str, Any]:
    release_lock = output_dir.with_name(f"{output_dir.name}.release.lock")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        with release_lock.open("xb"):
            pass
    except FileExistsError as exc:
        raise ValueError(
            "grounded D1 release is locked; an active or stale publisher "
            "requires inspection"
        ) from exc
    try:
        return _publish_release_locked(
            output_dir=output_dir,
            masters=masters,
            phase2_rows=phase2_rows,
            phase3_rows=phase3_rows,
            d1_manifest_sha=d1_manifest_sha,
            labels_manifest_sha=labels_manifest_sha,
        )
    finally:
        release_lock.unlink(missing_ok=True)


def _publish_release_locked(
    *,
    output_dir: Path,
    masters: list[dict[str, Any]],
    phase2_rows: list[dict[str, Any]],
    phase3_rows: list[dict[str, Any]],
    d1_manifest_sha: str,
    labels_manifest_sha: str,
) -> dict[str, Any]:
    if output_dir.exists():
        raise ValueError(f"output release already exists: {output_dir}")
    pending = output_dir.with_name(f"{output_dir.name}.pending")
    if pending.exists():
        raise ValueError(f"stale pending release requires inspection: {pending}")
    pending.parent.mkdir(parents=True, exist_ok=True)
    pending.mkdir()
    try:
        files = {
            "master": (pending / "master.jsonl", masters),
            "phase2": (pending / "phase2.jsonl", phase2_rows),
            "phase3": (pending / "phase3.jsonl", phase3_rows),
        }
        file_evidence: dict[str, Any] = {}
        for view, (path, rows) in files.items():
            _write_jsonl(path, rows)
            artifact_sha = _sha256(path)
            manifest = {
                "schema_version": "d1_grounded_view_release.v1",
                "dataset": "D1",
                "view": view,
                "immutable": True,
                "complete": True,
                "rows": len(rows),
                "artifact_sha256": artifact_sha,
                "source_d1_manifest_sha256": d1_manifest_sha,
                "call_labels_manifest_sha256": labels_manifest_sha,
                "view_consistency_valid": True,
            }
            manifest_path = Path(f"{path}.manifest.json")
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            file_evidence[view] = {
                "path": path.name,
                "artifact_sha256": artifact_sha,
                "manifest": manifest_path.name,
                "manifest_sha256": _sha256(manifest_path),
            }
        release_manifest = {
            "schema_version": "d1_grounded_views_release.v1",
            "dataset": "D1",
            "immutable": True,
            "complete": True,
            "rows": len(masters),
            "source_d1_manifest_sha256": d1_manifest_sha,
            "call_labels_manifest_sha256": labels_manifest_sha,
            "view_consistency_valid": True,
            "files": file_evidence,
        }
        release_manifest_path = pending / "release_manifest.json"
        release_manifest_path.write_text(
            json.dumps(release_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(pending, output_dir)
        return {
            "status": "released",
            "output_dir": str(output_dir),
            "rows": len(masters),
            "release_manifest_sha256": _sha256(
                output_dir / "release_manifest.json"
            ),
        }
    except Exception:
        shutil.rmtree(pending, ignore_errors=True)
        raise


def _verify_input_manifest(path: Path, *, artifact_path: Path, rows: int) -> str:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"input manifest must be an object: {path}")
    if value.get("immutable") is not True or value.get("complete") is not True:
        raise ValueError(f"input manifest is not immutable and complete: {path}")
    if value.get("artifact_sha256") != _sha256(artifact_path):
        raise ValueError(f"input manifest artifact SHA mismatch: {path}")
    if value.get("rows") != rows:
        raise ValueError(f"input manifest row count mismatch: {path}")
    return _sha256(path)


def _contexts(value: Any) -> tuple[RedfishContext, ...]:
    if not isinstance(value, list):
        raise ValueError("D1 api_context must be a list")
    result: list[RedfishContext] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError("D1 api_context item must be an object")
        result.append(RedfishContext(
            rest_api=_required_string(item.get("rest_api"), "rest_api"),
            allowed_methods=item.get("allowed_methods", []),
            operation_names=item.get("operation_names", []),
            argument_schema=_mapping(item, "argument_schema"),
            json=_mapping(item, "json"),
        ))
    return tuple(result)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank JSONL row")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            rows.append(value)
    if not rows:
        raise ValueError(f"JSONL input is empty: {path}")
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _row_id(row: Mapping[str, Any]) -> str:
    return _required_string(_mapping(row, "metadata").get("row_id"), "D1 row_id")


def _mapping(source: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = source.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be an object")
    return value


def _required_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _require_exact_fields(
    value: Mapping[str, Any],
    expected: set[str],
    label: str,
) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} must contain exactly {sorted(expected)}")


if __name__ == "__main__":
    raise SystemExit(main())
