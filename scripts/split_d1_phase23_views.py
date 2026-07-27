#!/usr/bin/env python3
"""Create aligned, disjoint Phase 2/3 train and robustness-heldout releases."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from igc.ds.rest_goal_contract import (  # noqa: E402
    render_call_example,
    render_rest_api_list_example,
)
from igc.modules.train.phase3_promotion import validate_phase_views  # noqa: E402


_VARIANTS = (
    "base",
    "api_context_shuffled",
    "json_key_order_shuffled",
    "target_serialization_reversed",
    "irrelevant_distractors_added",
)
_ARGUMENT_CLASSES = {
    "read_only_empty",
    "patch_scalar",
    "patch_nested",
    "post_no_arguments",
    "post_one_argument",
    "post_multiple_arguments",
    "delete_no_body",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse full aligned views, their manifests, and one new release path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spec",
        default="configs/data/d1_phase23_split.yaml",
    )
    parser.add_argument("--phase2-jsonl", required=True)
    parser.add_argument("--phase2-manifest", required=True)
    parser.add_argument("--phase3-jsonl", required=True)
    parser.add_argument("--phase3-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Validate, partition, augment, and atomically publish aligned views."""
    args = parse_args(argv)
    try:
        result = build_split_release(
            spec_path=Path(args.spec),
            phase2_path=Path(args.phase2_jsonl),
            phase2_manifest_path=Path(args.phase2_manifest),
            phase3_path=Path(args.phase3_jsonl),
            phase3_manifest_path=Path(args.phase3_manifest),
            output_dir=Path(args.output_dir),
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    except (OSError, TypeError, ValueError) as exc:
        print(json.dumps({
            "status": "blocked",
            "error": str(exc),
            "safe_next_step": (
                "repair the grounded D1 views/split spec or choose a new output path"
            ),
        }, sort_keys=True), file=sys.stderr)
        return 2


def build_split_release(
    *,
    spec_path: Path,
    phase2_path: Path,
    phase2_manifest_path: Path,
    phase3_path: Path,
    phase3_manifest_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Build one immutable aligned split release from full Phase 2/3 views."""
    spec = _load_spec(spec_path)
    phase2_rows = _read_jsonl(phase2_path)
    phase3_rows = _read_jsonl(phase3_path)
    phase2_manifest_sha = _verify_manifest(
        phase2_manifest_path,
        artifact_path=phase2_path,
        rows=len(phase2_rows),
        view="phase2",
    )
    phase3_manifest_sha = _verify_manifest(
        phase3_manifest_path,
        artifact_path=phase3_path,
        rows=len(phase3_rows),
        view="phase3",
    )
    phase2_by_id = _rows_by_id(phase2_rows, phase=2)
    phase3_by_id = _rows_by_id(phase3_rows, phase=3)
    if set(phase2_by_id) != set(phase3_by_id):
        raise ValueError("full Phase 2/3 view row IDs do not match")
    for row_id in phase2_by_id:
        validate_phase_views(phase2_by_id[row_id], phase3_by_id[row_id])

    train_ids, heldout_ids = _partition_ids(
        sorted(phase2_by_id),
        seed=spec["seed"],
        heldout_fraction=spec["heldout_fraction"],
    )
    if len(train_ids) < spec["min_train_rows"]:
        raise ValueError("D1 train split does not meet min_train_rows")
    if len(heldout_ids) < spec["min_heldout_rows"]:
        raise ValueError("D1 held-out split does not meet min_heldout_rows")
    _validate_heldout_coverage(
        [phase2_by_id[row_id] for row_id in heldout_ids],
        [phase3_by_id[row_id] for row_id in heldout_ids],
        spec=spec,
    )

    context_pool = _context_pool(phase2_rows)
    phase2_heldout: list[dict[str, Any]] = []
    phase3_heldout: list[dict[str, Any]] = []
    heldout_pairs: list[dict[str, Any]] = []
    for row_id in heldout_ids:
        for variant in spec["robustness_variants"]:
            phase2, phase3 = _variant_pair(
                phase2_by_id[row_id],
                phase3_by_id[row_id],
                variant=variant,
                context_pool=context_pool,
            )
            render_rest_api_list_example(phase2)
            render_call_example(phase3)
            validate_phase_views(phase2, phase3)
            phase2_heldout.append(phase2)
            phase3_heldout.append(phase3)
            heldout_pairs.append({"phase2": phase2, "phase3": phase3})

    return _publish(
        output_dir=output_dir,
        spec_path=spec_path,
        phase2_manifest_sha=phase2_manifest_sha,
        phase3_manifest_sha=phase3_manifest_sha,
        train_source_ids=train_ids,
        heldout_source_ids=heldout_ids,
        phase2_train=[phase2_by_id[row_id] for row_id in train_ids],
        phase3_train=[phase3_by_id[row_id] for row_id in train_ids],
        phase2_heldout=phase2_heldout,
        phase3_heldout=phase3_heldout,
        heldout_pairs=heldout_pairs,
        variants=spec["robustness_variants"],
    )


def _load_spec(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    fields = {
        "version",
        "name",
        "seed",
        "heldout_fraction",
        "min_train_rows",
        "min_heldout_rows",
        "min_heldout_rows_per_vendor_or_model",
        "min_empty_set_heldout_rows",
        "required_phase3_argument_classes",
        "robustness_variants",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"split spec must contain exactly {sorted(fields)}")
    if value.get("version") != 1:
        raise ValueError("split spec version must be 1")
    seed = value.get("seed")
    if not isinstance(seed, str) or not seed:
        raise ValueError("split spec seed must be a non-empty string")
    fraction = value.get("heldout_fraction")
    if not isinstance(fraction, (int, float)) or not 0.0 < float(fraction) < 1.0:
        raise ValueError("heldout_fraction must be between zero and one")
    for key in (
        "min_train_rows",
        "min_heldout_rows",
        "min_heldout_rows_per_vendor_or_model",
        "min_empty_set_heldout_rows",
    ):
        if not isinstance(value.get(key), int) or value[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    classes = value.get("required_phase3_argument_classes")
    if not isinstance(classes, list) or set(classes) != _ARGUMENT_CLASSES:
        raise ValueError("required_phase3_argument_classes must match the contract")
    variants = value.get("robustness_variants")
    if not isinstance(variants, list) or tuple(variants) != _VARIANTS:
        raise ValueError("robustness_variants must match the contract")
    return {
        **dict(value),
        "heldout_fraction": float(fraction),
        "robustness_variants": tuple(variants),
    }


def _partition_ids(
    row_ids: Sequence[str],
    *,
    seed: str,
    heldout_fraction: float,
) -> tuple[list[str], list[str]]:
    train: list[str] = []
    heldout: list[str] = []
    cutoff = int(heldout_fraction * (1 << 256))
    for row_id in row_ids:
        digest = hashlib.sha256(f"{seed}\0{row_id}".encode("utf-8")).digest()
        (heldout if int.from_bytes(digest, "big") < cutoff else train).append(row_id)
    if set(train) & set(heldout) or set(train) | set(heldout) != set(row_ids):
        raise ValueError("D1 partition is not disjoint and complete")
    return train, heldout


def _validate_heldout_coverage(
    phase2_rows: Sequence[Mapping[str, Any]],
    phase3_rows: Sequence[Mapping[str, Any]],
    *,
    spec: Mapping[str, Any],
) -> None:
    groups: Counter[str] = Counter()
    empty_rows = 0
    argument_classes: set[str] = set()
    for row in phase2_rows:
        metadata = _mapping(row, "metadata")
        raw_groups = metadata.get("heldout_vendor_or_model")
        if not isinstance(raw_groups, list) or not raw_groups:
            raise ValueError("D1 row is missing heldout_vendor_or_model list")
        if not all(isinstance(group, str) and group for group in raw_groups):
            raise ValueError("heldout_vendor_or_model must contain non-empty strings")
        groups.update(set(raw_groups))
        if not _mapping(row, "y_true").get("rest_api_list"):
            empty_rows += 1
    minimum = spec["min_heldout_rows_per_vendor_or_model"]
    if not groups or min(groups.values()) < minimum:
        raise ValueError("held-out vendor/model groups do not meet the configured minimum")
    if empty_rows < spec["min_empty_set_heldout_rows"]:
        raise ValueError("held-out split lacks enough real judged empty-set rows")
    for row in phase3_rows:
        for call in _mapping(row, "y_true").get("calls", []):
            argument_classes.add(_argument_class(call))
    required = set(spec["required_phase3_argument_classes"])
    if not required <= argument_classes:
        raise ValueError(
            "held-out split lacks required Phase 3 argument classes: "
            f"{sorted(required - argument_classes)}"
        )


def _variant_pair(
    phase2_source: Mapping[str, Any],
    phase3_source: Mapping[str, Any],
    *,
    variant: str,
    context_pool: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    phase2 = copy.deepcopy(dict(phase2_source))
    phase3 = copy.deepcopy(dict(phase3_source))
    row_id = _row_id(phase2)
    inference_case_id = _digest(f"{row_id}\0{variant}")
    for row in (phase2, phase3):
        metadata = dict(_mapping(row, "metadata"))
        metadata["semantic_case_id"] = row_id
        metadata["robustness_variant"] = variant
        metadata["inference_case_id"] = inference_case_id
        row["metadata"] = metadata

    phase2_x = dict(_mapping(phase2, "x"))
    phase3_x = dict(_mapping(phase3, "x"))
    if variant == "api_context_shuffled":
        phase2_x["api_context"] = list(reversed(phase2_x["api_context"]))
        phase3_x["api_context"] = list(reversed(phase3_x["api_context"]))
    elif variant == "json_key_order_shuffled":
        phase2_x["api_context"] = _reverse_context_json(phase2_x["api_context"])
        phase3_x["api_context"] = _reverse_context_json(phase3_x["api_context"])
    elif variant == "target_serialization_reversed":
        phase2_target = dict(_mapping(phase2, "y_true"))
        phase2_target["rest_api_list"] = list(
            reversed(phase2_target["rest_api_list"])
        )
        phase2["y_true"] = phase2_target
        phase3_target = dict(_mapping(phase3, "y_true"))
        phase3_target["calls"] = list(reversed(phase3_target["calls"]))
        phase3["y_true"] = phase3_target
    elif variant == "irrelevant_distractors_added":
        present = {
            item["rest_api"] for item in phase2_x.get("api_context", [])
        }
        available = sorted(set(context_pool) - present)
        if not available:
            raise ValueError("no irrelevant distractor is available for robustness")
        offset = int(_digest(row_id).removeprefix("sha256:"), 16) % len(available)
        extra = copy.deepcopy(dict(context_pool[available[offset]]))
        phase2_x["api_context"] = [*phase2_x["api_context"], extra]
        phase3_x["api_context"] = [
            *phase3_x["api_context"],
            copy.deepcopy(extra),
        ]
    elif variant != "base":
        raise ValueError(f"unsupported robustness variant: {variant}")
    phase2["x"] = phase2_x
    phase3["x"] = phase3_x
    return phase2, phase3


def _publish(
    *,
    output_dir: Path,
    spec_path: Path,
    phase2_manifest_sha: str,
    phase3_manifest_sha: str,
    train_source_ids: Sequence[str],
    heldout_source_ids: Sequence[str],
    phase2_train: Sequence[Mapping[str, Any]],
    phase3_train: Sequence[Mapping[str, Any]],
    phase2_heldout: Sequence[Mapping[str, Any]],
    phase3_heldout: Sequence[Mapping[str, Any]],
    heldout_pairs: Sequence[Mapping[str, Any]],
    variants: Sequence[str],
) -> dict[str, Any]:
    release_lock = output_dir.with_name(f"{output_dir.name}.release.lock")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        with release_lock.open("xb"):
            pass
    except FileExistsError as exc:
        raise ValueError(
            "D1 Phase 2/3 split release is locked; an active or stale publisher "
            "requires inspection"
        ) from exc
    try:
        return _publish_locked(
            output_dir=output_dir,
            spec_path=spec_path,
            phase2_manifest_sha=phase2_manifest_sha,
            phase3_manifest_sha=phase3_manifest_sha,
            train_source_ids=train_source_ids,
            heldout_source_ids=heldout_source_ids,
            phase2_train=phase2_train,
            phase3_train=phase3_train,
            phase2_heldout=phase2_heldout,
            phase3_heldout=phase3_heldout,
            heldout_pairs=heldout_pairs,
            variants=variants,
        )
    finally:
        release_lock.unlink(missing_ok=True)


def _publish_locked(
    *,
    output_dir: Path,
    spec_path: Path,
    phase2_manifest_sha: str,
    phase3_manifest_sha: str,
    train_source_ids: Sequence[str],
    heldout_source_ids: Sequence[str],
    phase2_train: Sequence[Mapping[str, Any]],
    phase3_train: Sequence[Mapping[str, Any]],
    phase2_heldout: Sequence[Mapping[str, Any]],
    phase3_heldout: Sequence[Mapping[str, Any]],
    heldout_pairs: Sequence[Mapping[str, Any]],
    variants: Sequence[str],
) -> dict[str, Any]:
    if output_dir.exists():
        raise ValueError(f"output release already exists: {output_dir}")
    pending = output_dir.with_name(f"{output_dir.name}.pending")
    if pending.exists():
        raise ValueError(f"stale pending release requires inspection: {pending}")
    pending.parent.mkdir(parents=True, exist_ok=True)
    pending.mkdir()
    try:
        entries = {
            "phase2_train": (phase2_train, "phase2", "train", train_source_ids),
            "phase3_train": (phase3_train, "phase3", "train", train_source_ids),
            "phase2_heldout": (
                phase2_heldout,
                "phase2",
                "heldout",
                heldout_source_ids,
            ),
            "phase3_heldout": (
                phase3_heldout,
                "phase3",
                "heldout",
                heldout_source_ids,
            ),
        }
        files: dict[str, Any] = {}
        for name, (rows, view, split, source_ids) in entries.items():
            path = pending / f"{name}.jsonl"
            _write_jsonl(path, rows)
            manifest = {
                "schema_version": "d1_phase23_split_view.v1",
                "dataset": "D1",
                "view": view,
                "split": split,
                "immutable": True,
                "complete": True,
                "rows": len(rows),
                "artifact_sha256": _sha256(path),
                "source_full_manifest_sha256": (
                    phase2_manifest_sha if view == "phase2" else phase3_manifest_sha
                ),
                "source_row_ids": list(source_ids),
                "robustness_variants": list(variants) if split == "heldout" else [],
            }
            manifest_path = Path(f"{path}.manifest.json")
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            files[name] = {
                "path": path.name,
                "manifest": manifest_path.name,
                "artifact_sha256": manifest["artifact_sha256"],
                "manifest_sha256": _sha256(manifest_path),
            }
        pairs_path = pending / "heldout_view_pairs.jsonl"
        _write_jsonl(pairs_path, heldout_pairs)
        files["heldout_view_pairs"] = {
            "path": pairs_path.name,
            "artifact_sha256": _sha256(pairs_path),
        }
        release = {
            "schema_version": "d1_phase23_split_release.v1",
            "dataset": "D1",
            "immutable": True,
            "complete": True,
            "split_spec_sha256": _sha256(spec_path),
            "phase2_full_manifest_sha256": phase2_manifest_sha,
            "phase3_full_manifest_sha256": phase3_manifest_sha,
            "train_source_rows": len(train_source_ids),
            "heldout_source_rows": len(heldout_source_ids),
            "disjoint": not bool(set(train_source_ids) & set(heldout_source_ids)),
            "files": files,
        }
        release_path = pending / "release_manifest.json"
        release_path.write_text(
            json.dumps(release, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(pending, output_dir)
        return {
            "status": "released",
            "output_dir": str(output_dir),
            "release_manifest_sha256": _sha256(
                output_dir / "release_manifest.json"
            ),
        }
    except Exception:
        shutil.rmtree(pending, ignore_errors=True)
        raise


def _rows_by_id(
    rows: Sequence[Mapping[str, Any]],
    *,
    phase: int,
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    renderer = render_rest_api_list_example if phase == 2 else render_call_example
    for row in rows:
        renderer(row)
        row_id = _row_id(row)
        if row_id in result:
            raise ValueError(f"duplicate full-view row_id: {row_id}")
        result[row_id] = row
    return result


def _verify_manifest(
    path: Path,
    *,
    artifact_path: Path,
    rows: int,
    view: str,
) -> str:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"manifest must be an object: {path}")
    if value.get("immutable") is not True or value.get("complete") is not True:
        raise ValueError(f"manifest is not immutable and complete: {path}")
    if value.get("dataset") != "D1" or value.get("view") != view:
        raise ValueError(f"manifest view identity is invalid: {path}")
    if value.get("rows") != rows or value.get("artifact_sha256") != _sha256(artifact_path):
        raise ValueError(f"manifest data evidence mismatch: {path}")
    return _sha256(path)


def _context_pool(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        for context in _mapping(row, "x").get("api_context", []):
            api = context.get("rest_api") if isinstance(context, Mapping) else None
            if isinstance(api, str) and api:
                result.setdefault(api, context)
    return result


def _reverse_context_json(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("api_context must be a list")
    output: list[dict[str, Any]] = []
    for item in value:
        copied = copy.deepcopy(dict(item))
        copied["json"] = _reverse_mapping(copied.get("json", {}))
        output.append(copied)
    return output


def _reverse_mapping(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _reverse_mapping(item)
            for key, item in reversed(list(value.items()))
        }
    if isinstance(value, list):
        return [_reverse_mapping(item) for item in value]
    return value


def _argument_class(call: Mapping[str, Any]) -> str:
    method = call.get("http_method")
    arguments = call.get("arguments")
    if method in {"GET", "HEAD"}:
        return "read_only_empty"
    if method == "DELETE" and not arguments:
        return "delete_no_body"
    if method == "POST":
        if not arguments:
            return "post_no_arguments"
        return "post_one_argument" if len(arguments) == 1 else "post_multiple_arguments"
    if method == "PATCH":
        return (
            "patch_nested"
            if any(isinstance(value, (Mapping, list)) for value in arguments.values())
            else "patch_scalar"
        )
    return "unsupported"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank row")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            rows.append(value)
    if not rows:
        raise ValueError(f"JSONL input is empty: {path}")
    return rows


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), separators=(",", ":")))
            handle.write("\n")


def _mapping(source: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = source.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be an object")
    return value


def _row_id(row: Mapping[str, Any]) -> str:
    value = _mapping(row, "metadata").get("row_id")
    if not isinstance(value, str) or not value.startswith("sha256:"):
        raise ValueError("metadata.row_id must be a canonical SHA-256 id")
    digest = value.removeprefix("sha256:")
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest.lower()
    ):
        raise ValueError("metadata.row_id must be a canonical SHA-256 id")
    return value


def _digest(value: str) -> str:
    return f"sha256:{hashlib.sha256(value.encode('utf-8')).hexdigest()}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


if __name__ == "__main__":
    raise SystemExit(main())
