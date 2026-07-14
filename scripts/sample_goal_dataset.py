#!/usr/bin/env python3
"""Sample and summarize a built GoalExtractor / GoalEncoder dataset.

This local inspection helper reads the JSONL artifacts produced by
``scripts/build_goal_dataset.py`` or ``scripts/build_goal_dataset_lab.sh`` and
prints a small, human-readable view of what is inside. It does not call model
endpoints, does not require a GPU, and hides full verifier payloads by default.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Iterable

# Running as ``python scripts/sample_goal_dataset.py`` puts scripts/ on sys.path;
# add the repo root so ``import igc`` works without installation.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from igc.ds.goal_dataset import GoalSurface, GoalTextExample


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        default="",
        help="Directory containing goal_dataset_manifest.json and JSONL outputs.",
    )
    parser.add_argument(
        "--dataset-tar",
        default="",
        help="Tar.gz artifact containing goal_dataset_manifest.json and JSONL outputs.",
    )
    parser.add_argument(
        "--surfaces",
        default="",
        help="Explicit goal_surfaces.jsonl path; overrides --dataset-dir default.",
    )
    parser.add_argument(
        "--text",
        default="",
        help="Explicit goal_text_examples.jsonl path; overrides --dataset-dir default.",
    )
    parser.add_argument(
        "--manifest",
        default="",
        help="Explicit goal_dataset_manifest.json path; overrides --dataset-dir default.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum rows to print from each selected dataset file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Deterministic sampling seed.",
    )
    parser.add_argument(
        "--family",
        default="",
        help="Optional GoalRef family filter such as power, boot, network, bios.",
    )
    parser.add_argument(
        "--vendor",
        default="",
        help="Optional vendor filter for surfaces such as dell, hpe, supermicro.",
    )
    parser.add_argument(
        "--mode",
        choices=("both", "surfaces", "text"),
        default="both",
        help="Which artifact type to sample.",
    )
    parser.add_argument(
        "--show-verifier",
        action="store_true",
        help="Print full verifier payloads. Off by default to keep output compact.",
    )
    return parser.parse_args(argv)


def _safe_extract_dataset_tar(path: Path, target: Path) -> None:
    """Extract a dataset tarball without allowing path traversal."""
    with tarfile.open(path, "r:gz") as tar:
        for member in tar.getmembers():
            member_path = Path(member.name)
            if (
                member_path.is_absolute()
                or ".." in member_path.parts
                or member.issym()
                or member.islnk()
            ):
                raise SystemExit(f"unsafe path in dataset tar: {member.name}")
        tar.extractall(target)


def _dataset_path(args: argparse.Namespace, name: str, explicit: str) -> Path | None:
    """Resolve an artifact path from either explicit args or ``--dataset-dir``."""
    if explicit:
        return Path(explicit)
    if args.dataset_dir:
        return Path(args.dataset_dir) / name
    return None


def _read_json(path: Path | None) -> dict[str, Any]:
    """Read a JSON object if the path exists, otherwise return an empty dict."""
    if path is None or not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise SystemExit(f"manifest is not a JSON object: {path}")
    return data


def _read_jsonl(path: Path | None) -> list[dict[str, Any]]:
    """Read a JSON Lines file into dictionaries."""
    if path is None or not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            stripped = line.strip()
            if not stripped:
                continue
            data = json.loads(stripped)
            if not isinstance(data, dict):
                raise SystemExit(f"row {line_no} is not a JSON object: {path}")
            rows.append(data)
    return rows


def _sample(rows: list[Any], limit: int, seed: int) -> list[Any]:
    """Return a deterministic sample without changing row order when small."""
    if limit <= 0 or not rows:
        return []
    if len(rows) <= limit:
        return rows
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(rows)), limit))
    return [rows[index] for index in indices]


def _goal_family(goal_ref: dict[str, Any]) -> str:
    """Read a GoalRef family from serialized data."""
    return str(goal_ref.get("family") or "")


def _filter_surfaces(
    rows: Iterable[GoalSurface],
    family: str,
    vendor: str,
) -> list[GoalSurface]:
    """Apply optional family/vendor filters to surface rows."""
    filtered = []
    for row in rows:
        if family and row.goal_ref.family != family:
            continue
        if vendor and row.vendor != vendor:
            continue
        filtered.append(row)
    return filtered


def _filter_text(rows: Iterable[GoalTextExample], family: str) -> list[GoalTextExample]:
    """Apply optional family filter to text rows."""
    if not family:
        return list(rows)
    return [
        row for row in rows
        if any(ref.family == family for ref in row.goal_refs)
    ]


def _print_manifest(manifest: dict[str, Any]) -> None:
    """Print compact manifest fields when present."""
    if not manifest:
        print("manifest: <missing>")
        return
    print("manifest:")
    for key in (
        "capture_records",
        "goal_surfaces",
        "unique_goal_ids",
        "text_examples",
    ):
        if key in manifest:
            print(f"  {key}: {manifest[key]}")
    vendors = manifest.get("vendors") or []
    if vendors:
        print(f"  vendors: {', '.join(str(vendor) for vendor in vendors)}")


def _format_value(value: Any) -> str:
    """Render compact scalar/list values for terminal inspection."""
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True)


def _print_surfaces(rows: list[GoalSurface], show_verifier: bool) -> None:
    """Print sampled surface rows."""
    print(f"surface sample ({len(rows)} row(s)):")
    if not rows:
        print("  <none>")
        return
    for index, row in enumerate(rows, 1):
        allowed = len(tuple(row.allowed_values))
        print(f"  {index}. {row.goal_ref.goal_id}")
        print(
            "     "
            f"vendor={row.vendor or 'unknown'} "
            f"source={row.source or 'unknown'} "
            f"resource_type={row.goal_ref.resource_type}"
        )
        print(
            "     "
            f"path={row.fact_path} "
            f"target={_format_value(row.target_value)} "
            f"current={_format_value(row.current_value)} "
            f"allowed_values={allowed}"
        )
        print(f"     resource_uri={row.resource_uri}")
        print(f"     verifier_kind={row.verifier.get('kind', '<missing>')}")
        if show_verifier:
            print(f"     verifier={json.dumps(dict(row.verifier), sort_keys=True)}")


def _print_text(rows: list[GoalTextExample]) -> None:
    """Print sampled text rows."""
    print(f"text sample ({len(rows)} row(s)):")
    if not rows:
        print("  <none>")
        return
    for index, row in enumerate(rows, 1):
        print(f"  {index}. {row.text}")
        print(f"     source={row.text_source} split={row.split}")
        print("     goal_ids:")
        for ref in row.goal_refs:
            print(f"       - {ref.goal_id}")
        if row.dependencies:
            print("     dependencies:")
            for dep in row.dependencies:
                print(
                    "       - "
                    f"{dep.before_goal_id} {dep.relation} {dep.after_goal_id}"
                )


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint.

    :return: process-style exit code.
    """
    args = parse_args(argv)
    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    if args.dataset_tar:
        temp_dir = tempfile.TemporaryDirectory(prefix="igc-goal-dataset-")
        _safe_extract_dataset_tar(Path(args.dataset_tar), Path(temp_dir.name))
        args.dataset_dir = temp_dir.name

    manifest_path = _dataset_path(args, "goal_dataset_manifest.json", args.manifest)
    surfaces_path = _dataset_path(args, "goal_surfaces.jsonl", args.surfaces)
    text_path = _dataset_path(args, "goal_text_examples.jsonl", args.text)

    manifest = _read_json(manifest_path)
    surface_rows = [
        GoalSurface.from_dict(row) for row in _read_jsonl(surfaces_path)
    ]
    text_rows = [
        GoalTextExample.from_dict(row) for row in _read_jsonl(text_path)
    ]

    surface_rows = _filter_surfaces(surface_rows, args.family, args.vendor)
    text_rows = _filter_text(text_rows, args.family)

    _print_manifest(manifest)
    if args.mode in {"both", "surfaces"}:
        print()
        _print_surfaces(_sample(surface_rows, args.limit, args.seed), args.show_verifier)
    if args.mode in {"both", "text"}:
        print()
        _print_text(_sample(text_rows, args.limit, args.seed))
    if temp_dir is not None:
        temp_dir.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# Author: Mus mbayramo@stanford.edu
