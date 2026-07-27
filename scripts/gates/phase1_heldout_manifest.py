#!/usr/bin/env python3
"""Build and verify the immutable manifest for a materialized Phase 1 held-out JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from igc.ds.phase1_render import validate_phase1_row  # noqa: E402


class ManifestError(ValueError):
    """Raised when held-out rows disagree with the full corpus split manifest."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the exact full-manifest, held-out JSONL, and output paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-manifest", required=True)
    parser.add_argument("--heldout-jsonl", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Validate exact split membership and emit a sanitized manifest."""
    args = parse_args(argv)
    try:
        full_manifest_path = Path(args.full_manifest)
        heldout_path = Path(args.heldout_jsonl)
        full_manifest = _json_object(full_manifest_path)
        expected_ids = full_manifest.get("heldout_row_ids")
        if not isinstance(expected_ids, list) or not expected_ids:
            raise ManifestError("full manifest has no heldout_row_ids")
        if len(expected_ids) != len(set(expected_ids)):
            raise ManifestError("full manifest heldout_row_ids are not unique")

        observed_ids: list[str] = []
        source_counts: Counter[str] = Counter()
        with heldout_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise ManifestError(f"held-out line {line_number} is blank")
                row = json.loads(line)
                if not isinstance(row, Mapping):
                    raise ManifestError(f"held-out line {line_number} is not an object")
                try:
                    validate_phase1_row(row)
                except ValueError as exc:
                    raise ManifestError(
                        f"held-out line {line_number} violates D0: {exc}"
                    ) from exc
                metadata = row.get("metadata")
                source = (
                    metadata.get("source_corpus")
                    if isinstance(metadata, Mapping)
                    else None
                )
                rest_api = row.get("x", {}).get("rest_api")
                if not isinstance(source, str) or not source:
                    raise ManifestError(f"held-out line {line_number} lacks source")
                if not isinstance(rest_api, str) or not rest_api:
                    raise ManifestError(f"held-out line {line_number} lacks REST API")
                row_id = _record_id(source, rest_api)
                if metadata.get("row_id") != row_id:
                    raise ManifestError(
                        f"held-out line {line_number} metadata.row_id disagrees "
                        "with source_corpus and rest_api"
                    )
                observed_ids.append(row_id)
                source_counts[source] += 1

        if observed_ids != expected_ids:
            raise ManifestError(
                "held-out JSONL row IDs/order disagree with the full corpus manifest"
            )
        eval_count = full_manifest.get("eval_count")
        if eval_count != len(observed_ids):
            raise ManifestError("held-out row count disagrees with full manifest eval_count")
        required_corpora = full_manifest.get("required_heldout_sources")
        if not isinstance(required_corpora, list) or not required_corpora:
            raise ManifestError(
                "full manifest required_heldout_sources must be a non-empty list"
            )
        if len(required_corpora) != len(set(required_corpora)):
            raise ManifestError("full manifest required_heldout_sources are not unique")
        if not all(isinstance(item, str) and item for item in required_corpora):
            raise ManifestError(
                "full manifest required_heldout_sources must contain non-empty strings"
            )
        if not set(required_corpora) <= set(source_counts):
            raise ManifestError("held-out JSONL is missing an eligible source")
        expected_source_counts = full_manifest.get("heldout_by_source")
        if not isinstance(expected_source_counts, Mapping):
            raise ManifestError("full manifest heldout_by_source must be an object")
        if dict(source_counts) != dict(expected_source_counts):
            raise ManifestError(
                "held-out corpus counts disagree with full manifest heldout_by_source"
            )
        full_source_counts = full_manifest.get("by_source")
        if (
            not isinstance(full_source_counts, Mapping)
            or not all(
                isinstance(name, str)
                and name
                and isinstance(count, int)
                and not isinstance(count, bool)
                and count > 0
                for name, count in full_source_counts.items()
            )
        ):
            raise ManifestError("full manifest by_source must map sources to positive counts")
        if not set(required_corpora) <= set(full_source_counts):
            raise ManifestError("full manifest by_source is missing a required held-out source")
        if any(
            source_counts[corpus] > full_source_counts[corpus]
            for corpus in required_corpora
        ):
            raise ManifestError("held-out source count exceeds full corpus source count")
        source_registry_sha = full_manifest.get("source_registry_sha")
        source_manifest_shas = full_manifest.get("source_manifest_shas")
        if not _is_sha256(source_registry_sha):
            raise ManifestError("full manifest source_registry_sha is missing or invalid")
        if (
            not isinstance(source_manifest_shas, Mapping)
            or not source_manifest_shas
            or not all(
                isinstance(name, str)
                and name
                and _is_sha256(digest)
                for name, digest in source_manifest_shas.items()
            )
        ):
            raise ManifestError(
                "full manifest source_manifest_shas must map sources to SHA-256 digests"
            )

        payload = {
            "schema_version": "phase1_heldout_manifest.v1",
            "immutable": True,
            "complete": True,
            "artifact_sha256": _sha256(heldout_path),
            "approved_heldout_rows": len(observed_ids),
            "required_corpora": required_corpora,
            "rows_by_corpus": dict(sorted(source_counts.items())),
            "full_rows_by_corpus": {
                corpus: full_source_counts[corpus]
                for corpus in sorted(required_corpora)
            },
            "row_ids": observed_ids,
            "full_corpus_manifest_sha256": _sha256(full_manifest_path),
            "source_registry_sha256": source_registry_sha,
            "source_manifest_shas": dict(sorted(source_manifest_shas.items())),
        }
        _write_json_atomic(Path(args.output_json), payload)
        print(json.dumps({
            "status": "pass",
            "rows": len(observed_ids),
            "output": args.output_json,
        }, sort_keys=True))
        return 0
    except (json.JSONDecodeError, ManifestError, OSError, TypeError, ValueError) as exc:
        print(json.dumps({
            "status": "blocked",
            "error": str(exc),
            "safe_next_step": "rebuild the deterministic corpus split in the approved remote job",
        }), file=sys.stderr)
        return 2


def _record_id(source: str, rest_api: str) -> str:
    digest = hashlib.sha256(f"{source}\0{rest_api}".encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ManifestError(f"{path} must contain a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        return False
    digest = value.removeprefix("sha256:")
    return len(digest) == 64 and all(
        char in "0123456789abcdef" for char in digest.lower()
    )


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise ManifestError(f"immutable held-out manifest already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = Path(f"{path}.pending")
    if pending.exists():
        raise ManifestError(
            f"stale held-out manifest pending file requires inspection: {pending}"
        )
    pending.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    pending.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
