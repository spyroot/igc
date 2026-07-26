#!/usr/bin/env python3
"""Gate one real released D1 dataset. Audience: agent."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from igc.modules.train.d1_promotion import (  # noqa: E402
    D1PromotionError,
    evaluate_d1_promotion,
)


def parse_args(argv=None) -> argparse.Namespace:
    """Parse exact release and quality evidence paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default="configs/phase2_labelled_requests.yaml")
    parser.add_argument("--dataset-jsonl", required=True)
    parser.add_argument("--release-manifest", required=True)
    parser.add_argument("--heldout-jsonl", required=True)
    parser.add_argument("--heldout-manifest", required=True)
    parser.add_argument("--build-metrics", required=True)
    parser.add_argument("--judge-calibration-metrics", required=True)
    parser.add_argument("--artifact-evidence", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    """Evaluate D1 promotion and emit a bounded verdict."""
    args = parse_args(argv)
    try:
        spec = yaml.safe_load(Path(args.spec).read_text(encoding="utf-8"))
        heldout_rows = _jsonl_rows(Path(args.heldout_jsonl))
        result = evaluate_d1_promotion(
            release_manifest=_json(args.release_manifest),
            observed_artifact_sha=_sha256(Path(args.dataset_jsonl)),
            build_metrics=_json(args.build_metrics),
            calibration_metrics=_json(args.judge_calibration_metrics).get("metrics", {}),
            build_thresholds=spec["acceptance"],
            calibration_thresholds=spec["judge_calibration"],
            artifact_evidence=_json(args.artifact_evidence),
            observed_full_manifest_sha=_sha256(Path(args.release_manifest)),
            observed_heldout_manifest_sha=_sha256(Path(args.heldout_manifest)),
            observed_heldout_sha=_sha256(Path(args.heldout_jsonl)),
            observed_heldout_rows=heldout_rows,
        )
        Path(args.output_json).write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"status": result["status"], "output": args.output_json}))
        return 0 if result["status"] == "pass" else 1
    except (KeyError, OSError, TypeError, ValueError, D1PromotionError) as exc:
        print(json.dumps({
            "status": "blocked",
            "error": str(exc),
            "safe_next_step": "supply the real release, calibration, lineage, and smoke evidence",
        }), file=sys.stderr)
        return 2


def _json(path: str) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _jsonl_rows(path: Path) -> int:
    rows = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank held-out row")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: held-out row must be an object")
            rows += 1
    if rows == 0:
        raise ValueError(f"{path}: held-out JSONL is empty")
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
