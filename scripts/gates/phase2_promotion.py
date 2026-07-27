#!/usr/bin/env python3
"""Gate a real Phase 2 checkpoint with held-out set metrics. Audience: agent."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from igc.modules.train.phase2_promotion import (  # noqa: E402
    Phase2PromotionError,
    evaluate_phase2_promotion,
)
from igc.modules.train.promotion_evidence import (  # noqa: E402
    validate_configured_hard_checks,
)


def parse_args(argv=None) -> argparse.Namespace:
    """Parse exact evidence inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default="configs/inference/phase2_goal_extractor_promotion.yaml")
    parser.add_argument("--heldout-jsonl", required=True)
    parser.add_argument("--heldout-manifest", required=True)
    parser.add_argument("--source-full-manifest", required=True)
    parser.add_argument("--source-full-jsonl", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--split-release-manifest", required=True)
    parser.add_argument("--artifact-evidence", required=True)
    parser.add_argument("--run-report", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    """Run the promotion gate and write one verdict artifact."""
    args = parse_args(argv)
    try:
        spec = yaml.safe_load(Path(args.spec).read_text(encoding="utf-8"))
        heldout_path = Path(args.heldout_jsonl)
        rows = _jsonl(args.heldout_jsonl)
        evidence = _json(args.artifact_evidence)
        result = evaluate_phase2_promotion(
            rows=rows,
            thresholds=spec["thresholds"],
            artifact_evidence=evidence,
            run_report=_json(args.run_report),
            observed_source_full_manifest_sha=_sha256(
                Path(args.source_full_manifest)
            ),
            observed_source_full_sha=_sha256(Path(args.source_full_jsonl)),
            observed_train_manifest_sha=_sha256(Path(args.train_manifest)),
            observed_train_sha=_sha256(Path(args.train_jsonl)),
            observed_heldout_manifest_sha=_sha256(Path(args.heldout_manifest)),
            observed_heldout_sha=_sha256(heldout_path),
            observed_split_release_sha=_sha256(
                Path(args.split_release_manifest)
            ),
        )
        validate_configured_hard_checks(
            result.get("checks"),
            spec.get("hard_checks"),
        )
        Path(args.output_json).write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"status": result["status"], "output": args.output_json}))
        return 0 if result["status"] == "pass" else 1
    except (KeyError, OSError, TypeError, ValueError, Phase2PromotionError) as exc:
        print(json.dumps({
            "status": "blocked",
            "error": str(exc),
            "safe_next_step": "provide complete real held-out and artifact evidence",
        }), file=sys.stderr)
        return 2


def _json(path: str) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _jsonl(path: str) -> list[dict]:
    rows = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: row must be an object")
        rows.append(value)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


if __name__ == "__main__":
    raise SystemExit(main())
