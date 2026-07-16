#!/usr/bin/env python3
"""Evaluate Phase 1 held-out prediction JSONL artifacts offline."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from igc.modules.train.phase1_acceptance import (  # noqa: E402
    AcceptanceSpec,
    Phase1AcceptanceError,
    evaluate_acceptance,
)
from igc.modules.train.phase1_golden import (  # noqa: E402
    Phase1GoldenError,
    build_phase1_golden_payload,
    write_json,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, help="YAML metric/threshold spec.")
    parser.add_argument(
        "--baseline-jsonl",
        required=True,
        help="Already-produced baseline prediction JSONL artifact.",
    )
    parser.add_argument(
        "--model-jsonl",
        required=True,
        help="Already-produced model_x prediction JSONL artifact.",
    )
    parser.add_argument(
        "--metrics-out",
        required=True,
        help="Output JSON path for compact baseline/model_x metric tables.",
    )
    parser.add_argument(
        "--evidence-out",
        required=True,
        help="Output JSON path for artifact summaries and acceptance checks.",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Return 0 after writing evidence even if acceptance checks fail.",
    )
    return parser.parse_args(argv)


def load_spec(path: str | Path) -> dict[str, Any]:
    """Load the YAML spec as a mapping."""
    with Path(path).open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise Phase1AcceptanceError("spec root must be a mapping")
    return data


def build_outputs(
    *,
    spec_path: str | Path,
    baseline_jsonl: str | Path,
    model_jsonl: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the metrics and evidence payloads for CLI callers."""
    spec_root = load_spec(spec_path)
    spec = AcceptanceSpec.from_mapping(spec_root)
    metrics_payload = build_phase1_golden_payload(
        baseline_jsonl=baseline_jsonl,
        model_jsonl=model_jsonl,
        ece_bins=spec.ece_bins,
    )
    acceptance = evaluate_acceptance(metrics_payload, spec)
    evidence = {
        "schema_version": "phase1_inference_gate_evidence.v1",
        "phase": 1,
        "task": metrics_payload["task"],
        "metric_namespace": metrics_payload["metric_namespace"],
        "spec": {
            "name": spec_root.get("name", ""),
            "schema_version": spec_root.get("schema_version", ""),
            "ece_bins": spec.ece_bins,
            "thresholds": dict(spec.thresholds),
        },
        "artifacts": {
            "baseline": metrics_payload["evidence"]["baseline"]["artifact"],
            "model_x": metrics_payload["evidence"]["model_x"]["artifact"],
        },
        "counts": {
            "baseline": metrics_payload["evidence"]["baseline"]["counts"],
            "model_x": metrics_payload["evidence"]["model_x"]["counts"],
        },
        "comparison": metrics_payload["comparison"],
        "acceptance": acceptance,
    }
    metrics_out = {
        "schema_version": metrics_payload["schema_version"],
        "phase": metrics_payload["phase"],
        "task": metrics_payload["task"],
        "metric_namespace": metrics_payload["metric_namespace"],
        "metrics": metrics_payload["metrics"],
        "comparison": metrics_payload["comparison"],
    }
    return metrics_out, evidence


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = parse_args(argv)
    try:
        metrics_payload, evidence = build_outputs(
            spec_path=args.spec,
            baseline_jsonl=args.baseline_jsonl,
            model_jsonl=args.model_jsonl,
        )
        write_json(args.metrics_out, metrics_payload)
        write_json(args.evidence_out, evidence)
    except (OSError, Phase1AcceptanceError, Phase1GoldenError) as exc:
        print(f"PHASE1 INFERENCE GATE ERROR: {exc}", file=sys.stderr)
        return 2

    status = evidence["acceptance"]["status"]
    print(json.dumps({"status": status, "evidence_out": args.evidence_out}, sort_keys=True))
    if status != "pass" and not args.report_only:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# Author: Mus mbayramo@stanford.edu
