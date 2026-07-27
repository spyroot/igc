#!/usr/bin/env python3
"""Combine Phase 1 quality, lineage, reload, and best-artifact evidence."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from igc.modules.train.phase1_promotion import (  # noqa: E402
    Phase1PromotionError,
    evaluate_phase1_promotion,
)
from igc.modules.train.promotion_evidence import (  # noqa: E402
    validate_configured_hard_checks,
)


def parse_args(argv=None) -> argparse.Namespace:
    """Parse required promotion evidence paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--full-metrics", required=True)
    parser.add_argument("--full-evidence", required=True)
    parser.add_argument("--golden-metrics", required=True)
    parser.add_argument("--golden-evidence", required=True)
    parser.add_argument("--instruction-retention-evidence", required=True)
    parser.add_argument("--run-report", required=True)
    parser.add_argument("--heldout-manifest", required=True)
    parser.add_argument("--model-load-evidence", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args(argv)


def _json(path: str) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase1PromotionError(f"{path} must contain a JSON object")
    return value


def main(argv=None) -> int:
    """Evaluate promotion and write one self-contained verdict."""
    args = parse_args(argv)
    try:
        spec = yaml.safe_load(Path(args.spec).read_text(encoding="utf-8"))
        result = evaluate_phase1_promotion(
            thresholds=spec["thresholds"],
            full_metrics=_json(args.full_metrics),
            full_evidence=_json(args.full_evidence),
            golden_metrics=_json(args.golden_metrics),
            golden_evidence=_json(args.golden_evidence),
            retention_evidence=_json(args.instruction_retention_evidence),
            run_report=_json(args.run_report),
            heldout_manifest=_json(args.heldout_manifest),
            load_evidence=_json(args.model_load_evidence),
        )
        validate_configured_hard_checks(
            result.get("checks"),
            spec.get("hard_checks"),
        )
        Path(args.output_json).write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (KeyError, OSError, TypeError, ValueError, Phase1PromotionError) as exc:
        print(f"PHASE1 PROMOTION ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": result["status"], "output": args.output_json}))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
