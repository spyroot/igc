#!/usr/bin/env python3
"""Compare foundation and model_x human-command generation on fixed k=1/2/3 rows."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from igc.modules.train.instruction_retention import (  # noqa: E402
    InstructionRetentionError,
    compare_retention_artifacts,
    load_retention_jsonl,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--foundation-jsonl", required=True)
    parser.add_argument("--model-jsonl", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the paired retention gate and write its evidence."""
    args = parse_args(argv)
    try:
        spec = yaml.safe_load(Path(args.spec).read_text(encoding="utf-8"))
        max_drop = float(spec["promotion"]["max_judge_accept_rate_drop"])
        comparison = compare_retention_artifacts(
            load_retention_jsonl(args.foundation_jsonl),
            load_retention_jsonl(args.model_jsonl),
        )
        observed_drop = -float(comparison["delta"]["judge_acceptance_rate"])
        passed = observed_drop <= max_drop
        payload = {
            "schema_version": "phase1_instruction_retention_evidence.v1",
            "status": "pass" if passed else "fail",
            "comparison": comparison,
            "check": {
                "name": "max_judge_accept_rate_drop",
                "observed": observed_drop,
                "threshold": max_drop,
                "passed": passed,
            },
        }
        Path(args.output_json).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (KeyError, OSError, TypeError, ValueError, InstructionRetentionError) as exc:
        print(f"PHASE1 INSTRUCTION RETENTION ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": payload["status"], "output": args.output_json}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
