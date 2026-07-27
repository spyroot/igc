#!/usr/bin/env python3
"""Evaluate labelled private-judge calibration evidence. Audience: agent."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from igc.ds.phase2_labelled_requests import (  # noqa: E402
    evaluate_judge_calibration,
    judge_calibration_passes,
    load_phase2_labelled_requests_spec,
    parse_pro_judge_result,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse deterministic calibration inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default="configs/phase2_labelled_requests.yaml")
    parser.add_argument("--calibration-jsonl", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the calibration gate and emit one bounded JSON verdict."""
    args = parse_args(argv)
    try:
        spec = load_phase2_labelled_requests_spec(args.spec)
        examples: list[tuple[Any, list[str], bool]] = []
        for line_number, line in enumerate(
            Path(args.calibration_jsonl).read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"line {line_number}: calibration row must be an object")
            selected = row.get("selected_api_set")
            human_accept = row.get("human_accept")
            judge_raw = row.get("judge_raw")
            if not isinstance(selected, list) or not all(isinstance(api, str) for api in selected):
                raise ValueError(f"line {line_number}: selected_api_set must be list[str]")
            if not isinstance(human_accept, bool) or not isinstance(judge_raw, str):
                raise ValueError(
                    f"line {line_number}: human_accept must be bool and judge_raw must be string"
                )
            examples.append((parse_pro_judge_result(judge_raw), selected, human_accept))
        metrics = evaluate_judge_calibration(examples)
        passed = judge_calibration_passes(spec, metrics)
        result = {
            "schema_version": "d1_judge_calibration.v1",
            "status": "pass" if passed else "fail",
            "thresholds": dict(spec.judge_calibration_thresholds),
            "metrics": metrics,
        }
        Path(args.output_json).write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"status": result["status"], "output": args.output_json}))
        return 0 if passed else 1
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(
            json.dumps({
                "status": "blocked",
                "error": str(exc),
                "safe_next_step": "repair the labelled calibration JSONL or YAML spec",
            }),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
