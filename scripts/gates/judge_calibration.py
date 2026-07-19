"""Gate: judge.calibration — the Pro judge must agree with a hand-labelled suite.

Before the Pro judge is trusted at scale to accept/reject Phase 2 D1 drafts, it
must pass a small human-authored calibration suite
(``configs/gates/judge_calibration_suite.yaml``) covering correct single/multi-API
commands and the failure modes: reversed explicit order, missing API, extra API,
duplicate, ambiguous, method-incompatible, and nonsense. The judge's accept/reject
verdict is scored against the hand-labelled ``expected_accepted``; a single
**false-accept** (judge accepts a case that must be rejected) in any dangerous
category fails the gate — a lenient judge would silently poison the dataset.

Offline (CI): each case carries a ``reference_judge_response`` (the JSON a
calibrated judge emits); the gate parses it with the canonical
``parse_pro_judge_result`` and validates the suite + scoring deterministically.
Cluster run: pass a real judge callable (Pro) via ``run(...)``; the verdict is
scored the same way. The real Pro run is BLOCKED while the Brain/GB300 surface is
unavailable.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml

from igc.ds.phase2_labelled_requests import parse_pro_judge_result

JudgeFn = Callable[[Mapping[str, Any]], str]


def _reference_judge(case: Mapping[str, Any]) -> str:
    """Default offline judge: return the case's committed reference response."""
    return str(case["reference_judge_response"])


def load_suite(path: Path) -> dict[str, Any]:
    """Load and minimally validate the calibration suite."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "cases" not in data:
        raise ValueError(f"calibration suite malformed: {path}")
    return data


def judge_verdict(raw: str) -> bool:
    """Reduce a raw judge response to a single accept/reject decision.

    Accept only when the judge explicitly accepts, does not flag nonsense, and the
    response parsed cleanly. This mirrors what enters the accepted D1 dataset.

    :param raw: raw judge JSON string.
    :return: True if the judge would let the row into the dataset.
    """
    result = parse_pro_judge_result(raw)
    return bool(result.accepted and not result.nonsense and not result.invalid_json)


def score_case(case: Mapping[str, Any], raw: str) -> dict[str, Any]:
    """Score one case: the judge verdict vs the hand-labelled expectation."""
    verdict = judge_verdict(raw)
    expected = bool(case["expected_accepted"])
    return {
        "id": case["id"],
        "category": case["category"],
        "judge_accepted": verdict,
        "expected_accepted": expected,
        "agree": verdict == expected,
        "false_accept": verdict and not expected,   # accepted something that must be rejected
        "false_reject": (not verdict) and expected,  # rejected something that must be accepted
    }


def run(suite: Mapping[str, Any], judge_fn: JudgeFn = _reference_judge) -> dict[str, Any]:
    """Run the calibration suite through ``judge_fn`` and build the report."""
    danger = set(suite.get("zero_false_accept_categories", []))
    rows = [score_case(c, judge_fn(c)) for c in suite["cases"]]

    false_accepts = [r for r in rows if r["false_accept"]]
    dangerous_false_accepts = [r for r in false_accepts if r["category"] in danger]
    total = len(rows)
    agree = sum(1 for r in rows if r["agree"])

    per_category: dict[str, dict[str, int]] = {}
    for r in rows:
        pc = per_category.setdefault(r["category"], {"total": 0, "agree": 0})
        pc["total"] += 1
        pc["agree"] += 1 if r["agree"] else 0

    return {
        "gate": "judge.calibration",
        "total": total,
        "agreement": agree,
        "accuracy": (agree / total) if total else 0.0,
        "false_accepts": [r["id"] for r in false_accepts],
        "false_rejects": [r["id"] for r in rows if r["false_reject"]],
        "dangerous_false_accepts": [r["id"] for r in dangerous_false_accepts],
        "per_category": per_category,
        "rows": rows,
    }


def is_calibrated(report: Mapping[str, Any], *, min_accuracy: float = 1.0) -> bool:
    """The judge is calibrated only with zero dangerous false-accepts and >= min_accuracy."""
    return not report["dangerous_false_accepts"] and report["accuracy"] >= min_accuracy


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the judge-calibration gate (offline reference mode)."""
    parser = argparse.ArgumentParser(description="Score the Pro judge against the calibration suite.")
    parser.add_argument("--suite", default="configs/gates/judge_calibration_suite.yaml")
    parser.add_argument("--out", default="reports/gate-report-judge-calibration.json")
    parser.add_argument("--min-accuracy", type=float, default=1.0)
    args = parser.parse_args(argv)

    report = run(load_suite(Path(args.suite)))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps({k: report[k] for k in ("accuracy", "dangerous_false_accepts", "false_rejects")}, indent=2))
    if not is_calibrated(report, min_accuracy=args.min_accuracy):
        print("BLOCKER: judge is not calibrated against the suite.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# Author: Mus mbayramo@stanford.edu
