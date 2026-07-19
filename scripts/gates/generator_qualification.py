"""Gate: generator.qualification — model_x draft quality on a fixed API sample.

Before a large Phase 2 D1 build, model_x is run on a small fixed API sample and the
drafts are judged; this gate measures the baseline the full build is compared
against:

* valid_text_rate       — drafts that are well-formed operator text (not junk)
* judge_acceptance_rate  — drafts the judge accepts
* duplicate_intent_rate  — drafts whose extracted API list repeats an API
* unsupported_intent_rate— drafts referencing an API outside the allowed context
* ambiguous_command_rate — drafts too vague to map to a concrete API

If the generator is below/above the configured thresholds on the small sample, do
not scale the build. Offline (CI) scores the committed baseline records with the
canonical ``parse_pro_judge_result``; the real model_x + Pro-judge run is BLOCKED
while Brain/GB300 is off.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from igc.ds.phase2_labelled_requests import parse_pro_judge_result


def load_config(path: Path) -> dict[str, Any]:
    """Load and minimally validate the qualification config."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "records" not in data or "thresholds" not in data:
        raise ValueError(f"generator qualification config malformed: {path}")
    return data


def classify(record: Mapping[str, Any]) -> dict[str, bool]:
    """Classify one generation record via the canonical judge parser."""
    result = parse_pro_judge_result(str(record["judge_response"]))
    text = str(record.get("text", ""))
    allowed = record.get("allowed_methods", {}) or {}
    apis = list(result.rest_api_list)
    clean = not result.nonsense and not result.invalid_json
    return {
        "valid_text": bool(text.strip()) and clean,
        "accepted": bool(result.accepted) and clean,
        "duplicate_intent": len(apis) != len(set(apis)),
        "unsupported_intent": any(a not in allowed for a in apis),
        "ambiguous": clean and len(apis) == 0,
    }


def _rate(flag: str, rows: Sequence[Mapping[str, bool]]) -> float:
    """Fraction of rows where ``flag`` is true (0.0 for an empty sample)."""
    return (sum(1 for r in rows if r[flag]) / len(rows)) if rows else 0.0


def run(records: Sequence[Mapping[str, Any]], thresholds: Mapping[str, float]) -> dict[str, Any]:
    """Compute the five rates and evaluate them against the thresholds."""
    rows = [classify(r) for r in records]
    rates = {
        "valid_text_rate": _rate("valid_text", rows),
        "judge_acceptance_rate": _rate("accepted", rows),
        "duplicate_intent_rate": _rate("duplicate_intent", rows),
        "unsupported_intent_rate": _rate("unsupported_intent", rows),
        "ambiguous_command_rate": _rate("ambiguous", rows),
    }
    violations: list[str] = []
    checks = {
        "min_valid_text_rate": ("valid_text_rate", "min"),
        "min_judge_acceptance_rate": ("judge_acceptance_rate", "min"),
        "max_duplicate_intent_rate": ("duplicate_intent_rate", "max"),
        "max_unsupported_intent_rate": ("unsupported_intent_rate", "max"),
        "max_ambiguous_command_rate": ("ambiguous_command_rate", "max"),
    }
    for key, (rate_name, direction) in checks.items():
        if key not in thresholds:
            continue
        limit = float(thresholds[key])
        value = rates[rate_name]
        if direction == "min" and value < limit:
            violations.append(f"{rate_name}={value:.3f} < {limit}")
        if direction == "max" and value > limit:
            violations.append(f"{rate_name}={value:.3f} > {limit}")
    return {
        "gate": "generator.qualification",
        "sample_size": len(rows),
        "rates": rates,
        "thresholds": dict(thresholds),
        "violations": violations,
        "qualified": not violations,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the generator-qualification gate (offline baseline mode)."""
    parser = argparse.ArgumentParser(description="Qualify model_x drafts on a fixed API sample.")
    parser.add_argument("--config", default="configs/gates/generator_qualification.yaml")
    parser.add_argument("--out", default="reports/gate-report-generator-qualification.json")
    args = parser.parse_args(argv)

    config = load_config(Path(args.config))
    report = run(config["records"], config["thresholds"])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps({"rates": report["rates"], "violations": report["violations"]}, indent=2, sort_keys=True))
    if not report["qualified"]:
        print("BLOCKER: generator did not qualify on the fixed sample.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# Author: Mus mbayramo@stanford.edu
