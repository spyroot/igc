#!/usr/bin/env python3
"""Gate: dataset.phase3-combination-coverage — k=1/2/3 + curriculum coverage.

Phase 3 must train on one-, two-, and three-API combinations, not mostly
singleton examples, and the first curriculum must include the operation mix
(observe/mutate/invoke/delete), the argument-structure mix (zero-argument
function, one scalar, nested object, multiple arguments), and the vendor mix
(standard/OEM pairs where available). This gate reads a built Phase 3 dataset
(JSONL, one row per line with ``y_true.calls``), classifies every row into the
categories of ``configs/gates/phase3_combination_coverage.yaml``, and fails the
build when a required width or category is missing, or when singleton rows
dominate. Mention-order variants (different texts, same unordered combination)
are counted per canonical combination and enforced when the config asks.

Used by:
  tests/gates/test_phase3_combination_coverage.py  (offline gate; `pytest -q`)
  CLI: python scripts/gates/phase3_combination_coverage.py --dataset <rows.jsonl>

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "gates" / "phase3_combination_coverage.yaml"

_READ = {"GET", "HEAD"}       # observe methods
_MUTATE = {"PATCH", "PUT"}    # mutate methods
_INVOKE = {"POST"}            # invoke (action/function) methods
_DELETE = {"DELETE"}          # delete methods


def _is_oem(rest_api: str) -> bool:
    """True when the API path carries an OEM segment."""
    return "/oem/" in f"{rest_api.lower()}/"


def _is_scalar(value: Any) -> bool:
    """True for JSON scalars (str/number/bool/null) — not object/array."""
    return not isinstance(value, (dict, list))


def classify_row(calls: list[Mapping[str, Any]]) -> set[str]:
    """Classify one row's call set into curriculum categories.

    :param calls: the row's ``y_true.calls`` (each
        ``{rest_api, http_method, operation_name, arguments}``).
    :return: the set of category names this row satisfies (possibly several).
    """
    methods = {str(call.get("http_method", "")).upper() for call in calls}
    categories: set[str] = set()

    if calls and methods <= _READ:
        categories.add("all_get_observe")
    if methods & _READ and methods & _MUTATE:
        categories.add("observe_mutate")
    if methods & _READ and methods & _INVOKE:
        categories.add("observe_invoke")
    if methods & _MUTATE and methods & _INVOKE:
        categories.add("mutate_invoke")
    if methods & _DELETE and methods & _READ:
        categories.add("delete_observe")

    for call in calls:
        method = str(call.get("http_method", "")).upper()
        arguments = dict(call.get("arguments") or {})
        if method not in _READ and not arguments:
            categories.add("zero_argument_function")
        if len(arguments) == 1 and _is_scalar(next(iter(arguments.values()))):
            categories.add("one_scalar_argument")
        if any(isinstance(value, dict) for value in arguments.values()):
            categories.add("nested_object_argument")
        if len(arguments) >= 2:
            categories.add("multiple_arguments")

    oem_flags = [_is_oem(str(call.get("rest_api", ""))) for call in calls]
    if len(calls) >= 2:
        if not any(oem_flags):
            categories.add("standard_standard")
        if any(oem_flags) and not all(oem_flags):
            categories.add("standard_oem")
        if sum(oem_flags) >= 2:
            categories.add("oem_oem")

    return categories


def _combination_key(calls: list[Mapping[str, Any]]) -> str:
    """Canonical unordered identity of a call combination (order-independent)."""
    parts = sorted(
        (str(call.get("rest_api", "")), str(call.get("http_method", "")).upper())
        for call in calls
    )
    return json.dumps(parts)


def evaluate_coverage(
    rows: Iterable[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate width/category/variant coverage for a built Phase 3 dataset.

    :param rows: Phase 3 rows carrying ``y_true.calls`` and ``x.text``.
    :param config: parsed coverage config (see the YAML for field meanings).
    :return: report dict with counts, violations, and pass/fail.
    """
    width_counts: Counter[int] = Counter()
    category_counts: Counter[str] = Counter()
    texts_by_combination: dict[str, set[str]] = defaultdict(set)
    total_nonempty = 0
    singleton_rows = 0

    for row in rows:
        calls = list((row.get("y_true") or {}).get("calls") or [])
        width_counts[len(calls)] += 1
        if not calls:
            continue  # hard negatives never satisfy a width/category
        total_nonempty += 1
        if len(calls) == 1:
            singleton_rows += 1
        for category in classify_row(calls):
            category_counts[category] += 1
        text = str((row.get("x") or {}).get("text", ""))
        texts_by_combination[_combination_key(calls)].add(text)

    violations: list[str] = []
    for width in config.get("required_sample_widths", []):
        if width_counts.get(int(width), 0) < int(config.get("min_rows_per_width", 1)):
            violations.append(f"missing sample width k={width}")

    max_singleton = float(config.get("max_singleton_fraction", 1.0))
    singleton_fraction = (singleton_rows / total_nonempty) if total_nonempty else 0.0
    if total_nonempty and singleton_fraction > max_singleton:
        violations.append(
            f"singleton fraction {singleton_fraction:.2f} exceeds cap {max_singleton:.2f}"
            " (Phase 3 cannot be mostly singleton examples)"
        )

    for name, spec in (config.get("categories") or {}).items():
        spec = dict(spec or {})
        count = category_counts.get(name, 0)
        if spec.get("required", False) and count < int(spec.get("min_rows", 1)):
            violations.append(f"required curriculum category missing: {name}")

    min_variants = int(config.get("min_text_variants_per_combination", 1))
    thin_combinations = sorted(
        key for key, texts in texts_by_combination.items() if len(texts) < min_variants
    )
    if min_variants > 1 and thin_combinations:
        violations.append(
            f"{len(thin_combinations)} combination(s) below "
            f"{min_variants} mention-order text variants"
        )

    return {
        "rows_total": sum(width_counts.values()),          # all rows incl. hard negatives
        "width_counts": {str(k): v for k, v in sorted(width_counts.items())},
        "singleton_fraction": singleton_fraction,           # over non-empty rows only
        "category_counts": dict(sorted(category_counts.items())),
        "combination_count": len(texts_by_combination),     # distinct unordered combos
        "text_variants_per_combination": {
            key: len(texts) for key, texts in sorted(texts_by_combination.items())
        },
        "violations": violations,
        "passed": not violations,
    }


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load the coverage config YAML."""
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    :return: process exit code (0 = coverage passes).
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Phase 3 rows JSONL to check.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--report-out", default="", help="Optional JSON report path.")
    args = parser.parse_args(argv)

    rows = [
        json.loads(line)
        for line in Path(args.dataset).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    report = evaluate_coverage(rows, load_config(args.config))
    if args.report_out:
        out = Path(args.report_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for violation in report["violations"]:
        print(f"BLOCKER: {violation}", file=sys.stderr)
    if report["passed"]:
        print(
            f"OK: coverage passed — rows={report['rows_total']} "
            f"widths={report['width_counts']} "
            f"singleton_fraction={report['singleton_fraction']:.2f}"
        )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


# Author: Mus mbayramo@stanford.edu
