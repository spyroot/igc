"""CLI to summarize and compare Phase 1 training runs.

The training run writes a per-run ``report.json`` (a :class:`ResultBundle` from
:mod:`igc.modules.train.report`); this command turns those into the human
output the profile matrix exists for — a single-run summary or a cross-arm
comparison table with the fairness check and baseline deltas. It reads only
``report.json`` files, so it runs offline with no GPU or model.

Usage::

    python -m igc.modules.train.report_cli experiments/*/report.json
    python -m igc.modules.train.report_cli --baseline lora --json out.json run*/report.json

Author:
Mus mbayramo@stanford.edu
"""

import argparse
import glob
import json
import sys
from typing import List, Optional

from igc.modules.train.report import ResultBundle, compare


def _resolve_paths(patterns: List[str]) -> List[str]:
    """Expand path/glob arguments into a sorted, de-duplicated file list.

    :param patterns: literal paths or globs (e.g. ``experiments/*/report.json``).
    :return: sorted unique matches that exist on disk.
    """
    paths = set()
    for pattern in patterns:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            paths.update(matches)
        elif not glob.has_magic(pattern) and pattern:
            # a literal path with no matches is kept so the read raises a clear
            # error; an unmatched glob simply contributes nothing.
            paths.add(pattern)
    return sorted(paths)


def build_report(paths: List[str], baseline: Optional[str]) -> dict:
    """Read report bundles and render a comparison (or single-run summary).

    :param paths: resolved ``report.json`` paths.
    :param baseline: arm-label prefix to diff against, or None to skip deltas.
    :return: ``{"markdown": str, "comparison": dict}`` for printing/serialization.
    :raises FileNotFoundError: if no readable report is found.
    """
    bundles = [ResultBundle.read(p) for p in paths]
    if not bundles:
        raise FileNotFoundError("no report.json files matched")
    report = compare(bundles, baseline=baseline)
    return {"markdown": report.to_markdown(), "comparison": report.to_dict()}


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point: print the comparison table, optionally write the JSON.

    :param argv: CLI args (paths/globs, ``--baseline``, ``--json``).
    :return: process exit code (0 ok, 2 nothing matched).
    """
    parser = argparse.ArgumentParser(
        description="Summarize and compare igc Phase 1 training report.json files.")
    parser.add_argument("reports", nargs="+", help="report.json paths or globs.")
    parser.add_argument(
        "--baseline", default="lora",
        help="arm-label prefix to diff against (default: lora); 'none' to skip deltas.")
    parser.add_argument(
        "--json", default=None, help="also write the comparison dict to this path.")
    args = parser.parse_args(argv)

    paths = _resolve_paths(args.reports)
    baseline = None if args.baseline == "none" else args.baseline
    try:
        result = build_report(paths, baseline)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    print(result["markdown"])
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(result["comparison"], fh, indent=2, sort_keys=True, default=str)
        print(f"\nwrote {args.json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# Author: Mus mbayramo@stanford.edu
