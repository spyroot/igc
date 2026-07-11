#!/usr/bin/env python3
"""
Emit a machine-readable metrics snapshot for the current commit.

Records what is easy to forget across a fast-moving history: how many tests exist, the line
coverage, and the hot-path timings — one JSON object per commit that any two points in time
can be diffed. CI writes one snapshot per run (uploaded as an artifact and echoed into the
run summary), so "has coverage dropped / did a hot path get slower since last week?" is a
`git`-and-`diff` question, not an archaeology dig.

Used by: ``.github/workflows/ci.yml`` (the gate job) runs it after the coverage pass and
uploads ``metrics.json``; ``make metrics`` runs it locally. It shells out to ``pytest --co``
for the test count and to ``scripts/bench_hot_paths.py --section rl`` for timings, and reads
a ``coverage.json`` (from ``pytest --cov-report=json``) for the coverage percent.

Usage:
    python scripts/metrics_snapshot.py [--coverage-json coverage.json] [--out metrics.json]

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from typing import Dict, Optional

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _git(*args: str) -> str:
    """Return the stripped stdout of a git command, or ``""`` on failure."""
    try:
        return subprocess.check_output(["git", *args], cwd=_REPO_ROOT).decode().strip()
    except (subprocess.CalledProcessError, OSError):
        return ""


def parse_collected(pytest_co_output: str) -> int:
    """Parse the test count from ``pytest --co -q`` output.

    :param pytest_co_output: stdout of a collect-only run.
    :return: number of collected tests (0 if not found).
    """
    match = re.search(r"(\d+)\s+tests?\s+collected", pytest_co_output)
    if match:
        return int(match.group(1))
    return sum(1 for line in pytest_co_output.splitlines() if "::" in line)


def parse_bench_table(bench_output: str) -> Dict[str, float]:
    """Parse ``stage  seconds`` rows from a bench_hot_paths.py table.

    :param bench_output: stdout of the benchmark harness.
    :return: mapping of stage label to seconds.
    """
    timings: Dict[str, float] = {}
    for line in bench_output.splitlines():
        match = re.match(r"^(.*?\S)\s+(\d+\.\d+)\s*$", line)
        if match and match.group(1).lower() not in ("stage",):
            timings[match.group(1).strip()] = float(match.group(2))
    return timings


def read_coverage(coverage_json_path: Optional[str]) -> Optional[float]:
    """Read the total line-coverage percent from a ``coverage.json`` report.

    :param coverage_json_path: path to ``pytest --cov-report=json`` output, or ``None``.
    :return: percent covered rounded to 2 dp, or ``None`` when unavailable.
    """
    if not coverage_json_path or not os.path.isfile(coverage_json_path):
        return None
    data = json.load(open(coverage_json_path))
    return round(data.get("totals", {}).get("percent_covered", 0.0), 2)


def _run(cmd) -> str:
    """Run a command from the repo root and return combined stdout (best effort).

    Puts the repo root on ``PYTHONPATH`` so a plain ``python scripts/...`` child can import
    ``igc`` (running a script by path does not add the repo root to ``sys.path``).
    """
    env = {**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE", "OMP_NUM_THREADS": "1",
           "PYTHONPATH": _REPO_ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")}
    result = subprocess.run(cmd, cwd=_REPO_ROOT, capture_output=True, text=True, env=env)
    return result.stdout + result.stderr


def build_snapshot(coverage_json_path: Optional[str]) -> Dict:
    """Assemble the metrics snapshot for the current commit.

    :param coverage_json_path: optional ``coverage.json`` path for the coverage percent.
    :return: the snapshot dict.
    """
    collected = _run([sys.executable, "-m", "pytest", "--co", "-q"])
    bench = _run([sys.executable, "scripts/bench_hot_paths.py", "--section", "rl"])
    return {
        "commit": _git("rev-parse", "HEAD"),
        "commit_short": _git("rev-parse", "--short", "HEAD"),
        "commit_date": _git("show", "-s", "--format=%cI", "HEAD"),
        "num_tests": parse_collected(collected),
        "coverage_pct": read_coverage(coverage_json_path),
        "hot_paths_sec": parse_bench_table(bench),
    }


def to_markdown(snapshot: Dict) -> str:
    """Render a snapshot as a short markdown summary (for the CI run summary).

    :param snapshot: a snapshot dict from :func:`build_snapshot`.
    :return: markdown string.
    """
    lines = [
        f"### Metrics @ `{snapshot['commit_short']}`",
        "",
        f"- tests: **{snapshot['num_tests']}**",
        f"- coverage: **{snapshot['coverage_pct']}%**" if snapshot["coverage_pct"] is not None
        else "- coverage: (not measured)",
        "",
        "| hot path | seconds |",
        "|---|---|",
    ]
    for stage, seconds in snapshot["hot_paths_sec"].items():
        lines.append(f"| {stage} | {seconds:.4f} |")
    return "\n".join(lines)


def main() -> None:
    """CLI: build the snapshot, write JSON, print the markdown summary."""
    parser = argparse.ArgumentParser(description="Emit a per-commit metrics snapshot.")
    parser.add_argument("--coverage-json", default="coverage.json")
    parser.add_argument("--out", default="metrics.json")
    args = parser.parse_args()

    snapshot = build_snapshot(args.coverage_json)
    with open(args.out, "w") as handle:
        json.dump(snapshot, handle, indent=2, sort_keys=True)
    print(to_markdown(snapshot))


if __name__ == "__main__":
    main()

# Author: Mus mbayramo@stanford.edu
