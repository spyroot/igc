"""
Offline tests for the metrics-snapshot parsers.

Pins the pure parsers that turn tool output into a comparable snapshot: the pytest collected
count, the bench_hot_paths timing table, the coverage.json percent, and the markdown summary.
The subprocess-driving build is exercised in CI, not here. Pure stdlib + tmp_path.

Author:
Mus mbayramo@stanford.edu
"""

import json
from pathlib import Path

from scripts.metrics_snapshot import (
    parse_bench_table,
    parse_collected,
    read_coverage,
    to_markdown,
)


def test_parse_collected_from_summary():
    """The '<n> tests collected' summary line is parsed."""
    assert parse_collected("igc/x.py::test_a\n\n566 tests collected in 1.2s\n") == 566


def test_parse_collected_falls_back_to_node_ids():
    """Without a summary line, count lines carrying a '::' node id."""
    out = "tests/a.py::test_one\ntests/a.py::test_two\nnoise\n"
    assert parse_collected(out) == 2


def test_parse_bench_table():
    """Only 'label  seconds' rows are parsed; the header is ignored."""
    out = ("corpus=synthetic\nstage                     seconds\n"
           "q_learning_target [B,N]   0.0003\n"
           "pointer forward, naive    0.1926\n")
    table = parse_bench_table(out)
    assert table["q_learning_target [B,N]"] == 0.0003
    assert table["pointer forward, naive"] == 0.1926
    assert "stage" not in table


def test_read_coverage(tmp_path: Path):
    """Coverage percent comes from totals.percent_covered, rounded; None when absent."""
    path = tmp_path / "coverage.json"
    path.write_text(json.dumps({"totals": {"percent_covered": 47.5137}}))
    assert read_coverage(str(path)) == 47.51
    assert read_coverage(str(tmp_path / "missing.json")) is None
    assert read_coverage(None) is None


def test_to_markdown_includes_tests_coverage_and_paths():
    """The summary shows the commit, test count, coverage, and each hot-path row."""
    md = to_markdown({"commit_short": "abc1234", "num_tests": 566, "coverage_pct": 47.5,
                      "hot_paths_sec": {"neighbors": 0.001}})
    assert "abc1234" in md and "566" in md and "47.5%" in md and "neighbors" in md


def test_to_markdown_handles_missing_coverage():
    """A snapshot without coverage renders without crashing."""
    md = to_markdown({"commit_short": "abc1234", "num_tests": 10, "coverage_pct": None,
                      "hot_paths_sec": {}})
    assert "not measured" in md


# Author: Mus mbayramo@stanford.edu
