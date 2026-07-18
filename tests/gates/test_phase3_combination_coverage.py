"""Offline tests for the Phase 3 combination-coverage gate.

Phase 3 must train on k=1/2/3 combinations covering the first-curriculum
operation/argument/vendor categories, not mostly singletons; these tests prove
the classifier and the coverage checks on tiny fixture rows.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

SCRIPT = Path("scripts/gates/phase3_combination_coverage.py")


def _load_gate() -> ModuleType:
    """Load the gate script module for direct testing."""
    spec = importlib.util.spec_from_file_location("phase3_combination_coverage", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _row(text: str, calls: list[dict]) -> dict:
    """One minimal Phase 3 row for coverage evaluation."""
    return {"x": {"text": text}, "y_true": {"calls": calls}}


def _call(rest_api: str, method: str, arguments: dict | None = None) -> dict:
    """One bound call."""
    return {"rest_api": rest_api, "http_method": method, "arguments": dict(arguments or {})}


def _curriculum_rows() -> list[dict]:
    """A tiny dataset covering every required width and category."""
    return [
        # k=1: all-observe singleton.
        _row("read z", [_call("/api/z", "GET")]),
        # k=2: observe + mutate with a nested object argument.
        _row("set x and read z", [
            _call("/api/z", "GET"),
            _call("/api/x", "PATCH", {"Attributes": {"x": 1}}),
        ]),
        # k=3: observe + zero-argument invoke + delete.
        _row("reset a, read z, and remove w", [
            _call("/api/z", "GET"),
            _call("/api/a/Actions/Reset", "POST", {}),
            _call("/api/w", "DELETE", {"Confirm": True}),
        ]),
        # k=2: mutate + invoke, one scalar argument and multiple arguments.
        _row("set x to 1 and invoke b", [
            _call("/api/x", "PATCH", {"x": 1}),
            _call("/api/b/Actions/Do", "POST", {"a": 1, "b": 2}),
        ]),
    ]


def test_curriculum_fixture_passes_all_required_checks() -> None:
    """The curriculum fixture satisfies widths, categories, and the singleton cap."""
    gate = _load_gate()
    report = gate.evaluate_coverage(_curriculum_rows(), gate.load_config())

    assert report["violations"] == []
    assert report["passed"] is True
    assert report["width_counts"] == {"1": 1, "2": 2, "3": 1}
    assert report["singleton_fraction"] == 0.25
    for category in (
        "all_get_observe",
        "observe_mutate",
        "observe_invoke",
        "mutate_invoke",
        "delete_observe",
        "zero_argument_function",
        "one_scalar_argument",
        "nested_object_argument",
        "multiple_arguments",
        "standard_standard",
    ):
        assert report["category_counts"].get(category, 0) >= 1, category


def test_mostly_singletons_fails() -> None:
    """A dataset dominated by singleton rows violates the singleton cap."""
    gate = _load_gate()
    rows = _curriculum_rows() + [
        _row(f"read s{i}", [_call(f"/api/s{i}", "GET")]) for i in range(10)
    ]

    report = gate.evaluate_coverage(rows, gate.load_config())
    assert any("singleton fraction" in v for v in report["violations"])


def test_missing_width_fails() -> None:
    """A build without k=3 rows fails the required-width check."""
    gate = _load_gate()
    rows = [row for row in _curriculum_rows() if len(row["y_true"]["calls"]) != 3]

    report = gate.evaluate_coverage(rows, gate.load_config())
    assert any("k=3" in v for v in report["violations"])


def test_missing_required_category_fails() -> None:
    """Dropping the mutate+invoke row fails that required category."""
    gate = _load_gate()
    rows = [
        row for row in _curriculum_rows()
        if row["x"]["text"] != "set x to 1 and invoke b"
    ]
    # Keep k=2 present so ONLY the category violation fires.
    rows.append(_row("read z and read q", [_call("/api/z", "GET"), _call("/api/q", "GET")]))

    report = gate.evaluate_coverage(rows, gate.load_config())
    assert any("mutate_invoke" in v for v in report["violations"])


def test_hard_negative_rows_never_satisfy_widths() -> None:
    """Empty ([] calls) hard negatives are counted but satisfy no width/category."""
    gate = _load_gate()
    rows = [_row("nothing to do", [])]

    report = gate.evaluate_coverage(rows, gate.load_config())
    assert report["width_counts"] == {"0": 1}
    assert any("k=1" in v for v in report["violations"])


def test_oem_categories_are_counted() -> None:
    """OEM mixes are classified even though they are optional by default."""
    gate = _load_gate()
    rows = _curriculum_rows() + [
        _row("read the vendor extension and z", [
            _call("/api/z", "GET"),
            _call("/api/Oem/Vendor/Thing", "GET"),
        ]),
        _row("read two vendor extensions", [
            _call("/api/Oem/Vendor/A", "GET"),
            _call("/api/Oem/Vendor/B", "GET"),
        ]),
    ]

    report = gate.evaluate_coverage(rows, gate.load_config())
    assert report["category_counts"].get("standard_oem", 0) >= 1
    assert report["category_counts"].get("oem_oem", 0) >= 1
    assert report["passed"] is True


def test_mention_order_variants_same_combination() -> None:
    """Different mention orders count as text variants of ONE unordered combination."""
    gate = _load_gate()
    combo = [
        _call("/api/x", "PATCH", {"x": 1}),
        _call("/api/z", "GET"),
    ]
    rows = [
        _row("set x, then read z", combo),
        _row("read z after setting x", list(reversed(combo))),
        _row("update x and report z", combo),
    ]

    report = gate.evaluate_coverage(rows, gate.load_config())
    assert report["combination_count"] == 1
    assert list(report["text_variants_per_combination"].values()) == [3]

    config = dict(gate.load_config())
    config["min_text_variants_per_combination"] = 2
    thin = gate.evaluate_coverage([rows[0]], config)
    assert any("text variants" in v for v in thin["violations"])


def test_cli_writes_report_and_exit_codes(tmp_path: Path) -> None:
    """The CLI validates a JSONL dataset and writes the JSON report."""
    gate = _load_gate()
    good = tmp_path / "good.jsonl"
    good.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in _curriculum_rows()),
        encoding="utf-8",
    )
    report_path = tmp_path / "report.json"

    assert gate.main(["--dataset", str(good), "--report-out", str(report_path)]) == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["passed"] is True

    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps(_row("read z", [_call("/api/z", "GET")])) + "\n", encoding="utf-8")
    assert gate.main(["--dataset", str(bad)]) == 1


# Author: Mus mbayramo@stanford.edu
