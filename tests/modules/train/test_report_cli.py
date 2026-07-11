"""Offline tests for the training-report CLI.

``igc.modules.train.report`` has the library (ResultBundle/compare/markdown)
but had no executable surface — the profile-matrix comparison the run manifests
exist for could not be produced. These drive the CLI over fixture report.json
files: glob resolution, the comparison table, baseline handling, JSON output,
and the no-match exit code. CPU-only, no GPU/model.

Author:
Mus mbayramo@stanford.edu
"""

import json

import pytest

from igc.modules.train.report import ResultBundle, RunManifest
from igc.modules.train.report_cli import build_report, main


def _write_report(path, arm_method, rank, loss):
    """Write a minimal report.json for one arm."""
    bundle = ResultBundle(
        manifest=RunManifest(
            run_id=f"run-{arm_method}", profile="m1", model="gpt2",
            adapter_method=arm_method, adapter_rank=rank,
        ),
        metrics={"eval_loss": loss, "accuracy": 90.0},
    )
    bundle.write(str(path))


def test_build_report_compares_two_arms(tmp_path):
    """Two report.json files render a table with both arms and metrics."""
    _write_report(tmp_path / "a.json", "lora", 16, 1.5)
    _write_report(tmp_path / "b.json", "rslora", 32, 1.2)

    result = build_report(
        [str(tmp_path / "a.json"), str(tmp_path / "b.json")], baseline="lora"
    )
    assert "eval_loss" in result["markdown"]
    assert set(result["comparison"]["arms"]) == {"lora-r16", "rslora-r32"}
    # baseline deltas present for the non-baseline arm
    assert result["comparison"]["deltas"]["rslora-r32"]["eval_loss"] == pytest.approx(-0.3)


def test_main_globs_and_prints(tmp_path, capsys):
    """The CLI expands a glob and prints the markdown table."""
    _write_report(tmp_path / "run1-report.json", "lora", 16, 1.5)
    _write_report(tmp_path / "run2-report.json", "full_finetune", None, 1.1)

    code = main([str(tmp_path / "*-report.json"), "--baseline", "lora"])
    out = capsys.readouterr().out

    assert code == 0
    assert "| arm |" in out
    assert "lora-r16" in out


def test_main_writes_json(tmp_path):
    """--json serializes the comparison dict."""
    _write_report(tmp_path / "a.json", "lora", 16, 1.5)
    out_path = tmp_path / "cmp.json"

    main([str(tmp_path / "a.json"), "--json", str(out_path)])

    written = json.loads(out_path.read_text())
    assert "table" in written and "lora-r16" in written["arms"]


def test_main_no_match_exits_2(tmp_path, capsys):
    """No matching report yields a non-zero exit, not a traceback."""
    code = main([str(tmp_path / "nothing-*.json")])
    assert code == 2
    assert "no report.json" in capsys.readouterr().err


def test_baseline_none_skips_deltas(tmp_path):
    """--baseline none produces a table without deltas."""
    _write_report(tmp_path / "a.json", "lora", 16, 1.5)
    result = build_report([str(tmp_path / "a.json")], baseline=None)
    assert result["comparison"]["deltas"] == {}


# Author: Mus mbayramo@stanford.edu
