"""CLI tests for ``scripts/phase1_inference_gate.py``.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from igc.modules.base.metric_keys import PHASE1_FINETUNE, phase_metric

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "phase1_inference_gate.py"


def _load_script():
    """Import the script as a module without requiring PYTHONPATH setup."""
    spec = importlib.util.spec_from_file_location("phase1_inference_gate", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path: Path, *, exact: bool) -> Path:
    rest_api = "/redfish/v1/Systems/1"
    target = {"@odata.id": rest_api, "Name": "System"}
    prediction = target if exact else {"@odata.id": rest_api, "Name": "Wrong"}
    row = {
        "id": "row-1",
        "x": {"rest_api": rest_api, "json": target},
        "y_true": {"json": target},
        "y_pred": {"json": prediction},
        "generated_tokens": 10,
        "latency_sec": 1.0,
        "confidence": 0.9,
    }
    path.write_text(json.dumps(row) + "\n")
    return path


def _write_spec(path: Path, *, exact_delta: float = 0.0) -> Path:
    path.write_text(
        "\n".join(
            [
                "schema_version: phase1_golden_acceptance.v1",
                "metrics:",
                "  ece_bins: 10",
                "thresholds:",
                "  min_rows: 1",
                "  max_missing_target_rows: 0",
                "  max_missing_prediction_rows: 0",
                "  min_model_json_parse_rate: 1.0",
                "  min_model_odata_id_match_rate: 1.0",
                f"  min_exact_match_delta_vs_baseline: {exact_delta}",
            ]
        )
        + "\n"
    )
    return path


def test_phase1_inference_gate_writes_metrics_and_evidence(tmp_path: Path, capsys) -> None:
    """The CLI writes compact JSON outputs and returns 0 for passing thresholds."""
    script = _load_script()
    baseline = _write_jsonl(tmp_path / "baseline.jsonl", exact=False)
    model = _write_jsonl(tmp_path / "model_x.jsonl", exact=True)
    spec = _write_spec(tmp_path / "spec.yaml", exact_delta=1.0)
    metrics_out = tmp_path / "metrics.json"
    evidence_out = tmp_path / "evidence.json"

    code = script.main(
        [
            "--spec",
            str(spec),
            "--baseline-jsonl",
            str(baseline),
            "--model-jsonl",
            str(model),
            "--metrics-out",
            str(metrics_out),
            "--evidence-out",
            str(evidence_out),
        ]
    )

    assert code == 0
    assert json.loads(capsys.readouterr().out)["status"] == "pass"
    metrics = json.loads(metrics_out.read_text())
    evidence = json.loads(evidence_out.read_text())
    assert metrics["metrics"]["model_x"][
        phase_metric(PHASE1_FINETUNE, "eval", "json_exact_match_rate")
    ] == 1.0
    assert evidence["acceptance"]["status"] == "pass"
    assert evidence["artifacts"]["baseline"]["name"] == "baseline.jsonl"


def test_phase1_inference_gate_returns_one_on_acceptance_failure(tmp_path: Path) -> None:
    """Failed acceptance still writes evidence, then exits with code 1."""
    script = _load_script()
    baseline = _write_jsonl(tmp_path / "baseline.jsonl", exact=True)
    model = _write_jsonl(tmp_path / "model_x.jsonl", exact=True)
    spec = _write_spec(tmp_path / "spec.yaml", exact_delta=0.5)
    evidence_out = tmp_path / "evidence.json"

    code = script.main(
        [
            "--spec",
            str(spec),
            "--baseline-jsonl",
            str(baseline),
            "--model-jsonl",
            str(model),
            "--metrics-out",
            str(tmp_path / "metrics.json"),
            "--evidence-out",
            str(evidence_out),
        ]
    )

    assert code == 1
    assert json.loads(evidence_out.read_text())["acceptance"]["status"] == "fail"


def test_phase1_inference_gate_returns_two_for_missing_input(tmp_path: Path) -> None:
    """Missing JSONL input returns the documented error code."""
    script = _load_script()
    spec = _write_spec(tmp_path / "spec.yaml")

    code = script.main(
        [
            "--spec",
            str(spec),
            "--baseline-jsonl",
            str(tmp_path / "missing-baseline.jsonl"),
            "--model-jsonl",
            str(tmp_path / "missing-model.jsonl"),
            "--metrics-out",
            str(tmp_path / "metrics.json"),
            "--evidence-out",
            str(tmp_path / "evidence.json"),
        ]
    )

    assert code == 2


# Author: Mus mbayramo@stanford.edu
