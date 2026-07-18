"""Script tests for the offline Phase 3 argument extraction smoke.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from igc.modules.base.metric_keys import PHASE3_ARGUMENT_EXTRACT, phase_metric

SCRIPT = Path("scripts/build_phase3_argument_smoke.py")


def _load_script() -> ModuleType:
    """Load the script module for direct ``main(argv)`` testing."""
    spec = importlib.util.spec_from_file_location("build_phase3_argument_smoke", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _base_args(tmp_path: Path) -> list[str]:
    """Return common CLI args for fixture tests."""
    return [
        "--output-jsonl",
        str(tmp_path / "out" / "phase3_argument_smoke.jsonl"),
        "--metrics-out",
        str(tmp_path / "out" / "metrics.json"),
    ]


def _read_jsonl(path: Path) -> list[dict]:
    """Read all non-blank JSONL rows."""
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _metric(group: str, name: str | None = None) -> str:
    """Return a Phase 3 argument-extraction metric key."""
    return phase_metric(PHASE3_ARGUMENT_EXTRACT, group, name)


def _exact_prediction_lines(script: ModuleType) -> str:
    """Return exact fake predictions for the built-in smoke rows."""
    return "".join(
        json.dumps({"calls": row["y_true"]["calls"]}, sort_keys=True) + "\n"
        for row in script.default_phase3_smoke_rows()
    )


def test_cli_mock_mode_writes_rendered_rows_and_metrics(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Mock mode parses exact fake predictions and writes registered smoke metrics."""
    script = _load_script()
    output = tmp_path / "out" / "phase3_argument_smoke.jsonl"
    metrics_path = tmp_path / "out" / "metrics.json"

    code = script.main(_base_args(tmp_path))
    stdout = capsys.readouterr().out

    assert code == 0
    assert "dataset=phase3_argument_extractor_smoke" in stdout
    assert "accepted=2" in stdout
    assert f"output_jsonl={output}" in stdout
    rows = _read_jsonl(output)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert len(rows) == 2
    assert rows[0]["task"] == "text_and_rest_api_list_to_calls"
    assert rows[0]["rendered"]["renderer"] == "render_call_example"
    assert rows[0]["rendered"]["target_char_start"] == rows[0]["rendered"]["prompt_char_count"]
    assert json.loads(rows[0]["rendered"]["target_json"]) == rows[0]["y_true"]
    # No call handoff from a Phase 3 producer: order is RL-oracle evidence.
    assert "inference" not in rows[0]
    assert "inference" not in rows[1]
    patch_call = next(
        call for call in rows[1]["y_true"]["calls"] if call["http_method"] == "PATCH"
    )
    assert patch_call["arguments"] == {"Attributes": {"BootMode": "Uefi"}}
    assert metrics["dataset"] == "phase3_argument_extractor_smoke"
    assert metrics[_metric("smoke", "rows_total")] == 2
    assert metrics[_metric("smoke", "parsed_total")] == 2
    assert metrics[_metric("smoke", "accepted_total")] == 2
    assert metrics[_metric("smoke", "weights_role")] == "argument_extractor"
    assert metrics[_metric("eval", "call_set_exact_match_rate")] == 1.0
    assert metrics[_metric("eval", "method_exact_match_rate")] == 1.0
    assert metrics[_metric("eval", "arguments_exact_match_rate")] == 1.0
    assert metrics[_metric("eval", "arguments_json_validity_rate")] == 1.0
    assert metrics[_metric("eval", "required_argument_coverage_rate")] == 1.0
    assert metrics[_metric("eval", "no_argument_accuracy_rate")] == 1.0
    assert metrics[_metric("eval", "unsafe_argument_rejection_rate")] == 1.0
    assert metrics[_metric("eval", "invalid_method_rate")] == 0.0
    assert metrics["thresholds_pass"] is True


def test_cli_file_provider_uses_local_fake_predictions(tmp_path: Path) -> None:
    """File mode stays offline and accepts local exact-match prediction fixtures."""
    script = _load_script()
    predictions = tmp_path / "predictions.jsonl"
    output = tmp_path / "out" / "phase3_argument_smoke.jsonl"
    metrics_path = tmp_path / "out" / "metrics.json"
    predictions.write_text(_exact_prediction_lines(script), encoding="utf-8")

    code = script.main(
        _base_args(tmp_path)
        + [
            "--provider-mode",
            "file",
            "--predictions-jsonl",
            str(predictions),
        ],
    )

    assert code == 0
    rows = _read_jsonl(output)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert rows[0]["evaluation"]["parsed"] is True
    assert rows[1]["evaluation"]["call_set_exact_match"] is True
    assert metrics[_metric("smoke", "accepted_total")] == 2


def test_cli_extra_predicted_call_fails_set_match(tmp_path: Path) -> None:
    """A distinct extra predicted call fails the set match — extras are never hidden."""
    script = _load_script()
    rows = script.default_phase3_smoke_rows()
    # Distinct extra call: a duplicate would already fail via duplicate detection;
    # a DISTINCT extra proves set-cardinality mismatch is what fails the row.
    extra_calls = list(rows[0]["y_true"]["calls"]) + [{
        "rest_api": "/redfish/v1/Chassis",
        "http_method": "GET",
        "arguments": {},
    }]
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
        json.dumps({"calls": extra_calls}, sort_keys=True)
        + "\n"
        + json.dumps({"calls": rows[1]["y_true"]["calls"]}, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    metrics_path = tmp_path / "out" / "metrics.json"

    code = script.main(
        _base_args(tmp_path)
        + [
            "--provider-mode",
            "file",
            "--predictions-jsonl",
            str(predictions),
            "--allow-threshold-failure",
        ],
    )

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert code == 0
    assert metrics[_metric("eval", "call_set_exact_match_rate")] == 0.5
    assert metrics[_metric("eval", "rest_api_set_match_rate")] == 0.5
    assert metrics[_metric("smoke", "accepted_total")] == 1
    assert metrics["thresholds_pass"] is False


def test_cli_invalid_method_returns_nonzero_after_writing_metrics(tmp_path: Path) -> None:
    """A method illegal for the row's evidence fails thresholds with inspectable artifacts."""
    script = _load_script()
    rows = script.default_phase3_smoke_rows()
    bad_calls = [dict(call) for call in rows[0]["y_true"]["calls"]]
    # PATCH is not legal for this read-only fixture API — the prediction still
    # PARSES (structure is valid); legality fails against the row's evidence.
    bad_calls[0]["http_method"] = "PATCH"
    bad_calls[0]["arguments"] = {"Illegal": True}
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
        json.dumps({"calls": bad_calls}, sort_keys=True)
        + "\n"
        + json.dumps({"calls": rows[1]["y_true"]["calls"]}, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "out" / "phase3_argument_smoke.jsonl"
    metrics_path = tmp_path / "out" / "metrics.json"

    code = script.main(
        _base_args(tmp_path)
        + [
            "--provider-mode",
            "file",
            "--predictions-jsonl",
            str(predictions),
        ],
    )

    rows_out = _read_jsonl(output)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert code == 2
    assert rows_out[0]["evaluation"]["parsed"] is True
    assert rows_out[0]["evaluation"]["invalid_method_rate"] == 0.5
    assert rows_out[0]["evaluation"]["call_set_exact_match"] is False
    assert metrics[_metric("eval", "invalid_method_rate")] == 0.25
    assert metrics[_metric("smoke", "parsed_total")] == 2
    assert metrics[_metric("smoke", "accepted_total")] == 1
    assert metrics["thresholds_pass"] is False


def test_cli_file_provider_requires_predictions_file(tmp_path: Path) -> None:
    """File mode must name the local fake prediction JSONL."""
    script = _load_script()

    with pytest.raises(SystemExit, match="predictions-jsonl"):
        script.main(_base_args(tmp_path) + ["--provider-mode", "file"])


def test_spec_validation_rejects_wrong_weights_role(tmp_path: Path) -> None:
    """The smoke profile is locked to the argument_extractor role."""
    script = _load_script()
    spec_path = tmp_path / "bad.yaml"
    spec_text = Path("configs/inference/phase3_argument_extractor_smoke.yaml").read_text(
        encoding="utf-8",
    )
    spec_path.write_text(
        spec_text.replace("weights_role: argument_extractor", "weights_role: model_x"),
        encoding="utf-8",
    )

    with pytest.raises(script.Phase3ArgumentSmokeSpecError, match="weights_role"):
        script.load_phase3_argument_smoke_spec(spec_path)


# Author: Mus mbayramo@stanford.edu
