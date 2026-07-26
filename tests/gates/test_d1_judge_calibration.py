"""Offline tests for ``scripts/gates/d1_judge_calibration.py``."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "gates" / "d1_judge_calibration.py"
SPEC = Path(__file__).resolve().parents[2] / "configs" / "phase2_labelled_requests.yaml"
API = "/redfish/v1/Systems/1"


def _load_script():
    """Import the gate without invoking its CLI entrypoint."""
    spec = importlib.util.spec_from_file_location("d1_judge_calibration_gate", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _judge_raw(*, accepted: bool) -> str:
    """Return one strict judge verdict in the gate's raw JSON string field."""
    return json.dumps(
        {
            "accepted": accepted,
            "natural": accepted,
            "nonsense": False,
            "ambiguous": False,
            "duplicate_intent": False,
            "extra_intents": False,
            "method_semantics_valid": True,
            "covered_api_set": [API] if accepted else [],
            "reason": "fixture",
        },
        sort_keys=True,
    )


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def test_d1_judge_calibration_gate_emits_positive_negative_example_counts(
    tmp_path: Path,
    capsys,
) -> None:
    """Calibration metrics include total, human-positive, and human-negative rows."""
    script = _load_script()
    calibration = _write_jsonl(
        tmp_path / "calibration.jsonl",
        [
            {
                "selected_api_set": [API],
                "human_accept": True,
                "judge_raw": _judge_raw(accepted=True),
            },
            {
                "selected_api_set": [API],
                "human_accept": False,
                "judge_raw": _judge_raw(accepted=False),
            },
        ],
    )
    output = tmp_path / "metrics.json"

    rc = script.main(
        [
            "--spec",
            str(SPEC),
            "--calibration-jsonl",
            str(calibration),
            "--output-json",
            str(output),
        ]
    )

    assert rc == 0
    assert json.loads(capsys.readouterr().out)["status"] == "pass"
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["metrics"]["positive_examples"] == 1
    assert payload["metrics"]["negative_examples"] == 1
    assert payload["metrics"]["examples"] == 2


def test_d1_judge_calibration_gate_rejects_one_sided_human_labels(
    tmp_path: Path,
    capsys,
) -> None:
    """A calibration source with only human-positive rows is not admissible evidence."""
    script = _load_script()
    calibration = _write_jsonl(
        tmp_path / "calibration.jsonl",
        [
            {
                "selected_api_set": [API],
                "human_accept": True,
                "judge_raw": _judge_raw(accepted=True),
            }
        ],
    )

    rc = script.main(
        [
            "--spec",
            str(SPEC),
            "--calibration-jsonl",
            str(calibration),
            "--output-json",
            str(tmp_path / "metrics.json"),
        ]
    )

    assert rc == 2
    assert "human-accepted and human-rejected" in capsys.readouterr().err
