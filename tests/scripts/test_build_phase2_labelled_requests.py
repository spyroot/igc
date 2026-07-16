"""Offline tests for ``scripts/build_phase2_labelled_requests.py``."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from igc.modules.base.metric_keys import phase_metric

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "build_phase2_labelled_requests.py"
CONFIG = REPO_ROOT / "configs" / "phase2_labelled_requests.yaml"


def _load_script():
    """Import the script as a module without requiring PYTHONPATH setup."""
    spec = importlib.util.spec_from_file_location("build_phase2_labelled_requests", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_capture(root: Path, *, rest_api: str = "/redfish/v1/Systems/1") -> None:
    """Write one tiny Redfish resource capture."""
    root.mkdir(parents=True)
    filename = rest_api.strip("/").replace("/", "_")
    (root / f"_{filename}.json").write_text(json.dumps({
        "@odata.id": rest_api,
        "@odata.type": "#ComputerSystem.v1_20_0.ComputerSystem",
        "Name": "System 1",
        "PowerState": "On",
    }))


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write object rows as JSONL."""
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _read_jsonl(path: Path) -> list[dict]:
    """Read JSONL object rows."""
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_phase2_labelled_requests_cli_writes_rows_and_metrics(tmp_path: Path) -> None:
    """The CLI binds offline provider JSONL to accepted rows and metric keys."""
    script = _load_script()
    capture = tmp_path / "capture"
    _write_capture(capture)
    drafts = tmp_path / "drafts.jsonl"
    judges = tmp_path / "judges.jsonl"
    output = tmp_path / "phase2_labelled_requests.jsonl"
    metrics_out = tmp_path / "metrics.json"
    _write_jsonl(drafts, [
        {"text": "check the first system"},
        {"text": "nonsense"},
    ])
    _write_jsonl(judges, [
        {
            "accepted": True,
            "rest_api_list": ["/redfish/v1/Systems/1"],
            "nonsense": False,
            "reason": "matches the sampled system",
        },
        {
            "accepted": False,
            "rest_api_list": [],
            "nonsense": True,
            "reason": "not a useful operator request",
        },
    ])

    code = script.main([
        "--spec", str(CONFIG),
        "--capture-root", str(capture),
        "--vendor", "dell",
        "--drafts-jsonl", str(drafts),
        "--judges-jsonl", str(judges),
        "--output-jsonl", str(output),
        "--metrics-out", str(metrics_out),
        "--sample-width", "1",
    ])

    assert code == 0
    rows = _read_jsonl(output)
    assert len(rows) == 1
    row = rows[0]
    assert row["dataset"] == "phase2_labelled_requests"
    assert row["task"] == "text_to_rest_api_set"
    assert row["x"]["text"] == "check the first system"
    assert row["y_true"]["rest_api_set"] == ["/redfish/v1/Systems/1"]
    assert row["validation"]["rest_api_set_match"] is True
    metrics = json.loads(metrics_out.read_text())
    namespace = "phase2_labelled_requests"
    assert metrics["drafted_rows"] == 2
    assert metrics["accepted_rows"] == 1
    assert metrics["rejected_rows"] == 1
    assert metrics["metrics"][phase_metric(namespace, "build", "draft_total")] == 2
    assert metrics["metrics"][phase_metric(namespace, "eval", "nonsense_rate")] == 0.5
    assert metrics["metrics"][phase_metric(namespace, "vendor", "source_corpus")] == (
        "dell/phase2_capture"
    )


def test_phase2_labelled_requests_cli_loads_redfish_ctl_manifest_root(
    tmp_path: Path,
) -> None:
    """The CLI can consume a redfish_ctl manifest plus materialized corpus root."""
    script = _load_script()
    manifest = tmp_path / "manifest.v1.json"
    materialized = tmp_path / "materialized"
    capture = materialized / "dataset" / "dell_xr8620t_full_corpus"
    _write_capture(capture, rest_api="/redfish/v1/Managers/1")
    (capture / "rest_api_map.v1.json").write_text(json.dumps({
        "allowed_methods_mapping": {
            "/redfish/v1/Managers/1": ["GET", "HEAD", "PATCH"],
        },
    }))
    manifest.write_text(json.dumps({
        "corpora": [{
            "id": "dell-xr8620t",
            "kind": "dataset",
            "vendor": "dell",
            "archive": "dell_xr8620t_full_corpus.tar.gz",
        }],
    }))
    drafts = tmp_path / "drafts.jsonl"
    judges = tmp_path / "judges.jsonl"
    output = tmp_path / "phase2.jsonl"
    metrics_out = tmp_path / "metrics.json"
    _write_jsonl(drafts, [{"text": "inspect the manager"}])
    _write_jsonl(judges, [{
        "accepted": True,
        "rest_api_list": ["/redfish/v1/Managers/1"],
        "nonsense": False,
    }])

    code = script.main([
        "--spec", str(CONFIG),
        "--corpus-manifest", str(manifest),
        "--corpus-root", str(materialized),
        "--corpus-id", "dell-xr8620t",
        "--drafts-jsonl", str(drafts),
        "--judges-jsonl", str(judges),
        "--output-jsonl", str(output),
        "--metrics-out", str(metrics_out),
        "--sample-width", "1",
    ])

    assert code == 0
    row = _read_jsonl(output)[0]
    assert row["x"]["records"][0]["rest_api"] == "/redfish/v1/Managers/1"
    assert row["x"]["records"][0]["allowed_methods"] == ["GET", "HEAD", "PATCH"]
    assert json.loads(metrics_out.read_text())["metrics"][
        phase_metric("phase2_labelled_requests", "vendor", "source_corpus")
    ] == "dell/dell-xr8620t"
