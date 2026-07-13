"""Offline tests for ``scripts/build_goal_dataset.py``.

Author:
Mus mbayramo@stanford.edu
"""

import importlib.util
import json
from pathlib import Path

from igc.ds.goal_dataset import read_goal_surfaces, read_goal_text_examples

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "build_goal_dataset.py"


def _load_script():
    """Import the script as a module without requiring PYTHONPATH setup."""
    spec = importlib.util.spec_from_file_location("build_goal_dataset", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_capture(root: Path) -> None:
    """Write tiny ComputerSystem and ManagerNetworkProtocol captures."""
    root.mkdir()
    (root / "_redfish_v1_Systems_1.json").write_text(json.dumps({
        "@odata.id": "/redfish/v1/Systems/1",
        "@odata.type": "#ComputerSystem.v1_20_0.ComputerSystem",
        "PowerState": "Off",
        "PowerState@Redfish.AllowableValues": ["On", "Off"],
    }))
    (root / "_redfish_v1_Managers_1_NetworkProtocol.json").write_text(json.dumps({
        "@odata.id": "/redfish/v1/Managers/1/NetworkProtocol",
        "@odata.type": "#ManagerNetworkProtocol.v1_10_0.ManagerNetworkProtocol",
        "NTP": {"ProtocolEnabled": False},
    }))


def test_build_goal_dataset_writes_surfaces_and_text_drafts(tmp_path: Path) -> None:
    """CLI builds deterministic Y rows and attaches supplied/generated X rows."""
    script = _load_script()
    capture = tmp_path / "capture"
    _write_capture(capture)
    surfaces_out = tmp_path / "goal_surfaces.jsonl"
    text_out = tmp_path / "goal_text_examples.jsonl"

    code = script.main([
        "--capture-root", str(capture),
        "--vendor", "dell",
        "--source", "real_dell",
        "--surfaces-out", str(surfaces_out),
        "--text-out", str(text_out),
        "--paraphrase-mode", "static",
        "--goal-id", "power.computer_system.PowerState.eq.On",
        "--goal-id", "network.manager_network_protocol.NTP.ProtocolEnabled.eq.True",
        "--static-text", "boot server and set ntp",
        "--static-text", "set ntp then boot server",
    ])

    assert code == 0
    surfaces = read_goal_surfaces(surfaces_out)
    assert {surface.goal_ref.goal_id for surface in surfaces} >= {
        "power.computer_system.PowerState.eq.On",
        "network.manager_network_protocol.NTP.ProtocolEnabled.eq.True",
    }
    examples = read_goal_text_examples(text_out)
    assert [example.text for example in examples] == [
        "boot server and set ntp",
        "set ntp then boot server",
    ]
    example = examples[0]
    assert [ref.goal_id for ref in example.goal_refs] == [
        "power.computer_system.PowerState.eq.On",
        "network.manager_network_protocol.NTP.ProtocolEnabled.eq.True",
    ]
    assert example.metadata["validation"] == "llm_generated_unvalidated"


def test_build_goal_dataset_can_generate_text_for_every_atomic_goal(
    tmp_path: Path,
) -> None:
    """Lab jobs can ask for X drafts over every discovered atomic GoalRef."""
    script = _load_script()
    capture = tmp_path / "capture"
    _write_capture(capture)
    surfaces_out = tmp_path / "goal_surfaces.jsonl"
    text_out = tmp_path / "goal_text_examples.jsonl"
    manifest_out = tmp_path / "manifest.json"

    code = script.main([
        "--capture-root", str(capture),
        "--surfaces-out", str(surfaces_out),
        "--text-out", str(text_out),
        "--manifest-out", str(manifest_out),
        "--paraphrase-mode", "static",
        "--generate-all-goals",
        "--count", "1",
        "--static-text", "generic operator request",
    ])

    assert code == 0
    surfaces = read_goal_surfaces(surfaces_out)
    examples = read_goal_text_examples(text_out)
    unique_goal_ids = {surface.goal_ref.goal_id for surface in surfaces}
    assert len(examples) == len(unique_goal_ids)
    assert {example.goal_refs[0].goal_id for example in examples} == unique_goal_ids
    manifest = json.loads(manifest_out.read_text())
    assert manifest["capture_records"] == 2
    assert manifest["unique_goal_ids"] == len(unique_goal_ids)
    assert manifest["text_examples"] == len(unique_goal_ids)
