"""Offline tests for ``scripts/build_goal_dataset.py``.

Author:
Mus mbayramo@stanford.edu
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

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
    root.mkdir(parents=True)
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


def test_build_goal_dataset_infers_vendor_from_known_redfish_ctl_roots(
    tmp_path: Path,
) -> None:
    """Multi-root lab builds keep vendor provenance for held-out splits."""
    script = _load_script()
    redfish_ctl = tmp_path / "redfish_ctl" / "tests"
    dell = redfish_ctl / "idrac_fixtures"
    supermicro = redfish_ctl / "supermicro_fixtures"
    hpe = redfish_ctl / "hpe_fixtures"
    for root in (dell, supermicro, hpe):
        _write_capture(root)

    surfaces_out = tmp_path / "goal_surfaces.jsonl"
    manifest_out = tmp_path / "manifest.json"

    code = script.main([
        "--capture-root", str(dell),
        "--capture-root", str(supermicro),
        "--capture-root", str(hpe),
        "--source", "full_redfish_corpus",
        "--surfaces-out", str(surfaces_out),
        "--manifest-out", str(manifest_out),
    ])

    assert code == 0
    manifest = json.loads(manifest_out.read_text())
    assert manifest["vendors"] == ["dell", "hpe", "supermicro"]
    surfaces = read_goal_surfaces(surfaces_out)
    assert {surface.vendor for surface in surfaces} == {
        "dell",
        "hpe",
        "supermicro",
    }


def test_build_goal_dataset_vendor_inference_avoids_substring_false_positive() -> None:
    """A misleading directory name should not silently relabel a capture."""
    script = _load_script()

    assert script._infer_vendor_from_root("/tmp/not_dell_capture/10.0.0.1") == ""
    assert script._infer_vendor_from_root("/tmp/hpe_dl360_full_corpus/10.0.0.2") == "hpe"


def test_build_goal_dataset_loads_allowed_methods_from_rest_api_map(
    tmp_path: Path,
) -> None:
    """Full corpora keep same-run method maps for action/reward consumers."""
    script = _load_script()
    capture = tmp_path / "dell_xr8620t_full_corpus" / "10.0.0.1"
    _write_capture(capture)
    (capture / "corpus_manifest.json").write_text(json.dumps({
        "artifact_type": "full_training",
        "json_file_count": 2,
    }))
    np.save(
        capture / "rest_api_map.npy",
        {
            "url_file_mapping": {
                "/redfish/v1/Systems/1": "_redfish_v1_Systems_1.json",
                "/redfish/v1/Managers/1/NetworkProtocol": (
                    "_redfish_v1_Managers_1_NetworkProtocol.json"
                ),
            },
            "allowed_methods_mapping": {
                "/redfish/v1/Systems/1": ["GET", "HEAD", "PATCH"],
                "/redfish/v1/Managers/1/NetworkProtocol": ["GET", "HEAD", "PATCH"],
            },
        },
    )
    surfaces_out = tmp_path / "goal_surfaces.jsonl"

    code = script.main([
        "--capture-root", str(capture),
        "--source", "full_redfish_corpus",
        "--surfaces-out", str(surfaces_out),
    ])

    assert code == 0
    power_surface = next(
        surface
        for surface in read_goal_surfaces(surfaces_out)
        if surface.goal_ref.goal_id == "power.computer_system.PowerState.eq.On"
    )
    assert power_surface.vendor == "dell"
    assert power_surface.provenance["allowed_methods"] == ["GET", "HEAD", "PATCH"]


def test_build_goal_dataset_ignores_method_map_without_manifest(tmp_path: Path) -> None:
    """Do not pickle-load loose rest_api_map.npy files outside full-corpus roots."""
    script = _load_script()
    capture = tmp_path / "dell_xr8620t_full_corpus" / "10.0.0.1"
    _write_capture(capture)
    np.save(
        capture / "rest_api_map.npy",
        {
            "allowed_methods_mapping": {
                "/redfish/v1/Systems/1": ["GET", "HEAD", "PATCH"],
            },
        },
    )
    surfaces_out = tmp_path / "goal_surfaces.jsonl"

    code = script.main([
        "--capture-root", str(capture),
        "--source", "full_redfish_corpus",
        "--surfaces-out", str(surfaces_out),
    ])

    assert code == 0
    power_surface = next(
        surface
        for surface in read_goal_surfaces(surfaces_out)
        if surface.goal_ref.goal_id == "power.computer_system.PowerState.eq.On"
    )
    assert "allowed_methods" not in power_surface.provenance


def test_build_goal_dataset_rejects_non_dict_rest_api_map(tmp_path: Path) -> None:
    """Malformed full-corpus map payloads fail before writing a dataset."""
    script = _load_script()
    capture = tmp_path / "dell_xr8620t_full_corpus" / "10.0.0.1"
    _write_capture(capture)
    (capture / "corpus_manifest.json").write_text(json.dumps({
        "artifact_type": "full_training",
        "json_file_count": 2,
    }))
    np.save(capture / "rest_api_map.npy", ["not", "a", "mapping"])
    surfaces_out = tmp_path / "goal_surfaces.jsonl"

    with pytest.raises(SystemExit, match="rest_api_map.npy is not a dict"):
        script.main([
            "--capture-root", str(capture),
            "--source", "full_redfish_corpus",
            "--surfaces-out", str(surfaces_out),
        ])
