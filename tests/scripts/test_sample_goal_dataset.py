"""Offline tests for ``scripts/sample_goal_dataset.py``.

Author:
Mus mbayramo@stanford.edu
"""

import importlib.util
import json
import tarfile
from pathlib import Path

import pytest

from igc.ds.goal_dataset import GoalRef, GoalSurface, GoalTextExample

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "sample_goal_dataset.py"


def _load_script():
    """Import the script as a module without requiring PYTHONPATH setup."""
    spec = importlib.util.spec_from_file_location("sample_goal_dataset", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write JSON Lines test data."""
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_sample_goal_dataset_prints_manifest_surfaces_and_text_examples(
    tmp_path: Path,
    capsys,
) -> None:
    """Sampler summarizes a built dataset without printing full verifier payloads."""
    script = _load_script()
    goal_ref = GoalRef(
        goal_id="power.computer_system.PowerState.eq.On",
        family="power",
        resource_type="computer_system",
        property_path="PowerState",
        target_value="On",
    )
    surface = GoalSurface(
        goal_ref=goal_ref,
        vendor="dell",
        source="real_dell",
        resource_uri="/redfish/v1/Systems/1",
        resource_type="#ComputerSystem.v1_20_0.ComputerSystem",
        fact_path="PowerState",
        target_value="On",
        current_value="Off",
        verifier={
            "kind": "state_eq",
            "resource_uri": "/redfish/v1/Systems/1",
            "property_path": "PowerState",
        },
    )
    text = GoalTextExample(
        text="turn the server on",
        goal_refs=(goal_ref,),
        text_source="llm_paraphrase",
    )
    dataset = tmp_path / "goal_dataset"
    dataset.mkdir()
    (dataset / "goal_dataset_manifest.json").write_text(json.dumps({
        "capture_records": 1,
        "goal_surfaces": 1,
        "text_examples": 1,
        "unique_goal_ids": 1,
        "vendors": ["dell"],
    }))
    _write_jsonl(dataset / "goal_surfaces.jsonl", [surface.to_dict()])
    _write_jsonl(dataset / "goal_text_examples.jsonl", [text.to_dict()])

    code = script.main(["--dataset-dir", str(dataset), "--limit", "1"])

    assert code == 0
    output = capsys.readouterr().out
    assert "capture_records: 1" in output
    assert "goal_surfaces: 1" in output
    assert "vendors: dell" in output
    assert "power.computer_system.PowerState.eq.On" in output
    assert "turn the server on" in output
    assert "verifier:" not in output


def test_sample_goal_dataset_filters_surface_family(tmp_path: Path, capsys) -> None:
    """Family filters keep local inspection focused."""
    script = _load_script()
    dataset = tmp_path / "goal_dataset"
    dataset.mkdir()
    refs = [
        GoalRef(
            goal_id="power.computer_system.PowerState.eq.On",
            family="power",
            resource_type="computer_system",
            property_path="PowerState",
            target_value="On",
        ),
        GoalRef(
            goal_id="network.manager_network_protocol.NTP.ProtocolEnabled.eq.True",
            family="network",
            resource_type="manager_network_protocol",
            property_path="NTP.ProtocolEnabled",
            target_value=True,
        ),
    ]
    rows = [
        GoalSurface(
            goal_ref=ref,
            vendor="hpe",
            source="real_hpe",
            resource_uri="/redfish/v1/R",
            resource_type="#R.v1_0_0.R",
            fact_path=ref.property_path,
            target_value=ref.target_value,
        ).to_dict()
        for ref in refs
    ]
    _write_jsonl(dataset / "goal_surfaces.jsonl", rows)

    code = script.main([
        "--dataset-dir",
        str(dataset),
        "--family",
        "network",
        "--limit",
        "5",
    ])

    assert code == 0
    output = capsys.readouterr().out
    assert "network.manager_network_protocol.NTP.ProtocolEnabled.eq.True" in output
    assert "power.computer_system.PowerState.eq.On" not in output


def test_sample_goal_dataset_reads_tar_artifact(tmp_path: Path, capsys) -> None:
    """Committed LFS tarballs can be sampled without manual extraction."""
    script = _load_script()
    dataset = tmp_path / "goal_dataset"
    dataset.mkdir()
    goal_ref = GoalRef(
        goal_id="power.computer_system.PowerState.eq.On",
        family="power",
        resource_type="computer_system",
        property_path="PowerState",
        target_value="On",
    )
    surface = GoalSurface(
        goal_ref=goal_ref,
        vendor="dell",
        source="full_redfish_corpus",
        resource_uri="/redfish/v1/Systems/1",
        resource_type="#ComputerSystem.v1_20_0.ComputerSystem",
        fact_path="PowerState",
        target_value="On",
    )
    (dataset / "goal_dataset_manifest.json").write_text(json.dumps({
        "capture_records": 1,
        "goal_surfaces": 1,
        "text_examples": 0,
        "unique_goal_ids": 1,
        "vendors": ["dell"],
    }))
    _write_jsonl(dataset / "goal_surfaces.jsonl", [surface.to_dict()])
    _write_jsonl(dataset / "goal_text_examples.jsonl", [])
    artifact = tmp_path / "goal_dataset.tar.gz"
    with tarfile.open(artifact, "w:gz") as tar:
        for name in (
            "goal_dataset_manifest.json",
            "goal_surfaces.jsonl",
            "goal_text_examples.jsonl",
        ):
            tar.add(dataset / name, arcname=name)

    code = script.main(["--dataset-tar", str(artifact), "--limit", "1"])

    assert code == 0
    output = capsys.readouterr().out
    assert "vendors: dell" in output
    assert "power.computer_system.PowerState.eq.On" in output


def test_sample_goal_dataset_rejects_tar_link_entries(tmp_path: Path) -> None:
    """Dataset tar extraction rejects links before writing any files."""
    script = _load_script()
    artifact = tmp_path / "unsafe_goal_dataset.tar.gz"
    with tarfile.open(artifact, "w:gz") as tar:
        info = tarfile.TarInfo("unsafe-link")
        info.type = tarfile.SYMTYPE
        info.linkname = "/tmp"
        tar.addfile(info)

    with pytest.raises(SystemExit, match="unsafe path in dataset tar"):
        script.main(["--dataset-tar", str(artifact)])
