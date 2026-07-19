"""Offline tests for the CPU/GPU surface-separation gate.

The committed pipeline + matrix must pass; representative violations (a cpu
job requesting a GPU, a gpu job in an MR pipeline, a wrong runner tag, a
malformed evidence manifest) must fail.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import yaml

SCRIPT = Path("scripts/gates/ci_surface_guard.py")


def _load_gate() -> ModuleType:
    """Load the gate script module for direct testing."""
    spec = importlib.util.spec_from_file_location("ci_surface_guard", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _ci() -> dict:
    """The committed .gitlab-ci.yml, parsed."""
    return yaml.safe_load(Path(".gitlab-ci.yml").read_text(encoding="utf-8"))


def test_committed_pipeline_and_matrix_pass() -> None:
    """The committed matrix, pipeline, and (absent) evidence all validate."""
    gate = _load_gate()
    matrix = gate.load_matrix()

    assert gate.check_matrix(matrix) == []
    assert gate.check_gitlab_ci(_ci(), matrix) == []
    assert gate.check_evidence(matrix) == []


def test_cpu_job_requesting_gpu_fails() -> None:
    """A homelab job referencing a GPU marker violates policy.no-gpu-homelab."""
    gate = _load_gate()
    matrix = gate.load_matrix()
    ci = _ci()
    ci["lint"]["script"].append("nvidia-smi")

    violations = gate.check_gitlab_ci(ci, matrix)
    assert any("no-gpu-homelab" in v for v in violations)


def test_wrong_runner_tag_fails() -> None:
    """A cpu-surface job on any tag but homelab-k8s fails."""
    gate = _load_gate()
    matrix = gate.load_matrix()
    ci = _ci()
    ci["lint"]["tags"] = ["gb300"]

    violations = gate.check_gitlab_ci(ci, matrix)
    assert any("must carry exactly tag" in v for v in violations)


def test_gpu_job_in_merge_request_pipeline_fails() -> None:
    """A gpu-surface job wired into MR pipelines fails (no live GPU from MRs)."""
    gate = _load_gate()
    matrix = gate.load_matrix()
    ci = _ci()
    ci["gb300-surface"]["rules"] = [
        {"if": '$CI_PIPELINE_SOURCE == "merge_request_event"'},
    ]

    violations = gate.check_gitlab_ci(ci, matrix)
    assert any("merge-request pipeline" in v for v in violations)


def test_job_without_surface_variable_fails() -> None:
    """Every job must declare its surface explicitly."""
    gate = _load_gate()
    matrix = gate.load_matrix()
    ci = _ci()
    del ci["lint"]["variables"]

    violations = gate.check_gitlab_ci(ci, matrix)
    assert any("missing IGC_SURFACE" in v for v in violations)


def test_malformed_evidence_manifest_fails(tmp_path: Path) -> None:
    """A GPU evidence manifest missing schema fields is rejected."""
    gate = _load_gate()
    matrix = gate.load_matrix()
    evidence = tmp_path / "gpu-evidence"
    evidence.mkdir()
    (evidence / "run1.json").write_text(
        json.dumps({"pass_name": "ddp_same_node_sanity", "status": "pass"}),
        encoding="utf-8",
    )

    violations = gate.check_evidence(matrix, evidence)
    assert any("missing field commit_sha" in v for v in violations)


def test_unknown_gpu_pass_in_evidence_fails(tmp_path: Path) -> None:
    """Evidence naming a pass outside the matrix is rejected."""
    gate = _load_gate()
    matrix = gate.load_matrix()
    evidence = tmp_path / "gpu-evidence"
    evidence.mkdir()
    manifest = {field: "x" for field in matrix["evidence_schema"]["required_fields"]}
    manifest["pass_name"] = "made_up_pass"
    manifest["status"] = "pass"
    (evidence / "run1.json").write_text(json.dumps(manifest), encoding="utf-8")

    violations = gate.check_evidence(matrix, evidence)
    assert any("unknown gpu pass" in v for v in violations)


def test_unfrozen_matrix_cpu_tag_fails(tmp_path: Path) -> None:
    """Changing the binding homelab-k8s cpu tag fails the matrix sanity check."""
    gate = _load_gate()
    matrix = gate.load_matrix()
    matrix["runner_tags"]["cpu"] = "anywhere"

    violations = gate.check_matrix(matrix)
    assert any("homelab-k8s" in v for v in violations)


# Author: Mus mbayramo@stanford.edu
