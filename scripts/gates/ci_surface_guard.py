#!/usr/bin/env python3
"""Gate: ci.surface-guard — CPU and GPU passes never mix surfaces.

The pass matrix (``configs/ci/pass_matrix.yaml``) is the single source for
where every validation pass runs: cpu passes on the shared-code CIs (GitHub
Actions + the internal GitLab ``homelab-k8s`` in-cluster runner, which has NO
GPU), gpu passes only on approved GB300 slot containers with their results
recorded as evidence manifests. This gate enforces that mechanically:

* every ``.gitlab-ci.yml`` job carries exactly the runner tag its surface
  demands (cpu -> ``homelab-k8s``; gpu -> ``gb300``, a tag with no home-lab
  runner so a mis-tagged job cannot schedule);
* no home-lab (cpu-tagged) job requests a GPU (``nvidia.com/gpu``, ``--gpus``,
  CUDA device envs) — the ``policy.no-gpu-homelab`` rule;
* gpu-surface jobs never run in merge-request pipelines (no live GPU work from
  an MR) — they may only validate committed evidence;
* committed GPU evidence manifests under ``reports/gpu-evidence/`` conform to
  the matrix's evidence schema, and their pass names exist in the matrix.

Used by:
  tests/gates/test_ci_surface_guard.py  (offline gate; runs in `pytest -q`)
  .gitlab-ci.yml cpu-gate stage
  CLI: python scripts/gates/ci_surface_guard.py

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
MATRIX_PATH = REPO_ROOT / "configs" / "ci" / "pass_matrix.yaml"
GITLAB_CI_PATH = REPO_ROOT / ".gitlab-ci.yml"
EVIDENCE_DIR = REPO_ROOT / "reports" / "gpu-evidence"

# Patterns that mean "this job wants a GPU". Any of these in a cpu-tagged job
# is a policy.no-gpu-homelab violation.
_GPU_MARKERS = ("nvidia.com/gpu", "--gpus", "CUDA_VISIBLE_DEVICES", "nvidia-smi")

# GitLab CI reserved top-level keys that are not jobs.
_RESERVED = {
    "stages", "variables", "default", "include", "workflow", "image",
    "services", "before_script", "after_script", "cache",
}


def load_matrix(path: Path = MATRIX_PATH) -> dict[str, Any]:
    """Load the pass matrix."""
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _jobs(ci: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Extract job definitions from a parsed .gitlab-ci.yml."""
    return {
        name: value
        for name, value in ci.items()
        if name not in _RESERVED and not name.startswith(".") and isinstance(value, Mapping)
    }


def check_gitlab_ci(
    ci: Mapping[str, Any],
    matrix: Mapping[str, Any],
) -> list[str]:
    """Validate the GitLab pipeline against the pass matrix.

    :param ci: parsed .gitlab-ci.yml content.
    :param matrix: parsed pass matrix.
    :return: list of human-readable violations; empty means compliant.
    """
    violations: list[str] = []
    tags = matrix["runner_tags"]
    cpu_tag, gpu_tag = tags["cpu"], tags["gpu"]

    for name, job in _jobs(ci).items():
        job_tags = list(job.get("tags") or [])
        surface = str(job.get("variables", {}).get("IGC_SURFACE", ""))
        if surface not in ("cpu", "gpu"):
            violations.append(f"job {name}: missing IGC_SURFACE variable (cpu|gpu)")
            continue
        expected_tag = cpu_tag if surface == "cpu" else gpu_tag
        if job_tags != [expected_tag]:
            violations.append(
                f"job {name}: surface {surface} must carry exactly tag "
                f"[{expected_tag}], got {job_tags}"
            )
        blob = json.dumps(job)
        if surface == "cpu":
            for marker in _GPU_MARKERS:
                if marker in blob:
                    violations.append(
                        f"job {name}: cpu-surface job references GPU marker {marker!r} "
                        "(policy.no-gpu-homelab)"
                    )
        else:
            # gpu-surface jobs must never run in merge-request pipelines.
            rules = json.dumps(job.get("rules", []))
            if "merge_request_event" in rules:
                violations.append(
                    f"job {name}: gpu-surface job runs in a merge-request pipeline "
                    "(no live GPU work from MRs)"
                )
            if not job.get("when") == "manual" and not job.get("rules"):
                violations.append(
                    f"job {name}: gpu-surface job must be manual or rule-gated"
                )
    return violations


def check_evidence(matrix: Mapping[str, Any], evidence_dir: Path = EVIDENCE_DIR) -> list[str]:
    """Validate committed GPU evidence manifests against the matrix schema.

    An empty/missing evidence directory is fine — evidence requiredness is
    enforced at phase-acceptance time, not per commit; this check only rejects
    MALFORMED evidence so a bad manifest cannot masquerade as proof.
    """
    violations: list[str] = []
    if not evidence_dir.exists():
        return violations
    required = list(matrix["evidence_schema"]["required_fields"])
    gpu_passes = {
        name for name, spec in matrix["passes"].items() if spec.get("surface") == "gpu"
    }
    for manifest_path in sorted(evidence_dir.glob("*.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            violations.append(f"{manifest_path.name}: invalid JSON ({exc.msg})")
            continue
        for field in required:
            if field not in manifest:
                violations.append(f"{manifest_path.name}: missing field {field}")
        pass_name = manifest.get("pass_name")
        if pass_name is not None and pass_name not in gpu_passes:
            violations.append(
                f"{manifest_path.name}: unknown gpu pass {pass_name!r}"
            )
        if manifest.get("status") not in ("pass", "fail", None):
            violations.append(f"{manifest_path.name}: status must be pass|fail")
    return violations


def check_matrix(matrix: Mapping[str, Any]) -> list[str]:
    """Sanity-check the matrix itself: two surfaces, binding cpu tag."""
    violations: list[str] = []
    if matrix.get("runner_tags", {}).get("cpu") != "homelab-k8s":
        violations.append("runner_tags.cpu must be 'homelab-k8s' (binding)")
    for name, spec in matrix.get("passes", {}).items():
        if spec.get("surface") not in ("cpu", "gpu"):
            violations.append(f"pass {name}: surface must be cpu|gpu")
        if spec.get("surface") == "gpu" and spec.get("evidence") != "required":
            violations.append(f"pass {name}: gpu pass must declare evidence: required")
    return violations


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    _ = argv
    matrix = load_matrix()
    violations = check_matrix(matrix)
    if GITLAB_CI_PATH.exists():
        ci = yaml.safe_load(GITLAB_CI_PATH.read_text(encoding="utf-8"))
        violations += check_gitlab_ci(ci, matrix)
    violations += check_evidence(matrix)
    for violation in violations:
        print(f"BLOCKER: {violation}", file=sys.stderr)
    if not violations:
        print("OK: CPU/GPU surface separation holds.")
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())


# Author: Mus mbayramo@stanford.edu
