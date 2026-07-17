"""Gate runner: repo.contract-authority — one command that proves the Phase 2/3 contract.

This runner is the single entry point a Phase 2/3 PR (or CI) invokes to produce the
observable gate evidence required before that PR can be accepted. It composes the
existing repo-level gates and the offline contract/output/envelope tests into one
sanitized JSON report:

* ``repo.contract-single-source`` — one canonical contract module, no forked
  namespaces, no parallel islands (``scripts/gates/contract_single_source.py``).
* ``repo.schema-snapshot`` — the approved Phase 2/3 row schema has not drifted
  (``scripts/gates/schema_snapshot.py``); reports ``bootstrap`` until the snapshot
  is committed, which is not a failure.
* ``offline.contract-tests`` — the canonical contract row/parse/eval tests
  (``tests/ds/test_rest_goal_contract.py``).
* ``offline.gate-tests`` — the repo-gate + inference-envelope tests
  (``tests/gates/``), including the deterministic exact-match envelope layer.
* ``offline.phase123-conformance`` — a deterministic Phase 1 -> D1 -> Phase 2
  -> Phase 3 key/shape/output walk, with fixture model and judge responses.

It writes ``reports/gate-report-contract-authority.json`` with a pass/fail (or
``bootstrap``) status per check and an overall verdict. The report is SANITIZED:
it carries only check names, statuses, return codes, and a scrubbed one-line
summary — never IPs, hostnames, tokens, file bodies, or environment values. The
exit code is non-zero if any check FAILED (``bootstrap`` does not fail the run).

Execution surface (requirement 5). This runner is intended for CI
(``.github/workflows/ci.yml`` gate job) or an approved remote/container
container — NOT the operator laptop, per the standing test-execution guard. It
does no host detection and adds no laptop-refusal logic on purpose: the surface
is enforced by where it is invoked (CI / approved container), and any real-model
step lives behind the ``@pytest.mark.gpu`` marker that the offline pytest run
excludes, so this runner performs no GPU work itself.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT = "reports/gate-report-contract-authority.json"
DEFAULT_REGISTRY = "configs/contracts/rest_goal.yaml"
DEFAULT_SNAPSHOT = "schemas/snapshots/rest_goal_contract.shape.json"

# Offline pytest targets composed into the authority gate. Kept relative so the
# report never leaks an absolute checkout path.
_CONTRACT_TEST_TARGETS = ("tests/ds/test_rest_goal_contract.py",)
_GATE_TEST_TARGETS = ("tests/gates",)
_PHASE123_REPORT = "reports/gate-report-phase123-conformance.json"

# Redact IPv4 addresses so a scrubbed pytest summary can never leak an endpoint.
_IPV4_RE = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")


def _load_gate(name: str, relpath: str) -> Any:
    """Load a sibling gate script (not an installed package) by file path.

    :param name: module name to register the loaded script under.
    :param relpath: repo-relative path to the gate script.
    :return: the loaded module object.
    """
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relpath)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load gate script: {relpath}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _sanitize(text: str, limit: int = 200) -> str:
    """Reduce arbitrary tool output to a short, IP-free, single-line summary.

    :param text: raw stdout/stderr from a check.
    :param limit: maximum characters to retain.
    :return: a scrubbed one-line summary safe to commit/log.
    """
    # Prefer the pytest result line (e.g. "5 passed in 0.12s") when present.
    summary = ""
    for line in reversed(text.strip().splitlines()):
        stripped = line.strip().strip("= ")
        if stripped:
            summary = stripped
            break
    summary = _IPV4_RE.sub("[redacted-ip]", summary)
    summary = " ".join(summary.split())
    return summary[:limit]


def _run_single_source(registry: Path) -> dict[str, Any]:
    """Run ``repo.contract-single-source`` in-process and record its verdict."""
    css = _load_gate("contract_single_source", "scripts/gates/contract_single_source.py")
    try:
        violations = css.check(registry, REPO_ROOT)
    except (OSError, ValueError, SystemExit) as exc:
        return {
            "name": "repo.contract-single-source",
            "status": "fail",
            "detail": f"registry error: {_sanitize(str(exc))}",
        }
    if violations:
        return {
            "name": "repo.contract-single-source",
            "status": "fail",
            "detail": f"{len(violations)} violation(s); first: {_sanitize(violations[0])}",
        }
    return {"name": "repo.contract-single-source", "status": "pass", "detail": "one canonical contract"}


def _run_schema_snapshot(snapshot: Path) -> dict[str, Any]:
    """Run ``repo.schema-snapshot`` in-process; bootstrap is not a failure."""
    ss = _load_gate("schema_snapshot", "scripts/gates/schema_snapshot.py")
    try:
        code = ss.check(snapshot)
    except Exception as exc:  # noqa: BLE001 - gate converts any script failure into evidence.
        return {
            "name": "repo.schema-snapshot",
            "status": "fail",
            "detail": f"snapshot error: {_sanitize(str(exc))}",
        }
    if code == ss.EXIT_OK:
        return {"name": "repo.schema-snapshot", "status": "pass", "detail": "schema matches snapshot"}
    if code == ss.EXIT_BOOTSTRAP:
        return {
            "name": "repo.schema-snapshot",
            "status": "bootstrap",
            "detail": "snapshot not committed yet; run --update on an approved surface",
        }
    return {"name": "repo.schema-snapshot", "status": "fail", "detail": "schema drifted from snapshot"}


def _run_pytest(name: str, targets: tuple[str, ...]) -> dict[str, Any]:
    """Run an offline pytest subset in a child process and record its verdict.

    :param name: report check name for this pytest subset.
    :param targets: repo-relative pytest paths (gpu-marked tests stay excluded by
        ``pytest.ini`` ``addopts``, so no GPU is touched here).
    :return: check record with pass/fail status, return code, and scrubbed summary.
    """
    env = dict(os.environ)
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env.setdefault("HF_DATASETS_OFFLINE", "1")
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *targets],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    status = "pass" if completed.returncode == 0 else "fail"
    summary = _sanitize(completed.stdout or completed.stderr)
    return {
        "name": name,
        "status": status,
        "returncode": completed.returncode,
        "detail": summary,
    }


def _run_phase123_conformance() -> dict[str, Any]:
    """Run the deterministic Phase 1->D1->Phase 2->Phase 3 conformance gate."""

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/gates/phase123_conformance.py",
            "--report",
            _PHASE123_REPORT,
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    return {
        "name": "offline.phase123-conformance",
        "status": "pass" if completed.returncode == 0 else "fail",
        "returncode": completed.returncode,
        "detail": _sanitize(completed.stdout or completed.stderr),
    }


def run(registry: Path, snapshot: Path) -> dict[str, Any]:
    """Run every contract-authority check and assemble the report structure.

    :param registry: path to the contract registry YAML.
    :param snapshot: path to the committed schema-shape snapshot.
    :return: report mapping with per-check records and an overall verdict.
    """
    checks = [
        _run_single_source(registry),
        _run_schema_snapshot(snapshot),
        _run_pytest("offline.contract-tests", _CONTRACT_TEST_TARGETS),
        _run_pytest("offline.gate-tests", _GATE_TEST_TARGETS),
        _run_phase123_conformance(),
    ]
    failed = [c for c in checks if c["status"] == "fail"]
    return {
        "gate": "repo.contract-authority",
        "overall": "fail" if failed else "pass",
        "failed_count": len(failed),
        "checks": checks,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the ``repo.contract-authority`` gate runner."""
    parser = argparse.ArgumentParser(
        description="Run the Phase 2/3 contract-authority gate and write a sanitized report."
    )
    parser.add_argument("--registry", default=DEFAULT_REGISTRY, help="Contract registry YAML (repo-relative).")
    parser.add_argument("--snapshot", default=DEFAULT_SNAPSHOT, help="Schema-shape snapshot (repo-relative).")
    parser.add_argument("--report", default=DEFAULT_REPORT, help="Sanitized JSON report output path (repo-relative).")
    args = parser.parse_args(argv)

    report = run(REPO_ROOT / args.registry, REPO_ROOT / args.snapshot)

    report_path = REPO_ROOT / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    for check in report["checks"]:
        print(f"[{check['status'].upper():9}] {check['name']}: {check.get('detail', '')}")
    print(f"contract-authority: {report['overall'].upper()} (report: {args.report})")
    return 1 if report["overall"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())


# Author: Mus mbayramo@stanford.edu
