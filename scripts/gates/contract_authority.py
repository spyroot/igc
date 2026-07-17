"""Gate runner: the Phase 1/2/3 contract-authority evidence layer.

Composes the repo-level contract gates and emits ONE sanitized evidence report a
Phase 2/3 PR must attach (or be marked BLOCKED). It runs the two in-process source
gates directly:

* ``repo.contract-single-source`` (``contract_single_source.check``) — one canonical
  Phase 2/3 contract, no forked namespaces, no retired parallel-contract modules.
* ``repo.schema-snapshot`` (``schema_snapshot.check``) — the canonical row shape has
  not drifted from the committed snapshot.

The dataset-shape, output-schema, and inference-envelope conformance TESTS run in the
normal offline pytest step (they live under ``tests/gates`` + the canonical
``tests/ds`` suite), so this runner does NOT re-run them by default — pass ``--full``
to include them when the runner is the standalone evidence job (GitLab k8s runner or a
GB300 container).

Execution surface (requirement 5): this runner refuses to run on the operator laptop.
It requires ``IGC_GATE_SURFACE`` to be one of ``ci`` / ``container`` / ``k8s`` (set by
the CI job or the k8s Job spec), matching the standing "no tests on the laptop" guard.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

_ALLOWED_SURFACES = {"ci", "container", "k8s"}


def _load(name: str, path: Path):
    """Load a sibling gate script (scripts/gates is not an importable package)."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load gate module {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _require_remote_surface() -> str:
    """Refuse to run unless on an approved remote surface (never the laptop).

    :return: the resolved surface string.
    :raises SystemExit: if ``IGC_GATE_SURFACE`` is not an approved remote surface.
    """
    surface = os.environ.get("IGC_GATE_SURFACE", "").strip().lower()
    if surface not in _ALLOWED_SURFACES:
        print(
            "BLOCKER: contract-authority gate must run on an approved remote surface, "
            "not the operator laptop. Set IGC_GATE_SURFACE to one of "
            f"{sorted(_ALLOWED_SURFACES)} in the CI job or k8s Job spec.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return surface


def _run_full_conformance(repo_root: Path) -> dict[str, Any]:
    """Optionally run the conformance test suites and capture pass/fail.

    :param repo_root: repository root.
    :return: ``{"ran": bool, "returncode": int, "targets": [...]}``.
    """
    targets = [
        "tests/gates",
        "tests/ds/test_rest_goal_contract.py",
        "tests/ds/test_phase2_labelled_requests.py",
    ]
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *targets],
        cwd=repo_root,
        env={**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE", "OMP_NUM_THREADS": "1"},
    )
    return {"ran": True, "returncode": proc.returncode, "targets": targets}


def run(repo_root: Path, *, full: bool, snapshot: Path, registry: Path) -> dict[str, Any]:
    """Run the in-process contract gates and build the sanitized report.

    :param repo_root: repository root.
    :param full: also run the offline conformance test suites.
    :param snapshot: schema-snapshot path.
    :param registry: contract registry path.
    :return: sanitized report dict (names + statuses only; no secrets/IPs).
    """
    gates_dir = repo_root / "scripts" / "gates"
    css = _load("contract_single_source", gates_dir / "contract_single_source.py")
    ss = _load("schema_snapshot", gates_dir / "schema_snapshot.py")

    checks: dict[str, str] = {}

    single_source = css.check(registry, repo_root)
    checks["contract.single-source"] = "PASS" if not single_source else "FAIL"

    snapshot_code = ss.check(snapshot)
    checks["repo.schema-snapshot"] = {
        ss.EXIT_OK: "PASS",
        ss.EXIT_FAIL: "FAIL",
        ss.EXIT_BOOTSTRAP: "BOOTSTRAP",  # snapshot not committed yet (known pending)
    }.get(snapshot_code, "FAIL")

    report: dict[str, Any] = {
        "gate": "contract-authority",
        "surface": os.environ.get("IGC_GATE_SURFACE", ""),
        "commit": os.environ.get("GITHUB_SHA", ""),
        "checks": checks,
        "single_source_violations": single_source,
    }
    if full:
        report["conformance"] = _run_full_conformance(repo_root)
    return report


def _is_hard_failure(report: dict[str, Any]) -> bool:
    """A FAIL in any in-process gate, or a non-zero conformance run, is a hard fail.

    BOOTSTRAP (schema snapshot not committed yet) is a known pending state, not a fail.
    """
    if "FAIL" in report["checks"].values():
        return True
    conformance = report.get("conformance")
    if conformance and conformance.get("returncode", 0) != 0:
        return True
    return False


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the contract-authority gate runner."""
    parser = argparse.ArgumentParser(description="Run the Phase 1/2/3 contract-authority gate and emit evidence.")
    parser.add_argument("--repo-root", default=".", help="Repository root.")
    parser.add_argument("--registry", default="configs/contracts/rest_goal.yaml", help="Contract registry path.")
    parser.add_argument("--snapshot", default="schemas/snapshots/rest_goal_contract.shape.json", help="Schema snapshot path.")
    parser.add_argument("--out", default="reports/gate-report-contract-authority.json", help="Sanitized report output.")
    parser.add_argument("--full", action="store_true", help="Also run the offline conformance test suites.")
    args = parser.parse_args(argv)

    _require_remote_surface()
    repo_root = Path(args.repo_root).resolve()
    report = run(
        repo_root,
        full=args.full,
        snapshot=repo_root / args.snapshot,
        registry=repo_root / args.registry,
    )

    out_path = repo_root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(report["checks"], indent=2, sort_keys=True))
    print(f"wrote {out_path}")
    if _is_hard_failure(report):
        print("BLOCKER: contract-authority gate failed.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# Author: Mus mbayramo@stanford.edu
