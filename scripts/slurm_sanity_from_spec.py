#!/usr/bin/env python
"""Opt-in Slurm scheduler sanity check from an igc run spec.

Default mode is dry-run. Live mode submits a tiny scheduler job, polls at a
bounded interval, verifies a sentinel in the job log, and returns a clear
blocker instead of silently trusting scheduler state.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import time
from typing import Callable, Protocol

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from igc.shared.run_spec import (
    RunSpec,
    RunSpecError,
    load_run_spec,
    slurm_output_path,
    slurm_sanity_settings,
)


class Runner(Protocol):
    """Protocol for Slurm command execution, mockable in unit tests."""

    def run(self, cmd: list[str]) -> tuple[int, str, str]:
        """Run a command and return ``(returncode, stdout, stderr)``."""


class SubprocessRunner:
    """Subprocess-backed runner for live Slurm commands."""

    def run(self, cmd: list[str]) -> tuple[int, str, str]:
        """Run a command and capture text output."""
        result = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            check=False,
        )
        return result.returncode, result.stdout, result.stderr


def render_slurm_sanity(spec: RunSpec) -> str:
    """Render the Slurm sanity commands without executing them."""
    settings = slurm_sanity_settings(spec)
    sbatch_cmd = _sbatch_command(spec)
    return "\n".join([
        "# igc slurm sanity dry-run",
        " ".join(sbatch_cmd),
        "# poll: sacct -n -X -j <jobid> --format=State",
        f"# expect log sentinel: {settings['sentinel']}",
        f"# timeout_seconds: {settings['timeout_seconds']}",
        f"# poll_seconds: {settings['poll_seconds']}",
        "",
    ])


def run_live(
        spec_path: str | Path,
        *,
        runner: Runner | None = None,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], float] = time.monotonic,
) -> int:
    """Submit and verify a tiny Slurm sanity job.

    :param spec_path: YAML run spec.
    :param runner: command runner, injectable for tests.
    :param sleep: sleep function, injectable for tests.
    :param now: monotonic clock, injectable for tests.
    :return: 0 for pass, 1 for scheduler/log blocker, 2 for spec error.
    """
    try:
        spec = load_run_spec(spec_path)
    except RunSpecError as exc:
        print(f"RUN SPEC ERROR: {exc}")
        return 2

    if spec.backend != "slurm":
        print("RUN SPEC ERROR: slurm sanity requires backend: slurm")
        return 2

    settings = slurm_sanity_settings(spec)
    if not settings["enabled"]:
        print("RUN SPEC ERROR: sanity.slurm.enabled must be true for live sanity")
        return 2

    poll_seconds = int(settings["poll_seconds"])
    if poll_seconds < 1:
        print("RUN SPEC ERROR: sanity.slurm.poll_seconds must be >= 1")
        return 2

    runner = runner or SubprocessRunner()
    rc, out, err = runner.run(_sbatch_command(spec))
    if rc != 0:
        print(f"BLOCKER: sbatch failed rc={rc}: {(err or out).strip()}")
        return 1

    job_id = out.strip().splitlines()[0] if out.strip() else ""
    if not job_id:
        print("BLOCKER: sbatch did not return a job id")
        return 1

    timeout = int(settings["timeout_seconds"])
    deadline = now() + timeout
    terminal_failures = {"BOOT_FAIL", "CANCELLED", "DEADLINE", "FAILED", "NODE_FAIL", "OUT_OF_MEMORY", "TIMEOUT"}

    while True:
        state = _query_state(runner, job_id)
        if state == "COMPLETED":
            return _verify_log(spec, job_id, str(settings["sentinel"]))
        if state in terminal_failures:
            print(f"BLOCKER: Slurm job {job_id} ended in {state}")
            return 1
        sleep(poll_seconds)
        if now() >= deadline:
            runner.run(["scancel", job_id])
            print(f"BLOCKER: Slurm job {job_id} timed out waiting for completion")
            return 1


def _sbatch_command(spec: RunSpec) -> list[str]:
    settings = slurm_sanity_settings(spec)
    resources = spec.resources
    slurm = spec.slurm
    cmd = [
        "sbatch",
        "--parsable",
        f"--job-name={spec.name}-sanity",
        f"--nodes={resources.get('nodes', 1)}",
        f"--gres=gpu:{resources.get('gpus', 1)}",
        f"--output={slurm_output_path(spec)}",
        f"--wrap={settings['command']}",
    ]
    if resources.get("wall_time"):
        cmd.insert(-1, f"--time={resources['wall_time']}")
    if slurm.get("partition"):
        cmd.insert(-1, f"--partition={slurm['partition']}")
    if slurm.get("account"):
        cmd.insert(-1, f"--account={slurm['account']}")
    return cmd


def _query_state(runner: Runner, job_id: str) -> str:
    rc, out, err = runner.run(["sacct", "-n", "-X", "-j", job_id, "--format=State"])
    if rc != 0:
        print(f"BLOCKER: sacct failed rc={rc}: {(err or out).strip()}")
        return "FAILED"
    text = out.strip().splitlines()[0] if out.strip() else "UNKNOWN"
    return text.split()[0]


def _verify_log(spec: RunSpec, job_id: str, sentinel: str) -> int:
    path = Path(slurm_output_path(spec, job_id=job_id))
    if not path.exists():
        print(f"BLOCKER: expected Slurm log missing: {path}")
        return 1
    text = path.read_text(encoding="utf-8", errors="replace")
    if sentinel not in text:
        print(f"BLOCKER: Slurm log {path} missing sentinel {sentinel!r}")
        return 1
    print(f"slurm sanity OK job={job_id} log={path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, help="YAML run spec")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--live", action="store_true")
    args = parser.parse_args(argv)

    try:
        spec = load_run_spec(args.spec)
        if args.dry_run:
            sys.stdout.write(render_slurm_sanity(spec))
            return 0
    except RunSpecError as exc:
        print(f"RUN SPEC ERROR: {exc}", file=sys.stderr)
        return 2

    return run_live(args.spec)


if __name__ == "__main__":
    raise SystemExit(main())
