"""Offline tests for the opt-in Slurm sanity checker.

The live checker is exercised with fake runners and temporary log files only.

Author:
Mus mbayramo@stanford.edu
"""

import importlib.util
import pathlib
import textwrap


ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "slurm_sanity_from_spec.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("slurm_sanity_from_spec", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_slurm_spec(tmp_path):
    log_path = tmp_path / "slurm-%j.out"
    path = tmp_path / "slurm.yaml"
    path.write_text(
        textwrap.dedent(
            f"""
            version: 1
            name: slurm-sanity
            backend: slurm
            image:
              ref: igc-train:test
            runtime:
              command: [python, igc_main.py, --help]
            paths:
              output: {tmp_path}
            resources:
              gpus: 1
              nodes: 1
            slurm:
              output: {log_path}
            sanity:
              slurm:
                enabled: true
                command: "printf IGC_SLURM_SANITY_OK"
                sentinel: IGC_SLURM_SANITY_OK
                timeout_seconds: 2
                poll_seconds: 1
            """,
        ),
        encoding="utf-8",
    )
    return path, tmp_path / "slurm-123.out"


class FakeRunner:
    """Tiny subprocess stand-in for Slurm commands."""

    def __init__(self, state, log_path=None, sentinel="IGC_SLURM_SANITY_OK"):
        self.state = state
        self.log_path = log_path
        self.sentinel = sentinel
        self.calls = []

    def run(self, cmd):
        self.calls.append(cmd)
        exe = pathlib.Path(cmd[0]).name
        if exe == "sbatch":
            if self.log_path is not None:
                self.log_path.write_text(f"{self.sentinel}\n", encoding="utf-8")
            return 0, "123\n", ""
        if exe == "sacct":
            return 0, f"{self.state}\n", ""
        if exe == "scancel":
            return 0, "", ""
        return 127, "", f"unexpected command: {cmd}"


def test_live_slurm_sanity_passes_when_job_completes_and_log_matches(tmp_path):
    """COMPLETED plus sentinel log is a pass."""
    module = _load_module()
    spec_path, log_path = _write_slurm_spec(tmp_path)
    runner = FakeRunner("COMPLETED", log_path=log_path)
    assert module.run_live(spec_path, runner=runner, sleep=lambda _: None) == 0
    assert any(call[0] == "sbatch" for call in runner.calls)


def test_live_slurm_sanity_fails_on_failed_job(tmp_path):
    """FAILED scheduler state returns a blocker status."""
    module = _load_module()
    spec_path, log_path = _write_slurm_spec(tmp_path)
    runner = FakeRunner("FAILED", log_path=log_path)
    assert module.run_live(spec_path, runner=runner, sleep=lambda _: None) == 1


def test_live_slurm_sanity_reports_sbatch_failure(tmp_path):
    """A scheduler submission failure is surfaced without polling."""
    module = _load_module()
    spec_path, _ = _write_slurm_spec(tmp_path)

    class SbatchFailure(FakeRunner):
        def run(self, cmd):
            self.calls.append(cmd)
            if cmd[0] == "sbatch":
                return 12, "", "invalid partition"
            raise AssertionError(f"unexpected poll after sbatch failure: {cmd}")

    assert module.run_live(spec_path, runner=SbatchFailure("COMPLETED"), sleep=lambda _: None) == 1


def test_live_slurm_sanity_fails_when_log_is_missing(tmp_path):
    """A completed job without its expected log is not trusted."""
    module = _load_module()
    spec_path, _ = _write_slurm_spec(tmp_path)
    runner = FakeRunner("COMPLETED", log_path=None)
    assert module.run_live(spec_path, runner=runner, sleep=lambda _: None) == 1


def test_live_slurm_sanity_times_out_without_hammering(tmp_path):
    """Polling stops at the timeout and uses the configured interval."""
    module = _load_module()
    spec_path, log_path = _write_slurm_spec(tmp_path)
    runner = FakeRunner("RUNNING", log_path=log_path)
    sleeps = []
    assert module.run_live(spec_path, runner=runner, sleep=sleeps.append, now=iter([0, 1, 3]).__next__) == 1
    assert sleeps == [1, 1]


# Author: Mus mbayramo@stanford.edu
