"""Offline contract tests for the GPU dataset-isolation sweep wrapper.

The real sweep runs on GB300 with ``torchrun`` and per-case timeouts. These
tests keep that behavior mocked: fake ``nvidia-smi`` controls the detected
world sizes and fake ``timeout`` returns PASS/DEADLOCK outcomes from the
environment variables the wrapper sets for each case.

Author:
Mus mbayramo@stanford.edu
"""

import os
import stat
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "gpu_dataset_isolation.sh"


def _write_executable(path: Path, text: str) -> None:
    """Write a small fake command into ``path`` and make it executable."""
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _run_sweep(tmp_path: Path, timeout_body: str) -> subprocess.CompletedProcess[str]:
    """Run the wrapper with fake GPU/timeout commands and a stable environment."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_executable(
        bin_dir / "nvidia-smi",
        "#!/usr/bin/env bash\n"
        "if [ \"${1:-}\" = \"-L\" ]; then\n"
        "  printf 'GPU 0\\nGPU 1\\n'\n"
        "fi\n",
    )
    _write_executable(
        bin_dir / "timeout",
        "#!/usr/bin/env bash\n"
        "shift\n"
        f"{timeout_body}\n",
    )

    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "WORLDS": "2",
        "CASE_TIMEOUT": "1",
    }
    return subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def test_gpu_dataset_isolation_wrapper_exits_nonzero_on_mismatch(tmp_path):
    """Any expected-vs-got mismatch must make the sweep fail."""
    result = _run_sweep(
        tmp_path,
        # Force a PASS case to time out. The wrapper should report MISMATCH and
        # exit nonzero so CI/automation cannot mistake it for a usable sweep.
        'if [ "$DS_SIZE" = "128" ]; then exit 124; fi\n'
        "exit 0",
    )

    assert result.returncode != 0
    assert "MISMATCH" in result.stdout


def test_gpu_dataset_isolation_wrapper_passes_when_expectations_match(tmp_path):
    """The deterministic matrix exits cleanly when PASS and DEADLOCK cases match."""
    result = _run_sweep(
        tmp_path,
        'if [ "$DROP_LAST" = "0" ]; then exit 124; fi\n'
        "exit 0",
    )

    assert result.returncode == 0
    rows = [line for line in result.stdout.splitlines() if not line.startswith("summary:")]
    assert not any(line.rstrip().endswith("MISMATCH") for line in rows)
    assert "summary:" in result.stdout


# Author: Mus mbayramo@stanford.edu
