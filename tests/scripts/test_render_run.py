"""Offline CLI tests for run rendering and launch refusal.

The command-line wrappers must be safe by default: rendering is allowed, but a
real Docker or Slurm launch requires an explicit live path added in a later PR.

Author:
Mus mbayramo@stanford.edu
"""

import subprocess
import sys
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _write_spec(tmp_path, backend="docker"):
    slurm = """
      partition: ${IGC_SLURM_PARTITION}
      output: ${IGC_OUTPUT_DIR}/slurm-%j.out
    """ if backend == "slurm" else "{}"
    path = tmp_path / f"{backend}.yaml"
    path.write_text(
        textwrap.dedent(
            f"""
            version: 1
            name: {backend}-cli
            backend: {backend}
            image:
              ref: igc-train:test
              pull_policy: if_missing
            runtime:
              command:
                - python
                - igc_main.py
                - --help
            paths:
              code: ${{IGC_CODE_DIR}}
              data: ${{IGC_DATA_DIR}}
              output: ${{IGC_OUTPUT_DIR}}
            resources:
              gpus: 1
              nodes: 1
            docker:
              env_files:
                - ${{IGC_DOCKER_ENV_FILE}}
            slurm:
        """
        )
        + textwrap.indent(textwrap.dedent(slurm).strip() + "\n", "      "),
        encoding="utf-8",
    )
    return path


def _run(*args):
    return subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_render_run_prints_docker_plan(tmp_path):
    """The renderer prints a Docker dry-run without touching Docker."""
    result = _run("scripts/render_run.py", "--spec", str(_write_spec(tmp_path)))
    assert result.returncode == 0, result.stderr
    assert "backend: docker" in result.stdout
    assert "docker run" in result.stdout
    assert "${IGC_DOCKER_ENV_FILE}" in result.stdout


def test_render_run_reports_validation_errors(tmp_path):
    """Invalid specs fail gracefully with a short spec error."""
    bad = tmp_path / "bad.yaml"
    bad.write_text("backend: docker\nunknown: true\n", encoding="utf-8")
    result = _run("scripts/render_run.py", "--spec", str(bad))
    assert result.returncode == 2
    assert "RUN SPEC ERROR:" in result.stderr


def test_launch_run_refuses_live_launch_without_dry_run(tmp_path):
    """The first implementation slice cannot mutate Docker or Slurm."""
    result = _run("scripts/launch_run.py", "--spec", str(_write_spec(tmp_path)))
    assert result.returncode == 2
    assert "refusing live launch" in result.stderr


def test_launch_run_dry_run_delegates_to_renderer(tmp_path):
    """Dry-run launch prints the same safe command plan."""
    result = _run(
        "scripts/launch_run.py",
        "--spec",
        str(_write_spec(tmp_path, backend="slurm")),
        "--dry-run",
    )
    assert result.returncode == 0, result.stderr
    assert "backend: slurm" in result.stdout
    assert "sbatch" in result.stdout


# Author: Mus mbayramo@stanford.edu
