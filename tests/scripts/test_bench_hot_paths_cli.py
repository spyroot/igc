"""CLI smoke tests for ``scripts/bench_hot_paths.py``.

The script is used by Makefile profiling targets, so it must be runnable by
file path from the repository root without relying on an installed package.

Author:
Mus mbayramo@stanford.edu
"""

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "bench_hot_paths.py"


def test_bench_hot_paths_help_runs_from_repo_root() -> None:
    """Path-based invocation reaches argparse instead of failing on imports."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--section" in result.stdout


# Author: Mus mbayramo@stanford.edu
