"""CLI smoke tests for ``scripts/bench_hot_paths.py``.

The benchmark is documented as a file-path command, so it must be runnable from
the repository root without relying on an installed package or manual
``PYTHONPATH``.

Author:
Mus mbayramo@stanford.edu
"""

from __future__ import annotations

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
