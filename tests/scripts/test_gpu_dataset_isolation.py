"""Offline boundary tests for ``scripts/gpu_dataset_isolation.py``.

The multi-rank NCCL deadlock case is intentionally opt-in on GB300. These tests
exercise the same harness in single-rank CPU mode so edge cases around zero
batches, exact divisibility, and indivisible tails stay covered in the default
gate.

Author:
Mus mbayramo@stanford.edu
"""

import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
HARNESS = REPO_ROOT / "scripts" / "gpu_dataset_isolation.py"


def _load_harness():
    """Import the script as a module without requiring it to be on PYTHONPATH."""
    spec = importlib.util.spec_from_file_location("gpu_dataset_isolation", HARNESS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "ds_size,batch,drop_last,expected_batches",
    [
        (3, 4, 1, "batches=0"),
        (8, 4, 1, "batches=2"),
        (9, 4, 1, "batches=2"),
        (9, 4, 0, "batches=3"),
    ],
)
def test_single_rank_boundary_cases_exit_cleanly(
    monkeypatch,
    capsys,
    ds_size,
    batch,
    drop_last,
    expected_batches,
):
    """Single-rank harness handles zero/exact/tail batches without a GPU."""
    harness = _load_harness()
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.setenv("DS_SIZE", str(ds_size))
    monkeypatch.setenv("BATCH", str(batch))
    monkeypatch.setenv("DROP_LAST", str(drop_last))
    monkeypatch.setenv("EPOCHS", "1")

    with pytest.raises(SystemExit) as exc:
        harness.main()

    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert expected_batches in out
    assert "PASS" in out


def test_env_int_defaults_when_unset(monkeypatch):
    """Missing integer env knobs use the caller-provided default."""
    harness = _load_harness()
    monkeypatch.delenv("IGC_MISSING_INT", raising=False)

    assert harness._env_int("IGC_MISSING_INT", 17) == 17


# Author: Mus mbayramo@stanford.edu
