"""Offline guards for GB300 NCCL launch defaults.

MNNVL requires working IMEX channels. Slot11 showed MNNVL as available but not
usable, and NCCL failed before the M3 smoke could construct DDP until
``NCCL_MNNVL_ENABLE=0`` was set. Keep MNNVL opt-in in scripts while leaving
CUMEM default-on.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _read_script(name: str) -> str:
    """Read a repo script by basename."""
    return (REPO_ROOT / "scripts" / name).read_text(encoding="utf-8")


def test_gb300_launch_keeps_mnnvl_opt_in_after_private_env_source():
    """Private env sourcing must not silently force MNNVL back on."""
    src = _read_script("gb300_launch.sh")

    assert 'IGC_NCCL_MNNVL="${IGC_NCCL_MNNVL:-${NCCL_MNNVL_ENABLE:-0}}"' in src
    assert 'export NCCL_MNNVL_ENABLE="${_igc_nccl_mnnvl}"' in src
    assert 'export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-1}"' in src


def test_gb300_sanity_uses_safe_mnnvl_default():
    """Distributed sanity should use MNNVL=0 unless the caller opts in."""
    src = _read_script("gb300_sanity_check.sh")

    assert "-e NCCL_MNNVL_ENABLE=${NCCL_MNNVL_ENABLE:-0}" in src
    assert "-e NCCL_CUMEM_ENABLE=${NCCL_CUMEM_ENABLE:-1}" in src


def test_slurm_launcher_keeps_mnnvl_opt_in_after_private_env_source():
    """The Slurm path should preserve the safe MNNVL default through env sourcing."""
    src = _read_script("train_igc.sbatch")

    assert 'export IGC_NCCL_MNNVL="${IGC_NCCL_MNNVL:-${NCCL_MNNVL_ENABLE:-0}}"' in src
    assert 'export NCCL_MNNVL_ENABLE="${IGC_NCCL_MNNVL}"' in src
    assert 'export NCCL_MNNVL_ENABLE="${_igc_nccl_mnnvl}"' in src
    assert 'export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-1}"' in src


# Author: Mus mbayramo@stanford.edu
