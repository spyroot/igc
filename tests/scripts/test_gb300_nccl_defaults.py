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


def test_gb300_launch_keeps_nccl_defaults_after_private_env_source():
    """Private env sourcing must not silently rewrite NCCL transport defaults."""
    src = _read_script("gb300_launch.sh")

    assert 'IGC_NCCL_CUMEM="${IGC_NCCL_CUMEM:-${NCCL_CUMEM_ENABLE:-1}}"' in src
    assert 'NCCL_CUMEM_ENABLE="${IGC_NCCL_CUMEM}"' in src
    assert 'IGC_NCCL_MNNVL="${IGC_NCCL_MNNVL:-${NCCL_MNNVL_ENABLE:-0}}"' in src
    assert 'NCCL_MNNVL_ENABLE="${IGC_NCCL_MNNVL}"' in src
    assert 'export NCCL_CUMEM_ENABLE="${IGC_NCCL_CUMEM}"' in src
    assert 'export NCCL_MNNVL_ENABLE="${IGC_NCCL_MNNVL}"' in src
    assert src.index('[ -f "${IGC_HF_ENV}" ]') < src.index(
        'export NCCL_CUMEM_ENABLE="${IGC_NCCL_CUMEM}"'
    )
    assert src.index('[ -f "${IGC_HF_ENV}" ]') < src.index(
        'export NCCL_MNNVL_ENABLE="${IGC_NCCL_MNNVL}"'
    )


def test_gb300_sanity_uses_safe_mnnvl_default():
    """Distributed sanity should use MNNVL=0 unless the caller opts in."""
    src = _read_script("gb300_sanity_check.sh")

    assert "-e NCCL_MNNVL_ENABLE=${NCCL_MNNVL_ENABLE:-0}" in src
    assert "-e NCCL_CUMEM_ENABLE=${NCCL_CUMEM_ENABLE:-1}" in src


def test_slurm_launcher_keeps_nccl_defaults_after_private_env_source():
    """The Slurm path should preserve NCCL defaults through env sourcing."""
    src = _read_script("train_igc.sbatch")

    host_hf_source = src.index('[ -f "${IGC_DIR}/.internal/hf.env" ]')
    inner_hf_source = src.index("[ -f .internal/hf.env ]")
    assert 'export IGC_NCCL_CUMEM="${IGC_NCCL_CUMEM:-${NCCL_CUMEM_ENABLE:-1}}"' in src
    assert 'export NCCL_CUMEM_ENABLE="${IGC_NCCL_CUMEM}"' in src
    assert 'export NCCL_CUMEM_ENABLE="${_igc_nccl_cumem}"' in src
    assert '_igc_nccl_cumem="${IGC_NCCL_CUMEM:-${NCCL_CUMEM_ENABLE:-1}}"' in src
    assert 'export IGC_NCCL_MNNVL="${IGC_NCCL_MNNVL:-${NCCL_MNNVL_ENABLE:-0}}"' in src
    assert 'export NCCL_MNNVL_ENABLE="${IGC_NCCL_MNNVL}"' in src
    assert 'export NCCL_MNNVL_ENABLE="${_igc_nccl_mnnvl}"' in src
    assert host_hf_source < src.index('export NCCL_CUMEM_ENABLE="${IGC_NCCL_CUMEM}"', host_hf_source)
    assert host_hf_source < src.index('export NCCL_MNNVL_ENABLE="${IGC_NCCL_MNNVL}"', host_hf_source)
    assert inner_hf_source < src.index('export NCCL_CUMEM_ENABLE="${_igc_nccl_cumem}"')
    assert inner_hf_source < src.index('export NCCL_MNNVL_ENABLE="${_igc_nccl_mnnvl}"')


# Author: Mus mbayramo@stanford.edu
