"""Offline tests for ``scripts/rl_env_sanity.py``.

The real proof is an opt-in GB300 ``torchrun --nproc_per_node=4`` run. These
tests cover the CPU helper surface so CI can validate fixture construction,
rollout shape contracts, replay insertion, and the CUDA-required guard without
starting distributed workers.

Author:
Mus mbayramo@stanford.edu
"""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "rl_env_sanity.py"


def _load_sanity():
    """Import the script as a module without requiring it on PYTHONPATH."""
    spec = importlib.util.spec_from_file_location("rl_env_sanity", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_write_tiny_fixtures_builds_mapping(tmp_path):
    """The live harness starts from a tiny local REST mapping."""
    sanity = _load_sanity()

    mapping = sanity.write_tiny_fixtures(tmp_path, rank=3)

    pairs = dict(mapping.get_rest_api_mappings())
    assert set(pairs) == {sanity.ROOT_URI, sanity.SYSTEM_URI, sanity.CHASSIS_URI}
    assert all(Path(path).exists() for path in pairs.values())
    assert mapping.lookup_rest_api_to_method(sanity.ROOT_URI) == "GET"
    assert mapping.num_actions == 3


def test_rollout_actions_preserve_vector_env_batch_axis(tmp_path):
    """A rollout action batch has one row per vector sub-environment."""
    sanity = _load_sanity()
    env = sanity.build_env(tmp_path, rank=0, num_envs=4, max_episode=8)

    actions = sanity.rollout_actions(env, step=1)

    assert actions.shape == (4, 3 + 6)
    assert torch.all(actions[:, -6:].sum(dim=1) == 1)
    assert torch.all(actions[:, :3].sum(dim=1) == 1)


def test_run_rollout_collects_replay_and_shape_stats(tmp_path):
    """CPU helper rollout exercises env step, replay add/sample, and shape stats."""
    sanity = _load_sanity()

    stats = sanity.run_rollout(
        num_envs=2,
        steps=3,
        device=torch.device("cpu"),
        rank=1,
        world=1,
        base_dir=tmp_path,
    )

    assert stats.transitions == 6
    assert stats.replay_len == 3
    assert stats.terminals == 1
    assert stats.truncated == 0
    assert stats.obs_shape == (2, 2, 3)
    assert stats.action_shape == (2, 9)
    assert stats.sample_state_shape == (3, 2, 2, 3)
    assert stats.sample_done_shape == (3, 2)


def test_stats_and_shape_tensors_are_reduction_ready(tmp_path):
    """Stats pack into tensors suitable for distributed all_reduce."""
    sanity = _load_sanity()
    stats = sanity.run_rollout(
        num_envs=2,
        steps=2,
        device=torch.device("cpu"),
        rank=0,
        world=1,
        base_dir=tmp_path,
    )

    values = sanity.stats_tensor(stats, torch.device("cpu"))
    shapes = sanity.shape_tensor(stats, torch.device("cpu"))

    assert values.dtype == torch.float64
    assert values.tolist()[:4] == [4.0, 2.0, 1.0, 0.0]
    assert shapes.dtype == torch.long
    assert shapes.tolist() == [2, 2, 3, 9, 2, 3, 2]


def test_main_requires_cuda_unless_cpu_development_is_allowed(monkeypatch):
    """The script must not accidentally make a CPU run look like a GPU proof."""
    sanity = _load_sanity()
    monkeypatch.setattr(sanity.torch.cuda, "is_available", lambda: False)

    with pytest.raises(SystemExit, match="CUDA not available"):
        sanity.main(["--num-envs", "1", "--steps", "1"])


def test_main_allow_cpu_runs_single_process_smoke(monkeypatch, capsys):
    """The explicit CPU escape hatch supports local smoke testing."""
    sanity = _load_sanity()
    monkeypatch.setattr(sanity.torch.cuda, "is_available", lambda: False)
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)

    assert sanity.main(["--allow-cpu", "--num-envs", "1", "--steps", "1"]) == 0
    assert "PASS world=1" in capsys.readouterr().out


# Author: Mus mbayramo@stanford.edu
