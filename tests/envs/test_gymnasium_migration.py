"""Offline regressions for the gym → gymnasium migration of igc/envs.

The three env modules previously imported legacy ``gym`` 0.26 and would
hard-crash under gymnasium 1.x at three verified points: the removed
``gym.vector.utils.spaces`` alias, ``VectorEnv.__init__(num_envs, ...)``
(gymnasium's base defines no such signature), and the removed
``step_async``/``step_wait`` dispatch in ``VectorEnv.step``. These pin the
migrated surface: gymnasium base classes, gym-0.26-compatible batched spaces
(``single_*`` + ``batch_space``), and a ``step`` that dispatches through the
async/wait pair. CPU-only, no source-tree scan beyond igc/envs.

Author:
Mus mbayramo@stanford.edu
"""

import pathlib
import re

import gymnasium

import igc.envs.rest_gym_base as base_mod
import igc.envs.rest_gym_batch_env as batch_mod
from igc.envs.rest_gym_base import RestApiBaseEnv
from igc.envs.rest_gym_batch_env import VectorizedRestApiEnv
from igc.envs.rest_gym_env import RestApiEnv

_ENVS_DIR = pathlib.Path(base_mod.__file__).resolve().parent


def test_no_legacy_gym_imports_remain():
    """No module under igc/envs imports legacy gym (gymnasium only)."""
    pattern = re.compile(r"^\s*(import gym$|import gym\s|from gym[.\s])", re.M)
    for path in _ENVS_DIR.glob("*.py"):
        assert not pattern.search(path.read_text()), f"legacy gym import in {path.name}"


def test_env_classes_extend_gymnasium():
    """Both env classes are gymnasium citizens."""
    assert issubclass(RestApiBaseEnv, gymnasium.Env)
    assert issubclass(RestApiEnv, gymnasium.Env)
    assert issubclass(VectorizedRestApiEnv, gymnasium.vector.VectorEnv)


def test_spaces_come_from_gymnasium():
    """The spaces alias resolves to gymnasium's spaces module."""
    assert base_mod.spaces is gymnasium.spaces


def test_vector_env_step_dispatches_async_wait():
    """step() drives step_async + step_wait (gymnasium's base no longer does)."""
    calls = []
    env = VectorizedRestApiEnv.__new__(VectorizedRestApiEnv)
    env.step_async = lambda actions: calls.append(("async", actions))
    env.step_wait = lambda: calls.append(("wait", None)) or "result"

    assert env.step([1, 2]) == "result"
    assert calls == [("async", [1, 2]), ("wait", None)]


def test_batch_space_semantics_match_gym_026(monkeypatch):
    """single_* spaces are per-env; public spaces are batched over num_envs."""
    env = VectorizedRestApiEnv.__new__(VectorizedRestApiEnv)
    single = gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(3,))
    env.observation_space = single
    env.action_space = single
    # replicate exactly what __init__ does after the migration
    gymnasium.vector.VectorEnv.__init__(env)
    env.num_envs = 2
    env.single_observation_space = env.observation_space
    env.single_action_space = env.action_space
    env.observation_space = batch_mod.batch_space(env.single_observation_space, 2)
    env.action_space = batch_mod.batch_space(env.single_action_space, 2)

    assert env.single_observation_space.shape == (3,)
    assert env.observation_space.shape == (2, 3)
    assert env.observation_space.shape[-1] == 3  # consumers key on shape[-1]


# Author: Mus mbayramo@stanford.edu
