"""Offline regression for goal-rollout termination in ``IgcAgentTrainer.train_goal``.

``VectorizedRestApiEnv.step`` returns ``done`` (goal reached or horizon) as its third
value. The rollout must end as soon as any env reports the episode is over; a regression
where ``done`` was unpacked into an unused name let the loop always run to
``max_episode_len`` and replay past-terminal transitions. These tests drive the rollout
with tiny CPU stubs (no model, no real env) and assert the step count matches the env's
own terminal signal.

Author:
Mus mbayramo@stanford.edu
"""
import torch
import pytest

from igc.modules.rl.q_targets import q_learning_target
from igc.modules.igc_train_agent import IgcAgentTrainer


class _FakeEnv:
    """Minimal single-env stub: reports ``done`` after a fixed number of steps."""

    def __init__(self, state_dim: int, done_after: int):
        self.num_envs = 1
        self._state = torch.zeros(1, state_dim)
        self._done_after = done_after
        self.step_calls = 0

    def reset(self, goal=None, goal_type=None):
        return self._state.clone(), [{}]

    def add_goal_state(self, _state):
        return None

    def encode_batched_rest_api_method(self, _method, num_envs):
        return torch.zeros(num_envs, 2)

    def concat_batch_rest_api_method(self, api_one_hot, method_one_hot):
        return torch.cat([api_one_hot, method_one_hot], dim=1)

    def step(self, _action):
        self.step_calls += 1
        is_done = self.step_calls >= self._done_after
        terminated = torch.tensor([is_done])
        truncated = torch.tensor([False])
        rewards = torch.zeros(1)
        info = [{"goal_reached": torch.tensor(False)}]
        return self._state.clone(), rewards, terminated, truncated, info


class _FakeDataset:
    """Supplies a fixed one-hot action batch for the epsilon-random branch."""

    def __init__(self, api_dim: int):
        self._api_dim = api_dim

    def sample_batch_of_rest_api(self, num_envs):
        return None, None, torch.zeros(num_envs, self._api_dim)


class _DeadEndTerminalEnv(_FakeEnv):
    """Single-env stub: first step is a negative terminal dead end."""

    def __init__(self, state_dim: int):
        super().__init__(state_dim=state_dim, done_after=1)

    def step(self, _action):
        self.step_calls += 1
        terminated = torch.tensor([True])
        truncated = torch.tensor([False])
        rewards = torch.tensor([-0.5])
        info = [{"goal_reached": torch.tensor(False)}]
        return self._state.clone(), rewards, terminated, truncated, info


def _make_trainer(state_dim: int, max_episode_len: int, done_after: int):
    """Build a trainer with train_goal's dependencies stubbed (bypasses __init__)."""
    trainer = IgcAgentTrainer.__new__(IgcAgentTrainer)
    trainer.env = _FakeEnv(state_dim, done_after)
    trainer.dataset = _FakeDataset(api_dim=3)
    trainer.current_goal = {"state": torch.zeros(1, state_dim)}
    trainer.max_episode_len = max_episode_len
    return trainer


def test_rollout_stops_on_env_done():
    """A first-step env terminal ends the rollout immediately, not at max_episode_len."""
    trainer = _make_trainer(state_dim=4, max_episode_len=10, done_after=1)
    trainer.train_goal(epsilon=1.0)
    assert trainer.env.step_calls == 1


def test_rollout_stops_midway_when_env_terminates():
    """The rollout ends on the step the env first reports done (before the horizon)."""
    trainer = _make_trainer(state_dim=4, max_episode_len=10, done_after=3)
    trainer.train_goal(epsilon=1.0)
    assert trainer.env.step_calls == 3


def test_rollout_truncates_at_max_episode_len():
    """With no env terminal, the horizon guard still caps the rollout length."""
    trainer = _make_trainer(state_dim=4, max_episode_len=5, done_after=999)
    trainer.train_goal(epsilon=1.0)
    assert trainer.env.step_calls == 5


@pytest.mark.xfail(
    reason=(
        "IgcAgentTrainer.train_goal drops terminated/truncated flags, so "
        "update_replay_buffer rebuilds done from reward >= 1.0 and bootstraps "
        "through negative terminal dead ends."
    ),
    strict=True,
)
def test_negative_terminal_mask_survives_rollout_for_replay():
    """A negative terminal must be stored as done=True, not inferred from reward."""
    trainer = IgcAgentTrainer.__new__(IgcAgentTrainer)
    trainer.env = _DeadEndTerminalEnv(state_dim=4)
    trainer.dataset = _FakeDataset(api_dim=3)
    trainer.current_goal = {"state": torch.zeros(1, 4)}
    trainer.max_episode_len = 10

    episode, _, _ = trainer.train_goal(epsilon=1.0)
    _, _, reward, _, _, terminated, truncated = episode[0]

    assert reward.tolist() == [-0.5]
    assert terminated.tolist() == [True]
    assert truncated.tolist() == [False]
    target = q_learning_target(
        reward,
        terminated.to(reward.dtype),
        next_q=torch.tensor([[10.0]]),
        gamma=0.9,
    )
    torch.testing.assert_close(target, reward)


# Author: Mus mbayramo@stanford.edu
