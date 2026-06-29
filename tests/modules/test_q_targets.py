"""Offline CPU tests for the goal-conditioned DQN+HER math (igc.modules.rl.q_targets).

Pins the two correctness fixes the legacy agent lacked: a terminal bootstrap mask
that preserves a {0, +1} success reward (instead of clipping the target to <= 0),
and HER "future" relabeling that draws goals from an achieved future next-state.

Author:
Mus mbayramo@stanford.edu
"""
import numpy as np
import torch

from igc.modules.rl.q_targets import q_learning_target, relabel_future


def test_terminal_masks_bootstrap_and_keeps_success_reward():
    """At a terminal (done=1) the target equals reward — a +1 success is preserved,
    not clipped away, and the next-state Q is not bootstrapped."""
    reward = torch.tensor([1.0, 0.0])
    done = torch.tensor([1.0, 0.0])
    next_q = torch.tensor([[5.0, 9.0], [5.0, 9.0]])  # max == 9 for both rows
    gamma = 0.9
    target = q_learning_target(reward, done, next_q, gamma)
    # row 0 terminal: target == reward == 1.0 (no bootstrap)
    assert torch.isclose(target[0], torch.tensor(1.0))
    # row 1 non-terminal: 0 + 0.9 * 9 == 8.1 (keeps bootstrapping)
    assert torch.isclose(target[1], torch.tensor(8.1))


def test_truncation_keeps_bootstrapping():
    """A non-terminal step (truncation -> done=0) still bootstraps through next_q."""
    reward = torch.tensor([0.0])
    done = torch.tensor([0.0])
    next_q = torch.tensor([[2.0, 4.0]])
    target = q_learning_target(reward, done, next_q, gamma=0.5)
    assert torch.isclose(target[0], torch.tensor(2.0))  # 0 + 0.5 * 4


def _reward_match(state, goal):
    """Toy env reward: 1.0 where state equals goal elementwise, else 0.0 (shape [B])."""
    return torch.all(state == goal, dim=1).to(torch.float32)


def test_relabel_future_uses_achieved_future_next_state():
    """The substituted goal comes from a FUTURE step's achieved next-state (index 3),
    not the episode's final pre-action state."""
    # episode tuples: (state, action, reward, next_state, goal); index 3 == next_state
    def step(next_state_val):
        s = torch.zeros(1, 2)
        return (s, torch.zeros(1, 4), torch.zeros(1), torch.full((1, 2), float(next_state_val)), s)

    episode = [step(1), step(2), step(3)]
    rng = np.random.default_rng(0)
    # relabel timestep 0; achieved next-state of this transition is value 1
    out = relabel_future(episode, 0, episode[0][3], num_relabeled=8,
                         reward_fn=_reward_match, rng=rng)
    assert len(out) == 8
    # every sampled goal must be an achieved next-state from t' in [0, 2] -> {1,2,3}
    goal_vals = {float(g[0, 0].item()) for g, _, _ in out}
    assert goal_vals.issubset({1.0, 2.0, 3.0})


def test_relabel_future_reward_and_done_consistent():
    """When the substituted goal equals this transition's achieved next-state the
    recomputed reward is 1.0 and done is 1.0; otherwise both are 0.0."""
    achieved = torch.full((1, 2), 1.0)

    def step(val):
        s = torch.zeros(1, 2)
        return (s, torch.zeros(1, 4), torch.zeros(1), torch.full((1, 2), float(val)), s)

    # only the future step at index 0 matches `achieved` (value 1); index 1 is value 5
    episode = [step(1), step(5)]

    class _FixedRng:
        def __init__(self, vals):
            self._vals = list(vals)

        def integers(self, low, high):
            return self._vals.pop(0)

    out = relabel_future(episode, 0, achieved, num_relabeled=2,
                         reward_fn=_reward_match, rng=_FixedRng([0, 1]))
    (_, r0, d0), (_, r1, d1) = out
    assert float(r0.item()) == 1.0 and float(d0.item()) == 1.0   # goal == achieved
    assert float(r1.item()) == 0.0 and float(d1.item()) == 0.0   # goal != achieved


# Author: Mus mbayramo@stanford.edu
