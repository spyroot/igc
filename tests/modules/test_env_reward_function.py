"""Offline regressions for the HER reward function in ``igc_train_agent``.

``env_reward_function`` previously reduced the goal match across the whole
batch (``torch.any``) and broadcast the result, so one env row matching a
relabeled goal rewarded every row — poisoning replay with false terminal
successes. It also used exact equality while the vectorized env's
``check_goal`` uses ``allclose(rtol=1e-3, atol=1e-3)``, so HER and the env
could disagree on success. These pin the per-row, tolerance-aligned contract.

Author:
Mus mbayramo@stanford.edu
"""

import torch

from igc.modules.igc_train_agent import env_reward_function


def test_reward_is_per_row_no_batch_leak():
    """Only the matching row is rewarded — no cross-batch success leak."""
    goal = torch.tensor([1.0, 2.0, 3.0])
    state = torch.stack([goal, torch.tensor([9.0, 9.0, 9.0])])

    reward = env_reward_function(state, goal)

    assert reward.shape == (2,)
    assert reward.tolist() == [1.0, 0.0]


def test_reward_zero_when_nothing_matches():
    """A batch with no matching row earns zero reward everywhere."""
    goal = torch.tensor([1.0, 2.0, 3.0])
    state = torch.full((3, 3), 7.0)

    assert env_reward_function(state, goal).tolist() == [0.0, 0.0, 0.0]


def test_reward_matches_env_check_goal_tolerance():
    """A near-match within the env's rtol/atol counts as success."""
    goal = torch.tensor([1.0, 2.0, 3.0])
    near = goal * (1.0 + 5e-4)  # inside rtol=1e-3
    far = goal * 1.1  # outside
    state = torch.stack([near, far])

    assert env_reward_function(state, goal).tolist() == [1.0, 0.0]


def test_reward_dtype_is_float32():
    """Reward stays float32 for replay-buffer stacking."""
    goal = torch.tensor([1.0])
    state = torch.tensor([[1.0], [0.0]])
    assert env_reward_function(state, goal).dtype == torch.float32


# Author: Mus mbayramo@stanford.edu
