"""Offline CPU tests for the replay Buffer's done flag (igc.modules.igc_experience_buffer).

The buffer now stores and returns a per-transition ``done`` so the DQN target can
mask bootstrapping at terminals. Covers an explicit tensor done and the scalar
fallback for experiences added without one.

Author:
Mus mbayramo@stanford.edu
"""
import pytest
import torch

from igc.modules.igc_experience_buffer import Buffer


def _exp(done, value=0.0):
    """A tiny experience: state[1,2], action[1,3], reward[1], next_state[1,2], done."""
    state = torch.full((1, 2), value)
    action = torch.full((1, 3), value)
    reward = torch.tensor([value])
    next_state = torch.full((1, 2), value + 10.0)
    return state, action, reward, next_state, done


def test_sample_batch_returns_done_tensor():
    """sample_batch yields a 5-tuple whose done aligns with reward shape."""
    buf = Buffer(size=10, sample_size=2)
    buf.add(*_exp(torch.ones(1)))
    buf.add(*_exp(torch.zeros(1)))
    state, action, reward, next_state, done = buf.sample_batch()
    assert done.shape == reward.shape
    assert set(done.flatten().tolist()) <= {0.0, 1.0}


def test_add_default_done_is_zero():
    """Adding without a done flag (the default) yields a 0.0 terminal mask."""
    buf = Buffer(size=10, sample_size=1)
    buf.add(torch.zeros(1, 2), torch.zeros(1, 3), torch.ones(1), torch.zeros(1, 2))
    _, _, reward, _, done = buf.sample_batch()
    assert done.shape == reward.shape
    assert float(done.sum().item()) == 0.0


@pytest.mark.parametrize("sample_api", ["sample", "sample_batch"])
def test_mixed_scalar_and_tensor_done_flags_keep_reward_shape(sample_api):
    """Scalar bools and tensor done flags are returned in reward-aligned order."""
    buf = Buffer(size=5, sample_size=5)
    buf.add(*_exp(True, value=1.0))
    buf.add(*_exp(False, value=2.0))
    buf.add(*_exp(torch.tensor([1.0]), value=3.0))

    _, _, reward, _, done = getattr(buf, sample_api)()

    assert done.shape == reward.shape
    assert done.flatten().tolist() == [1.0, 0.0, 1.0]


def test_sample_batch_returns_only_capacity_retained_experiences():
    """The deque capacity evicts oldest transitions before batch reshaping."""
    buf = Buffer(size=2, sample_size=3)
    buf.add(*_exp(False, value=1.0))
    buf.add(*_exp(True, value=2.0))
    buf.add(*_exp(False, value=3.0))

    state, action, reward, next_state, done = buf.sample_batch()

    assert state.tolist() == [[2.0, 2.0], [3.0, 3.0]]
    assert action.tolist() == [[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]
    assert reward.tolist() == [2.0, 3.0]
    assert next_state.tolist() == [[12.0, 12.0], [13.0, 13.0]]
    assert done.tolist() == [1.0, 0.0]


def test_sample_batch_can_return_transition_origins():
    """Origin metadata lets DQN metrics split original replay from HER replay."""
    buf = Buffer(size=5, sample_size=5)
    buf.add(*_exp(False, value=1.0), origin="original")
    buf.add(*_exp(True, value=2.0), origin="her")

    *_, origins = buf.sample_batch(with_origin=True)

    assert origins == ("original", "her")


# Author: Mus mbayramo@stanford.edu
