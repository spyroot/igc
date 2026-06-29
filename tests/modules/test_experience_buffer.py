"""Offline CPU tests for the replay Buffer's done flag (igc.modules.igc_experience_buffer).

The buffer now stores and returns a per-transition ``done`` so the DQN target can
mask bootstrapping at terminals. Covers an explicit tensor done and the scalar
fallback for experiences added without one.

Author:
Mus mbayramo@stanford.edu
"""
import torch

from igc.modules.igc_experience_buffer import Buffer


def _exp(done):
    """A tiny experience: state[1,2], action[1,3], reward[1], next_state[1,2], done."""
    return (torch.zeros(1, 2), torch.zeros(1, 3), torch.ones(1), torch.zeros(1, 2), done)


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


# Author: Mus mbayramo@stanford.edu
