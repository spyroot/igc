"""Offline regression for ragged done-flag stacking in ``Buffer._stack_done``.

A replay batch can mix a scalar ``[]`` terminal tensor with a ``[1]`` one (HER relabeling
and the ``add`` default produce different shapes). ``torch.stack`` fails on ragged entries,
so the buffer coerces every done flag to its sample's reward shape first. These CPU tests
pin that normalization; they are the 1:1 regression for that source change.

Author:
Mus mbayramo@stanford.edu
"""
import torch

from igc.modules.igc_experience_buffer import Buffer


def test_stack_done_normalizes_mixed_tensor_shapes():
    """A ``[1]`` done mixed with a scalar ``[]`` done stacks without a ragged error."""
    reward_batch = torch.tensor([0.0, 0.0])
    samples = [
        (None, None, torch.tensor(0.0), None, torch.tensor([1.0])),
        (None, None, torch.tensor(0.0), None, torch.tensor(0.0)),
    ]

    done = Buffer._stack_done(samples, reward_batch)

    assert done.shape == reward_batch.shape
    assert done.reshape(-1).tolist() == [1.0, 0.0]


def test_stack_done_broadcasts_scalar_bool_and_zero_dim():
    """Plain ``bool`` and 0-dim tensor done values broadcast to the reward shape."""
    reward_batch = torch.zeros(2, 1)
    samples = [
        (None, None, torch.zeros(1), None, True),
        (None, None, torch.zeros(1), None, torch.tensor(1.0)),
    ]

    done = Buffer._stack_done(samples, reward_batch)

    assert done.shape == (2, 1)
    assert done.reshape(-1).tolist() == [1.0, 1.0]


def test_stack_done_missing_flag_defaults_to_not_done():
    """A sample added without a done flag contributes a zero (non-terminal) mask."""
    reward_batch = torch.tensor([0.0])
    samples = [(None, None, torch.tensor(0.0), None)]

    done = Buffer._stack_done(samples, reward_batch)

    assert done.shape == reward_batch.shape
    assert done.reshape(-1).tolist() == [0.0]


# Author: Mus mbayramo@stanford.edu
