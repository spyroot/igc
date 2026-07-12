"""Offline guard for the epoch-boundary save-collective deadlock (the "epoch-2 hang").

Under accelerate multi-rank training, the epoch boundary in
``igc/modules/llm_train_state_encoder.py`` calls ``broadcast_flag`` +
``accelerator.get_state_dict`` — collectives EVERY rank must reach together (its own
comment: "every rank participates or the fleet deadlocks"). Ranks reach them together
only if they process the SAME number of train batches. A train ``DataLoader`` without
``drop_last=True`` gives ranks a different batch count on an indivisible dataset, so one
rank hits the save collective while another is still looping -> NCCL deadlock at the first
save-eligible epoch ("epoch-2, one rank diverges post-save").

These run on CPU (no GPU/NCCL): they pin the batch-count invariant and assert the igc train
DataLoaders are built with ``drop_last=True``. The full multi-rank deadlock reproduction
lives in ``scripts/gpu_dataset_isolation.py`` (GPU, torchrun).

Author:
Mus mbayramo@stanford.edu
"""
import inspect
import re

import pytest
import torch
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset


def _batches_per_rank(n: int, batch_size: int, world: int, drop_last: bool):
    """Count DataLoader batches each rank would see for a size-n dataset."""
    ds = TensorDataset(torch.arange(n))
    counts = []
    for rank in range(world):
        sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=False, drop_last=drop_last)
        loader = DataLoader(ds, batch_size=batch_size, sampler=sampler, drop_last=drop_last)
        counts.append(sum(1 for _ in loader))
    return counts


@pytest.mark.parametrize("n,bs,world", [(100, 8, 4), (97, 8, 4), (2352, 16, 8), (10, 4, 3), (1, 4, 2)])
def test_drop_last_gives_equal_batch_counts_per_rank(n, bs, world):
    """drop_last=True => every rank sees the same #batches (the anti-deadlock invariant)."""
    counts = _batches_per_rank(n, bs, world, drop_last=True)
    assert len(set(counts)) == 1, f"unequal batch counts with drop_last=True: n={n} bs={bs} world={world} -> {counts}"


def test_igc_train_dataloaders_use_drop_last():
    """Regression guard: every train DataLoader in the encoder trainer sets drop_last=True.

    This is the exact fix for the epoch-boundary save-collective deadlock; if a future edit
    drops it, this test fails before a 72-GPU run hangs.
    """
    from igc.modules import llm_train_state_encoder as trainer_mod
    src = inspect.getsource(trainer_mod)
    blocks = re.findall(r"train_dataloader = DataLoader\((.*?)\n        \)", src, re.S)
    assert blocks, "no train_dataloader = DataLoader(...) construction found"
    for i, block in enumerate(blocks):
        assert "drop_last=True" in block, (
            f"train_dataloader #{i} is missing drop_last=True — this reintroduces the "
            f"epoch-boundary save-collective deadlock")


# Author: Mus mbayramo@stanford.edu
