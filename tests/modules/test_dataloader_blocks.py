"""Offline boundary tests for trainer DataLoader blocks.

These tests pin the small CPU-safe pieces that feed every later GPU run:
collation of tokenized rows and sampler selection. They do not instantiate the
heavy trainers or load models.

Author:
Mus mbayramo@stanford.edu
"""

import pytest
import torch

from igc.modules.igc_train_auto_state_encoder import AutoencoderTrainer
from igc.modules.llm_train_state_encoder import LlmEmbeddingsTrainer


def _sample(i: int, *, extra: bool = False) -> dict[str, torch.Tensor]:
    """Build one tiny tokenized row."""
    row = {
        "input_ids": torch.tensor([i, i + 1, i + 2]),
        "attention_mask": torch.tensor([1, 1, 0]),
    }
    if extra:
        row["labels"] = torch.tensor([99, 99, 99])
    return row


@pytest.mark.parametrize(
    "collate",
    [
        LlmEmbeddingsTrainer.custom_collate_fn,
        AutoencoderTrainer.custom_collate_fn,
    ],
)
def test_collate_stacks_only_model_inputs(collate):
    """Collate stacks ids/masks and ignores unrelated dataset columns."""
    batch = collate([_sample(0, extra=True), _sample(10, extra=True)])

    assert set(batch) == {"input_ids", "attention_mask"}
    assert batch["input_ids"].shape == (2, 3)
    assert batch["attention_mask"].shape == (2, 3)
    assert batch["input_ids"].tolist() == [[0, 1, 2], [10, 11, 12]]


@pytest.mark.parametrize(
    "collate",
    [
        LlmEmbeddingsTrainer.custom_collate_fn,
        AutoencoderTrainer.custom_collate_fn,
    ],
)
def test_collate_rejects_missing_attention_mask(collate):
    """A malformed row fails before training silently drops the mask."""
    row = _sample(0)
    row.pop("attention_mask")

    with pytest.raises(KeyError, match="attention_mask"):
        collate([row])


@pytest.mark.parametrize(
    "collate",
    [
        LlmEmbeddingsTrainer.custom_collate_fn,
        AutoencoderTrainer.custom_collate_fn,
    ],
)
def test_collate_rejects_empty_micro_batch(collate):
    """An empty micro-batch is invalid and fails loudly."""
    with pytest.raises(RuntimeError, match="stack expects a non-empty"):
        collate([])


# Author: Mus mbayramo@stanford.edu
