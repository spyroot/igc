"""Offline tests for build_causal_lm_labels (the double-shift fix in _train).

A HF CausalLM shifts labels internally, so the trainer must pass FULL-length,
un-pre-shifted labels — pinning that here guards against the regression where labels
were pre-shifted and then shifted again (training to predict two tokens ahead).
Pad/padding positions are ignored via -100 without ever corrupting input_ids.

Author:
Mus mbayramo@stanford.edu
"""

from types import SimpleNamespace

import torch

from igc.modules.llm_train_state_encoder import (
    LlmEmbeddingsTrainer,
    build_causal_lm_labels,
    causal_lm_labels_from_batch,
)


def test_labels_are_full_length_and_not_preshifted() -> None:
    """Labels match input_ids shape (HF does the one shift), not a truncated target."""
    input_ids = torch.tensor([[5, 6, 7, 8]])
    attention_mask = torch.ones_like(input_ids)
    labels = build_causal_lm_labels(input_ids, attention_mask, pad_token_id=0)
    assert labels.shape == input_ids.shape
    assert torch.equal(labels, input_ids)  # nothing ignored when fully attended, no pads


def test_padding_and_pad_token_ignored() -> None:
    """Attention-padding and pad-token positions become -100 in the labels."""
    pad = 0
    input_ids = torch.tensor([[5, 6, pad, pad]])
    attention_mask = torch.tensor([[1, 1, 0, 0]])
    labels = build_causal_lm_labels(input_ids, attention_mask, pad_token_id=pad)
    assert labels[0].tolist() == [5, 6, -100, -100]


def test_input_ids_never_mutated() -> None:
    """-100 goes only into labels; input_ids stays a valid embedding index tensor."""
    input_ids = torch.tensor([[5, 6, 0]])
    attention_mask = torch.tensor([[1, 1, 0]])
    before = input_ids.clone()
    build_causal_lm_labels(input_ids, attention_mask, pad_token_id=0)
    assert torch.equal(input_ids, before)


def test_pad_token_none_masks_only_attention_padding() -> None:
    """With no pad id, only attention-padding positions are ignored."""
    input_ids = torch.tensor([[5, 6, 7]])
    attention_mask = torch.tensor([[1, 1, 0]])
    labels = build_causal_lm_labels(input_ids, attention_mask, pad_token_id=None)
    assert labels[0].tolist() == [5, 6, -100]


def test_dataset_labels_are_used_without_rebuilding() -> None:
    """Prompt-masked dataset labels win over legacy whole-sequence label building."""
    input_ids = torch.tensor([[5, 6, 0]])
    attention_mask = torch.tensor([[1, 1, 0]])
    provided = torch.tensor([[-100, 6, -100]])
    batch = {"labels": provided}

    labels = causal_lm_labels_from_batch(batch, input_ids, attention_mask, pad_token_id=0)

    assert torch.equal(labels, provided)


def test_custom_collate_preserves_optional_labels() -> None:
    """Phase 1 batches include labels while legacy batches keep the old two keys."""
    samples = [
        {
            "input_ids": torch.tensor([1, 2]),
            "attention_mask": torch.tensor([1, 1]),
            "labels": torch.tensor([-100, 2]),
        },
        {
            "input_ids": torch.tensor([3, 4]),
            "attention_mask": torch.tensor([1, 1]),
            "labels": torch.tensor([-100, 4]),
        },
    ]

    batch = LlmEmbeddingsTrainer.custom_collate_fn(samples)

    assert set(batch) == {"input_ids", "attention_mask", "labels"}
    assert batch["labels"].tolist() == [[-100, 2], [-100, 4]]


def test_custom_collate_ignores_unmasked_legacy_extra_labels() -> None:
    """A stray legacy labels column is ignored unless it is prompt-masked."""
    samples = [
        {
            "input_ids": torch.tensor([1, 2]),
            "attention_mask": torch.tensor([1, 1]),
            "labels": torch.tensor([99, 99]),
        },
        {
            "input_ids": torch.tensor([3, 4]),
            "attention_mask": torch.tensor([1, 1]),
            "labels": torch.tensor([99, 99]),
        },
    ]

    batch = LlmEmbeddingsTrainer.custom_collate_fn(samples)

    assert set(batch) == {"input_ids", "attention_mask"}


def test_custom_collate_ignores_all_ignored_extra_labels() -> None:
    """All-ignored labels are invalid Phase 1 rows, not usable prompt masks."""
    samples = [
        {
            "input_ids": torch.tensor([1, 2, 3]),
            "attention_mask": torch.tensor([1, 1, 1]),
            "labels": torch.tensor([-100, -100, -100]),
        },
        {
            "input_ids": torch.tensor([4, 5, 6]),
            "attention_mask": torch.tensor([1, 1, 1]),
            "labels": torch.tensor([-100, -100, -100]),
        },
    ]

    batch = LlmEmbeddingsTrainer.custom_collate_fn(samples)

    assert set(batch) == {"input_ids", "attention_mask"}


def test_validate_returns_percent_for_legacy_metric_contract() -> None:
    """Validation returns 0-100 percent; namespaced W&B accuracy divides by 100."""

    class TinyValidationModel(torch.nn.Module):
        """Return deterministic logits with one correct and one wrong token."""

        def forward(self, input_ids, attention_mask):
            logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], 8)
            logits[:, 0, 2] = 10.0
            logits[:, 1, 4] = 10.0
            return SimpleNamespace(logits=logits)

    trainer = LlmEmbeddingsTrainer.__new__(LlmEmbeddingsTrainer)
    trainer.model = TinyValidationModel()
    trainer._device = torch.device("cpu")
    trainer._accelerator = None
    trainer.tokenizer = SimpleNamespace(pad_token_id=0)
    trainer.is_accelerator = False
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1, 0]]),
        "labels": torch.tensor([[-100, 2, 3, -100]]),
    }

    accuracy = LlmEmbeddingsTrainer.validate(trainer, [batch])

    assert accuracy == 50.0
    assert accuracy / 100.0 == 0.5


# Author: Mus mbayramo@stanford.edu
