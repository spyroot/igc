"""Offline tests for build_causal_lm_labels (the double-shift fix in _train).

A HF CausalLM shifts labels internally, so the trainer must pass FULL-length,
un-pre-shifted labels — pinning that here guards against the regression where labels
were pre-shifted and then shifted again (training to predict two tokens ahead).
Pad/padding positions are ignored via -100 without ever corrupting input_ids.

Author:
Mus mbayramo@stanford.edu
"""

import torch

from igc.modules.llm_train_state_encoder import build_causal_lm_labels


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


# Author: Mus mbayramo@stanford.edu
