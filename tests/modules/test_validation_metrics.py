"""Offline regressions for the encoder validation metrics.

``compute_accuracy`` double-shifted (callers already shift inputs vs targets
by one, so it measured two-tokens-ahead) and masked LOGIT VALUES against -100
(a no-op), counting every pad position as wrong — and this metric drives
best-checkpoint selection. Perplexity weighted the mean loss by ALL positions
while dividing by non-pad ones. Pins the aligned, pad-aware contract.

Author:
Mus mbayramo@stanford.edu
"""

import torch

from igc.modules.llm_train_state_encoder import LlmEmbeddingsTrainer


def _aligned_logits(targets, vocab=7):
    """Logits whose argmax equals targets wherever targets are valid."""
    batch, seq = targets.shape
    logits = torch.zeros(batch, seq, vocab)
    for b in range(batch):
        for t in range(seq):
            idx = targets[b, t].item()
            logits[b, t, idx if idx >= 0 else 0] = 5.0
    return logits


def test_perfect_predictions_score_one():
    """Aligned argmax == targets gives 100% (old double-shift gave less)."""
    targets = torch.tensor([[1, 2, 3, 4]])
    accuracy = LlmEmbeddingsTrainer.compute_accuracy(
        _aligned_logits(targets), targets, original_mask=None
    )
    assert accuracy == 1.0


def test_pad_positions_do_not_deflate_accuracy():
    """-100 labels are excluded, not counted as wrong."""
    targets = torch.tensor([[1, 2, -100, -100]])
    accuracy = LlmEmbeddingsTrainer.compute_accuracy(
        _aligned_logits(targets), targets, original_mask=None
    )
    assert accuracy == 1.0


def test_wrong_predictions_counted_over_valid_only():
    """One wrong of two valid positions = 0.5 regardless of padding."""
    targets = torch.tensor([[1, 2, -100]])
    logits = _aligned_logits(targets)
    logits[0, 1] = torch.zeros(7)
    logits[0, 1, 5] = 5.0  # wrong prediction at the second valid slot
    accuracy = LlmEmbeddingsTrainer.compute_accuracy(logits, targets, None)
    assert accuracy == 0.5


def test_all_pad_batch_is_zero_not_nan():
    """A fully-padded batch returns 0.0 instead of dividing by zero."""
    targets = torch.full((1, 3), -100)
    accuracy = LlmEmbeddingsTrainer.compute_accuracy(
        torch.zeros(1, 3, 7), targets, None
    )
    assert accuracy == 0.0


# Author: Mus mbayramo@stanford.edu
