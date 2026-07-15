"""
Regression guard for the distributed eval-metric gather fix.

A 4-GPU FSDP run hung at ~epoch 8 on an ``_ALLGATHER_BASE`` NCCL collective (600s watchdog): each
rank validated only its own eval shard, so ``validate()`` returned a RANK-LOCAL accuracy, the ranks
disagreed on the "best" metric, and they entered the epoch-boundary save collective asymmetrically.
The fix all-reduces the eval counts so every rank computes the SAME global accuracy, and tracks the
best on every rank (not only rank 0). These source-inspection guards mirror the existing
``test_dist_batch_invariant`` pattern — they keep the fix from silently regressing without needing a
multi-GPU box in the offline gate.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

import inspect

from igc.modules.llm_train_state_encoder import LlmEmbeddingsTrainer


def test_validate_all_reduces_the_metric():
    """validate() must all-reduce eval counts/loss so every rank sees one global metric."""
    src = inspect.getsource(LlmEmbeddingsTrainer.validate)
    assert "accelerator.reduce" in src, (
        "validate() must reduce (correct, total, loss_sum, loss_tokens) across ranks so every "
        "rank computes the same global eval metric; rank-local best values deadlock the save."
    )


def test_best_metric_tracked_on_all_ranks_not_only_rank_zero():
    """The best-metric update must run on ALL ranks, not under is_rank_zero.

    If only rank 0 tracks the best, the non-zero ranks keep stale metrics, compute best-save state
    differently, and enter the save collective asymmetrically -> the _ALLGATHER_BASE hang.
    """
    src = inspect.getsource(LlmEmbeddingsTrainer)
    assert src.count("self._best_validation_metric = selection_metric") == 1, (
        "expected exactly one best-metric assignment (the rank-0-only one was removed)"
    )
    idx = src.index("self._best_validation_metric = selection_metric")
    preceding = src[:idx]
    assert preceding.rfind("if improved:") > preceding.rfind("if self.is_rank_zero()"), (
        "the best-metric update must be gated on improved on every rank, i.e. BEFORE / "
        "outside the is_rank_zero() block, so all ranks agree on 'best'."
    )


def test_phase1_best_metric_is_eval_loss_minimize_with_min_delta():
    """Phase 1 must select checkpoints by lower eval loss, not higher token accuracy."""
    src = inspect.getsource(LlmEmbeddingsTrainer._train)
    assert "self._select_best_by_eval_loss" in src
    assert "selection_metric = validation_eval_loss" in src
    assert "self._early_stopping_min_delta" in src
    assert "< self._best_validation_metric" in src


# Author: Mus mbayramo@stanford.edu
