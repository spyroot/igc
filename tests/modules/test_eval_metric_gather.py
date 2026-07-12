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
    """validate() must all-reduce the eval counts so accuracy is a single global value."""
    src = inspect.getsource(LlmEmbeddingsTrainer.validate)
    assert "accelerator.reduce" in src, (
        "validate() must reduce (correct, total) across ranks so every rank computes the same "
        "global accuracy; a rank-local accuracy makes ranks disagree on 'best' and deadlock the save."
    )


def test_best_metric_tracked_on_all_ranks_not_only_rank_zero():
    """The best-metric update must run on ALL ranks (gated on is_best_accuracy), not under is_rank_zero.

    If only rank 0 tracks the best, the non-zero ranks keep -inf, compute is_best_accuracy
    differently, and enter the save collective asymmetrically -> the _ALLGATHER_BASE hang.
    """
    src = inspect.getsource(LlmEmbeddingsTrainer)
    assert src.count("self._best_validation_metric = validation_accuracy") == 1, (
        "expected exactly one best-metric assignment (the rank-0-only one was removed)"
    )
    idx = src.index("self._best_validation_metric = validation_accuracy")
    preceding = src[:idx]
    assert preceding.rfind("if is_best_accuracy:") > preceding.rfind("if self.is_rank_zero()"), (
        "the best-metric update must be gated on is_best_accuracy on every rank, i.e. BEFORE / "
        "outside the is_rank_zero() block, so all ranks agree on 'best'."
    )


# Author: Mus mbayramo@stanford.edu
