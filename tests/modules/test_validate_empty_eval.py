"""Offline regression: validate() tolerates an empty eval shard.

Under FSDP with drop_last + a small eval set, a rank can be handed 0 eval batches.
The accuracy computation divided correct/total unconditionally, so that rank crashed
with ZeroDivisionError mid-epoch — which, on a multi-GPU run, takes the whole fleet
down at a collective. validate() now returns 0.0 for an empty shard. (The companion
epoch-boundary barrier fix is distributed-only and is exercised on the 4-GPU run.)

Author:
Mus mbayramo@stanford.edu
"""

import types

from igc.modules.llm_train_state_encoder import LlmEmbeddingsTrainer


def test_validate_empty_eval_returns_zero_not_crash():
    """0 eval batches -> 0.0 accuracy, no ZeroDivisionError."""
    trainer = LlmEmbeddingsTrainer.__new__(LlmEmbeddingsTrainer)
    trainer.model = types.SimpleNamespace(eval=lambda: None)

    # empty dataloader -> the batch loop never runs -> total_predictions == 0
    assert trainer.validate([]) == 0.0


# Author: Mus mbayramo@stanford.edu
