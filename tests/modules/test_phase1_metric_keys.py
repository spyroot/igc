"""Phase 1 metric-key registry must not advertise known dead tuples.

Author:
Mus mbayramo@stanford.edu
"""
from igc.modules.base.metric_keys import (
    PHASE1_PRETRAIN,
    PHASE1_WANDB_METRIC_KEYS,
    phase_metric,
)


def test_phase1_metric_registry_keeps_expected_live_keys() -> None:
    """The current Phase 1 registry still covers train/eval/throughput basics."""
    keys = set(PHASE1_WANDB_METRIC_KEYS)
    assert phase_metric(PHASE1_PRETRAIN, "train", "loss") in keys
    assert phase_metric(PHASE1_PRETRAIN, "train", "epoch_loss") in keys
    assert phase_metric(PHASE1_PRETRAIN, "eval", "token_accuracy") in keys
    assert phase_metric(PHASE1_PRETRAIN, "throughput", "train_tokens_per_sec") in keys


def test_phase1_metric_registry_omits_unwired_structural_keys() -> None:
    """Do not advertise structural/data metrics until a real producer lands."""
    keys = set(PHASE1_WANDB_METRIC_KEYS)
    assert phase_metric(PHASE1_PRETRAIN, "eval", "json_parse_rate") not in keys
    assert phase_metric(PHASE1_PRETRAIN, "eval", "json_exact_match_rate") not in keys
    assert phase_metric(PHASE1_PRETRAIN, "eval", "odata_id_match_rate") not in keys
    assert phase_metric(PHASE1_PRETRAIN, "data", "padding_ratio") not in keys


# Author: Mus mbayramo@stanford.edu
