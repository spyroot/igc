"""Phase 1 metric-key registry includes the committed acceptance producer keys.

Author:
Mus mbayramo@stanford.edu
"""
from igc.modules.base.metric_keys import (
    PHASE1_ACCEPTANCE_METRIC_KEYS,
    PHASE1_ALL_METRIC_KEYS,
    PHASE1_FINETUNE,
    PHASE1_WANDB_METRIC_KEYS,
    phase_metric,
)


def test_phase1_metric_registry_keeps_expected_live_keys() -> None:
    """The current Phase 1 registry still covers train/eval/throughput basics."""
    keys = set(PHASE1_WANDB_METRIC_KEYS)
    assert phase_metric(PHASE1_FINETUNE, "train", "loss") in keys
    assert phase_metric(PHASE1_FINETUNE, "train", "epoch_loss") in keys
    assert phase_metric(PHASE1_FINETUNE, "eval", "token_accuracy") in keys
    assert phase_metric(PHASE1_FINETUNE, "throughput", "train_tokens_per_sec") in keys


def test_phase1_metric_registry_includes_golden_acceptance_keys() -> None:
    """The offline held-out producer emits the acceptance metric family."""
    keys = set(PHASE1_ACCEPTANCE_METRIC_KEYS)
    assert phase_metric(PHASE1_FINETUNE, "eval", "json_parse_rate") in keys
    assert phase_metric(PHASE1_FINETUNE, "eval", "json_exact_match_rate") in keys
    assert phase_metric(PHASE1_FINETUNE, "eval", "odata_id_match_rate") in keys
    assert phase_metric(PHASE1_FINETUNE, "throughput", "eval_tokens_per_sec") in keys
    assert phase_metric(PHASE1_FINETUNE, "data", "padding_ratio") in keys
    assert phase_metric(PHASE1_FINETUNE, "calibration", "ece") in keys
    assert phase_metric(PHASE1_FINETUNE, "test", "latency_sec_p95") in keys


def test_phase1_all_metric_keys_combines_live_and_acceptance_families() -> None:
    """Consumers that want both families can opt into the combined registry."""
    all_keys = set(PHASE1_ALL_METRIC_KEYS)
    assert set(PHASE1_WANDB_METRIC_KEYS) < all_keys
    assert set(PHASE1_ACCEPTANCE_METRIC_KEYS) < all_keys


# Author: Mus mbayramo@stanford.edu
