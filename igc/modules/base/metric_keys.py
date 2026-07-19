"""Shared metric-key registry for staged IGC training.

Used by offline tests, launch code, and phase docs to keep Phase 1, Phase 2, and
Phase 3 metric names aligned with the existing ``MetricLogger`` / ``metric_factory``
path. This module defines strings only; it does not import W&B or open a live run.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

PHASE1_OBJECTIVE_PRETRAIN = "phase1_pretrain"
PHASE1_FINETUNE = "phase1_finetune"
PHASE2_GOAL_EXTRACT = "phase2_goal_extraction"
PHASE2_LABELLED_REQUESTS = "phase2_labelled_requests"
PHASE3_ARGUMENT_EXTRACT = "phase3_argument_extraction"


def phase_metric(namespace: str, group: str, name: str | None = None) -> str:
    """Return a stable slash-separated metric key.

    :param namespace: Phase namespace, for example :data:`PHASE1_FINETUNE`.
    :param group: Metric group or metric name.
    :param name: Optional metric name inside the group.
    :return: Slash-separated W&B/TensorBoard metric key.
    :raises ValueError: if any component is empty.
    """
    parts = (namespace, group) if name is None else (namespace, group, name)
    if any(not part for part in parts):
        raise ValueError("metric key components must be non-empty")
    return "/".join(parts)


PHASE1_WANDB_METRIC_KEYS = (
    phase_metric(PHASE1_FINETUNE, "train", "loss"),
    phase_metric(PHASE1_FINETUNE, "train", "epoch_loss"),
    phase_metric(PHASE1_FINETUNE, "train", "perplexity"),
    phase_metric(PHASE1_FINETUNE, "train", "epoch_perplexity"),
    phase_metric(PHASE1_FINETUNE, "train", "optimizer_step"),
    phase_metric(PHASE1_FINETUNE, "train", "tokens_processed"),
    phase_metric(PHASE1_FINETUNE, "eval", "loss"),
    phase_metric(PHASE1_FINETUNE, "eval", "perplexity"),
    phase_metric(PHASE1_FINETUNE, "eval", "token_accuracy"),
    phase_metric(PHASE1_FINETUNE, "eval", "key_presence_rate"),
    phase_metric(PHASE1_FINETUNE, "eval", "value_exact_match_rate"),
    phase_metric(PHASE1_FINETUNE, "throughput", "train_tokens_per_sec"),
    phase_metric(PHASE1_FINETUNE, "throughput", "train_samples_per_sec"),
    phase_metric(PHASE1_FINETUNE, "calibration", "nll"),
    phase_metric(PHASE1_FINETUNE, "calibration", "brier_score"),
)

PHASE1_ACCEPTANCE_METRIC_KEYS = (
    phase_metric(PHASE1_FINETUNE, "eval", "top_k_accuracy"),
    phase_metric(PHASE1_FINETUNE, "eval", "json_parse_rate"),
    phase_metric(PHASE1_FINETUNE, "eval", "json_exact_match_rate"),
    phase_metric(PHASE1_FINETUNE, "eval", "odata_id_match_rate"),
    phase_metric(PHASE1_FINETUNE, "throughput", "eval_tokens_per_sec"),
    phase_metric(PHASE1_FINETUNE, "throughput", "eval_samples_per_sec"),
    phase_metric(PHASE1_FINETUNE, "data", "padding_ratio"),
    phase_metric(PHASE1_FINETUNE, "data", "mean_sequence_length"),
    phase_metric(PHASE1_FINETUNE, "data", "max_sequence_length"),
    phase_metric(PHASE1_FINETUNE, "calibration", "log_prob_per_token"),
    phase_metric(PHASE1_FINETUNE, "calibration", "ece"),
    phase_metric(PHASE1_FINETUNE, "test", "latency_sec_p50"),
    phase_metric(PHASE1_FINETUNE, "test", "latency_sec_p95"),
    phase_metric(PHASE1_FINETUNE, "test", "memory_peak_mb"),
)

PHASE1_ALL_METRIC_KEYS = PHASE1_WANDB_METRIC_KEYS + PHASE1_ACCEPTANCE_METRIC_KEYS

# Phase 2 is an UNORDERED set task: set match is the primary metric; there are
# no order/* keys (execution order is RL-oracle evidence, not a Phase 2 output).
PHASE2_WANDB_METRIC_KEYS = (
    phase_metric(PHASE2_GOAL_EXTRACT, "train", "loss"),
    phase_metric(PHASE2_GOAL_EXTRACT, "train", "perplexity"),
    phase_metric(PHASE2_GOAL_EXTRACT, "train", "optimizer_step"),
    phase_metric(PHASE2_GOAL_EXTRACT, "eval", "set_match_rate"),
    phase_metric(PHASE2_GOAL_EXTRACT, "eval", "precision"),
    phase_metric(PHASE2_GOAL_EXTRACT, "eval", "recall"),
    phase_metric(PHASE2_GOAL_EXTRACT, "eval", "f1"),
    phase_metric(PHASE2_GOAL_EXTRACT, "eval", "invalid_rest_rate"),
    phase_metric(PHASE2_GOAL_EXTRACT, "eval", "hard_negative_accuracy"),
)

PHASE2_LABELLED_REQUESTS_WANDB_METRIC_KEYS = (
    phase_metric(PHASE2_LABELLED_REQUESTS, "draft_total"),
    phase_metric(PHASE2_LABELLED_REQUESTS, "accepted_total"),
    phase_metric(PHASE2_LABELLED_REQUESTS, "rejected_total"),
    phase_metric(PHASE2_LABELLED_REQUESTS, "nonsense_rate"),
    phase_metric(PHASE2_LABELLED_REQUESTS, "invalid_json_rate"),
    phase_metric(PHASE2_LABELLED_REQUESTS, "pro_accept_rate"),
    phase_metric(PHASE2_LABELLED_REQUESTS, "rest_api_set_match_rate"),
    phase_metric(PHASE2_LABELLED_REQUESTS, "empty_set_match_rate"),
    phase_metric(PHASE2_LABELLED_REQUESTS, "empty_set_expected_total"),
    phase_metric(PHASE2_LABELLED_REQUESTS, "sample_width", "k"),
    phase_metric(PHASE2_LABELLED_REQUESTS, "vendor", "source_corpus"),
    phase_metric(PHASE2_LABELLED_REQUESTS, "prompt_spec_version"),
    phase_metric(PHASE2_LABELLED_REQUESTS, "model_x", "artifact_sha"),
    phase_metric(PHASE2_LABELLED_REQUESTS, "judge", "model"),
    phase_metric(PHASE2_LABELLED_REQUESTS, "judge", "profile"),
)

# Phase 3 is an UNORDERED bound-call set task. The contract metric family:
# method accuracy, argument JSON validity, required-argument coverage,
# no-argument accuracy, and unsafe/unsupported-argument rejection — plus the
# overall call-set match. No order/* keys (order is RL-oracle evidence).
PHASE3_WANDB_METRIC_KEYS = (
    phase_metric(PHASE3_ARGUMENT_EXTRACT, "train", "loss"),
    phase_metric(PHASE3_ARGUMENT_EXTRACT, "train", "perplexity"),
    phase_metric(PHASE3_ARGUMENT_EXTRACT, "train", "optimizer_step"),
    phase_metric(PHASE3_ARGUMENT_EXTRACT, "eval", "call_set_exact_match_rate"),
    phase_metric(PHASE3_ARGUMENT_EXTRACT, "eval", "method_exact_match_rate"),
    phase_metric(PHASE3_ARGUMENT_EXTRACT, "eval", "arguments_json_validity_rate"),
    phase_metric(PHASE3_ARGUMENT_EXTRACT, "eval", "arguments_exact_match_rate"),
    phase_metric(PHASE3_ARGUMENT_EXTRACT, "eval", "required_argument_coverage_rate"),
    phase_metric(PHASE3_ARGUMENT_EXTRACT, "eval", "no_argument_accuracy_rate"),
    phase_metric(PHASE3_ARGUMENT_EXTRACT, "eval", "unsafe_argument_rejection_rate"),
)

PHASE23_WANDB_METRIC_KEYS = PHASE2_WANDB_METRIC_KEYS + PHASE3_WANDB_METRIC_KEYS


# Author: Mus mbayramo@stanford.edu
