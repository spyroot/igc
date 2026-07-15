"""Shared metric-key registry for staged IGC training.

Used by offline tests, launch code, and phase docs to keep Phase 1, Phase 2, and
Phase 3 metric names aligned with the existing ``MetricLogger`` / ``metric_factory``
path. This module defines strings only; it does not import W&B or open a live run.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

PHASE1_PRETRAIN = "phase1_pretrain"
PHASE2_GOAL_EXTRACT = "phase2_goal_extract"
PHASE2_LABELLED_REQUESTS = "phase2_labelled_requests"
PHASE3_ARGUMENT_EXTRACT = "phase3_argument_extract"


def phase_metric(namespace: str, group: str, name: str) -> str:
    """Return a stable ``<namespace>/<group>/<name>`` metric key.

    :param namespace: Phase namespace, for example :data:`PHASE1_PRETRAIN`.
    :param group: Metric group such as ``train`` or ``eval``.
    :param name: Metric name inside the group.
    :return: Slash-separated W&B/TensorBoard metric key.
    :raises ValueError: if any component is empty.
    """
    if not namespace or not group or not name:
        raise ValueError("metric namespace, group, and name must be non-empty")
    return f"{namespace}/{group}/{name}"


PHASE1_WANDB_METRIC_KEYS = (
    phase_metric(PHASE1_PRETRAIN, "train", "loss"),
    phase_metric(PHASE1_PRETRAIN, "train", "epoch_loss"),
    phase_metric(PHASE1_PRETRAIN, "train", "perplexity"),
    phase_metric(PHASE1_PRETRAIN, "train", "epoch_perplexity"),
    phase_metric(PHASE1_PRETRAIN, "train", "optimizer_step"),
    phase_metric(PHASE1_PRETRAIN, "train", "tokens_processed"),
    phase_metric(PHASE1_PRETRAIN, "eval", "loss"),
    phase_metric(PHASE1_PRETRAIN, "eval", "perplexity"),
    phase_metric(PHASE1_PRETRAIN, "eval", "token_accuracy"),
    phase_metric(PHASE1_PRETRAIN, "throughput", "train_tokens_per_sec"),
    phase_metric(PHASE1_PRETRAIN, "throughput", "train_samples_per_sec"),
)

PHASE2_WANDB_METRIC_KEYS = (
    phase_metric(PHASE2_GOAL_EXTRACT, "train", "loss"),
    phase_metric(PHASE2_GOAL_EXTRACT, "train", "perplexity"),
    phase_metric(PHASE2_GOAL_EXTRACT, "train", "optimizer_step"),
    phase_metric(PHASE2_GOAL_EXTRACT, "eval", "ordered_exact_match_rate"),
    phase_metric(PHASE2_GOAL_EXTRACT, "eval", "set_match_rate"),
    phase_metric(PHASE2_GOAL_EXTRACT, "eval", "precision"),
    phase_metric(PHASE2_GOAL_EXTRACT, "eval", "recall"),
    phase_metric(PHASE2_GOAL_EXTRACT, "eval", "f1"),
    phase_metric(PHASE2_GOAL_EXTRACT, "eval", "missing_allowed_methods_rate"),
    phase_metric(PHASE2_GOAL_EXTRACT, "order", "kendall_tau"),
    phase_metric(PHASE2_GOAL_EXTRACT, "order", "edit_distance"),
)

PHASE2_LABELLED_REQUESTS_WANDB_METRIC_KEYS = (
    phase_metric(PHASE2_LABELLED_REQUESTS, "build", "draft_total"),
    phase_metric(PHASE2_LABELLED_REQUESTS, "build", "accepted_total"),
    phase_metric(PHASE2_LABELLED_REQUESTS, "build", "rejected_total"),
    phase_metric(PHASE2_LABELLED_REQUESTS, "eval", "nonsense_rate"),
    phase_metric(PHASE2_LABELLED_REQUESTS, "eval", "invalid_json_rate"),
    phase_metric(PHASE2_LABELLED_REQUESTS, "eval", "pro_accept_rate"),
    phase_metric(PHASE2_LABELLED_REQUESTS, "eval", "rest_api_set_match_rate"),
    phase_metric(PHASE2_LABELLED_REQUESTS, "eval", "empty_set_match_rate"),
    phase_metric(PHASE2_LABELLED_REQUESTS, "sample_width", "k"),
    phase_metric(PHASE2_LABELLED_REQUESTS, "vendor", "source_corpus"),
    phase_metric(PHASE2_LABELLED_REQUESTS, "spec", "prompt_spec_version"),
    phase_metric(PHASE2_LABELLED_REQUESTS, "model", "model_x_artifact_sha"),
    phase_metric(PHASE2_LABELLED_REQUESTS, "judge", "model"),
    phase_metric(PHASE2_LABELLED_REQUESTS, "judge", "profile"),
)

PHASE3_WANDB_METRIC_KEYS = (
    phase_metric(PHASE3_ARGUMENT_EXTRACT, "train", "loss"),
    phase_metric(PHASE3_ARGUMENT_EXTRACT, "train", "perplexity"),
    phase_metric(PHASE3_ARGUMENT_EXTRACT, "train", "optimizer_step"),
    phase_metric(PHASE3_ARGUMENT_EXTRACT, "eval", "call_ordered_exact_match_rate"),
    phase_metric(PHASE3_ARGUMENT_EXTRACT, "eval", "method_exact_match_rate"),
    phase_metric(PHASE3_ARGUMENT_EXTRACT, "eval", "arguments_exact_match_rate"),
    phase_metric(PHASE3_ARGUMENT_EXTRACT, "eval", "readonly_empty_arguments_rate"),
    phase_metric(PHASE3_ARGUMENT_EXTRACT, "order", "kendall_tau"),
    phase_metric(PHASE3_ARGUMENT_EXTRACT, "order", "edit_distance"),
)

PHASE23_WANDB_METRIC_KEYS = PHASE2_WANDB_METRIC_KEYS + PHASE3_WANDB_METRIC_KEYS


# Author: Mus mbayramo@stanford.edu
