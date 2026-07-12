"""Metric names and summaries for the M6 RL/HER training stage.

The trainer writes these keys to the selected ``MetricLogger`` backend. Keeping
the names and the ratio math in one torch-only module makes dashboard contracts
unit-testable without a live environment or Weight & Biases run.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import torch

ORIGIN_ORIGINAL = "original"
ORIGIN_HER = "her"

HER_METRIC_KEYS = (
    "03_m6_her/relabelled_transition_ratio",
    "03_m6_her/relabelled_success_ratio",
    "03_m6_her/original_success_ratio",
    "03_m6_her/positive_reward_ratio_original",
    "03_m6_her/positive_reward_ratio_relabelled",
    "03_m6_her/future_goal_offset_mean_env_steps",
    "03_m6_her/future_goal_offset_p95_env_steps",
    "03_m6_her/unique_relabelled_goal_count",
    "03_m6_her/invalid_relabelled_goal_ratio",
    "03_m6_her/duplicate_goal_ratio",
    "03_m6_her/reward_recompute_error_count",
    "03_m6_her/goal_termination_recompute_error_count",
)

DQN_UPDATE_METRIC_KEYS = (
    "03_m6_dqn_update/loss_td_original_mean_per_transition",
    "03_m6_dqn_update/loss_td_her_mean_per_transition",
    "03_m6_dqn_update/td_error_abs_original_p90_reward",
    "03_m6_dqn_update/td_error_abs_her_p90_reward",
)

EVAL_ORIGINAL_GOALS_METRIC_KEYS = (
    "03_m6_eval_original_goals/success_ratio",
    "03_m6_eval_original_goals/return_undiscounted_mean_reward",
)

EVAL_HINDSIGHT_GOALS_METRIC_KEYS = (
    "03_m6_eval_hindsight_goals/success_ratio",
    "03_m6_eval_hindsight_goals/return_undiscounted_mean_reward",
)

RL_TRAIN_METRIC_KEYS = (
    "03_m6_rl_train/epsilon",
    "03_m6_rl_train/replay_buffer_size",
)


def summarize_her_metrics(
    *,
    original_rewards: Any,
    original_done: Any,
    relabelled_rewards: Any,
    relabelled_done: Any,
    future_offsets_env_steps: Any,
    relabelled_goals: Sequence[Any],
    invalid_relabelled_goal_count: int = 0,
    reward_recompute_error_count: int = 0,
    goal_termination_recompute_error_count: int = 0,
) -> dict[str, float]:
    """Summarize original and HER-relabelled replay samples.

    :param original_rewards: Rewards for original environment transitions.
    :param original_done: Terminal flags for original transitions.
    :param relabelled_rewards: Rewards recomputed under HER goals.
    :param relabelled_done: Terminal flags recomputed under HER goals.
    :param future_offsets_env_steps: Future-goal offsets in environment steps.
    :param relabelled_goals: Substituted goals used for HER samples.
    :param invalid_relabelled_goal_count: Additional invalid goal rows found by
        the caller, if it validates goals outside this helper.
    :param reward_recompute_error_count: Count of HER reward recomputation
        mismatches detected by the caller.
    :param goal_termination_recompute_error_count: Count of HER done-mask
        recomputation mismatches detected by the caller.
    :return: Metric-name to scalar-value mapping for RL/HER dashboards.
    """
    original_reward_t = _as_float_tensor(original_rewards)
    original_done_t = _as_float_tensor(original_done)
    relabelled_reward_t = _as_float_tensor(relabelled_rewards)
    relabelled_done_t = _as_float_tensor(relabelled_done)
    future_offsets_t = _as_float_tensor(future_offsets_env_steps)

    original_count = int(original_reward_t.numel())
    relabelled_count = int(relabelled_reward_t.numel())
    total_count = original_count + relabelled_count

    goal_keys, detected_invalid_goal_count = _goal_keys(relabelled_goals)
    invalid_goal_count = invalid_relabelled_goal_count + detected_invalid_goal_count
    unique_goal_count = len(set(goal_keys))
    duplicate_goal_count = max(len(goal_keys) - unique_goal_count, 0)

    metrics = {
        "03_m6_her/relabelled_transition_ratio": _safe_ratio(relabelled_count, total_count),
        "03_m6_her/relabelled_success_ratio": _safe_ratio(
            _count_positive(relabelled_done_t), relabelled_count
        ),
        "03_m6_her/original_success_ratio": _safe_ratio(
            _count_positive(original_done_t), original_count
        ),
        "03_m6_her/positive_reward_ratio_original": _safe_ratio(
            _count_positive(original_reward_t), original_count
        ),
        "03_m6_her/positive_reward_ratio_relabelled": _safe_ratio(
            _count_positive(relabelled_reward_t), relabelled_count
        ),
        "03_m6_her/future_goal_offset_mean_env_steps": _mean(future_offsets_t),
        "03_m6_her/future_goal_offset_p95_env_steps": _nearest_percentile(
            future_offsets_t, 0.95
        ),
        "03_m6_her/unique_relabelled_goal_count": float(unique_goal_count),
        "03_m6_her/invalid_relabelled_goal_ratio": _safe_ratio(
            invalid_goal_count, relabelled_count
        ),
        "03_m6_her/duplicate_goal_ratio": _safe_ratio(duplicate_goal_count, relabelled_count),
        "03_m6_her/reward_recompute_error_count": float(reward_recompute_error_count),
        "03_m6_her/goal_termination_recompute_error_count": float(
            goal_termination_recompute_error_count
        ),
        "03_m6_her/original_reward_mean": _mean(original_reward_t),
        "03_m6_her/relabelled_reward_mean": _mean(relabelled_reward_t),
    }
    return metrics


def summarize_dqn_update_metrics(
    *,
    td_errors: Any,
    td_losses: Any,
    origins: Sequence[str],
) -> dict[str, float]:
    """Summarize TD loss and TD-error percentiles by replay origin.

    :param td_errors: Per-transition ``Q(s,a) - target`` errors.
    :param td_losses: Per-transition squared TD losses.
    :param origins: One origin string per flattened transition.
    :return: Metric-name to scalar-value mapping for DQN update dashboards.
    """
    td_error_t = _as_float_tensor(td_errors)
    td_loss_t = _as_float_tensor(td_losses)
    if td_error_t.numel() != len(origins) or td_loss_t.numel() != len(origins):
        raise ValueError("td_errors, td_losses, and origins must have matching lengths")

    original_mask = torch.tensor(
        [origin == ORIGIN_ORIGINAL for origin in origins], dtype=torch.bool
    )
    her_mask = torch.tensor([origin == ORIGIN_HER for origin in origins], dtype=torch.bool)

    return {
        "03_m6_dqn_update/loss_td_original_mean_per_transition": _mean(td_loss_t[original_mask]),
        "03_m6_dqn_update/loss_td_her_mean_per_transition": _mean(td_loss_t[her_mask]),
        "03_m6_dqn_update/td_error_abs_original_p90_reward": _nearest_percentile(
            td_error_t[original_mask].abs(), 0.90
        ),
        "03_m6_dqn_update/td_error_abs_her_p90_reward": _nearest_percentile(
            td_error_t[her_mask].abs(), 0.90
        ),
    }


def summarize_eval_metrics(
    *,
    successes: Any,
    returns: Any,
    hindsight: bool,
) -> dict[str, float]:
    """Summarize goal-specific evaluation success and undiscounted return.

    :param successes: Success flags for the evaluated trajectories.
    :param returns: Undiscounted returns for the same trajectories.
    :param hindsight: Whether the trajectories use hindsight goals.
    :return: Metric-name to scalar-value mapping for original or hindsight eval.
    """
    success_t = _as_float_tensor(successes)
    return_t = _as_float_tensor(returns)
    keys = EVAL_HINDSIGHT_GOALS_METRIC_KEYS if hindsight else EVAL_ORIGINAL_GOALS_METRIC_KEYS
    return {
        keys[0]: _safe_ratio(_count_positive(success_t), int(success_t.numel())),
        keys[1]: _mean(return_t),
    }


def mean_metric_dicts(metric_dicts: Iterable[Mapping[str, float]]) -> dict[str, float]:
    """Average matching scalar metric keys across multiple update dictionaries.

    :param metric_dicts: Per-episode or per-update metric dictionaries.
    :return: Mean value per metric key, or an empty dictionary for no inputs.
    """
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for metric_dict in metric_dicts:
        for key, value in metric_dict.items():
            totals[key] = totals.get(key, 0.0) + float(value)
            counts[key] = counts.get(key, 0) + 1
    return {key: totals[key] / counts[key] for key in totals}


def log_metric_dict(metric_logger, metrics: Mapping[str, float], step: int) -> None:
    """Log finite scalar metrics to an ``igc`` metric logger.

    :param metric_logger: Metric logger instance, or ``None``.
    :param metrics: Metric-name to scalar-value mapping.
    :param step: Training step or epoch index.
    :return: ``None``.
    """
    if metric_logger is None:
        return
    for key, value in metrics.items():
        scalar = float(value)
        if math.isfinite(scalar):
            metric_logger.log_metric(key, scalar, step)


def _as_float_tensor(values: Any) -> torch.Tensor:
    """Convert tensors, scalars, and iterables of tensors/scalars to ``float32``."""
    if values is None:
        return torch.empty(0, dtype=torch.float32)
    if torch.is_tensor(values):
        return values.detach().cpu().to(torch.float32).reshape(-1)
    if isinstance(values, (int, float, bool)):
        return torch.tensor([float(values)], dtype=torch.float32)

    tensors = []
    for value in values:
        if torch.is_tensor(value):
            tensors.append(value.detach().cpu().to(torch.float32).reshape(-1))
        else:
            tensors.append(torch.tensor([float(value)], dtype=torch.float32))
    if not tensors:
        return torch.empty(0, dtype=torch.float32)
    return torch.cat(tensors, dim=0)


def _goal_keys(goals: Sequence[Any]) -> tuple[list[tuple[Any, ...]], int]:
    """Return hashable goal rows plus a count of invalid rows."""
    keys: list[tuple[Any, ...]] = []
    invalid_count = 0
    for goal in goals:
        if torch.is_tensor(goal):
            rows = _goal_tensor_rows(goal)
            for row in rows:
                if row.numel() == 0 or not torch.isfinite(row).all():
                    invalid_count += 1
                    continue
                keys.append(tuple(float(item) for item in row.tolist()))
            continue

        if goal is None:
            invalid_count += 1
            continue
        if isinstance(goal, (str, bytes)):
            keys.append((goal,))
            continue
        try:
            keys.append(tuple(goal))
        except TypeError:
            keys.append((goal,))
    return keys, invalid_count


def _goal_tensor_rows(goal: torch.Tensor) -> list[torch.Tensor]:
    """Flatten a goal tensor into one row per vectorized environment item."""
    tensor = goal.detach().cpu().to(torch.float32)
    if tensor.ndim == 0:
        return [tensor.reshape(1)]
    if tensor.ndim == 1:
        return [tensor.reshape(-1)]
    return list(tensor.reshape(tensor.shape[0], -1))


def _count_positive(values: torch.Tensor) -> int:
    """Count positive entries in a 1-D tensor."""
    if values.numel() == 0:
        return 0
    return int((values > 0.0).sum().item())


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    """Return ``numerator / denominator`` or ``0.0`` for an empty denominator."""
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def _mean(values: torch.Tensor) -> float:
    """Return a finite mean for a 1-D tensor, using ``0.0`` for empty inputs."""
    if values.numel() == 0:
        return 0.0
    return float(values.to(torch.float32).mean().item())


def _nearest_percentile(values: torch.Tensor, percentile: float) -> float:
    """Return a nearest-rank percentile, using ``0.0`` for empty inputs."""
    if values.numel() == 0:
        return 0.0
    if not 0.0 <= percentile <= 1.0:
        raise ValueError("percentile must be in [0.0, 1.0]")
    sorted_values = values.to(torch.float32).sort().values
    rank = max(math.ceil(percentile * sorted_values.numel()) - 1, 0)
    return float(sorted_values[rank].item())


# Author: Mus mbayramo@stanford.edu
