"""Offline tests for RL/HER metric namespaces and summary math.

The metric helpers feed Weight & Biases panels for M6. These tests pin the
public names separately from the trainer so dashboards do not drift back to
mixed original/HER curves.

Author:
Mus mbayramo@stanford.edu
"""

import torch

from igc.modules.rl.metrics import (
    DQN_UPDATE_METRIC_KEYS,
    EVAL_ORIGINAL_GOALS_METRIC_KEYS,
    HER_METRIC_KEYS,
    ORIGIN_HER,
    ORIGIN_ORIGINAL,
    summarize_dqn_update_metrics,
    summarize_her_metrics,
)


def test_metric_key_contract_keeps_m6_rl_namespaces_separate():
    """M6 dashboards use explicit RL/HER/DQN/eval prefixes."""
    assert HER_METRIC_KEYS == (
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
    assert DQN_UPDATE_METRIC_KEYS == (
        "03_m6_dqn_update/loss_td_original_mean_per_transition",
        "03_m6_dqn_update/loss_td_her_mean_per_transition",
        "03_m6_dqn_update/td_error_abs_original_p90_reward",
        "03_m6_dqn_update/td_error_abs_her_p90_reward",
    )
    assert EVAL_ORIGINAL_GOALS_METRIC_KEYS == (
        "03_m6_eval_original_goals/success_ratio",
        "03_m6_eval_original_goals/return_undiscounted_mean_reward",
    )

def test_summarize_her_metrics_counts_original_and_relabelled_samples():
    """HER summaries split original rewards from relabelled rewards and goals."""
    metrics = summarize_her_metrics(
        original_rewards=torch.tensor([1.0, 0.0]),
        original_done=torch.tensor([1.0, 0.0]),
        relabelled_rewards=torch.tensor([1.0, 0.0, 1.0]),
        relabelled_done=torch.tensor([1.0, 0.0, 1.0]),
        future_offsets_env_steps=torch.tensor([0.0, 2.0, 4.0]),
        relabelled_goals=[
            torch.tensor([[1.0, 2.0]]),
            torch.tensor([[1.0, 2.0]]),
            torch.tensor([[3.0, 4.0]]),
        ],
        invalid_relabelled_goal_count=1,
        reward_recompute_error_count=2,
        goal_termination_recompute_error_count=3,
    )

    assert metrics["03_m6_her/relabelled_transition_ratio"] == 3 / 5
    assert metrics["03_m6_her/relabelled_success_ratio"] == 2 / 3
    assert metrics["03_m6_her/original_success_ratio"] == 1 / 2
    assert metrics["03_m6_her/positive_reward_ratio_original"] == 1 / 2
    assert metrics["03_m6_her/positive_reward_ratio_relabelled"] == 2 / 3
    assert metrics["03_m6_her/future_goal_offset_mean_env_steps"] == 2.0
    assert metrics["03_m6_her/future_goal_offset_p95_env_steps"] == 4.0
    assert metrics["03_m6_her/unique_relabelled_goal_count"] == 2.0
    assert metrics["03_m6_her/invalid_relabelled_goal_ratio"] == 1 / 3
    assert metrics["03_m6_her/duplicate_goal_ratio"] == 1 / 3
    assert metrics["03_m6_her/reward_recompute_error_count"] == 2.0
    assert metrics["03_m6_her/goal_termination_recompute_error_count"] == 3.0


def test_summarize_dqn_update_metrics_splits_original_and_her_td_loss():
    """TD loss and TD-error panels are not mixed across replay origins."""
    metrics = summarize_dqn_update_metrics(
        td_errors=torch.tensor([1.0, -2.0, 4.0]),
        td_losses=torch.tensor([1.0, 4.0, 16.0]),
        origins=[ORIGIN_ORIGINAL, ORIGIN_HER, ORIGIN_HER],
    )

    assert metrics["03_m6_dqn_update/loss_td_original_mean_per_transition"] == 1.0
    assert metrics["03_m6_dqn_update/loss_td_her_mean_per_transition"] == 10.0
    assert metrics["03_m6_dqn_update/td_error_abs_original_p90_reward"] == 1.0
    assert metrics["03_m6_dqn_update/td_error_abs_her_p90_reward"] == 4.0


# Author: Mus mbayramo@stanford.edu
