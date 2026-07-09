"""Pure RL math for the goal-conditioned DQN+HER learner (torch-only, CPU-testable).

Two functions that carry the correctness-critical logic the legacy agent got wrong:

* :func:`q_learning_target` — the one-step bootstrap target with a *terminal* mask,
  so a terminal state contributes only its reward (no bootstrap) and a ``{0, +1}``
  success reward survives. The legacy target clipped its ceiling to ``0``, which
  erased every success, and bootstrapped through terminals (no done mask).
* :func:`relabel_future` — Hindsight Experience Replay (HER) "future" relabeling,
  drawing the substituted goal from an *achieved future next-state* rather than the
  episode's final *pre-action* state (the legacy bug), and recomputing the reward
  against this transition's own achieved next-state.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

from typing import Callable

import torch


def q_learning_target(
    reward: torch.Tensor,
    done: torch.Tensor,
    next_q: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """One-step Q-learning target with a terminal bootstrap mask.

    ``target = reward + gamma * (1 - done) * max_a' Q_target(s', a')``

    :param reward: reward tensor, shape ``[B]``.
    :param done: ``terminated`` flag, shape ``[B]`` — 1.0 at a *true* MDP terminal
        (the goal was reached). This is NOT truncation: an episode cut for a time
        or step limit must keep bootstrapping, so its ``done`` stays 0.0.
    :param next_q: target-network Q-values for the next state, shape ``[B, A]``.
        Illegal actions may be masked to ``-inf`` (the pointer-policy legality
        convention). A row that is *entirely* ``-inf`` is a dead-end next state
        with no legal action; it is treated as terminal — no bootstrap — so the
        target stays finite (``max`` over an all-``-inf`` row is ``-inf``).
    :param gamma: discount factor.
    :return: the target tensor, shape ``[B]``. At a terminal (``done=1`` or a
        dead-end next state with no legal action) the bootstrap term is masked
        out, so the target equals ``reward`` — a ``+1`` success is preserved
        instead of being clipped to ``<= 0`` as the legacy code did.
    """
    done = done.to(reward.dtype)
    max_next_q = next_q.max(dim=-1).values
    # A next state with no legal action (all actions masked to -inf) is a dead
    # end: treat it as terminal so we don't bootstrap a non-finite value into
    # the target. max_next_q is -inf exactly for those all-masked rows.
    dead_end = ~torch.isfinite(max_next_q)
    max_next_q = torch.where(dead_end, torch.zeros_like(max_next_q), max_next_q)
    done = torch.where(dead_end, torch.ones_like(done), done)
    return reward + gamma * (1.0 - done) * max_next_q


def relabel_future(
    episode: list,
    timestep: int,
    achieved_next_state: torch.Tensor,
    num_relabeled: int,
    reward_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    rng,
    *,
    goal_index: int = 3,
) -> list:
    """HER "future" relabeling for one transition.

    For the transition at ``timestep``, sample ``num_relabeled`` future indices
    ``t' in [timestep, T-1]`` and, for each, build a substituted goal from the
    *achieved next-state* of ``episode[t']`` (``episode[t'][goal_index]``), then
    recompute the reward by checking whether THIS transition's own achieved
    next-state (``achieved_next_state``) satisfies that substituted goal.

    This fixes two legacy bugs: the goal was taken from the episode's *final*
    timestep (not a sampled future one) and from its *pre-action* state (not the
    achieved next-state), so relabeled transitions almost never registered as
    successes.

    :param episode: the episode experience, a list of per-step tuples whose
        ``goal_index`` element is that step's achieved next-state.
    :param timestep: index of the transition being relabeled.
    :param achieved_next_state: ``s_{t+1}`` of this transition, compared against
        each substituted goal to recompute the reward.
    :param num_relabeled: how many future goals to sample (HER ``k``).
    :param reward_fn: ``reward_fn(state, goal) -> reward`` (``[B]``); the env's
        goal-reached reward, returning ``>= 1.0`` on a match.
    :param rng: a ``numpy`` ``Generator`` (``rng.integers(low, high)``,
        high-exclusive) used to sample future indices.
    :param goal_index: tuple position of a step's achieved next-state (default 3).
    :return: a list of ``(relabeled_goal, relabeled_reward, relabeled_done)``
        triples; ``relabeled_done`` is ``1.0`` where the substituted goal is met.
    """
    out = []
    horizon = len(episode)
    for _ in range(num_relabeled):
        future_idx = int(rng.integers(timestep, horizon))  # [timestep, T-1]
        relabeled_goal = episode[future_idx][goal_index]
        relabeled_reward = reward_fn(achieved_next_state, relabeled_goal)
        relabeled_done = (relabeled_reward >= 1.0).to(relabeled_reward.dtype)
        out.append((relabeled_goal, relabeled_reward, relabeled_done))
    return out


# Author: Mus mbayramo@stanford.edu
