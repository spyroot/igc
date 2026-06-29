"""Pointer / candidate-scoring policy head for the goal-conditioned tool-use agent.

This replaces the fixed-width Q-head (`igc.modules.igc_q_network.Igc_QNetwork`, whose
output dimension is ``num_actions`` and therefore grows with the discovered-URL count)
with a head whose output width is the number of *legal candidate actions* for the
current state — exactly the list ``GoalEnvironment.available_actions(obs)`` returns.

The state and goal are projected to a query ``q``; each candidate-action embedding is
projected to a key ``k_i``; the score is the dot product ``Q(s, a_i) = q . k_i``. The
output dimension is ``N`` (the local fan-out, tens), never the global catalog, so
adding tools / URLs / methods never widens the head and a brand-new tool is scorable
without an output-layer resize. Queries and keys are L2-normalized, so the dot product
is a cosine similarity.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def score_candidates(
    query: torch.Tensor, keys: torch.Tensor, mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Dot-product score of a query against candidate keys, padding masked to ``-inf``.

    :param query: float tensor ``[B, d]``.
    :param keys: float tensor ``[B, N, d]`` of candidate keys.
    :param mask: optional ``[B, N]`` tensor, 1 for real candidates and 0 for padding;
        padded positions are set to ``-inf`` so they are never selected.
    :return: ``[B, N]`` scores. The width is ``N`` — the number of candidates — not the
        global action count.
    """
    scores = torch.einsum("bd,bnd->bn", query, keys)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    return scores


def greedy_select(scores: torch.Tensor) -> torch.Tensor:
    """Greedy candidate index per batch element.

    :param scores: ``[B, N]`` candidate scores (padding already ``-inf``).
    :return: ``[B]`` index of the highest-scoring legal candidate.
    """
    return scores.argmax(dim=-1)


class StateQueryHead(nn.Module):
    """Project a pooled state and goal to an L2-normalized query vector.

    :param h_dim: backbone hidden size ``H``.
    :param q_dim: query/key dimension ``d``.
    """

    def __init__(self, h_dim: int, q_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * h_dim, 512), nn.GELU(), nn.LayerNorm(512), nn.Linear(512, q_dim)
        )

    def forward(self, state_h: torch.Tensor, goal_h: torch.Tensor) -> torch.Tensor:
        """:param state_h: ``[B, H]`` pooled state.
        :param goal_h: ``[B, H]`` pooled goal.
        :return: ``[B, d]`` L2-normalized query.
        """
        x = torch.cat((state_h, goal_h), dim=-1)
        return F.normalize(self.net(x), dim=-1)


class ActionProjector(nn.Module):
    """Project candidate-action embeddings to L2-normalized keys.

    :param h_dim: backbone hidden size ``H``.
    :param q_dim: query/key dimension ``d``.
    """

    def __init__(self, h_dim: int, q_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(h_dim, 512), nn.GELU(), nn.LayerNorm(512), nn.Linear(512, q_dim)
        )

    def forward(self, cand_h: torch.Tensor) -> torch.Tensor:
        """:param cand_h: ``[B, N, H]`` candidate-action embeddings.
        :return: ``[B, N, d]`` L2-normalized keys.
        """
        return F.normalize(self.net(cand_h), dim=-1)


class Igc_PointerQNetwork(nn.Module):
    """Candidate-scoring Q-head: output width = number of legal candidates, not num_actions.

    :param h_dim: backbone hidden size ``H``.
    :param q_dim: query/key dimension ``d``.
    """

    def __init__(self, h_dim: int, q_dim: int = 256):
        super().__init__()
        self.state_query = StateQueryHead(h_dim, q_dim)
        self.action_proj = ActionProjector(h_dim, q_dim)

    def forward(
        self,
        state_h: torch.Tensor,
        goal_h: torch.Tensor,
        cand_h: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Score each candidate action for each batch element.

        :param state_h: ``[B, H]`` pooled state.
        :param goal_h: ``[B, H]`` pooled goal.
        :param cand_h: ``[B, N, H]`` candidate-action embeddings.
        :param mask: optional ``[B, N]`` legality mask (1 real, 0 padding).
        :return: ``[B, N]`` candidate Q-scores; padded positions are ``-inf``. The width
            tracks ``N``, so the head never grows with the global action count.
        """
        query = self.state_query(state_h, goal_h)
        keys = self.action_proj(cand_h)
        return score_candidates(query, keys, mask)


# Author: Mus mbayramo@stanford.edu
