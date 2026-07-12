"""Offline CPU tests for the pointer / candidate-scoring policy head.

The key property is that the head's output width tracks the number of candidate
actions (``N``), never a global ``num_actions`` — this is the structural fix that
keeps the action space from exploding. Also checks dot-product scoring, masking,
greedy selection, and L2-normalized heads. Small tensors, CPU only.

Author:
Mus mbayramo@stanford.edu
"""
import torch
import pytest

from igc.modules.policy.pointer_policy import (
    ActionProjector,
    Igc_PointerQNetwork,
    StateQueryHead,
    greedy_select,
    score_candidates,
)


def test_score_picks_aligned_candidate():
    """The candidate whose key aligns with the query gets the highest score."""
    query = torch.tensor([[1.0, 0.0, 0.0]])
    keys = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.0, 0.0]]])
    scores = score_candidates(query, keys)
    assert scores.shape == (1, 3)
    assert torch.allclose(scores, torch.tensor([[1.0, 0.0, 0.5]]))
    assert int(greedy_select(scores)[0]) == 0


def test_mask_excludes_padding_even_if_highest():
    """A masked (padding) candidate is set to -inf and never selected."""
    query = torch.tensor([[1.0, 0.0]])
    keys = torch.tensor([[[0.1, 0.0], [0.0, 0.0], [5.0, 0.0]]])  # idx 2 has the top raw score
    mask = torch.tensor([[1, 1, 0]])  # but idx 2 is padding
    scores = score_candidates(query, keys, mask)
    assert scores[0, 2] == float("-inf")
    assert int(greedy_select(scores)[0]) == 0


@pytest.mark.xfail(
    reason="greedy_select currently argmaxes an all-masked row to candidate 0.",
    strict=True,
)
def test_greedy_select_rejects_all_masked_rows():
    """A row with no legal candidates must not emit a concrete candidate index."""
    scores = torch.tensor([[float("-inf"), float("-inf")]])

    with pytest.raises(ValueError, match="no legal candidates"):
        greedy_select(scores)


def test_output_width_tracks_num_candidates_not_global():
    """forward returns [B, N] for any N — the head never grows with a global count."""
    net = Igc_PointerQNetwork(h_dim=8, q_dim=4)
    b = 2
    for n in (1, 3, 64):
        state_h = torch.randn(b, 8)
        goal_h = torch.randn(b, 8)
        cand_h = torch.randn(b, n, 8)
        out = net(state_h, goal_h, cand_h)
        assert out.shape == (b, n)  # width = N, independent of any catalog size


def test_forward_respects_mask():
    """Padded candidates are -inf in the network's scores."""
    net = Igc_PointerQNetwork(h_dim=8, q_dim=4)
    state_h = torch.randn(2, 8)
    goal_h = torch.randn(2, 8)
    cand_h = torch.randn(2, 5, 8)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]])
    out = net(state_h, goal_h, cand_h, mask)
    assert torch.isinf(out[mask == 0]).all() and (out[mask == 0] < 0).all()
    assert torch.isfinite(out[mask == 1]).all()


def test_empty_candidate_set_is_well_shaped():
    """N=0 produces a [B, 0] score tensor without error."""
    net = Igc_PointerQNetwork(h_dim=8, q_dim=4)
    out = net(torch.randn(3, 8), torch.randn(3, 8), torch.randn(3, 0, 8))
    assert out.shape == (3, 0)


def test_heads_are_l2_normalized():
    """StateQueryHead and ActionProjector emit unit-norm queries/keys."""
    torch.manual_seed(0)
    q_head = StateQueryHead(h_dim=8, q_dim=16)
    a_proj = ActionProjector(h_dim=8, q_dim=16)
    q = q_head(torch.randn(4, 8), torch.randn(4, 8))
    k = a_proj(torch.randn(4, 7, 8))
    assert torch.allclose(q.norm(dim=-1), torch.ones(4), atol=1e-5)
    assert torch.allclose(k.norm(dim=-1), torch.ones(4, 7), atol=1e-5)


# Author: Mus mbayramo@stanford.edu
