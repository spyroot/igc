"""
RL training-path performance budgets (regression tripwires, `-m perf`).

Guards the CPU-offline RL critical sections — DQN target, HER relabel loop, replay data
feed, per-sample done stacking, candidate scoring — with absolute ceilings ~50-100x looser
than measured, so only an algorithmic regression trips them. Plus a machine-INDEPENDENT
ratio guard pinning the D-002 static-per-host key cache: projecting the unique candidate set
once must stay far cheaper than re-projecting B*N (measured 51x). Pure synthetic tensors on
CPU — no corpus, no GPU, no model download; safe to run in parallel while the GPU is busy.

Author:
Mus mbayramo@stanford.edu
"""

import time

import numpy as np
import pytest
import torch

from igc.modules.igc_experience_buffer import Buffer
from igc.modules.policy.pointer_policy import Igc_PointerQNetwork, score_candidates
from igc.modules.rl.q_targets import q_learning_target, relabel_future

pytestmark = pytest.mark.perf

_B, _N, _H, _T, _K = 256, 300, 768, 50, 8


def _seconds(fn):
    start = time.perf_counter()
    fn()
    return time.perf_counter() - start


def test_q_learning_target_budget():
    """The DQN target is vectorized: [B, N] batch well under 0.05s (measured ~0.0003s)."""
    reward, done = torch.rand(_B), (torch.rand(_B) > 0.9).float()
    next_q = torch.rand(_B, _N)
    assert _seconds(lambda: q_learning_target(reward, done, next_q, 0.99)) < 0.05


def test_her_full_episode_budget():
    """HER over a full episode (T*k relabels) stays a bounded loop: < 0.5s (measured ~0.004s)."""
    torch.manual_seed(0)
    episode = [(None, None, torch.rand(1), torch.rand(1, _H), None) for _ in range(_T)]
    achieved = torch.rand(1, _H)
    rng = np.random.default_rng(0)

    def reward_fn(state, goal):
        return torch.all(state == goal, dim=1).float()

    def run():
        for step in range(_T):
            relabel_future(episode, step, achieved, _K, reward_fn, rng)

    assert _seconds(run) < 0.5


def test_replay_data_feed_budget():
    """Replay sample + done-stacking (the per-step data feed) < 0.2s (measured ~0.0015s)."""
    buf = Buffer(size=10_000, sample_size=_B)
    for i in range(2_000):
        done_val = torch.rand(1) if i % 3 == 0 else False
        buf.add(torch.rand(_H), torch.rand(_N), torch.rand(1), torch.rand(_H), done_val)
    assert _seconds(buf.sample_batch) < 0.2


def test_score_candidates_einsum_budget():
    """The candidate-scoring einsum [B,N,d] < 0.3s (measured ~0.003s)."""
    query, keys = torch.rand(_B, 256), torch.rand(_B, _N, 256)
    mask = (torch.rand(_B, _N) > 0.3).float()
    assert _seconds(lambda: score_candidates(query, keys, mask)) < 0.3


def test_d002_key_cache_beats_reprojection():
    """D-002 property: projecting N unique candidates once is >= 10x cheaper than B*N.

    Machine-independent ratio (measured ~51x): guards against a training loop that calls the
    full pointer forward per step (re-projecting duplicated candidates) instead of caching the
    host's static keys and scoring with a per-step einsum.
    """
    torch.manual_seed(0)
    policy = Igc_PointerQNetwork(h_dim=_H, q_dim=256)
    policy.eval()
    state_h, goal_h = torch.rand(_B, _H), torch.rand(_B, _H)
    cand_h = torch.rand(_B, _N, _H)
    unique_cand = torch.rand(_N, _H)
    mask = (torch.rand(_B, _N) > 0.3).float()

    with torch.no_grad():
        naive = _seconds(lambda: policy(state_h, goal_h, cand_h, mask))

        def cached():
            query = policy.state_query(state_h, goal_h)
            keys = policy.action_proj(unique_cand.unsqueeze(0))
            return score_candidates(query, keys.expand(_B, -1, -1), mask)

        cached_t = _seconds(cached)

    assert naive / max(cached_t, 1e-6) >= 10.0


# Author: Mus mbayramo@stanford.edu
