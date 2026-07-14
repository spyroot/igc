#!/usr/bin/env python3
"""
Hot-path benchmark harness: the numbers behind every performance claim.

Benchmarks the code on the RL decision path — resource-graph build, all-node neighbor
expansion (the HER-relabel access pattern), candidate-cache construction, candidate
embedding, and zero-shot ranking — over a real fixture corpus from the data-collection
submodule, then prints a timing table and the top cumulative-time functions (the critical
sections) from cProfile. Run it before AND after touching any of these modules; paste the
table into the PR (TEAM_GUIDE: "Hot-path code ships with numbers").

Usage:
    python scripts/bench_hot_paths.py                       # default corpus
    python scripts/bench_hot_paths.py --corpus idrac_ctl/tests/hpe_fixtures
    python scripts/bench_hot_paths.py --profile             # + cProfile critical sections

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

import argparse
import cProfile
import io
import json
import pstats
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from igc.ds.sources import RedfishFixtureSource, TrustLevel
from igc.ds.sources.candidate_features import build_candidate_cache
from igc.ds.sources.resource_graph import RedfishResourceGraph
from igc.modules.eval.zero_shot_ranking import embed_candidates, top_k_hit_rate

DEFAULT_CORPUS = "idrac_ctl/tests/supermicro_fixtures"


def timed(label: str, fn: Callable, results: List[Tuple[str, float]]):
    """Run ``fn`` once, record wall time under ``label``, return its value.

    :param label: row name in the printed table.
    :param fn: zero-arg callable to time.
    :param results: accumulator of ``(label, seconds)`` rows.
    :return: whatever ``fn`` returns.
    """
    start = time.perf_counter()
    value = fn()
    results.append((label, time.perf_counter() - start))
    return value


def run_benchmarks(corpus: str) -> Tuple[List[Tuple[str, float]], Dict]:
    """Execute the hot-path suite over ``corpus``.

    :param corpus: fixture directory of captured Redfish JSON.
    :return: ``(timing rows, context info)``.
    """
    rows: List[Tuple[str, float]] = []
    records = timed("load records", lambda: list(
        RedfishFixtureSource(corpus, "bench", TrustLevel.REAL)), rows)
    graph = timed("graph build (from_records)", lambda: RedfishResourceGraph.from_records(records), rows)
    neighbors = timed("neighbors, ALL nodes (HER pattern)", lambda: {
        u: graph.neighbors(u) for u in graph.nodes}, rows)
    cache = timed("candidate cache build", lambda: build_candidate_cache(graph), rows)
    candidates = list(cache.values())
    timed("embed candidates (static, once/host)", lambda: embed_candidates(candidates), rows)
    states = {r.url.rstrip("/") or r.url: json.dumps(r.response, sort_keys=True) for r in records}
    truths = {u: set(neighbors[u]) for u in graph.nodes}
    timed("zero-shot rank+score, ALL states", lambda: top_k_hit_rate(states, candidates, truths, k=5), rows)
    info = {"corpus": corpus, "records": len(records), "nodes": len(graph.nodes),
            "candidates": len(candidates)}
    return rows, info


def print_report(rows: List[Tuple[str, float]], info: Dict) -> None:
    """Print the timing table.

    :param rows: ``(label, seconds)`` rows.
    :param info: corpus context.
    :return: ``None``.
    """
    print(f"corpus={info['corpus']}  records={info['records']} "
          f"nodes={info['nodes']} candidates={info['candidates']}")
    print(f"{'stage':40} {'seconds':>10}")
    for label, seconds in rows:
        print(f"{label:40} {seconds:10.4f}")


def _profile(fn: Callable, top: int = 15) -> None:
    """cProfile ``fn`` and print the top cumulative-time functions (critical sections).

    :param fn: zero-arg callable to profile.
    :param top: number of functions to show.
    :return: ``None``.
    """
    profiler = cProfile.Profile()
    profiler.enable()
    fn()
    profiler.disable()
    buffer = io.StringIO()
    pstats.Stats(profiler, stream=buffer).sort_stats("cumulative").print_stats(top)
    print("\n=== critical sections (cumulative) ===")
    print(buffer.getvalue())


def print_critical_sections(corpus: str, top: int = 12) -> None:
    """cProfile the data-gen suite and print the critical sections.

    :param corpus: fixture directory.
    :param top: number of functions to show.
    :return: ``None``.
    """
    _profile(lambda: run_benchmarks(corpus), top=top)


def run_rl_benchmarks(batch: int = 256, cand: int = 300, h_dim: int = 768,
                      horizon: int = 50, her_k: int = 8,
                      buffer_fill: int = 5000) -> Tuple[List[Tuple[str, float]], Dict]:
    """Benchmark the RL training critical sections on CPU (no GPU, no model download).

    Covers the DQN target math, the HER future-relabel nested loop, the replay-buffer data
    feed and its per-sample done-stacking loop, and the pointer-policy forward — the loops
    and feeds that, executed O(steps) times per training epoch, dominate wall-clock if any
    is accidentally superlinear. Everything runs on CPU with synthetic tensors, so it can be
    profiled in parallel while the GPU is busy.

    :param batch: replay sample size / batch dimension B.
    :param cand: candidate count N (pointer width, next_q action dim).
    :param h_dim: backbone hidden / state dimension H.
    :param horizon: episode length T (HER nested-loop depth).
    :param her_k: HER relabels per transition (k).
    :param buffer_fill: transitions preloaded into the replay buffer.
    :return: ``(timing rows, context info)``.
    """
    import numpy as np
    import torch

    from igc.modules.igc_experience_buffer import Buffer
    from igc.modules.policy.pointer_policy import Igc_PointerQNetwork, score_candidates
    from igc.modules.rl.q_targets import q_learning_target, relabel_future

    torch.manual_seed(0)
    rows: List[Tuple[str, float]] = []

    # 1. DQN one-step target with terminal mask (fully vectorized).
    reward = torch.rand(batch)
    done = (torch.rand(batch) > 0.9).to(torch.float32)
    next_q = torch.rand(batch, cand)
    timed("q_learning_target [B,N]", lambda: q_learning_target(reward, done, next_q, 0.99), rows)

    # 2. HER future relabeling over a full episode (nested loop: T timesteps x k relabels).
    goal_dim = h_dim
    episode = [(None, None, torch.rand(1), torch.rand(1, goal_dim), None) for _ in range(horizon)]
    achieved = torch.rand(1, goal_dim)
    rng = np.random.default_rng(0)

    def reward_fn(state, goal):
        return torch.all(state == goal, dim=1).to(torch.float32)

    def her_episode():
        out = []
        for step in range(horizon):
            out.extend(relabel_future(episode, step, achieved, her_k, reward_fn, rng))
        return out

    timed(f"HER relabel, full episode (T={horizon} x k={her_k})", her_episode, rows)

    # 3. Replay buffer: fill (setup, untimed) then time the data-feed sample + the done loop.
    buf = Buffer(size=buffer_fill * 2, sample_size=batch)
    for i in range(buffer_fill):
        done_val = torch.rand(1) if i % 3 == 0 else False   # exercise the _stack_done coercion
        buf.add(torch.rand(h_dim), torch.rand(cand), torch.rand(1), torch.rand(h_dim), done_val)
    timed("replay sample_batch (data feed)", buf.sample_batch, rows)
    samples = list(buf._buffer)[:batch]
    reward_batch = torch.stack([s[2] for s in samples], dim=0)
    timed("  _stack_done loop (isolated)", lambda: Buffer._stack_done(samples, reward_batch), rows)

    # 4. Fixed-width Q-network forward (the current agent path: state+goal cat -> num_actions).
    from igc.modules.igc_q_network import Igc_QNetwork
    qnet = Igc_QNetwork(input_dim=2 * h_dim, num_actions=cand)
    qnet.eval()
    qnet_input = torch.rand(batch, 2 * h_dim)
    with torch.no_grad():
        timed("fixed Q-network forward [B,2H]", lambda: qnet(qnet_input), rows)

    # 5. Pointer-policy forward on CPU (query + key projection + einsum scoring).
    policy = Igc_PointerQNetwork(h_dim=h_dim, q_dim=256)
    policy.eval()
    state_h = torch.rand(batch, h_dim)
    goal_h = torch.rand(batch, h_dim)
    cand_h = torch.rand(batch, cand, h_dim)
    mask = (torch.rand(batch, cand) > 0.3).to(torch.float32)
    with torch.no_grad():
        timed("pointer forward, re-project B*N (naive)", lambda: policy(state_h, goal_h, cand_h, mask), rows)
        # D-002 caching: a host's N candidates are static, so project the UNIQUE set once
        # per optimizer step and reuse across the whole batch (keys expand is a view, no copy).
        unique_cand = torch.rand(cand, h_dim)

        def cached_forward():
            query = policy.state_query(state_h, goal_h)
            keys = policy.action_proj(unique_cand.unsqueeze(0))          # project N once, not B*N
            return score_candidates(query, keys.expand(batch, -1, -1), mask)

        timed("pointer forward, project-N-once (D-002 cache)", cached_forward, rows)
        query = torch.rand(batch, 256)
        keys = torch.rand(batch, cand, 256)
        timed("  score_candidates einsum (isolated)", lambda: score_candidates(query, keys, mask), rows)

    info = {"corpus": f"synthetic RL  B={batch} N={cand} H={h_dim} T={horizon} k={her_k}",
            "records": buffer_fill, "nodes": horizon * her_k, "candidates": cand}
    return rows, info


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Benchmark igc hot paths over a real corpus.")
    parser.add_argument("--corpus", default=DEFAULT_CORPUS)
    parser.add_argument("--profile", action="store_true", help="also print cProfile critical sections")
    parser.add_argument("--section", choices=("data", "rl", "all"), default="all",
                        help="data-gen path, RL training path, or both (default)")
    args = parser.parse_args()
    if args.section in ("data", "all"):
        rows, info = run_benchmarks(args.corpus)
        print_report(rows, info)
        if args.profile:
            print_critical_sections(args.corpus)
    if args.section in ("rl", "all"):
        rl_rows, rl_info = run_rl_benchmarks()
        print()
        print_report(rl_rows, rl_info)
        if args.profile:
            _profile(lambda: run_rl_benchmarks())


if __name__ == "__main__":
    main()

# Author: Mus mbayramo@stanford.edu
