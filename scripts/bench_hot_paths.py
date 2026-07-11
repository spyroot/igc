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
import time
from typing import Callable, Dict, List, Tuple

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


def print_critical_sections(corpus: str, top: int = 12) -> None:
    """cProfile the whole suite and print the top cumulative-time functions.

    :param corpus: fixture directory.
    :param top: number of functions to show.
    :return: ``None``.
    """
    profiler = cProfile.Profile()
    profiler.enable()
    run_benchmarks(corpus)
    profiler.disable()
    buffer = io.StringIO()
    pstats.Stats(profiler, stream=buffer).sort_stats("cumulative").print_stats(top)
    print("\n=== critical sections (cumulative) ===")
    print(buffer.getvalue())


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Benchmark igc hot paths over a real corpus.")
    parser.add_argument("--corpus", default=DEFAULT_CORPUS)
    parser.add_argument("--profile", action="store_true", help="also print cProfile critical sections")
    args = parser.parse_args()
    rows, info = run_benchmarks(args.corpus)
    print_report(rows, info)
    if args.profile:
        print_critical_sections(args.corpus)


if __name__ == "__main__":
    main()

# Author: Mus mbayramo@stanford.edu
