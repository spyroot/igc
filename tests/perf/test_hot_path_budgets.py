"""
Hot-path performance budgets (regression tripwires, `-m perf`).

Each budget is ~50-100x looser than the measured number on the reference corpus, so a pass
never depends on machine speed — only an algorithmic regression (e.g. the O(V^2) neighbor
scan this suite was born from: 0.71s measured, budget 0.5s per 1,000 nodes) trips it.
Excluded from the default offline gate; run explicitly with `pytest -m perf` (the fixture
corpus is local to this checkout, so a minimal checkout without fixtures skips).

Author:
Mus mbayramo@stanford.edu
"""

import json
import os
import time

import pytest

from igc.ds.sources import RedfishFixtureSource, TrustLevel
from igc.ds.sources.candidate_features import build_candidate_cache
from igc.ds.sources.resource_graph import RedfishResourceGraph
from igc.modules.eval.zero_shot_ranking import embed_candidates

_CORPUS = "tests/supermicro_fixtures"

pytestmark = [
    pytest.mark.perf,
    pytest.mark.skipif(not os.path.isdir(_CORPUS),
                       reason="fixture corpus not present"),
]


@pytest.fixture(scope="module")
def corpus():
    """Records + graph for the reference corpus, built once per module."""
    records = list(RedfishFixtureSource(_CORPUS, "perf", TrustLevel.REAL))
    return records, RedfishResourceGraph.from_records(records)


def _seconds(fn):
    start = time.perf_counter()
    fn()
    return time.perf_counter() - start


def test_graph_build_budget(corpus):
    """Graph build stays linear-ish: <= 2s per 1,000 nodes (measured ~0.03s/1,499)."""
    records, graph = corpus
    budget = 2.0 * max(1, len(graph.nodes)) / 1000
    assert _seconds(lambda: RedfishResourceGraph.from_records(records)) < budget


def test_neighbors_all_nodes_budget(corpus):
    """The HER access pattern stays indexed: <= 0.5s per 1,000 nodes (O(V^2) was 0.71s)."""
    _, graph = corpus
    budget = 0.5 * max(1, len(graph.nodes)) / 1000
    assert _seconds(lambda: {u: graph.neighbors(u) for u in graph.nodes}) < budget


def test_candidate_cache_budget(corpus):
    """Cache build is a linear pass: <= 1s per 1,000 nodes."""
    _, graph = corpus
    budget = 1.0 * max(1, len(graph.nodes)) / 1000
    assert _seconds(lambda: build_candidate_cache(graph)) < budget


def test_candidate_embedding_budget(corpus):
    """Static per-host embedding: <= 20s per 1,000 candidates (measured ~4s/1,499)."""
    _, graph = corpus
    candidates = list(build_candidate_cache(graph).values())
    budget = 20.0 * max(1, len(candidates)) / 1000
    assert _seconds(lambda: embed_candidates(candidates)) < budget


def test_state_text_render_budget(corpus):
    """Rendering every state body to text stays cheap: <= 2s per 1,000 records."""
    records, _ = corpus
    budget = 2.0 * max(1, len(records)) / 1000
    assert _seconds(lambda: [json.dumps(r.response, sort_keys=True) for r in records]) < budget


# Author: Mus mbayramo@stanford.edu
