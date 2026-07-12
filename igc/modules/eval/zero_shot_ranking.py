"""
D-001/D-002 zero-shot go/no-go representation check.

Names, for a human: "D-001" (Decision 001 in ``docs/DECISIONS.md``) is the plan to SELECT an
action by ranking candidates with a pointer network and taking the top one; "D-002" is how
those candidates are REPRESENTED (text + graph features). This module is the cheap feasibility
test for both — with NO training, can a plain text-similarity ranker already put a resource's
true next-hops in the top-5? If yes, the learned ranker is worth building.

The problem it solves: before spending GPU time training a candidate ranker, prove the idea is sound
cheaply. If an untrained similarity ranker already puts the true next-hop in the top-5 most of the
time, the representation is good enough and the learned ranker is worth building; if not, fix the
representation first. It is a go/no-go gate run offline, not a runtime component the agent uses.

Deterministic zero-shot ranking of legal action candidates with a frozen character-trigram
text encoder: for each resource (state) in a walked Redfish tree, all host candidates are
ranked by similarity to the state text, and the module measures whether the state's TRUE
graph neighbors (the walked tree's real transitions) appear in the top-k. No training, no
torch, no randomness — numpy and the standard library only. Go/no-go per D-001: top-5 hit
rate >= 0.80 on a held-out vendor.

Not wired into any training or eval runtime: consumed only by ``scripts/bench_hot_paths.py``
(the embed + rank benchmark stages) and its unit/perf tests. This is a standalone one-off
feasibility gate whose >= 0.80 top-5 number decides whether a zero-cost frozen encoder can
recover true Redfish transitions before the D-001/D-002 ranking head is built.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

import hashlib
from typing import Dict, List, Set

import numpy as np


def trigram_embed(text: str, dim: int = 512) -> np.ndarray:
    """Compute a deterministic hashed character-trigram embedding.

    The text is lowercased and padded with a leading and trailing space. Every overlapping
    3-character window is hashed with blake2b (8-byte digest) and mapped to a bucket index
    modulo ``dim``; bucket counts are accumulated in a float64 vector, then L2-normalized.
    An all-zero vector stays zero. Stable across processes (no builtin ``hash()``).

    :param text: input string to embed.
    :param dim: embedding dimensionality.
    :return: L2-normalized float64 vector of shape ``(dim,)``.
    """
    processed = f" {text.lower()} "
    vec = np.zeros(dim, dtype=np.float64)
    if len(processed) < 3:
        return vec
    for i in range(len(processed) - 2):
        trigram = processed[i:i + 3]
        digest = hashlib.blake2b(trigram.encode("utf-8"), digest_size=8).digest()
        vec[int.from_bytes(digest, "big") % dim] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0.0:
        vec /= norm
    return vec


def candidate_text(candidate: Dict) -> str:
    """Render a candidate dict into its canonical text form for embedding.

    Joins the endpoint path tokens with ``/``, then appends the HTTP method, resource type,
    child relation name, and the literal ``action`` when the candidate exposes an action
    target — single spaces between parts.

    :param candidate: D-002 v1 candidate dict (see ``candidate_features.candidate_v1``).
    :return: canonical text representation.
    """
    parts: List[str] = [
        "/".join(candidate["endpoint_path_tokens"]),
        candidate["http_method"],
        candidate["resource_type"],
        candidate["child_relation_name"],
    ]
    if candidate.get("has_action_target", False):
        parts.append("action")
    return " ".join(parts)


def embed_candidates(candidates: List[Dict], dim: int = 512) -> np.ndarray:
    """Embed every candidate once (static per host — the D-002 caching contract).

    :param candidates: host candidate dicts.
    :param dim: embedding dimensionality.
    :return: float64 matrix of shape ``(len(candidates), dim)``.
    """
    if not candidates:
        return np.zeros((0, dim), dtype=np.float64)
    return np.stack([trigram_embed(candidate_text(c), dim=dim) for c in candidates])


def rank_candidates(state_text: str, candidates: List[Dict], dim: int = 512) -> List[int]:
    """Rank candidate indices by cosine similarity to the state text, descending.

    Embeddings are L2-normalized, so cosine equals the dot product. Ties break stably by
    ascending index.

    :param state_text: text representation of the current state.
    :param candidates: candidate dicts.
    :param dim: embedding dimensionality.
    :return: candidate indices, best first.
    """
    scores = embed_candidates(candidates, dim=dim) @ trigram_embed(state_text, dim=dim)
    return sorted(range(len(candidates)), key=lambda i: (-scores[i], i))


def top_k_hit_rate(states: Dict[str, str], candidates: List[Dict],
                   truths: Dict[str, Set[str]], k: int = 5, dim: int = 512) -> Dict:
    """Measure how often a state's true transitions rank in the top-k of all candidates.

    Candidate embeddings are computed ONCE and reused across states (the same static-per-host
    property the runtime cache relies on); a state's own endpoint is excluded from its
    ranking by score masking. A state counts as a hit when any top-k candidate URL is in its
    truth set; states with empty truth sets are skipped.

    :param states: ``{url: state_text}`` per resource.
    :param candidates: the full host candidate list.
    :param truths: ``{url: set of true next-endpoint urls}`` (graph neighbors).
    :param k: cutoff.
    :param dim: embedding dimensionality.
    :return: ``{"evaluated", "hits", "hit_rate", "k"}``.
    """
    embeddings = embed_candidates(candidates, dim=dim)
    urls = [c.get("url") for c in candidates]
    evaluated = 0
    hits = 0
    for state_url, state_text in states.items():
        truth_set = truths.get(state_url) or set()
        if not truth_set or embeddings.shape[0] == 0:
            continue
        scores = embeddings @ trigram_embed(state_text, dim=dim)
        for i, url in enumerate(urls):
            if url == state_url:
                scores[i] = -np.inf
        order = sorted(range(len(urls)), key=lambda i: (-scores[i], i))[:k]
        evaluated += 1
        if any(urls[i] in truth_set for i in order):
            hits += 1
    return {
        "evaluated": evaluated,
        "hits": hits,
        "hit_rate": hits / evaluated if evaluated else 0.0,
        "k": k,
    }


# Author: Mus mbayramo@stanford.edu
