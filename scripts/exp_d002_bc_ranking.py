"""
D-002 go/no-go experiment: does a learned bilinear projection recover Redfish transitions?

The frozen-encoder baseline (``igc/modules/eval/zero_shot_ranking.py``) scores a (state,
candidate) pair by the cosine of their character-trigram embeddings and was recorded NO-GO in
DECISIONS.md D-002 (top-5 0.293 in-domain Supermicro, 0.754 held-out HPE, bar 0.80). This
script tests the recorded next step: behavioural-clone a bilinear projection ``W`` on in-domain
graph transitions (Supermicro) so the score becomes ``s^T W c``, then re-run the SAME top-5
hit-rate zero-shot on the held-out vendor (HPE). The embeddings stay frozen; only ``W`` is
trained. Positives are a state's true graph neighbours; the loss is multi-positive listwise
softmax cross-entropy over the host candidate set with the state's own endpoint masked out.

Run (offline, CPU): ``python scripts/exp_d002_bc_ranking.py``

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

import argparse
import json
from typing import Dict, List

import numpy as np
import torch

from igc.ds.sources import RedfishFixtureSource, TrustLevel
from igc.ds.sources.candidate_features import build_candidate_cache
from igc.ds.sources.resource_graph import RedfishResourceGraph
from igc.modules.eval.zero_shot_ranking import candidate_text, trigram_embed

SUPERMICRO = "idrac_ctl/tests/supermicro_fixtures"
HPE = "idrac_ctl/tests/hpe_fixtures"

# Non-operational metadata: the Redfish JSON-schema store, message/attribute registries, and metric
# DEFINITIONS. An agent never navigates to these; a full "entire dump" walk is dominated by them and
# they distort the ranking metric (they are the near-identical sibling leaves that fill top-k).
_META_URL = ("/jsonschemas/", "/schemas/", "/schemastore/", "/registries/",
             "/metricdefinitions/", "/messageregistry")
_META_TYPE = {"jsonschemafile", "metricdefinition", "metricreportdefinition",
              "messageregistryfile", "attributeregistry", "schemafile"}


def _is_metadata(rec) -> bool:
    """True for schema/registry/metric-definition resources (not part of the action space)."""
    if any(seg in rec.url.lower() for seg in _META_URL):
        return True
    body = rec.response if isinstance(rec.response, dict) else {}
    t = str(body.get("@odata.type", "")).split(".")[0].lstrip("#").lower()
    return t in _META_TYPE


def load_corpus(corpus_dir: str, dim: int = 512, filter_metadata: bool = False) -> Dict:
    """Load a fixture corpus into frozen state/candidate embeddings + truth masks.

    :param corpus_dir: fixture directory of captured Redfish JSON.
    :param dim: trigram embedding dimensionality.
    :param filter_metadata: drop the schema/registry/metric-definition dump before building the graph.
    :return: dict with S (states x dim), C (candidates x dim), and boolean pos/self masks.
    """
    records = list(RedfishFixtureSource(corpus_dir, "exp", TrustLevel.REAL))
    if filter_metadata:
        records = [r for r in records if not _is_metadata(r)]
    graph = RedfishResourceGraph.from_records(records)
    candidates = list(build_candidate_cache(graph).values())
    cand_urls = [c["url"] for c in candidates]
    C = np.stack([trigram_embed(candidate_text(c), dim=dim) for c in candidates])

    state_text = {(r.url.rstrip("/") or r.url): json.dumps(r.response, sort_keys=True) for r in records}
    truths = {u: set(graph.neighbors(u)) for u in graph.nodes}

    # keep states that have at least one true neighbour present in the candidate set
    cand_url_set = set(cand_urls)
    state_urls: List[str] = [
        u for u, t in truths.items()
        if u in state_text and t and (t & cand_url_set)
    ]
    S = np.stack([trigram_embed(state_text[u], dim=dim) for u in state_urls])

    n_state, n_cand = len(state_urls), len(cand_urls)
    pos = np.zeros((n_state, n_cand), dtype=bool)
    selfm = np.zeros((n_state, n_cand), dtype=bool)
    for i, u in enumerate(state_urls):
        t = truths[u]
        for j, cu in enumerate(cand_urls):
            if cu in t:
                pos[i, j] = True
            if cu == u:
                selfm[i, j] = True
    return {
        "S": torch.tensor(S, dtype=torch.float32),
        "C": torch.tensor(C, dtype=torch.float32),
        "pos": torch.tensor(pos),
        "self": torch.tensor(selfm),
        "n_state": n_state, "n_cand": n_cand,
    }


def logits(data: Dict, W: torch.Tensor) -> torch.Tensor:
    """Bilinear score matrix ``S W C^T`` with the state's own endpoint masked to -inf."""
    z = data["S"] @ W @ data["C"].t()
    return z.masked_fill(data["self"], float("-inf"))


def top_k_hit_rate(data: Dict, W: torch.Tensor, k: int = 5) -> float:
    """Fraction of states whose true neighbours appear in the top-k of the ranked candidates."""
    z = logits(data, W)
    topk = z.topk(k, dim=1).indices
    hit = data["pos"].gather(1, topk).any(dim=1)
    return hit.float().mean().item()


def host_ce(data: Dict, W: torch.Tensor) -> torch.Tensor:
    """Multi-positive listwise softmax cross-entropy for one host's (states, candidates)."""
    z = logits(data, W)
    denom = torch.logsumexp(z, dim=1)                                  # all non-self candidates
    num = torch.logsumexp(z.masked_fill(~data["pos"], float("-inf")), dim=1)  # positives
    return -(num - denom).mean()


def train_W(train_hosts: List[Dict], steps: int, lr: float, l2i: float, seed: int = 0) -> torch.Tensor:
    """BC-train a bilinear projection shared across hosts, anchored near identity.

    :param train_hosts: one data dict per training host (candidate sets are per host).
    :param l2i: strength of the ``||W - I||_F^2`` anchor keeping ``W`` a small perturbation of
        cosine (regularizing toward the identity baseline, not toward zero).
    """
    torch.manual_seed(seed)
    dim = train_hosts[0]["S"].shape[1]
    eye = torch.eye(dim)
    W = torch.eye(dim, requires_grad=True)  # start at identity == the cosine baseline
    opt = torch.optim.Adam([W], lr=lr)
    for step in range(steps):
        opt.zero_grad()
        loss = sum(host_ce(h, W) for h in train_hosts) / len(train_hosts)
        loss = loss + l2i * ((W - eye) ** 2).sum()
        loss.backward()
        opt.step()
        if step % max(1, steps // 4) == 0 or step == steps - 1:
            with torch.no_grad():
                hr = sum(top_k_hit_rate(h, W) for h in train_hosts) / len(train_hosts)
            print(f"  step {step:4d}  loss={loss.item():.4f}  train top-5(avg)={hr:.3f}")
    return W.detach()


DELL_FIXT = "idrac_ctl/tests/idrac_fixtures"
DELL_FIXT_NOTE = "point --holdout at a real full walk under datasets/orig/<host> for a large held-out host"
GENERIC = "idrac_ctl/tests/generic_fixtures"

DEFAULT_TRAIN = ",".join([SUPERMICRO, DELL_FIXT, GENERIC])
DEFAULT_HOLDOUT = HPE                             # public fixture; pass --holdout <dir> for a large real walk


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=str, default=DEFAULT_TRAIN, help="comma-separated train corpora")
    ap.add_argument("--holdout", type=str, default=DEFAULT_HOLDOUT, help="comma-separated held-out corpora")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--l2i", type=float, default=0.1, help="||W-I||^2 anchor (0 = free-form W)")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--filter", action="store_true",
                    help="drop the schema/registry/metric-definition metadata dump (operational graph only)")
    args = ap.parse_args()
    bar, k = 0.80, args.k

    train_dirs = [d for d in args.train.split(",") if d]
    holdout_dirs = [d for d in args.holdout.split(",") if d]

    print(f"loading corpora (frozen trigram embeddings; filter_metadata={args.filter})...")
    train = [load_corpus(d, filter_metadata=args.filter) for d in train_dirs]
    holdout = [load_corpus(d, filter_metadata=args.filter) for d in holdout_dirs]
    for d, h in zip(train_dirs, train):
        print(f"  TRAIN   {d:45s} states={h['n_state']:5d} candidates={h['n_cand']}")
    for d, h in zip(holdout_dirs, holdout):
        print(f"  HOLDOUT {d:45s} states={h['n_state']:5d} candidates={h['n_cand']}")

    dim = holdout[0]["S"].shape[1]
    W_id = torch.eye(dim)
    print(f"\ntraining anchored multi-vendor W (l2i={args.l2i}, steps={args.steps}) on {len(train)} corpora ...")
    W = train_W(train, args.steps, args.lr, args.l2i, args.seed)

    print(f"\n=== held-out top-{k} hit rate (bar {bar:.2f}) ===")
    for d, h in zip(holdout_dirs, holdout):
        base = top_k_hit_rate(h, W_id, k=k)
        trained = top_k_hit_rate(h, W, k=k)
        verdict = "GO" if trained >= bar else "NO-GO"
        print(f"  {d:45s} baseline={base:.3f}  trained={trained:.3f}  ({trained - base:+.3f})  -> {verdict}")


if __name__ == "__main__":
    main()


# Author: Mus mbayramo@stanford.edu
