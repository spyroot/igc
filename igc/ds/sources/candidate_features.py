"""
D-002 v1 action-candidate featurizer with a static per-host cache.

Used by ``scripts/bench_hot_paths.py`` (the candidate-cache benchmark stage), feeding
``igc/modules/eval/zero_shot_ranking.py``. The candidate dict emitted here is a schema
contract with that ranker's ``candidate_text`` / ``embed_candidates`` — renaming a field
silently breaks ranking.

Turns graph nodes into the accepted candidate schema — path tokens, HTTP method, resource
type, child-relation name, action-target flag — for the D-001 pointer. Per D-002, every
field is static per host: the cache is built once from the walked tree and only *filtered*
by a state's legal catalog at decision time (HER relabeling re-scores cached candidates, it
never re-encodes them). Member ids in the path are kept as trailing tokens, never deleted —
erasing them would break goals that target a specific collection member.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from igc.ds.sources.resource_graph import GraphNode, RedfishResourceGraph

# a candidate is (url, METHOD); GET is always legal on a walked resource.
_DEFAULT_METHODS = ("GET",)


def path_tokens(url: str) -> List[str]:
    """Split a resource URL into path tokens, preserving member ids as their own tokens.

    :param url: canonical resource URL.
    :return: non-empty path segments in order.
    """
    return [seg for seg in url.split("/") if seg]


def candidate_v1(node: GraphNode, method: str) -> Dict:
    """Build the D-002 v1 candidate dict for one (endpoint, method) pair.

    :param node: the endpoint's graph node.
    :param method: HTTP method for this candidate.
    :return: the v1 schema — url, path tokens, method, resource type, relation, action flag.
    """
    return {
        "url": node.url,
        "endpoint_path_tokens": path_tokens(node.url),
        "http_method": method.upper(),
        "resource_type": node.resource_type,
        "child_relation_name": node.child_relation,
        "has_action_target": node.has_action_target,
    }


def build_candidate_cache(
    graph: RedfishResourceGraph,
    allowed_methods: Optional[Dict[str, List[str]]] = None,
) -> Dict[Tuple[str, str], Dict]:
    """Build the static per-host candidate cache for every walked (endpoint, method).

    Methods come from the discovery ``allowed_methods_mapping`` when provided (the ``.npy``
    contract half loaded by ``ds_rest_trajectories``); endpoints missing from the map fall
    back to GET, so a partial map never drops a walked endpoint.

    :param graph: the host's resource graph.
    :param allowed_methods: optional ``{url: [methods]}`` from the discovery map.
    :return: ``{(url, METHOD): candidate_v1}`` — filter per state, never rebuild per state.
    """
    allowed_methods = allowed_methods or {}
    cache: Dict[Tuple[str, str], Dict] = {}
    for url, node in graph.nodes.items():
        methods = allowed_methods.get(url) or _DEFAULT_METHODS
        for method in methods:
            method_upper = str(method).upper()
            cache[(url, method_upper)] = candidate_v1(node, method_upper)
    return cache


# Author: Mus mbayramo@stanford.edu
