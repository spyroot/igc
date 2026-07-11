"""
Redfish resource graph built from provenance-tagged capture records (D-002).

Builds the typed resource graph the D-002 candidate representation needs: nodes are walked
resources, edges are URL-prefix containment plus ``@odata.id`` references harvested from the
WHOLE body — the corpus census showed explicit ``Links`` sections cover only 10-14% of real
resources, so reference edges must come from anywhere in the JSON, not just ``Links``. Each
node carries the D-002 structural fields: resource type (from ``@odata.type``), depth, the
child-relation name by which it is reachable, and whether its body exposes an invokable
action target. The graph is static per host and safe to build from a partial crawl.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

from igc.ds.sources.base import SourceRecord


@dataclass
class GraphNode:
    """One walked resource in the host's Redfish tree.

    :param url: canonical resource URL (node key).
    :param resource_type: the body's ``@odata.type`` (empty when absent).
    :param depth: number of path segments below ``/``.
    :param child_relation: name by which this node is reachable — the referencing key in a
        parent body when one exists, else the node's collection segment (see
        :meth:`RedfishResourceGraph.from_records`).
    :param has_action_target: whether ``Actions`` in the body carries an invokable ``target``.
    :param body_refs: ``@odata.id`` references harvested from the whole body (deduped, no self).
    """
    url: str
    resource_type: str = ""
    depth: int = 0
    child_relation: str = ""
    has_action_target: bool = False
    body_refs: List[str] = field(default_factory=list)


def harvest_refs(body, self_url: str) -> List[str]:
    """Collect every ``@odata.id`` string reachable anywhere in a JSON body.

    The census rule: explicit ``Links`` sections are too sparse (10-14% coverage) to carry the
    graph, so references are harvested from all nested dicts/lists. Order of first appearance
    is preserved; duplicates and the resource's own URL are dropped.

    :param body: decoded JSON body (any nesting).
    :param self_url: the resource's own URL, excluded from the result.
    :return: ordered unique referenced URLs.
    """
    seen: Dict[str, None] = {}

    def _walk(obj) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key == "@odata.id" and isinstance(value, str) and value and value != self_url:
                    seen.setdefault(value.rstrip("/") or value, None)
                else:
                    _walk(value)
        elif isinstance(obj, list):
            for item in obj:
                _walk(item)

    _walk(body)
    return list(seen)


def has_action_target(body) -> bool:
    """Whether the body's ``Actions`` section exposes at least one invokable ``target``.

    :param body: decoded JSON body.
    :return: ``True`` when any ``Actions`` entry is a dict carrying a ``target`` key.
    """
    actions = body.get("Actions") if isinstance(body, dict) else None
    if not isinstance(actions, dict):
        return False
    return any(isinstance(v, dict) and "target" in v for v in actions.values())


class RedfishResourceGraph:
    """Typed resource graph over one (or one host's) set of capture records.

    Nodes key on canonical URL. Two edge families:

    * **containment** — URL-prefix parent/child (``/redfish/v1/Systems/1`` is a child of
      ``/redfish/v1/Systems``), derivable for 90-98% of real resources;
    * **references** — body-harvested ``@odata.id`` mentions (see :func:`harvest_refs`).
    """

    def __init__(self) -> None:
        self.nodes: Dict[str, GraphNode] = {}
        # parent bodies are needed once, to resolve referencing keys; kept out of GraphNode
        # so downstream consumers of nodes stay light.
        self._bodies: Dict[str, Dict] = {}

    @classmethod
    def from_records(cls, records: Iterable[SourceRecord]) -> "RedfishResourceGraph":
        """Build the graph from capture records (tolerant of partial crawls).

        ``child_relation`` resolution, in priority order: (1) the top-level key under which the
        parent's body references this node; (2) for a collection member whose last segment is
        an id, the parent's collection segment; (3) the node's own last path segment.

        :param records: provenance-tagged capture records for one host/corpus.
        :return: the populated graph.
        """
        graph = cls()
        for rec in records:
            url = rec.url.rstrip("/") or rec.url
            body = rec.response if isinstance(rec.response, dict) else {}
            graph._bodies[url] = body
            graph.nodes[url] = GraphNode(
                url=url,
                resource_type=str(body.get("@odata.type", "")),
                depth=len([seg for seg in url.split("/") if seg]),
                has_action_target=has_action_target(body),
                body_refs=harvest_refs(body, url),
            )
        for node in graph.nodes.values():
            node.child_relation = graph._resolve_relation(node.url)
        return graph

    def parent(self, url: str) -> Optional[str]:
        """The longest walked URL-prefix parent of ``url``, or ``None``.

        :param url: node URL.
        :return: parent URL when walked, else ``None``.
        """
        candidate = url.rstrip("/")
        while "/" in candidate.strip("/"):
            candidate = candidate.rsplit("/", 1)[0]
            if candidate in self.nodes:
                return candidate
        return None

    def children(self, url: str) -> List[str]:
        """Walked nodes whose nearest walked ancestor is ``url`` (containment children).

        :param url: node URL.
        :return: child URLs, sorted.
        """
        return sorted(u for u in self.nodes if u != url and self.parent(u) == url)

    def neighbors(self, url: str) -> List[str]:
        """True transition targets from ``url``: containment children, parent, and walked refs.

        :param url: node URL.
        :return: ordered unique neighbor URLs present in the graph.
        """
        out: Dict[str, None] = {}
        for child in self.children(url):
            out.setdefault(child, None)
        parent = self.parent(url)
        if parent:
            out.setdefault(parent, None)
        node = self.nodes.get(url)
        if node:
            for ref in node.body_refs:
                if ref in self.nodes and ref != url:
                    out.setdefault(ref, None)
        return list(out)

    def _resolve_relation(self, url: str) -> str:
        """Resolve the D-002 ``child_relation`` name for ``url`` (see :meth:`from_records`).

        :param url: node URL.
        :return: relation name; empty only for a root with no segments.
        """
        segments = [seg for seg in url.split("/") if seg]
        parent_url = self.parent(url)
        if parent_url:
            parent_body_key = self._referencing_key(parent_url, url)
            if parent_body_key:
                return parent_body_key
        if len(segments) >= 2 and self._looks_like_member_id(segments[-1]):
            return segments[-2]
        return segments[-1] if segments else ""

    def _referencing_key(self, parent_url: str, child_url: str) -> str:
        """Top-level key of the parent body whose subtree references ``child_url``.

        :param parent_url: walked parent URL.
        :param child_url: child URL to locate.
        :return: the referencing top-level key, or ``""``.
        """
        return _top_level_ref_key(self._bodies.get(parent_url), child_url) or ""

    @staticmethod
    def _looks_like_member_id(segment: str) -> bool:
        """Heuristic: a collection-member id segment (digits or short opaque token).

        :param segment: last URL path segment.
        :return: ``True`` for ids like ``1``, ``0``, ``CPU_0``, UUID-ish tokens.
        """
        return segment.isdigit() or any(ch.isdigit() for ch in segment)


def _top_level_ref_key(body, child_url: str) -> Optional[str]:
    """Top-level key of ``body`` whose subtree contains ``@odata.id == child_url``.

    :param body: decoded parent JSON body (or ``None``).
    :param child_url: URL to find.
    :return: key name or ``None``.
    """
    if not isinstance(body, dict):
        return None

    def _contains(obj) -> bool:
        if isinstance(obj, dict):
            odata = obj.get("@odata.id")
            if isinstance(odata, str) and odata.rstrip("/") == child_url:
                return True
            return any(_contains(v) for v in obj.values())
        if isinstance(obj, list):
            return any(_contains(i) for i in obj)
        return False

    for key, value in body.items():
        if key.startswith("@"):
            continue
        if _contains(value):
            return key
    return None


# Author: Mus mbayramo@stanford.edu
