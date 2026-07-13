"""Structured Redfish state contract for M1/M2 and downstream RL.

The normalized corpus still stores ``TrainingExample`` dictionaries for
compatibility, but M1/M2 now consume them through this typed state layer. The
contract is intentionally compact, deterministic, and public-safe: it preserves
resource graph, action affordances, resource text, provenance metadata, and a
slot for model-produced latents without embedding private raw captures or hosts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Dict, Iterable, List, Optional

METHOD_IDS = {
    "GET": 1,
    "HEAD": 2,
    "POST": 3,
    "PATCH": 4,
    "PUT": 5,
    "DELETE": 6,
}
METHOD_ORDER = ("GET", "HEAD", "POST", "PATCH", "PUT", "DELETE")


def _canonical(value: Any) -> str:
    """Deterministic JSON rendering for hashes and debug text."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _resource_kind(odata_type: str, keys: Iterable[str]) -> str:
    """Derive a compact kind from ``@odata.type`` or observed keys."""
    if odata_type:
        tail = str(odata_type).split(".")[-1]
        return tail.lstrip("#") or "resource"
    key_list = sorted(str(k) for k in keys)
    return "resource:" + ",".join(key_list[:4]) if key_list else "resource"


def stable_id(value: Any, modulo: int = 2**31 - 1) -> int:
    """Return a deterministic positive integer id for a feature value."""
    digest = hashlib.blake2b(str(value).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % modulo


def path_segment_hashes(uri: str, max_segments: int = 8) -> List[int]:
    """Stable hashes of URI path segments, padded/truncated for batching."""
    segments = [s for s in str(uri).split("/") if s]
    hashes = [stable_id(seg, modulo=1_000_003) for seg in segments[:max_segments]]
    while len(hashes) < max_segments:
        hashes.append(0)
    return hashes


def allowed_method_mask(methods: Iterable[str]) -> List[int]:
    """Fixed method mask in ``METHOD_ORDER`` order."""
    allowed = {str(m).upper() for m in methods}
    return [1 if method in allowed else 0 for method in METHOD_ORDER]


@dataclass(frozen=True)
class StateGraphNode:
    """One compact Redfish resource node."""

    id: int
    uri: str
    resource_type: str
    kind: str
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "uri": self.uri,
            "resource_type": self.resource_type,
            "kind": self.kind,
            "attributes": self.attributes,
        }


@dataclass(frozen=True)
class StateGraphEdge:
    """One compact relation between resources or resource/action affordances."""

    source: int
    target: int
    relation: str
    label: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "relation": self.relation,
            "label": self.label,
        }


@dataclass(frozen=True)
class StateRecord:
    """Minimum structured state record consumed by M1/M2."""

    state_id: str
    state_fingerprint: str
    source_snapshot: str
    resource_graph: Dict[str, Any]
    resource_node: Dict[str, Any]
    resource_text: str
    goal_context: Optional[Dict[str, Any]]
    action_affordances: Dict[str, Any]
    state_latent: Optional[Any]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state_id": self.state_id,
            "state_fingerprint": self.state_fingerprint,
            "source_snapshot": self.source_snapshot,
            "resource_graph": self.resource_graph,
            "resource_node": self.resource_node,
            "resource_text": self.resource_text,
            "goal_context": self.goal_context,
            "action_affordances": self.action_affordances,
            "state_latent": self.state_latent,
            "metadata": self.metadata,
        }


def candidate_feature_tensor_payload(candidates: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    """Convert candidate feature dicts into batchable per-field lists."""
    return {
        "resource_type_id": [c["resource_type_id"] for c in candidates],
        "parent_type_id": [c["parent_type_id"] for c in candidates],
        "relation_name_id": [c["relation_name_id"] for c in candidates],
        "depth_bucket": [c["depth_bucket"] for c in candidates],
        "method_id": [c["method_id"] for c in candidates],
        "has_action_target": [int(c["has_action_target"]) for c in candidates],
        "is_collection": [int(c["is_collection"]) for c in candidates],
        "is_oem": [int(c["is_oem"]) for c in candidates],
        "path_segment_hashes": [c["path_segment_hashes"] for c in candidates],
        "allowed_method_mask": [c["allowed_method_mask"] for c in candidates],
        "local_state_summary": [c["local_state_summary"] for c in candidates],
    }


def build_state_record(example: Dict[str, Any]) -> StateRecord:
    """Build a deterministic structured state from a normalized TrainingExample."""
    graph_before = example.get("resource_graph_before", {}) or {}
    response = example.get("response", {}) or {}
    action = example.get("request_or_action", {}) or {}
    allowed_methods = [str(m).upper() for m in (example.get("allowed_methods", []) or [])]
    expected_semantics = example.get("expected_semantics", {}) or {}

    nodes = _nodes_from_graph(graph_before)
    selected_node = _selected_node(nodes, action.get("url", ""))
    edges = _edges_from_response_refs(response, nodes)
    edges.extend(_action_edges([selected_node], allowed_methods))
    graph = {
        "nodes": [n.to_dict() for n in nodes],
        "edges": [e.to_dict() for e in edges],
        "node_count": len(nodes),
        "edge_count": len(edges),
    }
    action_affordances = {
        "allowed_methods": allowed_methods,
        "candidate_count": len(allowed_methods),
        "mutable_endpoint": bool(expected_semantics.get("mutable_endpoint", False)),
        "read_only": bool(expected_semantics.get("read_only", False)),
    }
    candidates = _candidate_features([selected_node], response, action, allowed_methods, action_affordances)
    action_affordances["templates"] = candidates
    action_affordances["feature_schema"] = [
        "resource_type_id",
        "parent_type_id",
        "relation_name_id",
        "depth_bucket",
        "method_id",
        "has_action_target",
        "is_collection",
        "is_oem",
        "path_segment_hashes",
        "allowed_method_mask",
        "local_state_summary",
    ]
    metadata = {
        "source": example.get("source", ""),
        "trust_level": example.get("trust_level", ""),
        "vendor": example.get("vendor", ""),
        "schema_version": example.get("schema_version", ""),
        "partition": (example.get("provenance", {}) or {}).get("partition", ""),
        "provenance": example.get("provenance", {}) or {},
    }
    source_snapshot = _source_snapshot(example)
    goal_context = _goal_context(example.get("goal_context"))

    fingerprint_payload = {
        "source_snapshot": source_snapshot,
        "resource_graph": graph,
        "action_affordances": action_affordances,
        "metadata": {k: metadata.get(k) for k in ("source", "trust_level", "vendor", "schema_version")},
        "goal_context": goal_context,
    }
    state_fingerprint = hashlib.blake2b(
        _canonical(fingerprint_payload).encode("utf-8"), digest_size=16).hexdigest()
    resource_text = render_resource_text(
        resource_node=selected_node.to_dict(),
        response=response,
        action=action,
        action_affordances=action_affordances,
        expected_semantics=expected_semantics,
        goal_context=goal_context,
    )

    return StateRecord(
        state_id=f"state:{state_fingerprint[:16]}",
        state_fingerprint=state_fingerprint,
        source_snapshot=source_snapshot,
        resource_graph=graph,
        resource_node=selected_node.to_dict(),
        resource_text=resource_text,
        goal_context=goal_context,
        action_affordances=action_affordances,
        state_latent=None,
        metadata=metadata,
    )


def render_resource_text(
    *,
    resource_node: Dict[str, Any],
    response: Dict[str, Any],
    action: Dict[str, Any],
    action_affordances: Dict[str, Any],
    expected_semantics: Dict[str, Any],
    goal_context: Optional[Dict[str, Any]],
) -> str:
    """Stable per-resource text consumed by M1 LM-compatible encoders.

    This intentionally avoids flattening the whole host dump or full graph into
    one prompt. M1 sees the selected resource JSON plus local typed features;
    M2 pools the structured graph/set outside this text path.
    """
    local_action_summary = {
        "allowed_methods": action_affordances.get("allowed_methods", []),
        "candidate_count": action_affordances.get("candidate_count", 0),
        "mutable_endpoint": action_affordances.get("mutable_endpoint", False),
        "read_only": action_affordances.get("read_only", False),
    }
    parts = [
        _section("GOAL_CONTEXT", goal_context or {}),
        f"REQUEST {action.get('method', 'GET')} {action.get('url', '')}",
        _section("RESOURCE_NODE", resource_node),
        _section("RESOURCE_JSON", response),
        _section("LEGAL_ACTION_SUMMARY", local_action_summary),
        _section("EXPECTED_SEMANTICS", expected_semantics),
    ]
    return "\n".join(parts)


def _nodes_from_graph(graph: Dict[str, Dict[str, Any]]) -> List[StateGraphNode]:
    nodes = []
    for uri in sorted(graph):
        summary = graph.get(uri, {}) or {}
        keys = summary.get("keys", []) or []
        resource_type = str(summary.get("@odata.type", ""))
        attrs = {
            "keys": list(keys),
            "odata_id": summary.get("@odata.id", uri),
            "resource_type_id": stable_id(resource_type),
            "is_collection": _is_collection_uri(str(uri), keys),
            "is_oem": _is_oem(resource_type, keys),
        }
        nodes.append(StateGraphNode(
            id=stable_id(uri, modulo=1_000_003),
            uri=str(uri),
            resource_type=resource_type,
            kind=_resource_kind(resource_type, keys),
            attributes=attrs,
        ))
    return nodes


def _selected_node(nodes: List[StateGraphNode], uri: str) -> StateGraphNode:
    if not nodes:
        return StateGraphNode(0, str(uri), "", "resource", {
            "keys": [],
            "odata_id": str(uri),
            "resource_type_id": stable_id(""),
            "is_collection": False,
            "is_oem": False,
        })
    for node in nodes:
        if node.uri == uri:
            return node
    return nodes[0]


def _edges_from_response_refs(response: Dict[str, Any], nodes: List[StateGraphNode]) -> List[StateGraphEdge]:
    uri_to_id = {n.uri: n.id for n in nodes}
    source_id = nodes[0].id if nodes else 0
    edges = []
    for ref in sorted(_harvest_refs(response)):
        if ref in uri_to_id and ref != nodes[0].uri:
            edges.append(StateGraphEdge(source=source_id, target=uri_to_id[ref], relation="reference"))
    return edges


def _action_edges(nodes: List[StateGraphNode], allowed_methods: List[str]) -> List[StateGraphEdge]:
    if not nodes:
        return []
    source_id = nodes[0].id
    base = max(n.id for n in nodes) + 1
    return [
        StateGraphEdge(source=source_id, target=base + idx, relation="action_capability", label=method)
        for idx, method in enumerate(allowed_methods)
    ]


def _candidate_features(
    nodes: List[StateGraphNode],
    response: Dict[str, Any],
    action: Dict[str, Any],
    allowed_methods: List[str],
    action_affordances: Dict[str, Any],
) -> List[Dict[str, Any]]:
    node = nodes[0] if nodes else StateGraphNode(0, str(action.get("url", "")), "", "resource", {})
    uri = str(action.get("url") or node.uri)
    parent_type = _parent_type(uri)
    relation_name = _relation_name(uri)
    local_summary = _local_state_summary(response, action_affordances)
    mask = allowed_method_mask(allowed_methods)
    has_action = _has_action_target(response)
    is_collection = bool(node.attributes.get("is_collection", False))
    is_oem = bool(node.attributes.get("is_oem", False))
    resource_type_id = int(node.attributes.get("resource_type_id", stable_id(node.resource_type)))
    parent_type_id = stable_id(parent_type)
    relation_name_id = stable_id(relation_name)
    depth_bucket = min(len([s for s in uri.split("/") if s]), 8)
    path_hashes = path_segment_hashes(uri)

    return [
        {
            "tool_name": "redfish",
            "op": method,
            "target": uri,
            "risk": "read" if method in {"GET", "HEAD"} else "mutating",
            "text": f"redfish {method} {uri}",
            "resource_type": node.resource_type,
            "parent_type": parent_type,
            "relation_name": relation_name,
            "resource_type_id": resource_type_id,
            "parent_type_id": parent_type_id,
            "relation_name_id": relation_name_id,
            "depth_bucket": depth_bucket,
            "method_id": METHOD_IDS.get(method, 0),
            "has_action_target": has_action,
            "is_collection": is_collection,
            "is_oem": is_oem,
            "path_segment_hashes": path_hashes,
            "allowed_method_mask": mask,
            "local_state_summary": local_summary,
        }
        for method in allowed_methods
    ]


def _parent_type(uri: str) -> str:
    parts = [p for p in uri.split("/") if p]
    return parts[-2] if len(parts) >= 2 else ""


def _relation_name(uri: str) -> str:
    parts = [p for p in uri.split("/") if p]
    return parts[-1] if parts else ""


def _is_collection_uri(uri: str, keys: Iterable[str]) -> bool:
    key_set = {str(k) for k in keys}
    return "Members" in key_set or uri.rstrip("/").split("/")[-1].endswith("s")


def _is_oem(resource_type: str, keys: Iterable[str]) -> bool:
    return "Oem" in {str(k) for k in keys} or "Oem" in str(resource_type)


def _has_action_target(response: Dict[str, Any]) -> bool:
    if not isinstance(response, dict):
        return False
    actions = response.get("Actions", {})
    if isinstance(actions, dict):
        return any(isinstance(v, dict) and "target" in v for v in actions.values())
    return False


def _local_state_summary(response: Dict[str, Any], action_affordances: Dict[str, Any]) -> List[int]:
    status = response.get("Status", {}) if isinstance(response, dict) else {}
    health = status.get("Health", "") if isinstance(status, dict) else ""
    state = status.get("State", "") if isinstance(status, dict) else ""
    return [
        stable_id(health, modulo=4096),
        stable_id(state, modulo=4096),
        int(bool(action_affordances.get("mutable_endpoint", False))),
        int(bool(action_affordances.get("read_only", False))),
    ]


def _harvest_refs(value: Any) -> set[str]:
    refs: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "@odata.id" and isinstance(item, str):
                refs.add(item)
            else:
                refs.update(_harvest_refs(item))
    elif isinstance(value, list):
        for item in value:
            refs.update(_harvest_refs(item))
    return refs


def _source_snapshot(example: Dict[str, Any]) -> str:
    provenance = example.get("provenance", {}) or {}
    for key in ("snapshot", "file", "source_file", "capture_id"):
        if provenance.get(key):
            return str(provenance[key])
    action = example.get("request_or_action", {}) or {}
    return f"{example.get('source', '')}:{action.get('method', 'GET')}:{action.get('url', '')}"


def _goal_context(value: Any) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    if not isinstance(value, dict):
        return {"goal_id": f"goal:{stable_id(value)}", "instruction": str(value)}
    payload = {
        "instruction": value.get("instruction", ""),
        "spec": value.get("spec", {}) or {},
        "constraints": value.get("constraints", []) or [],
        "plan": value.get("plan", []) or [],
    }
    return {
        "goal_id": "goal:" + hashlib.blake2b(
            _canonical(payload).encode("utf-8"), digest_size=8).hexdigest(),
        **payload,
    }


def _section(name: str, value: Any) -> str:
    return f"{name} {_canonical(value)}"
