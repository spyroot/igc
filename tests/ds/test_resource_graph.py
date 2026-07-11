"""
Offline tests for the Redfish resource graph and D-002 candidate featurizer.

Pins that the graph harvests @odata.id references from WHOLE bodies (not just Links — the
census showed Links sections cover only 10-14% of real resources), derives containment
parents/children from URL prefixes, resolves child_relation from the parent's referencing
key with collection-segment and last-segment fallbacks, flags action targets, tolerates a
partial crawl, and that the candidate cache is static per (url, method) with GET fallback
and preserved member-id tokens. Pure stdlib — no torch, no network.

Author:
Mus mbayramo@stanford.edu
"""

from igc.ds.sources.base import SourceRecord, TrustLevel
from igc.ds.sources.candidate_features import build_candidate_cache, candidate_v1, path_tokens
from igc.ds.sources.resource_graph import RedfishResourceGraph, harvest_refs, has_action_target


def _rec(url, body):
    """A REAL-tier SourceRecord with the given body."""
    return SourceRecord(url=url, response=body, source="real_test", trust_level=TrustLevel.REAL)


def _walk():
    """A small walked tree: root -> Systems collection -> member with action + deep ref."""
    return [
        _rec("/redfish/v1", {
            "@odata.id": "/redfish/v1", "@odata.type": "#ServiceRoot.v1.ServiceRoot",
            "Systems": {"@odata.id": "/redfish/v1/Systems"},
        }),
        _rec("/redfish/v1/Systems", {
            "@odata.id": "/redfish/v1/Systems", "@odata.type": "#Collection.Collection",
            "Members": [{"@odata.id": "/redfish/v1/Systems/1"}],
        }),
        _rec("/redfish/v1/Systems/1", {
            "@odata.id": "/redfish/v1/Systems/1",
            "@odata.type": "#ComputerSystem.v1.ComputerSystem",
            # ref buried OUTSIDE any Links section — the census-mandated harvest path
            "Status": {"Health": "OK", "Origin": {"@odata.id": "/redfish/v1/Chassis/1"}},
            "Actions": {"#ComputerSystem.Reset": {"target": "/redfish/v1/Systems/1/Actions/Reset"}},
        }),
        _rec("/redfish/v1/Chassis/1", {
            "@odata.id": "/redfish/v1/Chassis/1", "@odata.type": "#Chassis.v1.Chassis",
        }),
    ]


def test_harvest_refs_scans_whole_body_not_just_links():
    """References nested anywhere (here under Status.Origin) are harvested, self excluded."""
    body = _walk()[2].response
    refs = harvest_refs(body, "/redfish/v1/Systems/1")
    assert "/redfish/v1/Chassis/1" in refs
    assert "/redfish/v1/Systems/1" not in refs  # self excluded


def test_has_action_target():
    """Actions entries carrying a target flag the node; absent/empty Actions do not."""
    assert has_action_target(_walk()[2].response) is True
    assert has_action_target(_walk()[3].response) is False
    assert has_action_target({"Actions": {"#X.Y": "not-a-dict"}}) is False


def test_containment_parent_children_and_depth():
    """URL-prefix containment yields parents, children, and segment depth."""
    g = RedfishResourceGraph.from_records(_walk())
    assert g.parent("/redfish/v1/Systems/1") == "/redfish/v1/Systems"
    assert g.parent("/redfish/v1") is None
    assert g.children("/redfish/v1/Systems") == ["/redfish/v1/Systems/1"]
    assert g.nodes["/redfish/v1/Systems/1"].depth == 4


def test_child_relation_from_parent_referencing_key():
    """The relation is the parent's top-level key that references the child."""
    g = RedfishResourceGraph.from_records(_walk())
    assert g.nodes["/redfish/v1/Systems"].child_relation == "Systems"     # root refs via key
    assert g.nodes["/redfish/v1/Systems/1"].child_relation == "Members"   # collection Members


def test_child_relation_collection_fallback_without_parent_body():
    """A member id whose parent is unwalked falls back to the collection segment."""
    g = RedfishResourceGraph.from_records([
        _rec("/redfish/v1/Chassis/CPU_0", {"@odata.type": "#Chassis.v1.Chassis"}),
    ])
    assert g.nodes["/redfish/v1/Chassis/CPU_0"].child_relation == "Chassis"


def test_neighbors_union_children_parent_refs():
    """neighbors() = containment children + parent + walked body refs, deduped."""
    g = RedfishResourceGraph.from_records(_walk())
    n = g.neighbors("/redfish/v1/Systems/1")
    assert "/redfish/v1/Systems" in n          # parent
    assert "/redfish/v1/Chassis/1" in n        # harvested ref (walked)
    assert len(n) == len(set(n))


def test_partial_crawl_is_safe():
    """Refs to unwalked resources are kept in body_refs but never appear as neighbors."""
    g = RedfishResourceGraph.from_records(_walk()[:3])  # Chassis/1 NOT walked
    node = g.nodes["/redfish/v1/Systems/1"]
    assert "/redfish/v1/Chassis/1" in node.body_refs
    assert "/redfish/v1/Chassis/1" not in g.neighbors("/redfish/v1/Systems/1")


def test_candidate_v1_schema_and_member_id_tokens():
    """candidate_v1 carries the D-002 fields; member ids stay as trailing path tokens."""
    g = RedfishResourceGraph.from_records(_walk())
    c = candidate_v1(g.nodes["/redfish/v1/Systems/1"], "get")
    assert c["endpoint_path_tokens"] == ["redfish", "v1", "Systems", "1"]  # id kept
    assert c["http_method"] == "GET"
    assert c["resource_type"] == "#ComputerSystem.v1.ComputerSystem"
    assert c["child_relation_name"] == "Members"
    assert c["has_action_target"] is True
    assert path_tokens("/redfish/v1") == ["redfish", "v1"]


def test_candidate_cache_static_with_get_fallback_and_method_map():
    """The cache keys every (url, METHOD); mapped endpoints expand, unmapped fall back to GET."""
    g = RedfishResourceGraph.from_records(_walk())
    cache = build_candidate_cache(g, {"/redfish/v1/Systems/1": ["GET", "PATCH"]})
    assert ("/redfish/v1/Systems/1", "PATCH") in cache
    assert ("/redfish/v1/Systems/1", "GET") in cache
    assert ("/redfish/v1/Chassis/1", "GET") in cache            # fallback
    assert ("/redfish/v1/Chassis/1", "PATCH") not in cache
    assert len({id(v) for v in cache.values()}) == len(cache)   # distinct dicts


# Author: Mus mbayramo@stanford.edu
