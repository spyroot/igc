"""
Offline tests for the SourceRecord -> TrainingExample normalizer.

Independent verification of the normalization contract: a compact resource summary that drops
``@``-prefixed keys, a GET observation mapped to a (state, action, next_state, semantics) tuple
where before/after are equal-but-separate (a read does not mutate), read-only/idempotent
semantics, ``mutable_endpoint`` derived from allowed methods, provenance carried through, and a
JSON-serializable to_dict with the trust tier as its name. Pure stdlib — no torch, no network.

Author:
Mus mbayramo@stanford.edu
"""

import json

from igc.ds.sources.base import SourceRecord, TrustLevel
from igc.ds.sources.training_object import (
    TrainingExample,
    compact_resource,
    normalize,
    normalize_record,
)


def _rec(url="/redfish/v1/Systems/1", method="GET", allowed=None, body=None,
         trust=TrustLevel.REAL, vendor="dell"):
    """Build a SourceRecord for the normalizer under test."""
    if body is None:
        body = {"@odata.id": url, "@odata.type": "#ComputerSystem.v1.ComputerSystem",
                "Id": "1", "Name": "Sys"}
    return SourceRecord(url=url, response=body, source="real_dell", trust_level=trust,
                        method=method, allowed_methods=allowed, vendor=vendor,
                        schema_version="#ComputerSystem.v1.ComputerSystem",
                        provenance={"file": "x.json", "url_from": "odata"})


def test_compact_resource_summarizes_and_drops_odata_keys():
    """compact_resource keeps @odata.id/type and the sorted non-@ top-level keys."""
    r = compact_resource({"@odata.id": "/a", "@odata.type": "#T",
                          "Name": "n", "Id": "1", "@odata.etag": "x"})
    assert r == {"@odata.id": "/a", "@odata.type": "#T", "keys": ["Id", "Name"]}


def test_compact_resource_handles_empty_and_non_dict():
    """A non-dict or empty body yields the empty summary, not a crash."""
    empty = {"@odata.id": "", "@odata.type": "", "keys": []}
    assert compact_resource({}) == empty
    assert compact_resource(None) == empty


def test_normalize_get_is_a_nonmutating_observation():
    """A GET maps to a read-only tuple with before == after but separate objects."""
    ex = normalize_record(_rec(allowed=["GET", "PATCH"]))
    assert isinstance(ex, TrainingExample)
    assert ex.request_or_action == {"method": "GET", "url": "/redfish/v1/Systems/1", "body": None}
    assert list(ex.resource_graph_before) == ["/redfish/v1/Systems/1"]
    assert ex.resource_graph_before == ex.resource_graph_after
    assert ex.resource_graph_before is not ex.resource_graph_after  # equal, not aliased
    assert ex.resource_graph_before["/redfish/v1/Systems/1"]["keys"] == ["Id", "Name"]


def test_get_semantics_are_read_only_idempotent():
    """expected_semantics marks a GET read-only, idempotent, non-mutating, status 200."""
    sem = normalize_record(_rec(allowed=["GET", "PATCH"])).expected_semantics
    assert sem["method"] == "GET"
    assert sem["mutating"] is False and sem["read_only"] is True
    assert sem["idempotent"] is True and sem["expected_status"] == 200
    assert sem["mutable_endpoint"] is True  # PATCH allowed on the endpoint


def test_mutable_endpoint_and_allowed_methods_default():
    """mutable_endpoint is False for a read-only endpoint; None methods -> []."""
    read_only = normalize_record(_rec(allowed=["GET"]))
    assert read_only.expected_semantics["mutable_endpoint"] is False
    none_methods = normalize_record(_rec(allowed=None))
    assert none_methods.allowed_methods == []
    assert none_methods.expected_semantics["mutable_endpoint"] is False


def test_provenance_carried_through():
    """source / trust / schema / vendor / provenance survive normalization."""
    ex = normalize_record(_rec())
    assert ex.source == "real_dell" and ex.trust_level is TrustLevel.REAL
    assert ex.vendor == "dell" and ex.schema_version == "#ComputerSystem.v1.ComputerSystem"
    assert ex.provenance["url_from"] == "odata"


def test_to_dict_is_json_serializable_with_trust_name():
    """to_dict emits the trust tier as its name and is JSON-serializable."""
    d = normalize_record(_rec()).to_dict()
    assert d["trust_level"] == "REAL"
    json.dumps(d)  # must not raise


def test_normalize_maps_over_records():
    """normalize applies normalize_record to each record, preserving order."""
    exs = normalize([_rec(url="/a"), _rec(url="/b")])
    assert [list(e.resource_graph_before)[0] for e in exs] == ["/a", "/b"]


def test_head_is_read_only_post_is_mutating():
    """Method drives semantics: HEAD read-only/idempotent, a POST mutating/non-idempotent."""
    head = normalize_record(_rec(method="HEAD")).expected_semantics
    assert head["read_only"] is True and head["idempotent"] is True
    post = normalize_record(_rec(method="POST")).expected_semantics
    assert post["mutating"] is True and post["read_only"] is False and post["idempotent"] is False


# Author: Mus mbayramo@stanford.edu
