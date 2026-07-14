"""
Normalize provenance-tagged Redfish SourceRecords (GET observations) into uniform
TrainingExample tuples consumed by the state-encoder and RL agent.

Each SourceRecord carries a full Redfish response body, provenance metadata, and a trust level.
This module compacts the resource representation for the state graph, constructs the
(state, action, next_state, semantics) tuple, and preserves provenance end-to-end.

Consumers live inside ``igc.ds.sources``, not the model code: ``corpus_io.write_corpus``
serializes each ``TrainingExample.to_dict()`` to ``examples.jsonl``, and
``redfish_enum_space.normalize_enriched`` calls ``normalize_record``. That corpus reaches M1
training one hop downstream via ``CorpusJSONLDataset`` (the ``--corpus_dir`` path).

Author:
Mus mbayramo@stanford.edu
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from igc.ds.sources.base import SourceRecord, TrustLevel


def compact_resource(body: Dict[str, Any]) -> Dict[str, Any]:
    """Build a small faithful summary of a Redfish resource body.

    The full body remains in ``response``; this summary keeps the state graph light.

    :param body: Full Redfish response body (a JSON-like dict).
    :return: Dict with keys ``@odata.id``, ``@odata.type``, and ``keys`` (sorted
        top-level keys that do not start with ``@``).  If *body* is not a dict or is
        empty, ``@odata.id`` and ``@odata.type`` default to ``""`` and ``keys`` to ``[]``.
    """
    if not isinstance(body, dict) or not body:
        return {"@odata.id": "", "@odata.type": "", "keys": []}

    return {
        "@odata.id": body.get("@odata.id", ""),
        "@odata.type": body.get("@odata.type", ""),
        "keys": sorted(k for k in body if not k.startswith("@")),
    }


@dataclass
class TrainingExample:
    """Uniform (state, action, next_state, semantics) tuple with provenance.

    For a GET observation the resource graph is unchanged by the read, so
    ``resource_graph_after`` is a separate-but-equal copy of ``resource_graph_before``.
    """

    source: str  # where this example came from (host / dataset / fixture id)
    trust_level: TrustLevel  # data trust tier: REAL(5) > REPLAY(4) > SIM_VENDOR(3) > SIM_GENERIC(2) > SIM_DRIFT(1)
    schema_version: str  # Redfish schema version the resource declares
    # STATE *before* the action. TODAY this is a single-node dict {url: compact_resource}, NOT a
    # topology graph — it has no parent/subordinate edges (those live only in RedfishResourceGraph).
    resource_graph_before: Dict[str, Dict[str, Any]]
    request_or_action: Dict[str, Any]  # the action taken: {"method", "url", "body"} (the REST call)
    response: Dict[str, Any]  # raw Redfish response body returned — what the M1 encoder actually reads today
    resource_graph_after: Dict[str, Dict[str, Any]]  # STATE *after* the action; == before for a GET (a read never mutates)
    allowed_methods: List[str]  # HTTP methods the API permits on this URL (GET/POST/PATCH/DELETE...) -> the legal action set
    expected_semantics: Dict[str, Any]  # derived facts about the call: mutating? idempotent? expected status (see normalize_record)
    vendor: Optional[str]  # hardware vendor (Dell/Supermicro/HPE...) — used for cross-vendor train/eval splits
    provenance: Dict[str, Any] = field(default_factory=dict)  # audit trail: how/when/by-what this example was produced

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dict representation of the training example.

        ``trust_level`` is emitted as its ``.name`` string (e.g. ``"REAL"``); all other
        fields are returned as-is.

        :return: Dict suitable for JSON serialization.
        """
        return {
            "source": self.source,                              # provenance: origin id
            "trust_level": self.trust_level.name,               # tier name, e.g. "REAL" (IntEnum -> str for JSON)
            "schema_version": self.schema_version,              # Redfish schema version
            "resource_graph_before": self.resource_graph_before,  # state before (single-node dict today)
            "request_or_action": self.request_or_action,        # {"method","url","body"} — the REST call
            "response": self.response,                          # raw Redfish response body (what the encoder reads)
            "resource_graph_after": self.resource_graph_after,  # state after (== before for a GET)
            "allowed_methods": self.allowed_methods,            # HTTP methods legal on this URL -> action set
            "expected_semantics": self.expected_semantics,      # mutating? idempotent? expected status
            "vendor": self.vendor,                              # Dell/Supermicro/HPE... for vendor splits
            "provenance": self.provenance,                      # full audit trail
        }


def normalize_record(record: SourceRecord) -> TrainingExample:
    """Convert a single GET-observation SourceRecord into a TrainingExample.

    :param record: SourceRecord carrying a Redfish GET response and provenance.
    :return: TrainingExample with compacted resource graph, request metadata, and
        derived semantics.
    """
    graph: Dict[str, Dict[str, Any]] = {record.url: compact_resource(record.response)}

    method_upper = record.method.upper()
    mutating = method_upper not in ("GET", "HEAD")
    allowed = record.allowed_methods or []

    expected_semantics: Dict[str, Any] = {
        "method": method_upper,                    # normalized HTTP verb of this call
        "mutating": mutating,                      # True if the call changes server state (anything but GET/HEAD)
        "read_only": not mutating,                 # convenience inverse of `mutating`
        "idempotent": method_upper in ("GET", "HEAD", "PUT", "DELETE"),  # safe to retry: same effect if repeated
        "expected_status": 200,                    # HTTP status a healthy call should return
        "mutable_endpoint": any(                   # True if the URL allows ANY write method (a write-capable endpoint)
            m.upper() in ("POST", "PATCH", "PUT", "DELETE") for m in allowed
        ),
    }

    return TrainingExample(
        source=record.source,
        trust_level=record.trust_level,
        schema_version=record.schema_version,
        resource_graph_before=graph,
        request_or_action={
            "method": record.method,
            "url": record.url,
            "body": None,
        },
        response=record.response,
        resource_graph_after=dict(graph),  # separate-but-equal copy
        allowed_methods=allowed,
        expected_semantics=expected_semantics,
        vendor=record.vendor,
        provenance=record.provenance,
    )


def normalize(records: Iterable[SourceRecord]) -> List[TrainingExample]:
    """Normalize an iterable of SourceRecords into a list of TrainingExamples.

    :param records: Iterable of SourceRecord objects (typically GET observations).
    :return: List of TrainingExample instances, one per input record.
    """
    return [normalize_record(r) for r in records]


# Author: Mus mbayramo@stanford.edu
