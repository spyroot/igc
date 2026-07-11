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

    source: str
    trust_level: TrustLevel
    schema_version: str
    resource_graph_before: Dict[str, Dict[str, Any]]
    request_or_action: Dict[str, Any]
    response: Dict[str, Any]
    resource_graph_after: Dict[str, Dict[str, Any]]
    allowed_methods: List[str]
    expected_semantics: Dict[str, Any]
    vendor: Optional[str]
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dict representation of the training example.

        ``trust_level`` is emitted as its ``.name`` string (e.g. ``"REAL"``); all other
        fields are returned as-is.

        :return: Dict suitable for JSON serialization.
        """
        return {
            "source": self.source,
            "trust_level": self.trust_level.name,
            "schema_version": self.schema_version,
            "resource_graph_before": self.resource_graph_before,
            "request_or_action": self.request_or_action,
            "response": self.response,
            "resource_graph_after": self.resource_graph_after,
            "allowed_methods": self.allowed_methods,
            "expected_semantics": self.expected_semantics,
            "vendor": self.vendor,
            "provenance": self.provenance,
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
        "method": method_upper,
        "mutating": mutating,
        "read_only": not mutating,
        "idempotent": method_upper in ("GET", "HEAD", "PUT", "DELETE"),
        "expected_status": 200,
        "mutable_endpoint": any(
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
