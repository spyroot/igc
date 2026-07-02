"""
Provenance-tagged data sources for the IGC training pipeline.

Exposes the source contract (:class:`SourceAdapter`, :class:`SourceRecord`,
:class:`TrustLevel`) and the offline filesystem adapter
(:class:`RedfishFixtureSource`) that reads captured Redfish JSON corpora — real
vendor captures, the DMTF mockup replay tree, or ``~/.json_responses`` — into a
single trust-tagged record stream.

Author:
Mus mbayramo@stanford.edu
"""
from igc.ds.sources.base import (
    READ_METHODS,
    SourceAdapter,
    SourceRecord,
    TrustLevel,
)
from igc.ds.sources.redfish_fixture_source import RedfishFixtureSource
from igc.ds.sources.mixer import DataManifest, SourceMix, unit_hash
from igc.ds.sources.training_object import (
    TrainingExample,
    compact_resource,
    normalize,
    normalize_record,
)
from igc.ds.sources.corpus_io import (
    iter_examples,
    read_examples,
    read_manifest,
    write_corpus,
    write_examples,
    write_manifest,
)

__all__ = [
    "READ_METHODS",
    "SourceAdapter",
    "SourceRecord",
    "TrustLevel",
    "RedfishFixtureSource",
    "SourceMix",
    "DataManifest",
    "unit_hash",
    "TrainingExample",
    "compact_resource",
    "normalize",
    "normalize_record",
    "write_examples",
    "iter_examples",
    "read_examples",
    "write_manifest",
    "read_manifest",
    "write_corpus",
]

# Author: Mus mbayramo@stanford.edu
