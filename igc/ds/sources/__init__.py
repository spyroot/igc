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

__all__ = [
    "READ_METHODS",
    "SourceAdapter",
    "SourceRecord",
    "TrustLevel",
    "RedfishFixtureSource",
]

# Author: Mus mbayramo@stanford.edu
