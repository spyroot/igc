"""
Provenance-tagged data sources for the IGC training pipeline.

Exposes the source contract (:class:`SourceAdapter`, :class:`SourceRecord`,
:class:`TrustLevel`) and the offline filesystem adapter
(:class:`RedfishFixtureSource`) that reads captured Redfish JSON corpora — real
vendor captures, the DMTF mockup replay tree, or ``~/.json_responses`` — into a
single trust-tagged record stream, plus the enum-space extraction layer
(:class:`EnumSpaceIndex` and friends) that turns captured BIOS registries,
ActionInfo bodies, and inline allowable-value annotations into the per-slot
``arg_schema`` enum spaces the stage-2 argument decoder scores over.

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
from igc.ds.sources.redfish_ctl_manifest import (
    RedfishCorpusManifestError,
    RedfishCtlCorpusManifest,
    RedfishCtlManifestSource,
    load_redfish_ctl_manifest,
)
from igc.ds.sources.mixer import DataManifest, SourceMix, unit_hash
from igc.ds.sources.redfish_enum_space import (
    EnumSlot,
    EnumSpaceIndex,
    ResourceKind,
    classify_resource,
    normalize_enriched,
    slots_from_action_info,
    slots_from_attribute_registry,
    slots_from_inline_annotations,
    to_arg_schema,
)
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
    "RedfishCorpusManifestError",
    "RedfishCtlCorpusManifest",
    "RedfishCtlManifestSource",
    "load_redfish_ctl_manifest",
    "EnumSlot",
    "EnumSpaceIndex",
    "ResourceKind",
    "classify_resource",
    "normalize_enriched",
    "slots_from_action_info",
    "slots_from_attribute_registry",
    "slots_from_inline_annotations",
    "to_arg_schema",
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
