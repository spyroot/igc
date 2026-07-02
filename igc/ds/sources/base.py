"""
Data-source contracts for the IGC training pipeline.

A :class:`SourceAdapter` turns a corpus of Redfish observations — real captures
under ``~/.json_responses`` (written by ``idrac_ctl`` discovery), the DMTF
mockup replay tree, a vendor emulator, or a synthetic generator — into a stream
of provenance-tagged :class:`SourceRecord` objects. Every record carries where
it came from (``source``) and how much it can be trusted (:class:`TrustLevel`),
so the training/eval split can hold out real data as ground truth and weight or
filter the more synthetic tiers.

The trust ordering (real > replay > sim-vendor > sim-generic > sim-drift)
mirrors the data-provenance design: real captures validate semantics, while the
lower tiers add coverage and controlled variation for generalization.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional

from igc.modules.base.igc_abstract_logger import AbstractLogger


class TrustLevel(enum.IntEnum):
    """How much a record's response can be trusted as ground truth.

    Ordered high-to-low so callers can compare (``rec.trust_level >=
    TrustLevel.REAL``) and split evaluation by tier. ``REAL`` is captured from a
    live management controller; the ``SIM_*`` tiers are progressively more
    synthetic.
    """
    REAL = 5          # captured from a real management controller
    REPLAY = 4        # canonical DMTF mockup / recorded cassette replay
    SIM_VENDOR = 3    # vendor-flavored emulator (sushy + vendor capabilities)
    SIM_GENERIC = 2   # generic emulator / schema-driven synthesis
    SIM_DRIFT = 1     # deliberately perturbed / adversarial variation


# GET/HEAD are the safe, non-mutating observation methods; a captured fixture is
# always a response to one of these.
READ_METHODS = ("GET", "HEAD")


@dataclass
class SourceRecord:
    """One provenance-tagged Redfish observation.

    :param url: canonical resource URL (Redfish ``@odata.id`` when available).
    :param response: the decoded JSON body for the resource.
    :param source: source label, e.g. ``"real_dell"`` / ``"dmtf_mockup"``.
    :param trust_level: provenance tier (see :class:`TrustLevel`).
    :param method: HTTP method the response answers (captures are ``GET``).
    :param allowed_methods: methods the endpoint permits, when known (from the
        ``allowed_methods_mapping`` half of the ``.npy`` contract); else ``None``.
    :param vendor: originating vendor when known (``dell`` / ``hpe`` / ``supermicro``).
    :param schema_version: Redfish ``@odata.type`` of the resource, or ``""``.
    :param provenance: free-form trace (source file, how the URL was derived).
    """
    url: str
    response: Dict
    source: str
    trust_level: TrustLevel
    method: str = "GET"
    allowed_methods: Optional[List[str]] = None
    vendor: Optional[str] = None
    schema_version: str = ""
    provenance: Dict = field(default_factory=dict)


class SourceAdapter(AbstractLogger, ABC):
    """A stream of :class:`SourceRecord` from one data source.

    Concrete adapters (fixture directory, mockup replay, emulator, synthetic)
    implement only :meth:`iter_records`; the pipeline consumes the tagged stream
    uniformly regardless of tier. Inherits :class:`AbstractLogger` so adapters
    share the project's logging facility.
    """

    #: label attached to every emitted record (e.g. ``"real_hpe"``).
    source: str
    #: provenance tier attached to every emitted record.
    trust_level: TrustLevel

    @abstractmethod
    def iter_records(self) -> Iterator[SourceRecord]:
        """Yield provenance-tagged records for this source.

        :return: iterator over :class:`SourceRecord`.
        """
        raise NotImplementedError

    def __iter__(self) -> Iterator[SourceRecord]:
        """Iterate the source, delegating to :meth:`iter_records`."""
        return self.iter_records()


# Author: Mus mbayramo@stanford.edu
