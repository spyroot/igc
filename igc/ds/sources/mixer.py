"""
Multi-source mixer with a deterministic, trust-tier-aware train/eval split.

Composes several provenance-tagged Redfish data sources (real vendor captures, the DMTF
mockup replay tree, vendor/synthetic emulators) into a single training corpus. The eval split
is drawn ONLY from the highest-trust tier, so real captures serve as held-out ground truth
while the more synthetic tiers always feed training for coverage. The split is deterministic —
a stable hash of each source-qualified resource URL rather than an RNG — so it reproduces across runs and stays
stable as the corpus grows, and a serializable :class:`DataManifest` records the mix (for the
training run manifest and as a fair-comparison key).

The ``DataManifest`` produced here is serialized by ``corpus_io.write_corpus`` as the
``manifest.json`` sidecar and read back by ``CorpusJSONLDataset.run_manifest_fields``
(``igc/ds/corpus_dataset.py``) to stamp each run report's ``data_manifest`` / ``eval_split``
fair-comparison keys — reached in production when ``--corpus_dir`` is set.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from igc.ds.sources.base import SourceAdapter, SourceRecord, TrustLevel


def unit_hash(key: str, seed: int) -> float:
    """Map ``(seed, key)`` to a stable float in ``[0.0, 1.0)``.

    Uses blake2b so the value is identical across processes and runs (unlike the builtin
    ``hash()``); the first 8 digest bytes are read big-endian and divided by ``2 ** 64``.

    :param key: the string to hash (normally a source-qualified resource URL).
    :param seed: split seed, mixed into the digest.
    :return: a deterministic float in ``[0.0, 1.0)``.
    """
    digest = hashlib.blake2b(f"{seed}:{key}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") / 2 ** 64


@dataclass
class DataManifest:
    """A serializable summary of a composed corpus and its train/eval split.

    :param total: total records after dedup.
    :param train_count: records assigned to training.
    :param eval_count: records held out for evaluation.
    :param by_source: record count per source label.
    :param by_trust: record count per :class:`TrustLevel` name.
    :param by_vendor: record count per vendor (``"unknown"`` when unset).
    :param eval_trust_floor: name of the lowest trust tier eligible for eval.
    :param eval_fraction: fraction of eligible records held out for eval.
    :param seed: the split seed.
    :param sources: sorted unique source labels.
    :param train_row_ids: stable, non-URL identifiers assigned to training.
    :param heldout_row_ids: stable, non-URL identifiers reserved for evaluation.
    :param required_heldout_sources: sorted source labels whose trust tier makes
        them eligible for held-out evaluation, including a source that happened
        to receive zero rows under the deterministic split.
    :param heldout_by_source: observed held-out row count per source label.
    :param source_registry_sha: exact source-registry spec identity, when used.
    :param source_manifest_shas: exact upstream manifest identity per registry source.
    :param min_eval_per_source: minimum held-out rows per eligible source.
    :param phase1_transform: lossless chunk transform and original-resource lineage,
        when this manifest describes a materialized Phase 1 registry corpus.
    """
    total: int
    train_count: int
    eval_count: int
    by_source: Dict[str, int]
    by_trust: Dict[str, int]
    by_vendor: Dict[str, int]
    eval_trust_floor: str
    eval_fraction: float
    seed: int
    sources: List[str]
    train_row_ids: List[str]
    heldout_row_ids: List[str]
    required_heldout_sources: List[str] = field(default_factory=list)
    heldout_by_source: Dict[str, int] = field(default_factory=dict)
    source_registry_sha: str = ""
    source_manifest_shas: Dict[str, str] = field(default_factory=dict)
    min_eval_per_source: int = 0
    phase1_transform: Dict[str, object] = field(default_factory=dict)

    def content_hash(self) -> str:
        """Return a canonical SHA-256 identity over the manifest fields.

        Two manifests with equal contents hash equal, so this can key reproducibility and
        fair-comparison checks. Field (and nested-dict) order is canonicalized via sorted-key
        JSON before hashing.

        :return: a ``sha256:<64 lowercase hex>`` digest.
        """
        fields = dict(self.__dict__)
        if not fields["phase1_transform"]:
            fields.pop("phase1_transform")
        payload = json.dumps(fields, sort_keys=True, default=str)
        return f"sha256:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"

    def eval_split_id(self) -> str:
        """Return a SHA-256 identity for the exact held-out row-id set.

        Distinct from :meth:`content_hash` (which identifies the exact record mix): this
        identifies HOW the split was drawn, so two runs over the same corpus with the same
        policy share an ``eval_split`` id.

        :return: a ``sha256:<64 lowercase hex>`` digest.
        """
        payload = json.dumps(self.heldout_row_ids, sort_keys=True, separators=(",", ":"))
        return f"sha256:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"

    def to_run_manifest_fields(self) -> Dict[str, str]:
        """Fields to populate a training ``RunManifest`` from this mix.

        A run records its exact data mix by constructing
        ``RunManifest(..., **manifest.to_run_manifest_fields())`` so the report bundle's
        fair-comparison check can tell whether two runs trained on the same data + split.

        :return: ``{"data_manifest": content_hash, "eval_split": eval_split_id}``.
        """
        return {"data_manifest": self.content_hash(), "eval_split": self.eval_split_id()}


class SourceMix:
    """Compose sources into one corpus with a deterministic trust-tier train/eval split.

    :param adapters: source adapters to combine, in priority order (earlier wins a dedup tie).
    :param eval_fraction: fraction of eval-eligible records held out (``0.0``-``1.0``).
    :param eval_trust_floor: only records at or above this tier are eval-eligible.
    :param seed: seed for the deterministic split hash.
    :param dedup: when true, keep one record per source/URL pair (highest trust;
        first-seen on a tie), preserving different platform observations of the
        same standard Redfish endpoint.
    :param min_eval_per_source: deterministic minimum held-out count for each
        eligible source; a smaller source is held out in full.
    :raises ValueError: if ``eval_fraction`` is outside ``[0.0, 1.0]``.
    """

    def __init__(self, adapters: List[SourceAdapter], *, eval_fraction: float = 0.15,
                 eval_trust_floor: TrustLevel = TrustLevel.REAL, seed: int = 0,
                 dedup: bool = True, min_eval_per_source: int = 0):
        if not 0.0 <= eval_fraction <= 1.0:
            raise ValueError(f"eval_fraction must be in [0.0, 1.0], got {eval_fraction}")
        if (
            not isinstance(min_eval_per_source, int)
            or isinstance(min_eval_per_source, bool)
            or min_eval_per_source < 0
        ):
            raise ValueError("min_eval_per_source must be a non-negative integer")
        self._adapters = list(adapters)
        self.eval_fraction = eval_fraction
        self.eval_trust_floor = eval_trust_floor
        self.seed = seed
        self.dedup = dedup
        self.min_eval_per_source = min_eval_per_source
        self._records_cache: List[SourceRecord] = None

    def records(self) -> List[SourceRecord]:
        """Collect all records from every adapter, optionally deduped by URL.

        With dedup on, repeated copies of one URL within the same source collapse
        to the highest-trust copy. Different sources retain their own observation
        of a shared standard endpoint. The result is cached for repeated calls.

        :return: the composed list of records.
        """
        if self._records_cache is not None:
            return self._records_cache

        if not self.dedup:
            self._records_cache = [rec for adapter in self._adapters for rec in adapter.iter_records()]
            return self._records_cache

        best: Dict[tuple[str, str], SourceRecord] = {}
        order: List[tuple[str, str]] = []
        for adapter in self._adapters:
            for rec in adapter.iter_records():
                key = (rec.source, rec.url)
                current = best.get(key)
                if current is None:
                    best[key] = rec
                    order.append(key)
                elif rec.trust_level > current.trust_level:
                    best[key] = rec
        self._records_cache = [best[key] for key in order]
        return self._records_cache

    def split(self) -> Tuple[List[SourceRecord], List[SourceRecord]]:
        """Partition the corpus into ``(train, eval)`` deterministically.

        Eligible records first use the stable hash threshold. When a source falls
        below ``min_eval_per_source``, the lowest-hash records extend its held-out
        set to the required count, capped by all available rows. Input order is
        preserved within each output list.

        :return: ``(train_records, eval_records)``.
        """
        records = self.records()
        heldout_keys = {
            _record_key(record)
            for record in records
            if record.trust_level >= self.eval_trust_floor
            and unit_hash(_record_key(record), self.seed) < self.eval_fraction
        }
        if self.min_eval_per_source:
            eligible_by_source: Dict[str, List[SourceRecord]] = {}
            for record in records:
                if record.trust_level >= self.eval_trust_floor:
                    eligible_by_source.setdefault(record.source, []).append(record)
            for source_records in eligible_by_source.values():
                selected = sum(
                    _record_key(record) in heldout_keys
                    for record in source_records
                )
                required = min(self.min_eval_per_source, len(source_records))
                if selected >= required:
                    continue
                ranked = sorted(
                    source_records,
                    key=lambda record: (
                        unit_hash(_record_key(record), self.seed),
                        _record_key(record),
                    ),
                )
                heldout_keys.update(
                    _record_key(record) for record in ranked[:required]
                )
        train: List[SourceRecord] = []
        held_out: List[SourceRecord] = []
        for record in records:
            if _record_key(record) in heldout_keys:
                held_out.append(record)
            else:
                train.append(record)
        return train, held_out

    def manifest(self) -> DataManifest:
        """Summarize the composed corpus and split into a :class:`DataManifest`.

        :return: the manifest — counts by source / trust / vendor and train/eval sizes.
        """
        records = self.records()
        train, held_out = self.split()
        by_source: Dict[str, int] = {}
        by_trust: Dict[str, int] = {}
        by_vendor: Dict[str, int] = {}
        for rec in records:
            by_source[rec.source] = by_source.get(rec.source, 0) + 1
            by_trust[rec.trust_level.name] = by_trust.get(rec.trust_level.name, 0) + 1
            vendor = rec.vendor or "unknown"
            by_vendor[vendor] = by_vendor.get(vendor, 0) + 1
        required_heldout_sources = sorted({
            record.source
            for record in records
            if record.trust_level >= self.eval_trust_floor
        })
        heldout_by_source: Dict[str, int] = {}
        for record in held_out:
            heldout_by_source[record.source] = (
                heldout_by_source.get(record.source, 0) + 1
            )
        return DataManifest(
            total=len(records),
            train_count=len(train),
            eval_count=len(held_out),
            by_source=by_source,
            by_trust=by_trust,
            by_vendor=by_vendor,
            eval_trust_floor=self.eval_trust_floor.name,
            eval_fraction=self.eval_fraction,
            seed=self.seed,
            sources=sorted(by_source.keys()),
            train_row_ids=[_record_id(record) for record in train],
            heldout_row_ids=[_record_id(record) for record in held_out],
            required_heldout_sources=required_heldout_sources,
            heldout_by_source=heldout_by_source,
            min_eval_per_source=self.min_eval_per_source,
        )


def _record_id(record: SourceRecord) -> str:
    """Return a stable non-secret row identifier for one source/url pair."""
    digest = hashlib.sha256(
        f"{record.source}\0{record.url}".encode("utf-8")
    ).hexdigest()
    return f"sha256:{digest}"


def _record_key(record: SourceRecord) -> str:
    """Return the source-qualified key used by the deterministic split."""
    return f"{record.source}\0{record.url}"


# Author: Mus mbayramo@stanford.edu
