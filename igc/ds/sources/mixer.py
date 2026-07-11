"""
Multi-source mixer with a deterministic, trust-tier-aware train/eval split.

Composes several provenance-tagged Redfish data sources (real vendor captures, the DMTF
mockup replay tree, vendor/synthetic emulators) into a single training corpus. The eval split
is drawn ONLY from the highest-trust tier, so real captures serve as held-out ground truth
while the more synthetic tiers always feed training for coverage. The split is deterministic —
a stable hash of each resource URL rather than an RNG — so it reproduces across runs and stays
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
from dataclasses import dataclass
from typing import Dict, List, Tuple

from igc.ds.sources.base import SourceAdapter, SourceRecord, TrustLevel


def unit_hash(key: str, seed: int) -> float:
    """Map ``(seed, key)`` to a stable float in ``[0.0, 1.0)``.

    Uses blake2b so the value is identical across processes and runs (unlike the builtin
    ``hash()``); the first 8 digest bytes are read big-endian and divided by ``2 ** 64``.

    :param key: the string to hash (a resource URL).
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

    def content_hash(self) -> str:
        """Return a stable 16-hex-char hash over the manifest fields.

        Two manifests with equal contents hash equal, so this can key reproducibility and
        fair-comparison checks. Field (and nested-dict) order is canonicalized via sorted-key
        JSON before hashing.

        :return: a 16-character hex digest.
        """
        payload = json.dumps(self.__dict__, sort_keys=True, default=str)
        return hashlib.blake2b(payload.encode("utf-8"), digest_size=8).hexdigest()

    def eval_split_id(self) -> str:
        """Return a stable id of the eval-split policy (floor / fraction / seed).

        Distinct from :meth:`content_hash` (which identifies the exact record mix): this
        identifies HOW the split was drawn, so two runs over the same corpus with the same
        policy share an ``eval_split`` id.

        :return: e.g. ``"floor=REAL:frac=0.15:seed=0"``.
        """
        return f"floor={self.eval_trust_floor}:frac={self.eval_fraction}:seed={self.seed}"

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
    :param dedup: when true, keep one record per URL (highest trust; first-seen on a tie).
    :raises ValueError: if ``eval_fraction`` is outside ``[0.0, 1.0]``.
    """

    def __init__(self, adapters: List[SourceAdapter], *, eval_fraction: float = 0.15,
                 eval_trust_floor: TrustLevel = TrustLevel.REAL, seed: int = 0,
                 dedup: bool = True):
        if not 0.0 <= eval_fraction <= 1.0:
            raise ValueError(f"eval_fraction must be in [0.0, 1.0], got {eval_fraction}")
        self._adapters = list(adapters)
        self.eval_fraction = eval_fraction
        self.eval_trust_floor = eval_trust_floor
        self.seed = seed
        self.dedup = dedup
        self._records_cache: List[SourceRecord] = None

    def records(self) -> List[SourceRecord]:
        """Collect all records from every adapter, optionally deduped by URL.

        With dedup on, a URL seen in several sources collapses to its highest-trust copy
        (first-seen wins a trust tie, so adapter order is the tie-break); the record keeps its
        first-appearance position. The result is cached for repeated calls.

        :return: the composed list of records.
        """
        if self._records_cache is not None:
            return self._records_cache

        if not self.dedup:
            self._records_cache = [rec for adapter in self._adapters for rec in adapter.iter_records()]
            return self._records_cache

        best: Dict[str, SourceRecord] = {}
        order: List[str] = []
        for adapter in self._adapters:
            for rec in adapter.iter_records():
                current = best.get(rec.url)
                if current is None:
                    best[rec.url] = rec
                    order.append(rec.url)
                elif rec.trust_level > current.trust_level:
                    best[rec.url] = rec
        self._records_cache = [best[url] for url in order]
        return self._records_cache

    def split(self) -> Tuple[List[SourceRecord], List[SourceRecord]]:
        """Partition the corpus into ``(train, eval)`` deterministically.

        A record is held out for eval iff it is at or above ``eval_trust_floor`` AND
        ``unit_hash(url, seed) < eval_fraction``; everything else trains. Input order is
        preserved within each list.

        :return: ``(train_records, eval_records)``.
        """
        train: List[SourceRecord] = []
        held_out: List[SourceRecord] = []
        for rec in self.records():
            eligible = rec.trust_level >= self.eval_trust_floor
            if eligible and unit_hash(rec.url, self.seed) < self.eval_fraction:
                held_out.append(rec)
            else:
                train.append(rec)
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
        )


# Author: Mus mbayramo@stanford.edu
