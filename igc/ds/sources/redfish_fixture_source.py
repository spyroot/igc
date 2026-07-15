"""
Filesystem source adapter for captured Redfish JSON corpora.

:class:`RedfishFixtureSource` reads a directory of per-resource Redfish JSON
files, including a dataset artifact materialized by ``redfish_ctl corpus materialize``.
It also supports older capture roots such as ``~/.json_responses/<host>/`` and
the one-file-per-resource vendor fixture corpora, and yields provenance-tagged
:class:`SourceRecord` objects.

It keys each record on the resource's canonical ``@odata.id`` (present on the
large majority of captures and vendor-neutral); when a file lacks one it falls
back to reconstructing the URL from the discovery filename and records which
route it took in ``provenance``.

This adapter is offline by design: it consumes already-captured JSON and never
touches a live controller.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from igc.ds.sources.base import SourceAdapter, SourceRecord, TrustLevel


class RedfishFixtureSource(SourceAdapter):
    """Yield provenance-tagged records from a directory of Redfish JSON files.

    :param root: directory holding resource captures (``~`` expanded).
    :param source: source label stamped on every record (e.g. ``"real_dell"``).
    :param trust_level: provenance tier for every record (see :class:`TrustLevel`).
    :param vendor: originating vendor when known (``dell`` / ``hpe`` / ``supermicro``).
    :param allowed_methods_map: optional ``{url: [methods]}`` taken from the
        ``allowed_methods_mapping`` half of the ``.npy`` contract; when given,
        methods are matched to a record by its canonical URL.
    :param glob_pattern: filename glob selecting captures (default ``**/*.json``).
    :param corpus_id: stable corpus identity for provenance; defaults to ``source``.
    :param capture_id: optional capture identity from the manifest.
    :param model: optional hardware model from the manifest.
    """

    def __init__(
        self,
        root: str,
        source: str,
        trust_level: TrustLevel,
        vendor: Optional[str] = None,
        allowed_methods_map: Optional[Dict[str, List[str]]] = None,
        glob_pattern: str = "**/*.json",
        corpus_id: Optional[str] = None,
        capture_id: Optional[str] = None,
        model: Optional[str] = None,
    ):
        super().__init__(module_name=f"RedfishFixtureSource[{source}]")
        self.root = Path(os.path.expanduser(root))
        self.source = source
        self.trust_level = trust_level
        self.vendor = vendor
        self.corpus_id = corpus_id or source
        self.capture_id = capture_id
        self.model = model
        self._allowed = allowed_methods_map or {}
        self._glob = glob_pattern
        # Exposed after iteration for reporting / eval bookkeeping.
        self.num_emitted = 0
        self.num_skipped = 0

    @staticmethod
    def _archive_slug(entry: Dict) -> str:
        """Return the materialized directory slug for one redfish_ctl manifest row."""
        archive = entry.get("archive") or entry.get("archive_path")
        if isinstance(archive, str) and archive:
            name = Path(archive).name
            if name.endswith(".tar.gz"):
                return name[:-7]
            return Path(name).stem
        vendor = str(entry.get("vendor", "")).strip()
        model = str(entry.get("model", "")).strip()
        return f"{vendor}_{model}".strip("_")

    @staticmethod
    def _normalize_kind(kind: str) -> str:
        """Accept the former selector name as the dataset artifact kind."""
        return {"full": "dataset"}.get(str(kind).lower(), str(kind).lower())

    @staticmethod
    def _load_allowed_methods(root: Path) -> Dict[str, List[str]]:
        """Load the per-capture allowed-method map from portable JSON or legacy NPY."""
        portable = root / "rest_api_map.v1.json"
        if portable.is_file():
            with portable.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            methods = data.get("allowed_methods_mapping", {})
            return methods if isinstance(methods, dict) else {}

        legacy = root / "rest_api_map.npy"
        if legacy.is_file():
            import numpy as np

            data = np.load(legacy, allow_pickle=True).item()
            methods = data.get("allowed_methods_mapping", {})
            return methods if isinstance(methods, dict) else {}
        return {}

    @classmethod
    def from_redfish_ctl_manifest(
        cls,
        manifest_path: str,
        materialized_root: str,
        trust_level: TrustLevel = TrustLevel.REAL,
        kind: str = "dataset",
        corpus_ids: Optional[Sequence[str]] = None,
    ) -> List["RedfishFixtureSource"]:
        """Create sources for a materialized redfish_ctl corpus manifest.

        :param manifest_path: path to ``corpora/manifest.v1.json``.
        :param materialized_root: output root from ``redfish_ctl corpus materialize``.
        :param trust_level: provenance tier stamped on emitted records.
        :param kind: corpus kind to consume, normally ``"dataset"``.
        :param corpus_ids: optional allow-list of manifest IDs.
        :return: one source per selected, materialized manifest row.
        """
        manifest_file = Path(os.path.expanduser(manifest_path))
        materialized = Path(os.path.expanduser(materialized_root))
        with manifest_file.open("r", encoding="utf-8") as fh:
            manifest = json.load(fh)
        entries = manifest.get("corpora", [])
        selected = set(corpus_ids or [])
        normalized_kind = cls._normalize_kind(kind)
        sources: List[RedfishFixtureSource] = []
        for entry in entries:
            corpus_id = str(entry.get("id") or entry.get("corpus_id") or "")
            entry_kind = cls._normalize_kind(str(entry.get("kind", "")))
            if entry_kind != normalized_kind:
                continue
            if selected and corpus_id not in selected:
                continue
            slug = cls._archive_slug(entry)
            root = materialized / normalized_kind / slug
            if not root.is_dir():
                root = materialized / slug
            sources.append(
                cls(
                    str(root),
                    corpus_id or slug,
                    trust_level,
                    vendor=entry.get("vendor"),
                    allowed_methods_map=cls._load_allowed_methods(root),
                    corpus_id=corpus_id or slug,
                    capture_id=entry.get("capture_id"),
                    model=entry.get("model"),
                )
            )
        return sources

    @staticmethod
    def url_from_filename(name: str) -> str:
        """Reconstruct a Redfish URL from a discovery filename.

        redfish_ctl discovery encodes ``/redfish/v1/Systems/1`` as
        ``_redfish_v1_Systems_1.json``. This is a best-effort fallback used only
        when a file carries no ``@odata.id`` — segment IDs that themselves
        contain underscores cannot be recovered unambiguously this way.

        :param name: capture filename (with or without the ``.json`` suffix).
        :return: reconstructed URL path beginning with ``/``.
        """
        stem = name[:-5] if name.endswith(".json") else name
        return "/" + stem.strip("_").replace("_", "/")

    def _resolve_url(self, body: Dict, path: Path) -> Tuple[str, str]:
        """Return ``(url, url_source)`` preferring the canonical ``@odata.id``.

        :param body: decoded JSON resource.
        :param path: file the resource came from.
        :return: ``(url, "odata")`` when an ``@odata.id`` is present, otherwise
            ``(url, "filename")`` from :meth:`url_from_filename`.
        """
        odata = body.get("@odata.id")
        if isinstance(odata, str) and odata:
            return odata, "odata"
        return self.url_from_filename(path.name), "filename"

    @staticmethod
    def _is_metadata_json(path: Path, body: Dict) -> bool:
        """Return true for corpus metadata sidecars that are not Redfish resources."""
        if "@odata.id" in body:
            return False
        return path.name in {
            "corpus.json",
            "manifest.json",
            "manifest.v1.json",
            "rest_api_map.v1.json",
        }

    def iter_records(self) -> Iterator[SourceRecord]:
        """Walk the corpus and yield one record per parsable resource object.

        A missing directory yields nothing (a warning is logged, not raised, so
        a partial multi-vendor corpus still trains). Unparsable files and
        non-object JSON (e.g. a bare array) are counted in ``num_skipped`` and
        skipped. ``num_emitted`` / ``num_skipped`` are valid after iteration.

        :return: iterator over :class:`SourceRecord`.
        """
        self.num_emitted = 0
        self.num_skipped = 0
        if not self.root.is_dir():
            self.logger.warning(f"no such directory: {self.root}")
            return
        for path in sorted(self.root.glob(self._glob)):
            try:
                with open(path, "r") as fh:
                    body = json.load(fh)
            except (OSError, json.JSONDecodeError) as parse_err:
                self.num_skipped += 1
                self.logger.debug(f"skip {path.name}: {parse_err}")
                continue
            if not isinstance(body, dict):
                self.num_skipped += 1
                continue
            if self._is_metadata_json(path, body):
                self.num_skipped += 1
                continue
            url, url_source = self._resolve_url(body, path)
            rel_path = path.relative_to(self.root).as_posix()
            provenance = {
                "file": rel_path,
                "url_from": url_source,
                "corpus_id": self.corpus_id,
            }
            if self.capture_id:
                provenance["capture_id"] = self.capture_id
            if self.model:
                provenance["model"] = self.model
            self.num_emitted += 1
            yield SourceRecord(
                url=url,
                response=body,
                source=self.source,
                trust_level=self.trust_level,
                method="GET",
                allowed_methods=self._allowed.get(url),
                vendor=self.vendor,
                schema_version=str(body.get("@odata.type", "")),
                provenance=provenance,
            )


# Author: Mus mbayramo@stanford.edu
