"""
Filesystem source adapter for captured Redfish JSON corpora.

:class:`RedfishFixtureSource` reads a directory of per-resource Redfish JSON
files — the layout ``idrac_ctl`` discovery writes to ``~/.json_responses/<host>/``,
and the same one-file-per-resource layout used by the vendor fixture corpora
(``idrac_ctl/tests/{idrac,supermicro,hpe}_fixtures``) and the DMTF mockup tree —
and yields provenance-tagged :class:`SourceRecord` objects.

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
from typing import Dict, Iterator, List, Optional, Tuple

from igc.ds.sources.base import SourceAdapter, SourceRecord, TrustLevel


class RedfishFixtureSource(SourceAdapter):
    """Yield provenance-tagged records from a directory of Redfish JSON files.

    :param root: directory holding ``*.json`` resource captures (``~`` expanded).
    :param source: source label stamped on every record (e.g. ``"real_dell"``).
    :param trust_level: provenance tier for every record (see :class:`TrustLevel`).
    :param vendor: originating vendor when known (``dell`` / ``hpe`` / ``supermicro``).
    :param allowed_methods_map: optional ``{url: [methods]}`` taken from the
        ``allowed_methods_mapping`` half of the ``.npy`` contract; when given,
        methods are matched to a record by its canonical URL.
    :param glob_pattern: filename glob selecting captures (default ``*.json``).
    """

    def __init__(
        self,
        root: str,
        source: str,
        trust_level: TrustLevel,
        vendor: Optional[str] = None,
        allowed_methods_map: Optional[Dict[str, List[str]]] = None,
        glob_pattern: str = "*.json",
    ):
        super().__init__(module_name=f"RedfishFixtureSource[{source}]")
        self.root = Path(os.path.expanduser(root))
        self.source = source
        self.trust_level = trust_level
        self.vendor = vendor
        self._allowed = allowed_methods_map or {}
        self._glob = glob_pattern
        # Exposed after iteration for reporting / eval bookkeeping.
        self.num_emitted = 0
        self.num_skipped = 0

    @staticmethod
    def url_from_filename(name: str) -> str:
        """Reconstruct a Redfish URL from a discovery filename.

        idrac_ctl encodes ``/redfish/v1/Systems/1`` as
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
            url, url_source = self._resolve_url(body, path)
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
                provenance={"file": path.name, "url_from": url_source},
            )


# Author: Mus mbayramo@stanford.edu
