"""Source adapter for provider materialized redfish_ctl corpora.

The manifest is the authority for new Redfish corpora. This adapter never
assumes ``~/.json_responses``, an IP directory, a flat layout, or filename-based
URL reconstruction when the provider supplies ``@odata.id`` or a URL mapping.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping

from igc.ds.sources.base import SourceAdapter, SourceRecord, TrustLevel


SUPPORTED_MANIFEST_SCHEMA = "redfish-corpus-manifest/v1"


class RedfishCorpusManifestError(ValueError):
    """Raised when a redfish_ctl corpus manifest is invalid or unsupported."""


@dataclass(frozen=True)
class RedfishCtlCorpusManifest:
    """Validated provider corpus manifest metadata.

    :param path: manifest file path.
    :param schema: manifest schema version.
    :param root: materialized corpus root directory.
    :param corpus_id: stable corpus id.
    :param vendor: vendor label when known.
    :param model: platform model label when known.
    :param kind: provider-defined corpus kind, defaulting to ``full``.
    :param provider_revision: provider code/data revision.
    :param simulator_contract: simulator contract version paired with corpus.
    :param resources: provider-declared resource entries.
    :param action_capabilities: provider-declared simulatable actions.
    :param raw: raw manifest document for diagnostics.
    """
    path: Path
    schema: str
    root: Path
    corpus_id: str
    vendor: str | None
    model: str | None
    kind: str
    provider_revision: str
    simulator_contract: str
    resources: list[dict[str, Any]] = field(default_factory=list)
    action_capabilities: list[dict[str, Any]] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


def _schema_major(schema: str) -> str:
    """Return the manifest major version string."""
    match = re.match(r"^redfish-corpus-manifest/(?P<major>v\d+)(?:$|[.\-])", schema)
    if not match:
        raise RedfishCorpusManifestError(f"unsupported manifest schema: {schema!r}")
    return match.group("major")


def load_redfish_ctl_manifest(path: str | Path) -> RedfishCtlCorpusManifest:
    """Load and validate a redfish_ctl corpus manifest.

    :param path: manifest JSON path.
    :return: validated manifest dataclass.
    :raises RedfishCorpusManifestError: when required fields or versions fail.
    """
    manifest_path = Path(path)
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RedfishCorpusManifestError(f"cannot read redfish_ctl manifest {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise RedfishCorpusManifestError("redfish_ctl manifest must be a JSON object")
    schema = str(data.get("schema", ""))
    major = _schema_major(schema)
    if major != "v1":
        raise RedfishCorpusManifestError(
            f"unsupported manifest schema {schema!r}; expected {SUPPORTED_MANIFEST_SCHEMA}"
        )
    missing = [
        name
        for name in ("corpus_id", "provider_revision", "simulator_contract", "root")
        if not data.get(name)
    ]
    if missing:
        raise RedfishCorpusManifestError(
            f"redfish_ctl manifest missing required field(s): {', '.join(missing)}"
        )
    resources = data.get("resources") or []
    if not isinstance(resources, list):
        raise RedfishCorpusManifestError("redfish_ctl manifest field resources must be a list")
    actions = data.get("action_capabilities", data.get("actions", [])) or []
    if not isinstance(actions, list):
        raise RedfishCorpusManifestError(
            "redfish_ctl manifest field action_capabilities must be a list"
        )
    root = Path(str(data["root"]))
    if not root.is_absolute():
        root = manifest_path.parent / root
    return RedfishCtlCorpusManifest(
        path=manifest_path,
        schema=schema,
        root=root,
        corpus_id=str(data["corpus_id"]),
        vendor=data.get("vendor"),
        model=data.get("model"),
        kind=str(data.get("kind", "full")),
        provider_revision=str(data["provider_revision"]),
        simulator_contract=str(data["simulator_contract"]),
        resources=[dict(item) for item in resources],
        action_capabilities=[dict(item) for item in actions],
        raw=data,
    )


class RedfishCtlManifestSource(SourceAdapter):
    """Yield records from a materialized provider manifest.

    :param manifest_path: provider manifest path.
    :param corpus_id: optional exact corpus id filter.
    :param vendor: optional exact vendor filter.
    :param model: optional exact platform model filter.
    :param kind: exact corpus kind filter; defaults to ``full``.
    """

    _CONTROL_NAMES = {"manifest.json", "provider_manifest.json", "rest_api_map.v1.json"}

    def __init__(
        self,
        manifest_path: str | Path,
        corpus_id: str | None = None,
        vendor: str | None = None,
        model: str | None = None,
        kind: str | None = "full",
    ):
        manifest = load_redfish_ctl_manifest(manifest_path)
        super().__init__(module_name=f"RedfishCtlManifestSource[{manifest.corpus_id}]")
        self.manifest = manifest
        self.source = f"redfish_ctl:{manifest.corpus_id}"
        self.trust_level = TrustLevel.REAL
        self.num_emitted = 0
        self.num_skipped = 0
        if not manifest.root.is_dir():
            raise RedfishCorpusManifestError(
                f"redfish_ctl manifest root does not exist: {manifest.root}"
            )
        self._validate_filter("corpus_id", corpus_id, manifest.corpus_id)
        self._validate_filter("vendor", vendor, manifest.vendor)
        self._validate_filter("model", model, manifest.model)
        self._validate_filter("kind", kind, manifest.kind)
        self._file_to_url, self._allowed_methods, self._mapping_format = self._load_provider_map()

    @staticmethod
    def _validate_filter(field_name: str, expected: str | None, actual: str | None) -> None:
        """Validate an optional exact-match filter."""
        if expected is not None and expected != actual:
            raise RedfishCorpusManifestError(
                f"redfish_ctl manifest {field_name} mismatch: expected {expected!r}, got {actual!r}"
            )

    def _relative_mapping_path(self, file_path: str) -> str:
        """Normalize provider map file paths relative to the materialized root."""
        path = Path(file_path)
        if path.is_absolute():
            try:
                path = path.relative_to(self.manifest.root)
            except ValueError:
                return path.as_posix()
        return path.as_posix()

    def _load_provider_map(self) -> tuple[dict[str, str], dict[str, list[str]], str | None]:
        """Read provider URL/method maps, preferring ``rest_api_map.v1.json``."""
        path = self.manifest.root / "rest_api_map.v1.json"
        mapping_format = "rest_api_map.v1.json"
        if not path.is_file():
            path = self.manifest.root / "rest_api_map.npy"
            mapping_format = "rest_api_map.npy"
        if not path.is_file():
            return {}, {}, None
        try:
            if path.suffix == ".npy":
                import numpy as np

                data = np.load(path, allow_pickle=True).item()
            else:
                data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise RedfishCorpusManifestError(f"cannot read {path}: {exc}") from exc
        if not isinstance(data, dict):
            raise RedfishCorpusManifestError(f"provider map must be an object: {path}")
        url_to_file = data.get("url_file_mapping") or {}
        allowed = data.get("allowed_methods_mapping") or {}
        file_to_url: dict[str, str] = {}
        for url, file_path in dict(url_to_file).items():
            relative_file = self._relative_mapping_path(str(file_path))
            existing = file_to_url.get(relative_file)
            if existing is not None and existing != str(url):
                raise RedfishCorpusManifestError(
                    f"ambiguous provider map: {relative_file!r} maps to both "
                    f"{existing!r} and {str(url)!r}"
                )
            file_to_url[relative_file] = str(url)
        allowed_methods = {
            str(url): [str(method).upper() for method in methods]
            for url, methods in dict(allowed).items()
            if isinstance(methods, list)
        }
        return file_to_url, allowed_methods, mapping_format

    def _candidate_entries(self) -> list[tuple[Path, Mapping[str, Any]]]:
        """Return provider-declared resources or safe recursive JSON candidates."""
        if self.manifest.resources:
            return [
                (self.manifest.root / str(entry["file"]), entry)
                for entry in self.manifest.resources
                if entry.get("file")
            ]
        entries: list[tuple[Path, Mapping[str, Any]]] = []
        for path in sorted(self.manifest.root.rglob("*.json")):
            rel = path.relative_to(self.manifest.root).as_posix()
            name = path.name
            if name in self._CONTROL_NAMES or name.endswith(".control.json"):
                continue
            entries.append((path, {"file": rel}))
        return entries

    def _resolve_url(
        self,
        body: Mapping[str, Any],
        relative_file: str,
        entry: Mapping[str, Any],
    ) -> tuple[str | None, str]:
        """Resolve URL without filename reconstruction."""
        odata = body.get("@odata.id")
        if isinstance(odata, str) and odata:
            return odata, "odata"
        provider_url = entry.get("url") or self._file_to_url.get(relative_file)
        if isinstance(provider_url, str) and provider_url:
            return provider_url, "provider_map"
        return None, "missing"

    def iter_records(self) -> Iterator[SourceRecord]:
        """Yield manifest-backed Redfish observations.

        :return: iterator over :class:`SourceRecord`.
        """
        self.num_emitted = 0
        self.num_skipped = 0
        seen_urls: set[str] = set()
        for path, entry in self._candidate_entries():
            try:
                body = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                self.num_skipped += 1
                self.logger.debug(f"skip {path}: {exc}")
                continue
            if not isinstance(body, dict):
                self.num_skipped += 1
                continue
            relative_file = path.relative_to(self.manifest.root).as_posix()
            url, url_from = self._resolve_url(body, relative_file, entry)
            if url is None:
                self.num_skipped += 1
                continue
            if url in seen_urls:
                raise RedfishCorpusManifestError(
                    f"duplicate URL in redfish_ctl corpus manifest: {url}"
                )
            seen_urls.add(url)
            allowed = entry.get("allowed_methods") or self._allowed_methods.get(url)
            self.num_emitted += 1
            yield SourceRecord(
                url=url,
                response=dict(body),
                source=self.source,
                trust_level=self.trust_level,
                method=str(entry.get("method", "GET")).upper(),
                allowed_methods=[str(method).upper() for method in allowed] if allowed else None,
                vendor=self.manifest.vendor,
                schema_version=str(body.get("@odata.type", "")),
                provenance={
                    "file": relative_file,
                    "url_from": url_from,
                    "corpus_id": self.manifest.corpus_id,
                    "provider_revision": self.manifest.provider_revision,
                    "simulator_contract": self.manifest.simulator_contract,
                    "manifest_schema": self.manifest.schema,
                    "mapping_format": self._mapping_format,
                },
            )


# Author: Mus mbayramo@stanford.edu
