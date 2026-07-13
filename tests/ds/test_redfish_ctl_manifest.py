"""Offline tests for redfish_ctl corpus manifest ingestion.

The manifest source is the only default Redfish corpus entry point planned for
new RL work. These tests pin recursive materialization rules, provenance,
schema negotiation, and URL precedence without requiring provider LFS data.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from igc.ds.sources.base import TrustLevel
from igc.ds.sources.redfish_ctl_manifest import (
    RedfishCorpusManifestError,
    RedfishCtlManifestSource,
    load_redfish_ctl_manifest,
)


def _write_json(path: Path, body: object) -> None:
    """Write JSON while creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body), encoding="utf-8")


def _manifest(root: str = "corpus", **extra: object) -> dict[str, object]:
    """Return the smallest valid corpus manifest."""
    doc: dict[str, object] = {
        "schema": "redfish-corpus-manifest/v1",
        "corpus_id": "xr8620t-full",
        "vendor": "dell",
        "model": "xr8620t",
        "kind": "full",
        "provider_revision": "redfish-ctl-abc123",
        "simulator_contract": "redfish-simulator/v1",
        "root": root,
    }
    doc.update(extra)
    return doc


def test_manifest_source_uses_odata_then_provider_mapping(tmp_path: Path) -> None:
    """@odata.id wins over filenames; provider map fills absent @odata.id."""
    corpus = tmp_path / "corpus"
    _write_json(
        corpus / "nested" / "wrong_name.json",
        {
            "@odata.id": "/redfish/v1/Systems/System.Embedded.1",
            "@odata.type": "#ComputerSystem.v1_20_0.ComputerSystem",
            "Id": "System.Embedded.1",
        },
    )
    _write_json(
        corpus / "nested" / "no_odata.json",
        {
            "@odata.type": "#Chassis.v1_25_0.Chassis",
            "Id": "Chassis.1",
        },
    )
    _write_json(
        corpus / "rest_api_map.v1.json",
        {
            "url_file_mapping": {
                "/redfish/v1/Systems/System.Embedded.1": "nested/wrong_name.json",
                "/redfish/v1/Chassis/Chassis.1": "nested/no_odata.json",
            },
            "allowed_methods_mapping": {
                "/redfish/v1/Systems/System.Embedded.1": ["GET", "PATCH"],
                "/redfish/v1/Chassis/Chassis.1": ["GET", "HEAD"],
            },
        },
    )
    manifest_path = tmp_path / "manifest.json"
    _write_json(manifest_path, _manifest())

    records = list(RedfishCtlManifestSource(manifest_path).iter_records())
    by_url = {record.url: record for record in records}

    assert sorted(by_url) == [
        "/redfish/v1/Chassis/Chassis.1",
        "/redfish/v1/Systems/System.Embedded.1",
    ]
    assert by_url["/redfish/v1/Systems/System.Embedded.1"].provenance["url_from"] == "odata"
    assert by_url["/redfish/v1/Chassis/Chassis.1"].provenance["url_from"] == "provider_map"
    assert by_url["/redfish/v1/Systems/System.Embedded.1"].allowed_methods == ["GET", "PATCH"]
    assert by_url["/redfish/v1/Chassis/Chassis.1"].allowed_methods == ["GET", "HEAD"]
    assert all(record.source == "redfish_ctl:xr8620t-full" for record in records)
    assert all(record.trust_level is TrustLevel.REAL for record in records)


def test_manifest_duplicate_observation_urls_fail_clearly(tmp_path: Path) -> None:
    """Two resource files for the same URL are a corpus contract error."""
    corpus = tmp_path / "corpus"
    _write_json(corpus / "one.json", {"@odata.id": "/redfish/v1/Systems/1"})
    _write_json(corpus / "two.json", {"@odata.id": "/redfish/v1/Systems/1"})
    manifest_path = tmp_path / "manifest.json"
    _write_json(manifest_path, _manifest())

    source = RedfishCtlManifestSource(manifest_path)
    with pytest.raises(RedfishCorpusManifestError, match="duplicate URL"):
        list(source.iter_records())


def test_manifest_source_skips_control_json_during_recursive_discovery(tmp_path: Path) -> None:
    """Manifest and map files are control metadata, never observations."""
    corpus = tmp_path / "materialized"
    _write_json(corpus / "ServiceRoot.json", {"@odata.id": "/redfish/v1", "Id": "Root"})
    _write_json(corpus / "manifest.json", {"not": "an observation"})
    _write_json(corpus / "rest_api_map.v1.json", {"url_file_mapping": {}})
    _write_json(corpus / "nested" / "metadata.control.json", {"skip": True})
    manifest_path = tmp_path / "provider_manifest.json"
    _write_json(manifest_path, _manifest(root="materialized"))

    records = list(RedfishCtlManifestSource(manifest_path).iter_records())

    assert [record.url for record in records] == ["/redfish/v1"]
    assert records[0].provenance["file"] == "ServiceRoot.json"


def test_manifest_legacy_npy_map_is_fallback_only(tmp_path: Path) -> None:
    """Legacy rest_api_map.npy is used only when v1 JSON map is absent."""
    corpus = tmp_path / "corpus"
    _write_json(corpus / "legacy" / "no_odata.json", {"Id": "1"})
    np.save(
        corpus / "rest_api_map.npy",
        {
            "url_file_mapping": {"/redfish/v1/Managers/1": "legacy/no_odata.json"},
            "allowed_methods_mapping": {"/redfish/v1/Managers/1": ["GET"]},
        },
    )
    manifest_path = tmp_path / "manifest.json"
    _write_json(manifest_path, _manifest())

    records = list(RedfishCtlManifestSource(manifest_path).iter_records())

    assert [record.url for record in records] == ["/redfish/v1/Managers/1"]
    assert records[0].allowed_methods == ["GET"]
    assert records[0].provenance["mapping_format"] == "rest_api_map.npy"


def test_manifest_ambiguous_provider_map_file_fails_clearly(tmp_path: Path) -> None:
    """A no-odata file cannot safely map to two different provider URLs."""
    corpus = tmp_path / "corpus"
    _write_json(corpus / "shared.json", {"Id": "shared"})
    _write_json(
        corpus / "rest_api_map.v1.json",
        {
            "url_file_mapping": {
                "/redfish/v1/A": "shared.json",
                "/redfish/v1/B": "shared.json",
            },
            "allowed_methods_mapping": {},
        },
    )
    manifest_path = tmp_path / "manifest.json"
    _write_json(manifest_path, _manifest())

    with pytest.raises(RedfishCorpusManifestError, match="ambiguous"):
        RedfishCtlManifestSource(manifest_path)


def test_manifest_explicit_resources_allow_file_scoped_loading(tmp_path: Path) -> None:
    """Provider-declared resources can narrow loading without filename guessing."""
    corpus = tmp_path / "corpus"
    _write_json(corpus / "Systems" / "one.json", {"@odata.id": "/redfish/v1/Systems/1"})
    _write_json(corpus / "Systems" / "two.json", {"@odata.id": "/redfish/v1/Systems/2"})
    manifest_path = tmp_path / "manifest.json"
    _write_json(
        manifest_path,
        _manifest(
            resources=[
                {
                    "file": "Systems/two.json",
                    "url": "/provider/Systems/2",
                    "allowed_methods": ["GET", "PATCH"],
                }
            ]
        ),
    )

    records = list(RedfishCtlManifestSource(manifest_path).iter_records())

    assert [record.url for record in records] == ["/redfish/v1/Systems/2"]
    assert records[0].allowed_methods == ["GET", "PATCH"]
    assert records[0].provenance["url_from"] == "odata"


def test_manifest_filters_fail_closed_on_wrong_corpus_vendor_model_or_kind(tmp_path: Path) -> None:
    """Experiment filters must select the intended materialized corpus exactly."""
    (tmp_path / "corpus").mkdir()
    manifest_path = tmp_path / "manifest.json"
    _write_json(manifest_path, _manifest())

    with pytest.raises(RedfishCorpusManifestError, match="corpus_id"):
        RedfishCtlManifestSource(manifest_path, corpus_id="other")
    with pytest.raises(RedfishCorpusManifestError, match="vendor"):
        RedfishCtlManifestSource(manifest_path, vendor="hpe")
    with pytest.raises(RedfishCorpusManifestError, match="model"):
        RedfishCtlManifestSource(manifest_path, model="dl360")
    with pytest.raises(RedfishCorpusManifestError, match="kind"):
        RedfishCtlManifestSource(manifest_path, kind="smoke")


def test_manifest_unknown_major_version_is_rejected(tmp_path: Path) -> None:
    """Schema version negotiation fails before dataset loading starts."""
    manifest_path = tmp_path / "manifest.json"
    _write_json(manifest_path, _manifest(schema="redfish-corpus-manifest/v2"))

    with pytest.raises(RedfishCorpusManifestError, match="v2"):
        load_redfish_ctl_manifest(manifest_path)


def test_manifest_missing_required_field_is_rejected(tmp_path: Path) -> None:
    """Required provider provenance fields fail before source construction."""
    manifest_path = tmp_path / "manifest.json"
    body = _manifest()
    body.pop("provider_revision")
    _write_json(manifest_path, body)

    with pytest.raises(RedfishCorpusManifestError, match="provider_revision"):
        load_redfish_ctl_manifest(manifest_path)


def test_manifest_missing_materialized_root_fails_clearly(tmp_path: Path) -> None:
    """A manifest without its materialized corpus root is a setup blocker."""
    manifest_path = tmp_path / "manifest.json"
    _write_json(manifest_path, _manifest(root="missing-corpus"))

    with pytest.raises(RedfishCorpusManifestError, match="root"):
        RedfishCtlManifestSource(manifest_path)


def test_manifest_metadata_exposes_provider_and_simulator_contract(tmp_path: Path) -> None:
    """Run metadata can identify exact corpus, provider rev, and contracts."""
    manifest_path = tmp_path / "manifest.json"
    _write_json(manifest_path, _manifest(action_capabilities=[{"id": "systems.reset"}]))

    manifest = load_redfish_ctl_manifest(manifest_path)

    assert manifest.corpus_id == "xr8620t-full"
    assert manifest.provider_revision == "redfish-ctl-abc123"
    assert manifest.simulator_contract == "redfish-simulator/v1"
    assert manifest.action_capabilities == [{"id": "systems.reset"}]


# Author: Mus mbayramo@stanford.edu
