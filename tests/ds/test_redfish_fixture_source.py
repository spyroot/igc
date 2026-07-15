"""
Offline tests for the provenance-tagged Redfish fixture source.

Pins that :class:`RedfishFixtureSource` keys records on the canonical
``@odata.id`` (falling back to the discovery filename and recording which route
it took), stamps source/trust/vendor/schema provenance, attaches allowed methods
from a supplied ``.npy``-style map, skips unparsable and non-object JSON instead
of crashing, and treats a missing directory as empty. Also pins the trust-tier
ordering used to split evaluation. Pure stdlib — no torch, no network, no
checked-out redfish_ctl fixtures on disk.

Author:
Mus mbayramo@stanford.edu
"""

import json
from pathlib import Path

from igc.ds.sources import RedfishFixtureSource, SourceRecord, TrustLevel


def _write(root: Path, name: str, body) -> None:
    """Write ``body`` (dict or raw string) to ``root/name``."""
    root.mkdir(parents=True, exist_ok=True)
    text = body if isinstance(body, str) else json.dumps(body)
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _corpus(tmp_path: Path) -> Path:
    """A tiny mixed corpus: odata, no-odata, unparsable, and a bare array."""
    root = tmp_path / "capture"
    _write(root, "_redfish_v1_Systems_1.json",
           {"@odata.id": "/redfish/v1/Systems/1",
            "@odata.type": "#ComputerSystem.v1_20_0.ComputerSystem", "Id": "1"})
    _write(root, "_redfish_v1_Chassis_1.json",
           {"@odata.type": "#Chassis.v1_25_0.Chassis", "Id": "1"})  # no @odata.id
    _write(root, "_redfish_v1_broken.json", "{not valid json")
    _write(root, "_redfish_v1_array.json", [1, 2, 3])              # non-object
    return root


def test_odata_id_is_the_canonical_url(tmp_path: Path) -> None:
    """A record keys on @odata.id and marks provenance url_from='odata'."""
    src = RedfishFixtureSource(str(_corpus(tmp_path)), "real_dell", TrustLevel.REAL, vendor="dell")
    by_url = {r.url: r for r in src.iter_records()}
    rec = by_url["/redfish/v1/Systems/1"]
    assert rec.provenance["url_from"] == "odata"
    assert rec.schema_version == "#ComputerSystem.v1_20_0.ComputerSystem"
    assert rec.source == "real_dell" and rec.vendor == "dell"
    assert rec.trust_level is TrustLevel.REAL
    assert rec.method == "GET"


def test_filename_fallback_when_no_odata_id(tmp_path: Path) -> None:
    """Without @odata.id, the URL is reconstructed from the filename."""
    src = RedfishFixtureSource(str(_corpus(tmp_path)), "real_dell", TrustLevel.REAL)
    rec = next(r for r in src.iter_records()
               if r.provenance["file"] == "_redfish_v1_Chassis_1.json")
    assert rec.url == "/redfish/v1/Chassis/1"
    assert rec.provenance["url_from"] == "filename"


def test_unparsable_and_non_object_are_skipped(tmp_path: Path) -> None:
    """Broken JSON and a bare array are skipped and counted, not raised."""
    src = RedfishFixtureSource(str(_corpus(tmp_path)), "dmtf", TrustLevel.REPLAY)
    recs = list(src.iter_records())
    assert src.num_emitted == 2 and src.num_skipped == 2
    assert len(recs) == 2
    assert all(isinstance(r, SourceRecord) for r in recs)


def test_allowed_methods_matched_by_url(tmp_path: Path) -> None:
    """allowed_methods comes from the supplied map, keyed by canonical URL."""
    amap = {"/redfish/v1/Systems/1": ["GET", "PATCH"]}
    src = RedfishFixtureSource(str(_corpus(tmp_path)), "real_dell", TrustLevel.REAL,
                               allowed_methods_map=amap)
    by_url = {r.url: r for r in src.iter_records()}
    assert by_url["/redfish/v1/Systems/1"].allowed_methods == ["GET", "PATCH"]
    assert by_url["/redfish/v1/Chassis/1"].allowed_methods is None  # not in map


def test_allowed_methods_match_filename_fallback_url(tmp_path: Path) -> None:
    """The .npy method map still applies when URL comes from the filename."""
    amap = {"/redfish/v1/Chassis/1": ["GET", "HEAD"]}
    src = RedfishFixtureSource(str(_corpus(tmp_path)), "real_dell", TrustLevel.REAL,
                               allowed_methods_map=amap)
    rec = next(r for r in src.iter_records()
               if r.provenance["file"] == "_redfish_v1_Chassis_1.json")
    assert rec.url == "/redfish/v1/Chassis/1"
    assert rec.allowed_methods == ["GET", "HEAD"]


def test_glob_pattern_limits_selected_fixture_files(tmp_path: Path) -> None:
    """A custom glob selects only matching corpus files and skips sidecars."""
    root = tmp_path / "capture"
    _write(root, "_redfish_v1_Systems_1.json",
           {"@odata.id": "/redfish/v1/Systems/1", "Id": "1"})
    _write(root, "_redfish_v1_Systems_2.extra.json",
           {"@odata.id": "/redfish/v1/Systems/2", "Id": "2"})
    _write(root, "_redfish_v1_Chassis_1.txt",
           {"@odata.id": "/redfish/v1/Chassis/1", "Id": "1"})

    src = RedfishFixtureSource(str(root), "real_dell", TrustLevel.REAL,
                               glob_pattern="*_Systems_1.json")

    recs = list(src.iter_records())
    assert [r.url for r in recs] == ["/redfish/v1/Systems/1"]
    assert src.num_emitted == 1
    assert src.num_skipped == 0


def test_materialized_redfish_ctl_layout_is_recursive(tmp_path: Path) -> None:
    """A materialized dataset artifact may keep JSON below nested json_responses dirs."""
    root = tmp_path / "materialized" / "dataset" / "dell_xr8620t"
    _write(root, "corpus.json", {"corpus_id": "dell-xr8620t"})
    _write(root, "rest_api_map.v1.json", {
        "url_file_mapping": {
            "/redfish/v1/Systems/1": "json_responses/Systems/_redfish_v1_Systems_1.json"
        },
        "allowed_methods_mapping": {"/redfish/v1/Systems/1": ["GET", "PATCH"]},
    })
    _write(root, "json_responses/Systems/_redfish_v1_Systems_1.json",
           {"@odata.id": "/redfish/v1/Systems/1",
            "@odata.type": "#ComputerSystem.v1_20_0.ComputerSystem", "Id": "1"})
    _write(root, "json_responses/Schemas/Manifest.v1_1_0.json",
           {"@odata.id": "/redfish/v1/Schemas/Manifest.v1_1_0.json",
            "@odata.type": "#JsonSchemaFile.v1_0_0.JsonSchemaFile", "Id": "Manifest"})

    src = RedfishFixtureSource(str(root), "dell-xr8620t", TrustLevel.REAL, vendor="dell")
    recs = sorted(src.iter_records(), key=lambda rec: rec.url)

    assert [rec.url for rec in recs] == [
        "/redfish/v1/Schemas/Manifest.v1_1_0.json",
        "/redfish/v1/Systems/1",
    ]
    assert recs[1].provenance["file"] == "json_responses/Systems/_redfish_v1_Systems_1.json"
    assert recs[1].provenance["corpus_id"] == "dell-xr8620t"
    assert src.num_emitted == 2
    assert src.num_skipped == 2


def test_redfish_ctl_manifest_builds_vendor_sources(tmp_path: Path) -> None:
    """Manifest rows select materialized dataset corpora without assuming host dirs."""
    manifest = tmp_path / "manifest.v1.json"
    materialized = tmp_path / "build" / "corpora"
    rows = [
        ("dell-xr8620t", "dell", "xr8620t", "2023-06-17", "corpora/dataset/dell_xr8620t.tar.gz"),
        ("hpe-dl360", "hpe", "dl360", "2026-01-02", "corpora/dataset/hpe_dl360.tar.gz"),
        ("supermicro-x10sdv", "supermicro", "x10sdv", "2026-01-03", "corpora/dataset/supermicro_x10sdv.tar.gz"),
        ("supermicro-gb300", "supermicro", "gb300", "2026-01-04", "corpora/dataset/supermicro_gb300.tar.gz"),
    ]
    manifest.write_text(json.dumps({
        "schema_version": 1,
        "corpora": [
            {
                "id": corpus_id,
                "kind": "dataset",
                "vendor": vendor,
                "model": model,
                "capture_id": capture_id,
                "archive": archive,
            }
            for corpus_id, vendor, model, capture_id, archive in rows
        ],
    }))

    for corpus_id, _vendor, _model, _capture_id, archive in rows:
        slug = Path(archive).name[:-7]
        root = materialized / "dataset" / slug
        _write(root,
               "json_responses/_redfish_v1.json",
               {"@odata.id": f"/redfish/v1/{corpus_id}", "Id": corpus_id})
        _write(root,
               "rest_api_map.v1.json",
               {
                   "url_file_mapping": {
                       f"/redfish/v1/{corpus_id}": "json_responses/_redfish_v1.json",
                   },
                   "allowed_methods_mapping": {f"/redfish/v1/{corpus_id}": ["GET", "HEAD"]},
               })

    sources = RedfishFixtureSource.from_redfish_ctl_manifest(str(manifest), str(materialized))
    records = [next(source.iter_records()) for source in sources]

    assert [source.source for source in sources] == [row[0] for row in rows]
    assert [source.vendor for source in sources] == [row[1] for row in rows]
    assert [record.url for record in records] == [f"/redfish/v1/{row[0]}" for row in rows]
    assert [record.allowed_methods for record in records] == [["GET", "HEAD"]] * len(rows)
    assert [record.provenance["corpus_id"] for record in records] == [row[0] for row in rows]
    assert [record.provenance["capture_id"] for record in records] == [row[3] for row in rows]

    gb300_only = RedfishFixtureSource.from_redfish_ctl_manifest(
        str(manifest), str(materialized), corpus_ids=["supermicro-gb300"])
    assert [source.source for source in gb300_only] == ["supermicro-gb300"]


def test_url_from_filename_helper() -> None:
    """The filename->URL reconstruction strips markers and maps '_'->'/'."""
    f = RedfishFixtureSource.url_from_filename
    assert f("_redfish_v1_Systems_1.json") == "/redfish/v1/Systems/1"
    assert f("_redfish_v1_Chassis.json") == "/redfish/v1/Chassis"
    assert f("_redfish_v1") == "/redfish/v1"


def test_missing_directory_yields_nothing(tmp_path: Path) -> None:
    """A non-existent corpus directory yields no records and does not raise."""
    src = RedfishFixtureSource(str(tmp_path / "nope"), "real_hpe", TrustLevel.REAL)
    assert list(src.iter_records()) == []
    assert src.num_emitted == 0


def test_iterating_twice_resets_counters(tmp_path: Path) -> None:
    """Counters reflect the latest pass, so a source is safely re-iterable."""
    src = RedfishFixtureSource(str(_corpus(tmp_path)), "dmtf", TrustLevel.REPLAY)
    first = list(src)
    second = list(src)
    assert len(first) == len(second) == 2
    assert src.num_emitted == 2 and src.num_skipped == 2


def test_trust_tiers_are_ordered_high_to_low() -> None:
    """Real is the most trusted tier; drift the least — used for eval splits."""
    assert TrustLevel.REAL > TrustLevel.REPLAY > TrustLevel.SIM_VENDOR
    assert TrustLevel.SIM_VENDOR > TrustLevel.SIM_GENERIC > TrustLevel.SIM_DRIFT
    assert TrustLevel.REAL >= TrustLevel.REAL


# Author: Mus mbayramo@stanford.edu
