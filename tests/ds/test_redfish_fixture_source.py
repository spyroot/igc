"""
Offline tests for the provenance-tagged Redfish fixture source.

Pins that :class:`RedfishFixtureSource` keys records on the canonical
``@odata.id`` (falling back to the discovery filename and recording which route
it took), stamps source/trust/vendor/schema provenance, attaches allowed methods
from a supplied ``.npy``-style map, skips unparsable and non-object JSON instead
of crashing, and treats a missing directory as empty. Also pins the trust-tier
ordering used to split evaluation. Pure stdlib — no torch, no network, no
idrac_ctl fixtures on disk.

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
    (root / name).write_text(text)


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
