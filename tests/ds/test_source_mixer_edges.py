"""Additional edge coverage for trust-tier-aware source mixing."""

from igc.ds.sources import mixer as mixer_module
from igc.ds.sources.base import SourceAdapter, SourceRecord, TrustLevel
from igc.ds.sources.mixer import SourceMix


class _Source(SourceAdapter):
    """SourceAdapter test double backed by an in-memory record list."""

    def __init__(self, source, trust_level, records):
        super().__init__(module_name=f"edge-source[{source}]")
        self.source = source
        self.trust_level = trust_level
        self._records = records

    def iter_records(self):
        return iter(self._records)


def _record(url, source, trust_level):
    return SourceRecord(
        url=url,
        response={"@odata.id": url},
        source=source,
        trust_level=trust_level,
    )


def _source(source, trust_level, urls):
    return _Source(
        source,
        trust_level,
        [_record(url, source, trust_level) for url in urls],
    )


def test_single_record_below_eval_floor_always_trains():
    """A lone low-trust record is not held out even at eval_fraction=1.0."""
    mix = SourceMix(
        [_source("sim", TrustLevel.SIM_GENERIC, ["/redfish/v1/sim-only"])],
        eval_fraction=1.0,
        eval_trust_floor=TrustLevel.REAL,
    )

    train, held_out = mix.split()

    assert [record.url for record in train] == ["/redfish/v1/sim-only"]
    assert held_out == []


def test_eval_threshold_is_strictly_less_than_fraction(monkeypatch):
    """A hash equal to the fraction stays in train; below it is held out."""
    hash_by_url = {
        "/redfish/v1/exact": 0.5,
        "/redfish/v1/below": 0.499,
        "/redfish/v1/above": 0.501,
    }
    monkeypatch.setattr(
        mixer_module,
        "unit_hash",
        lambda key, seed: hash_by_url[key],
    )
    mix = SourceMix(
        [_source("real", TrustLevel.REAL, list(hash_by_url))],
        eval_fraction=0.5,
        seed=123,
    )

    train, held_out = mix.split()

    assert [record.url for record in train] == [
        "/redfish/v1/exact",
        "/redfish/v1/above",
    ]
    assert [record.url for record in held_out] == ["/redfish/v1/below"]


def test_all_same_trust_tier_split_preserves_source_order(monkeypatch):
    """Same-tier records split only by hash threshold and keep input order."""
    urls = ["/redfish/v1/a", "/redfish/v1/b", "/redfish/v1/c", "/redfish/v1/d"]
    hash_by_url = {
        "/redfish/v1/a": 0.10,
        "/redfish/v1/b": 0.90,
        "/redfish/v1/c": 0.20,
        "/redfish/v1/d": 0.80,
    }
    monkeypatch.setattr(
        mixer_module,
        "unit_hash",
        lambda key, seed: hash_by_url[key],
    )
    mix = SourceMix(
        [_source("replay", TrustLevel.REPLAY, urls)],
        eval_fraction=0.5,
        eval_trust_floor=TrustLevel.REPLAY,
    )

    train, held_out = mix.split()

    assert [record.url for record in train] == ["/redfish/v1/b", "/redfish/v1/d"]
    assert [record.url for record in held_out] == ["/redfish/v1/a", "/redfish/v1/c"]


def test_equal_trust_dedup_keeps_first_seen_record():
    """Equal-trust duplicate URLs keep the adapter-order winner."""
    first = _Source(
        "real_a",
        TrustLevel.REAL,
        [_record("/redfish/v1/shared", "real_a", TrustLevel.REAL)],
    )
    second = _Source(
        "real_b",
        TrustLevel.REAL,
        [_record("/redfish/v1/shared", "real_b", TrustLevel.REAL)],
    )

    records = SourceMix([first, second], dedup=True).records()

    assert len(records) == 1
    assert records[0].source == "real_a"
