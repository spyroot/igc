"""
Offline tests for the multi-source mixer and its trust-tier train/eval split.

Independent verification of the mixer contract: deterministic hashing, dedup that keeps the
highest-trust copy of a URL, an eval split drawn ONLY from the trusted tier (so ground truth
is held out while synthetic tiers always train), conservation of records across the split,
monotonic eval growth with eval_fraction, and a serializable manifest with a stable content
hash. Pure stdlib — no torch, no network, no fixtures on disk.

Author:
Mus mbayramo@stanford.edu
"""

import pytest

from igc.ds.sources.base import SourceAdapter, SourceRecord, TrustLevel
from igc.ds.sources.mixer import DataManifest, SourceMix, unit_hash


class _FakeSource(SourceAdapter):
    """A SourceAdapter that replays a fixed list of records (test double)."""

    def __init__(self, source, trust_level, records):
        super().__init__(module_name=f"fake[{source}]")
        self.source = source
        self.trust_level = trust_level
        self._records = records

    def iter_records(self):
        return iter(self._records)


def _rec(url, source, trust, vendor=None):
    """Build a minimal SourceRecord for the mixer under test."""
    return SourceRecord(url=url, response={"@odata.id": url}, source=source,
                        trust_level=trust, vendor=vendor)


def _real_source(n, source="real_dell", vendor="dell"):
    """A source of n REAL records with distinct urls."""
    return _FakeSource(source, TrustLevel.REAL,
                       [_rec(f"/redfish/v1/R/{i}", source, TrustLevel.REAL, vendor) for i in range(n)])


def test_unit_hash_deterministic_and_ranged():
    """unit_hash is stable, lands in [0,1), and varies with key and seed."""
    assert unit_hash("/a", 0) == unit_hash("/a", 0)
    assert 0.0 <= unit_hash("/a", 0) < 1.0
    assert unit_hash("/a", 0) != unit_hash("/b", 0)
    assert unit_hash("/a", 0) != unit_hash("/a", 1)


def test_records_dedup_keeps_highest_trust():
    """The same url from two tiers collapses to the highest-trust copy."""
    real = _FakeSource("real_dell", TrustLevel.REAL, [_rec("/x", "real_dell", TrustLevel.REAL)])
    mock = _FakeSource("dmtf", TrustLevel.REPLAY, [_rec("/x", "dmtf", TrustLevel.REPLAY)])
    mix = SourceMix([mock, real], dedup=True)  # mock first, but real must win
    recs = mix.records()
    assert len(recs) == 1
    assert recs[0].trust_level is TrustLevel.REAL and recs[0].source == "real_dell"


def test_records_no_dedup_keeps_all():
    """dedup=False keeps every copy of a repeated url."""
    real = _FakeSource("real_dell", TrustLevel.REAL, [_rec("/x", "real_dell", TrustLevel.REAL)])
    mock = _FakeSource("dmtf", TrustLevel.REPLAY, [_rec("/x", "dmtf", TrustLevel.REPLAY)])
    assert len(SourceMix([mock, real], dedup=False).records()) == 2


def test_split_is_deterministic():
    """Two calls to split() produce the identical partition."""
    mix = SourceMix([_real_source(30)], eval_fraction=0.3, seed=7)
    t1, e1 = mix.split()
    t2, e2 = mix.split()
    assert [r.url for r in e1] == [r.url for r in e2]
    assert [r.url for r in t1] == [r.url for r in t2]


def test_eval_only_from_trusted_tier():
    """Eval holds only trust >= floor; every sub-floor record trains."""
    real = _real_source(20)
    sim = _FakeSource("sim", TrustLevel.SIM_GENERIC,
                      [_rec(f"/redfish/v1/S/{i}", "sim", TrustLevel.SIM_GENERIC) for i in range(20)])
    mix = SourceMix([real, sim], eval_fraction=0.5, eval_trust_floor=TrustLevel.REAL, seed=1)
    train, ev = mix.split()
    assert all(r.trust_level >= TrustLevel.REAL for r in ev)
    assert all(r.source == "sim" for r in train if r.trust_level < TrustLevel.REAL)
    # none of the SIM_GENERIC records may appear in eval
    assert not any(r.source == "sim" for r in ev)


def test_eval_fraction_zero_and_one():
    """fraction 0 -> empty eval; fraction 1 -> all eligible records held out."""
    real = _real_source(15)
    assert SourceMix([real], eval_fraction=0.0).split()[1] == []
    train, ev = SourceMix([real], eval_fraction=1.0).split()
    assert len(ev) == 15 and train == []


def test_split_conserves_records():
    """train + eval is exactly records(), no loss or duplication."""
    mix = SourceMix([_real_source(25), _FakeSource(
        "sim", TrustLevel.SIM_VENDOR,
        [_rec(f"/redfish/v1/S/{i}", "sim", TrustLevel.SIM_VENDOR) for i in range(10)])],
        eval_fraction=0.4)
    train, ev = mix.split()
    assert sorted(r.url for r in train + ev) == sorted(r.url for r in mix.records())


def test_eval_fraction_is_monotone_superset():
    """A larger eval_fraction yields a superset of the smaller one's eval urls."""
    src = _real_source(40)
    small = {r.url for r in SourceMix([src], eval_fraction=0.2, seed=3).split()[1]}
    large = {r.url for r in SourceMix([src], eval_fraction=0.6, seed=3).split()[1]}
    assert small and small < large


def test_manifest_counts_and_breakdowns():
    """Manifest totals and by-source/trust/vendor breakdowns match the corpus."""
    real = _real_source(12, source="real_hpe", vendor="hpe")
    sim = _FakeSource("sim", TrustLevel.SIM_DRIFT,
                      [_rec(f"/redfish/v1/S/{i}", "sim", TrustLevel.SIM_DRIFT, vendor=None) for i in range(8)])
    mix = SourceMix([real, sim], eval_fraction=0.25, seed=2)
    m = mix.manifest()
    assert isinstance(m, DataManifest)
    assert m.total == 20 and m.train_count + m.eval_count == 20
    assert m.by_source == {"real_hpe": 12, "sim": 8}
    assert m.by_trust == {"REAL": 12, "SIM_DRIFT": 8}
    assert m.by_vendor == {"hpe": 12, "unknown": 8}
    assert m.sources == ["real_hpe", "sim"]
    assert m.eval_trust_floor == "REAL" and m.eval_fraction == 0.25 and m.seed == 2


def test_manifest_content_hash_stable_and_sensitive():
    """Identical manifests hash equal; a changed field changes the hash."""
    a = SourceMix([_real_source(10)], eval_fraction=0.2, seed=5).manifest()
    b = SourceMix([_real_source(10)], eval_fraction=0.2, seed=5).manifest()
    c = SourceMix([_real_source(10)], eval_fraction=0.3, seed=5).manifest()
    assert a.content_hash() == b.content_hash()
    assert a.content_hash() != c.content_hash()


def test_empty_adapters_are_safe():
    """No adapters -> empty split and a zeroed manifest, no crash."""
    mix = SourceMix([], eval_fraction=0.2)
    assert mix.records() == []
    assert mix.split() == ([], [])
    m = mix.manifest()
    assert m.total == 0 and m.train_count == 0 and m.eval_count == 0
    assert m.by_source == {} and m.sources == []


@pytest.mark.parametrize("bad", [-0.1, 1.5, 2.0])
def test_invalid_eval_fraction_raises(bad):
    """eval_fraction outside [0, 1] is rejected."""
    with pytest.raises(ValueError):
        SourceMix([], eval_fraction=bad)


def test_eval_split_id_encodes_policy():
    """eval_split_id depends on floor/fraction/seed, not on the corpus contents."""
    same_policy_a = SourceMix([_real_source(5)], eval_fraction=0.15, seed=0).manifest()
    same_policy_b = SourceMix([_real_source(9)], eval_fraction=0.15, seed=0).manifest()
    other_fraction = SourceMix([_real_source(5)], eval_fraction=0.30, seed=0).manifest()
    assert same_policy_a.eval_split_id() == same_policy_b.eval_split_id()
    assert same_policy_a.eval_split_id() != other_fraction.eval_split_id()
    assert "REAL" in same_policy_a.eval_split_id() and "0.15" in same_policy_a.eval_split_id()


def test_to_run_manifest_fields_feed_fair_comparison():
    """DataManifest fields populate RunManifest so the report's fairness check sees the mix."""
    from igc.modules.train.report import ResultBundle, RunManifest, compare

    dm = SourceMix([_real_source(20)], eval_fraction=0.15, seed=0).manifest()
    dm_same = SourceMix([_real_source(20)], eval_fraction=0.15, seed=0).manifest()
    dm_other = SourceMix([_real_source(50)], eval_fraction=0.15, seed=0).manifest()  # different corpus

    fields = dm.to_run_manifest_fields()
    assert set(fields) == {"data_manifest", "eval_split"}

    b1 = ResultBundle(manifest=RunManifest(run_id="r1", profile="p", model="m",
                      adapter_method="lora", **dm.to_run_manifest_fields()), metrics={"recall@1": 0.6})
    b2 = ResultBundle(manifest=RunManifest(run_id="r2", profile="p", model="m",
                      adapter_method="rslora", **dm_same.to_run_manifest_fields()), metrics={"recall@1": 0.7})
    b3 = ResultBundle(manifest=RunManifest(run_id="r3", profile="p", model="m",
                      adapter_method="dora", **dm_other.to_run_manifest_fields()), metrics={"recall@1": 0.9})

    assert not compare([b1, b2]).fairness_issues            # same mix + policy -> fair
    assert "data_manifest" in " ".join(compare([b1, b3]).fairness_issues)  # different mix -> flagged


# Author: Mus mbayramo@stanford.edu
