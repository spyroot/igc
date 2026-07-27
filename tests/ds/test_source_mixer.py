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

import hashlib
import json

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


def _row_id(source, url):
    """Mirror the public source-qualified row identity contract."""
    digest = hashlib.sha256(f"{source}\0{url}".encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


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


def test_records_dedup_keeps_highest_trust_inside_one_source():
    """The same source/url pair collapses to the highest-trust copy."""
    older = _rec("/x", "real_dell", TrustLevel.REPLAY)
    newer = _rec("/x", "real_dell", TrustLevel.REAL)
    mix = SourceMix(
        [_FakeSource("real_dell", TrustLevel.REAL, [older, newer])],
        dedup=True,
    )
    recs = mix.records()
    assert len(recs) == 1
    assert recs[0].trust_level is TrustLevel.REAL and recs[0].source == "real_dell"


def test_records_dedup_preserves_same_api_from_distinct_sources():
    """A common Redfish URL from Dell/HPE/GB300/X10/DSP remains source-distinct."""
    url = "/redfish/v1/Systems/1"
    sources = [
        ("real_dell", TrustLevel.REAL, "dell"),
        ("real_hpe", TrustLevel.REAL, "hpe"),
        ("gb300_lab", TrustLevel.REAL, "nvidia"),
        ("x10_lab", TrustLevel.REAL, "supermicro"),
        ("dsp_replay", TrustLevel.REPLAY, None),
    ]
    adapters = [
        _FakeSource(source, trust, [_rec(url, source, trust, vendor)])
        for source, trust, vendor in sources
    ]

    records = SourceMix(adapters, dedup=True).records()

    assert [(record.source, record.url) for record in records] == [
        (source, url) for source, _trust, _vendor in sources
    ]
    assert len({record.source for record in records}) == len(sources)


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


def test_default_min_eval_per_source_zero_preserves_threshold_split():
    """The default held-out floor does not move rows beyond the hash threshold."""
    real = _real_source(20, source="real_dell", vendor="dell")

    default_train, default_eval = SourceMix(
        [real],
        eval_fraction=0.25,
        seed=17,
    ).split()
    explicit_train, explicit_eval = SourceMix(
        [real],
        eval_fraction=0.25,
        seed=17,
        min_eval_per_source=0,
    ).split()

    assert [record.url for record in default_eval] == [
        record.url for record in explicit_eval
    ]
    assert [record.url for record in default_train] == [
        record.url for record in explicit_train
    ]


def test_min_eval_per_source_extends_each_real_source_deterministically():
    """Each eligible REAL source is extended to at least N lowest-hash rows."""
    dell = _real_source(6, source="real_dell", vendor="dell")
    hpe = _real_source(6, source="real_hpe", vendor="hpe")
    mix = SourceMix(
        [dell, hpe],
        eval_fraction=0.0,
        eval_trust_floor=TrustLevel.REAL,
        seed=23,
        min_eval_per_source=2,
    )

    train, heldout = mix.split()
    heldout_by_source = {
        source: [record.url for record in heldout if record.source == source]
        for source in ("real_dell", "real_hpe")
    }

    for source in ("real_dell", "real_hpe"):
        source_records = [record for record in mix.records() if record.source == source]
        expected = sorted(
            source_records,
            key=lambda record: (
                unit_hash(f"{record.source}\0{record.url}", 23),
                f"{record.source}\0{record.url}",
            ),
        )[:2]
        assert heldout_by_source[source] == [record.url for record in expected]
    assert len(heldout) == 4
    assert len(train) == 8
    assert SourceMix(
        [dell, hpe],
        eval_fraction=0.0,
        eval_trust_floor=TrustLevel.REAL,
        seed=23,
        min_eval_per_source=2,
    ).manifest().heldout_by_source == {"real_dell": 2, "real_hpe": 2}


def test_min_eval_per_source_holds_out_entire_small_eligible_source():
    """An eligible source smaller than the configured floor is fully held out."""
    small = _real_source(2, source="real_small", vendor="small")
    large = _real_source(5, source="real_large", vendor="large")

    train, heldout = SourceMix(
        [small, large],
        eval_fraction=0.0,
        eval_trust_floor=TrustLevel.REAL,
        seed=5,
        min_eval_per_source=3,
    ).split()

    assert [record.url for record in heldout if record.source == "real_small"] == [
        "/redfish/v1/R/0",
        "/redfish/v1/R/1",
    ]
    assert not any(record.source == "real_small" for record in train)
    assert sum(record.source == "real_large" for record in heldout) == 3


def test_min_eval_per_source_never_moves_replay_rows_to_heldout():
    """REPLAY/DSP rows remain train-only even when a REAL held-out floor is set."""
    real = _real_source(4, source="real_dell", vendor="dell")
    replay = _FakeSource(
        "dsp_replay",
        TrustLevel.REPLAY,
        [
            _rec(f"/redfish/v1/DSP/{index}", "dsp_replay", TrustLevel.REPLAY)
            for index in range(4)
        ],
    )

    train, heldout = SourceMix(
        [real, replay],
        eval_fraction=0.0,
        eval_trust_floor=TrustLevel.REAL,
        seed=7,
        min_eval_per_source=3,
    ).split()

    assert sum(record.source == "real_dell" for record in heldout) == 3
    assert not any(record.source == "dsp_replay" for record in heldout)
    assert sum(record.source == "dsp_replay" for record in train) == 4


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
    """Manifest totals and train/heldout source breakdowns match the corpus."""
    real = _real_source(12, source="real_hpe", vendor="hpe")
    sim = _FakeSource("sim", TrustLevel.SIM_DRIFT,
                      [_rec(f"/redfish/v1/S/{i}", "sim", TrustLevel.SIM_DRIFT, vendor=None) for i in range(8)])
    mix = SourceMix([real, sim], eval_fraction=0.25, seed=2)
    m = mix.manifest()
    _, heldout = mix.split()
    assert isinstance(m, DataManifest)
    assert m.total == 20 and m.train_count + m.eval_count == 20
    assert m.by_source == {"real_hpe": 12, "sim": 8}
    assert m.by_trust == {"REAL": 12, "SIM_DRIFT": 8}
    assert m.by_vendor == {"hpe": 12, "unknown": 8}
    assert m.sources == ["real_hpe", "sim"]
    assert m.required_heldout_sources == ["real_hpe"]
    assert m.heldout_by_source == {"real_hpe": len(heldout)}
    assert "sim" not in m.required_heldout_sources
    assert "sim" not in m.heldout_by_source
    assert m.eval_trust_floor == "REAL" and m.eval_fraction == 0.25 and m.seed == 2
    assert m.min_eval_per_source == 0


def test_required_heldout_sources_are_eval_eligible_not_all_sources():
    """Heldout requirements list only eval-eligible source labels, never train-only sources."""
    real_dell = _real_source(2, source="real_dell", vendor="dell")
    real_hpe = _real_source(3, source="real_hpe", vendor="hpe")
    replay = _FakeSource(
        "dmtf_replay",
        TrustLevel.REPLAY,
        [_rec(f"/redfish/v1/DMTF/{i}", "dmtf_replay", TrustLevel.REPLAY) for i in range(4)],
    )
    mix = SourceMix(
        [real_dell, real_hpe, replay],
        eval_fraction=1.0,
        eval_trust_floor=TrustLevel.REAL,
    )
    manifest = mix.manifest()

    assert manifest.sources == ["dmtf_replay", "real_dell", "real_hpe"]
    assert manifest.required_heldout_sources == ["real_dell", "real_hpe"]
    assert manifest.heldout_by_source == {"real_dell": 2, "real_hpe": 3}


def test_split_and_row_identities_are_source_qualified_for_same_api():
    """Split membership and row IDs preserve the source label for shared URLs."""
    url = "/redfish/v1/Systems/1"
    adapters = [
        _FakeSource(source, trust, [_rec(url, source, trust, vendor)])
        for source, trust, vendor in [
            ("real_dell", TrustLevel.REAL, "dell"),
            ("real_hpe", TrustLevel.REAL, "hpe"),
            ("gb300_lab", TrustLevel.REAL, "nvidia"),
            ("x10_lab", TrustLevel.REAL, "supermicro"),
            ("dsp_replay", TrustLevel.REPLAY, None),
        ]
    ]
    mix = SourceMix(
        adapters,
        eval_fraction=1.0,
        eval_trust_floor=TrustLevel.REPLAY,
        seed=11,
    )

    train, heldout = mix.split()
    manifest = mix.manifest()
    expected_ids = [_row_id(adapter.source, url) for adapter in adapters]
    expected_split_payload = json.dumps(
        expected_ids,
        sort_keys=True,
        separators=(",", ":"),
    )
    expected_split_id = (
        "sha256:"
        + hashlib.sha256(expected_split_payload.encode("utf-8")).hexdigest()
    )

    assert train == []
    assert [(record.source, record.url) for record in heldout] == [
        (adapter.source, url) for adapter in adapters
    ]
    assert manifest.heldout_row_ids == expected_ids
    assert len(set(manifest.heldout_row_ids)) == len(adapters)
    assert manifest.eval_split_id() == expected_split_id


def test_manifest_content_hash_stable_and_sensitive():
    """Identical manifests hash equal; a changed field changes the hash."""
    a = SourceMix([_real_source(10)], eval_fraction=0.2, seed=5).manifest()
    b = SourceMix([_real_source(10)], eval_fraction=0.2, seed=5).manifest()
    c = SourceMix([_real_source(10)], eval_fraction=0.3, seed=5).manifest()
    assert a.content_hash() == b.content_hash()
    assert a.content_hash() != c.content_hash()


def test_manifest_content_hash_is_exact_canonical_sha256():
    """DataManifest.content_hash is the SHA-256 of the canonical manifest JSON."""
    manifest = DataManifest(
        total=2,
        train_count=1,
        eval_count=1,
        by_source={"real_dell": 2},
        by_trust={"REAL": 2},
        by_vendor={"dell": 2},
        eval_trust_floor="REAL",
        eval_fraction=0.15,
        seed=0,
        sources=["real_dell"],
        required_heldout_sources=["real_dell"],
        heldout_by_source={"real_dell": 1},
        source_registry_sha="sha256:" + "3" * 64,
        source_manifest_shas={"real_dell": "sha256:" + "4" * 64},
        train_row_ids=["sha256:" + "1" * 64],
        heldout_row_ids=["sha256:" + "2" * 64],
    )
    fields = dict(manifest.__dict__)
    fields.pop("phase1_transform")
    payload = json.dumps(fields, sort_keys=True, default=str)
    expected = f"sha256:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"

    assert manifest.content_hash() == expected


def test_manifest_hash_changes_with_registry_and_upstream_manifest_lineage():
    """The semantic manifest identity includes source-registry and upstream manifest SHAs."""
    base = SourceMix([_real_source(10)], eval_fraction=0.2, seed=5).manifest()
    with_registry = DataManifest(
        **{
            **base.__dict__,
            "source_registry_sha": "sha256:" + "1" * 64,
            "source_manifest_shas": {"real_dell": "sha256:" + "2" * 64},
        }
    )

    assert with_registry.content_hash() != base.content_hash()


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


def test_eval_split_id_hashes_exact_heldout_row_ids():
    """eval_split_id is a SHA-256 identity over the exact held-out row IDs."""
    manifest = DataManifest(
        total=3,
        train_count=1,
        eval_count=2,
        by_source={"real_dell": 3},
        by_trust={"REAL": 3},
        by_vendor={"dell": 3},
        eval_trust_floor="REAL",
        eval_fraction=0.15,
        seed=0,
        sources=["real_dell"],
        required_heldout_sources=["real_dell"],
        heldout_by_source={"real_dell": 2},
        train_row_ids=["sha256:" + "1" * 64],
        heldout_row_ids=["sha256:" + "2" * 64, "sha256:" + "3" * 64],
    )
    payload = json.dumps(
        manifest.heldout_row_ids,
        sort_keys=True,
        separators=(",", ":"),
    )
    expected = f"sha256:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"
    changed = DataManifest(
        **{
            **manifest.__dict__,
            "heldout_row_ids": ["sha256:" + "3" * 64, "sha256:" + "2" * 64],
        }
    )

    assert manifest.eval_split_id() == expected
    assert changed.eval_split_id() != manifest.eval_split_id()


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
