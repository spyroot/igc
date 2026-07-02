"""
Offline tests for the training-corpus JSONL writer/reader.

Pins that examples round-trip through JSONL equal to their to_dict() form, that iter_examples
streams and skips blank lines, that the manifest sidecar round-trips, that write_corpus emits
both files with a correct count, and that empty input and nested output dirs are handled. Pure
stdlib + tmp_path — no torch, no network.

Author:
Mus mbayramo@stanford.edu
"""

from pathlib import Path

from igc.ds.sources.base import SourceRecord, TrustLevel
from igc.ds.sources.corpus_io import (
    iter_examples,
    read_examples,
    read_manifest,
    write_corpus,
    write_examples,
    write_manifest,
)
from igc.ds.sources.mixer import SourceMix
from igc.ds.sources.training_object import normalize


class _FakeSource:
    """Minimal source yielding fixed records (avoids the logger base in tests)."""

    def __init__(self, source, trust_level, records):
        self.source = source
        self.trust_level = trust_level
        self._records = records

    def iter_records(self):
        return iter(self._records)


def _examples(n=5):
    """n normalized TrainingExamples from synthetic REAL records."""
    recs = [SourceRecord(url=f"/redfish/v1/R/{i}", response={"@odata.id": f"/redfish/v1/R/{i}",
                         "@odata.type": "#T", "Id": str(i)}, source="real_dell",
                         trust_level=TrustLevel.REAL, allowed_methods=["GET"], vendor="dell")
            for i in range(n)]
    return normalize(recs)


def test_examples_round_trip(tmp_path: Path):
    """write_examples then read_examples reconstructs the to_dict() forms in order."""
    exs = _examples(4)
    path = str(tmp_path / "examples.jsonl")
    assert write_examples(exs, path) == 4
    back = read_examples(path)
    assert back == [e.to_dict() for e in exs]
    assert back[0]["trust_level"] == "REAL"


def test_iter_examples_streams_and_skips_blank_lines(tmp_path: Path):
    """iter_examples yields a generator and ignores blank/whitespace lines."""
    path = tmp_path / "e.jsonl"
    path.write_text('{"a": 1}\n\n   \n{"a": 2}\n')
    it = iter_examples(str(path))
    assert iter(it) is it  # a generator, not a materialized list
    assert list(it) == [{"a": 1}, {"a": 2}]


def test_manifest_round_trips(tmp_path: Path):
    """write_manifest/read_manifest preserve the manifest fields."""
    mix = SourceMix([_FakeSource("real_dell", TrustLevel.REAL,
                                 [SourceRecord(url=f"/r/{i}", response={}, source="real_dell",
                                               trust_level=TrustLevel.REAL) for i in range(6)])],
                    eval_fraction=0.2, seed=1)
    m = mix.manifest()
    path = str(tmp_path / "manifest.json")
    write_manifest(m, path)
    loaded = read_manifest(path)
    assert loaded["total"] == 6
    assert loaded["by_source"] == {"real_dell": 6}
    assert loaded["eval_trust_floor"] == "REAL"


def test_write_corpus_emits_both_files(tmp_path: Path):
    """write_corpus writes examples.jsonl + manifest.json into out_dir with a count."""
    exs = _examples(7)
    mix = SourceMix([_FakeSource("real_dell", TrustLevel.REAL,
                                 [SourceRecord(url=f"/r/{i}", response={}, source="real_dell",
                                               trust_level=TrustLevel.REAL) for i in range(7)])])
    out = tmp_path / "corpus"
    paths = write_corpus(exs, mix.manifest(), str(out))
    assert paths["count"] == "7"
    assert (out / "examples.jsonl").exists() and (out / "manifest.json").exists()
    assert len(read_examples(paths["examples"])) == 7
    assert read_manifest(paths["manifest"])["total"] == 7


def test_empty_and_nested_dirs(tmp_path: Path):
    """Empty input writes an empty file; missing parent dirs are created."""
    nested = str(tmp_path / "a" / "b" / "examples.jsonl")
    assert write_examples([], nested) == 0
    assert read_examples(nested) == []


# Author: Mus mbayramo@stanford.edu
