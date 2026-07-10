"""
Offline tests for the corpus tokenizer bridge (CorpusJSONLDataset).

Pins that a corpus written by write_corpus loads into fixed-length
input_ids/attention_mask items the trainer's collate stacks, that the trainer-facing
duck-type surface (tokenizer property, load_tokenizer, the masking no-ops) is present,
that run_manifest_fields round-trips the mixer's data_manifest/eval_split ids, and that
a missing corpus raises. Uses a fake tokenizer — no downloads, no network.

Author:
Mus mbayramo@stanford.edu
"""

from pathlib import Path

import pytest
import torch

from igc.ds.corpus_dataset import CorpusJSONLDataset
from igc.ds.sources.base import SourceRecord, TrustLevel
from igc.ds.sources.corpus_io import write_corpus
from igc.ds.sources.mixer import SourceMix
from igc.ds.sources.training_object import normalize


class _FakeTokenizer:
    """Minimal HF-like tokenizer: char-code ids, pads/truncates to max_length."""

    pad_token = "<pad>"
    eos_token = "<eos>"

    def __call__(self, text, padding=None, max_length=None, truncation=None,
                 return_tensors=None):
        ids = [ord(c) % 1000 + 1 for c in text][:max_length]
        mask = [1] * len(ids)
        while len(ids) < max_length:
            ids.append(0)
            mask.append(0)
        return {"input_ids": torch.tensor([ids]), "attention_mask": torch.tensor([mask])}


class _FakeSource:
    """Fixed-record source for building a small corpus."""

    def __init__(self, records):
        self.source = records[0].source if records else "real"
        self.trust_level = TrustLevel.REAL
        self._records = records

    def iter_records(self):
        return iter(self._records)


def _corpus_dir(tmp_path: Path, n=4) -> str:
    """Write a small normalized corpus (examples.jsonl + manifest.json)."""
    recs = [SourceRecord(url=f"/redfish/v1/S/{i}",
                         response={"@odata.id": f"/redfish/v1/S/{i}", "Id": str(i)},
                         source="real_dell", trust_level=TrustLevel.REAL,
                         allowed_methods=["GET"], vendor="dell") for i in range(n)]
    mix = SourceMix([_FakeSource(recs)], eval_fraction=0.25, seed=0)
    train, _ = mix.split()
    out = tmp_path / "corpus"
    write_corpus(normalize(train), mix.manifest(), str(out))
    return str(out)


def test_items_are_fixed_length_tensor_dicts(tmp_path: Path):
    """Every item is a {input_ids, attention_mask} pair of length max_len."""
    ds = CorpusJSONLDataset(_corpus_dir(tmp_path), max_len=64, tokenizer=_FakeTokenizer())
    assert len(ds) > 0
    item = ds[0]
    assert set(item) == {"input_ids", "attention_mask"}
    assert item["input_ids"].shape == (64,) and item["attention_mask"].shape == (64,)


def test_items_stack_like_the_trainer_collate(tmp_path: Path):
    """torch.stack over items works — the exact contract of custom_collate_fn."""
    ds = CorpusJSONLDataset(_corpus_dir(tmp_path), max_len=32, tokenizer=_FakeTokenizer())
    batch = {k: torch.stack([ds[i][k] for i in range(len(ds))]) for k in ("input_ids", "attention_mask")}
    assert batch["input_ids"].shape == (len(ds), 32)
    assert batch["attention_mask"].dtype == torch.int64


def test_trainer_duck_type_surface(tmp_path: Path):
    """The masking hooks and tokenizer surface the trainer touches all exist."""
    ds = CorpusJSONLDataset(_corpus_dir(tmp_path), max_len=16, tokenizer=_FakeTokenizer())
    for hook in ("disable_masking", "enable_masking", "mask_section", "mask_new_tokens",
                 "mask_targets", "mask_allowed_value", "mask_odata_id", "mask_targets_key",
                 "mask_objects", "mask_arrays"):
        getattr(ds, hook)()  # must not raise
    assert ds.tokenizer is not None
    ds.load_tokenizer()  # idempotent


def test_run_manifest_fields_round_trip(tmp_path: Path):
    """data_manifest/eval_split ids match what the mixer computed for this corpus."""
    ds = CorpusJSONLDataset(_corpus_dir(tmp_path), max_len=16, tokenizer=_FakeTokenizer())
    fields = ds.run_manifest_fields()
    assert set(fields) == {"data_manifest", "eval_split"}
    assert len(fields["data_manifest"]) == 16  # blake2b hex from DataManifest.content_hash
    assert fields["eval_split"] == "floor=REAL:frac=0.25:seed=0"


def test_missing_corpus_raises(tmp_path: Path):
    """A directory without examples.jsonl fails fast, not at first batch."""
    with pytest.raises(FileNotFoundError):
        CorpusJSONLDataset(str(tmp_path / "nope"), tokenizer=_FakeTokenizer())


# Author: Mus mbayramo@stanford.edu
