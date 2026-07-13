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

    def __init__(self):
        self.texts = []

    def __call__(self, text, padding=None, max_length=None, truncation=None,
                 return_tensors=None):
        self.texts.append(text)
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
                         response={
                             "@odata.id": f"/redfish/v1/S/{i}",
                             "@odata.type": "#ComputerSystem.v1.ComputerSystem",
                             "Id": str(i),
                             "Status": {"Health": "OK", "State": "Enabled"},
                             "Actions": {"#ComputerSystem.Reset": {"target": f"/redfish/v1/S/{i}/Actions/Reset"}},
                         },
                         source="real_dell", trust_level=TrustLevel.REAL,
                         allowed_methods=["GET", "PATCH"], vendor="dell") for i in range(n)]
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
    assert {"input_ids", "attention_mask", "graph_node_count", "graph_edge_count",
            "action_candidate_count", "candidate_mask", "candidate_resource_type_id",
            "candidate_parent_type_id", "candidate_relation_name_id",
            "candidate_depth_bucket", "candidate_method_id",
            "candidate_has_action_target", "candidate_is_collection", "candidate_is_oem",
            "candidate_path_segment_hashes", "candidate_allowed_method_mask",
            "candidate_local_state_summary", "state_fingerprint", "state_id"} <= set(item)
    assert item["input_ids"].shape == (64,) and item["attention_mask"].shape == (64,)
    assert item["graph_node_count"].item() >= 1
    assert item["action_candidate_count"].item() >= 1
    assert item["candidate_mask"].shape == (16,)
    assert item["candidate_path_segment_hashes"].shape == (16, 8)
    assert item["candidate_allowed_method_mask"].shape == (16, 6)
    assert item["candidate_local_state_summary"].shape == (16, 4)


def test_items_stack_like_the_trainer_collate(tmp_path: Path):
    """torch.stack over items works — the exact contract of custom_collate_fn."""
    ds = CorpusJSONLDataset(_corpus_dir(tmp_path), max_len=32, tokenizer=_FakeTokenizer())
    batch = {k: torch.stack([ds[i][k] for i in range(len(ds))]) for k in ("input_ids", "attention_mask")}
    assert batch["input_ids"].shape == (len(ds), 32)
    assert batch["attention_mask"].dtype == torch.int64


def test_training_example_graph_contract_is_tokenized(tmp_path: Path):
    """M1/M2 consume TrainingExample graph/state fields through the tokenized text."""
    tokenizer = _FakeTokenizer()
    ds = CorpusJSONLDataset(_corpus_dir(tmp_path), max_len=512, tokenizer=tokenizer)
    _ = ds[0]

    rendered = tokenizer.texts[0]
    assert "STATE_FINGERPRINT" not in rendered
    assert "RESOURCE_GRAPH" not in rendered
    assert "RESOURCE_NODE" in rendered
    assert "RESOURCE_JSON" in rendered
    assert "LEGAL_ACTION_SUMMARY" in rendered
    assert "EXPECTED_SEMANTICS" in rendered
    assert "/redfish/v1/S/" in rendered
    assert "GET" in rendered


def test_state_record_exposes_structured_contract(tmp_path: Path):
    """The dataset preserves the typed State contract next to token tensors."""
    ds = CorpusJSONLDataset(_corpus_dir(tmp_path), max_len=256, tokenizer=_FakeTokenizer())
    record = ds.state_record(0)
    assert record["state_id"].startswith("state:")
    assert len(record["state_fingerprint"]) == 32
    assert record["resource_graph"]["node_count"] >= 1
    assert record["resource_node"]["uri"].startswith("/redfish/v1/S/")
    assert record["resource_graph"]["edges"][0]["relation"] == "action_capability"
    assert record["action_affordances"]["candidate_count"] >= 1
    candidate = record["action_affordances"]["templates"][0]
    assert {"resource_type_id", "parent_type_id", "relation_name_id", "depth_bucket",
            "method_id", "has_action_target", "is_collection", "is_oem",
            "path_segment_hashes", "allowed_method_mask", "local_state_summary"} <= set(candidate)
    assert candidate["method_id"] == 1
    assert candidate["has_action_target"] is True
    assert record["action_affordances"]["candidate_count"] == 2
    assert record["action_affordances"]["templates"][1]["method_id"] == 4
    assert len(candidate["path_segment_hashes"]) == 8
    assert len(candidate["allowed_method_mask"]) == 6
    assert record["state_latent"] is None
    assert record["metadata"]["vendor"] == "dell"


def test_unrelated_graph_nodes_do_not_perturb_resource_text(tmp_path: Path):
    """M1 text is per-resource; unrelated graph nodes affect structured graph only."""
    example = {
        "source": "real_dell",
        "trust_level": "REAL",
        "schema_version": "#ComputerSystem.v1.ComputerSystem",
        "resource_graph_before": {
            "/redfish/v1/S/0": {
                "@odata.id": "/redfish/v1/S/0",
                "@odata.type": "#ComputerSystem.v1.ComputerSystem",
                "keys": ["Actions", "Id", "Status"],
            },
            "/redfish/v1/Chassis/9": {
                "@odata.id": "/redfish/v1/Chassis/9",
                "@odata.type": "#Chassis.v1.Chassis",
                "keys": ["Id"],
            },
        },
        "request_or_action": {"method": "GET", "url": "/redfish/v1/S/0", "body": None},
        "response": {
            "@odata.id": "/redfish/v1/S/0",
            "@odata.type": "#ComputerSystem.v1.ComputerSystem",
            "Id": "0",
            "Status": {"Health": "OK", "State": "Enabled"},
            "Actions": {"#ComputerSystem.Reset": {"target": "/redfish/v1/S/0/Actions/Reset"}},
        },
        "resource_graph_after": {},
        "allowed_methods": ["GET", "PATCH"],
        "expected_semantics": {
            "method": "GET",
            "mutating": False,
            "read_only": True,
            "idempotent": True,
            "expected_status": 200,
            "mutable_endpoint": True,
        },
        "vendor": "dell",
        "provenance": {},
    }
    from igc.ds.state_graph import build_state_record

    base_example = {
        **example,
        "resource_graph_before": {
            "/redfish/v1/S/0": example["resource_graph_before"]["/redfish/v1/S/0"],
        },
    }
    base = build_state_record(base_example)
    with_unrelated = build_state_record(example)
    assert with_unrelated.resource_text == base.resource_text
    assert with_unrelated.resource_graph["node_count"] == 2


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
