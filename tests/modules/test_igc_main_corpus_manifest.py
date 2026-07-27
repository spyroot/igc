"""
Offline regression tests for IgcMain corpus selection.

Pins that the redfish_ctl manifest flags feed a live dataset path and that
``IgcMain.run`` does not overwrite an already selected corpus dataset with the
legacy raw-capture dataset.

Author:
Mus mbayramo@stanford.edu
"""

import argparse
import json
from pathlib import Path

import pytest

from igc.ds.sources.base import SourceRecord, TrustLevel
from igc.ds.sources.mixer import DataManifest
from igc.modules.igc_main import IgcMain


class _FakeMetricLogger:
    """Small stand-in for MetricLogger; no files or backends."""

    def __init__(self, *args, **kwargs):
        pass


class _FakeCorpusDataset:
    """Capture the corpus directory IgcMain asks the tokenizer bridge to load."""

    instances = []

    def __init__(
        self,
        corpus_dir,
        default_tokenize=None,
        max_len=None,
        tokenizer=None,
        objective=None,
        phase1_structural_loss_profile="none",
        phase1_structural_loss_mode="train",
        phase1_structural_loss_seed=42,
    ):
        self.corpus_dir = Path(corpus_dir)
        self.default_tokenize = default_tokenize
        self.max_len = max_len
        self.input_tokenizer = tokenizer
        self.objective = objective
        self.phase1_structural_loss_profile = phase1_structural_loss_profile
        self.phase1_structural_loss_mode = phase1_structural_loss_mode
        self.phase1_structural_loss_seed = phase1_structural_loss_seed
        self.phase1_structural_loss_spec_sha = "sha256:" + "c" * 64
        self.tokenizer = tokenizer or object()
        suffix = "b" if tokenizer is not None else "a"
        self.data_sha256 = "sha256:" + suffix * 64
        self.eval_split_sha256 = ""
        _FakeCorpusDataset.instances.append(self)

    def set_eval_split_sha256(self, value):
        self.eval_split_sha256 = value


class _FakeSourceMix:
    """Deterministic train/heldout split for materialization tests."""

    def __init__(self, _sources):
        self.train_records = [
            SourceRecord(
                url="/redfish/v1/Systems/Train",
                response={
                    "@odata.id": "/redfish/v1/Systems/Train",
                    "@odata.type": "#ComputerSystem.v1_20_0.ComputerSystem",
                    "Id": "Train",
                },
                source="real_dell",
                trust_level=TrustLevel.REAL,
                allowed_methods=["GET", "HEAD", "PATCH"],
                vendor="dell",
                provenance={"corpus_id": "dell-xr8620t"},
            )
        ]
        self.heldout_records = [
            SourceRecord(
                url="/redfish/v1/Systems/Heldout",
                response={
                    "@odata.id": "/redfish/v1/Systems/Heldout",
                    "@odata.type": "#ComputerSystem.v1_20_0.ComputerSystem",
                    "Id": "Heldout",
                },
                source="real_dell",
                trust_level=TrustLevel.REAL,
                allowed_methods=["GET", "HEAD"],
                vendor="dell",
                provenance={"corpus_id": "dell-xr8620t"},
            )
        ]

    def split(self):
        return self.train_records, self.heldout_records

    def manifest(self):
        return DataManifest(
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
            train_row_ids=["sha256:" + "1" * 64],
            heldout_row_ids=["sha256:" + "2" * 64],
        )


def _specs(tmp_path: Path, **overrides):
    """Build the minimum argparse namespace IgcMain touches in these tests."""
    values = {
        "metric_report": "none",
        "json_data_dir": str(tmp_path / "json"),
        "dataset_dir": str(tmp_path / "datasets"),
        "corpus_dir": "",
        "corpus_manifest": "",
        "corpus_root": "",
        "corpus_kind": "dataset",
        "model_type": "gpt2",
        "seq_len": 32,
        "recreate_dataset": False,
        "do_consistency_check": False,
        "copy_llm": False,
        "test_llm": False,
        "train": "",
        "llm": None,
        "rl": None,
        "device": "cpu",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _write_json(path: Path, body) -> None:
    """Write a JSON body under ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body), encoding="utf-8")


def test_redfish_ctl_manifest_materializes_live_corpus(tmp_path, monkeypatch):
    """--corpus_manifest/--corpus_root build train+heldout corpora then load both."""
    monkeypatch.setattr("igc.modules.igc_main.MetricLogger", _FakeMetricLogger)
    monkeypatch.setattr("igc.ds.corpus_dataset.CorpusJSONLDataset", _FakeCorpusDataset)
    monkeypatch.setattr(
        "igc.ds.sources.RedfishFixtureSource.from_redfish_ctl_manifest",
        staticmethod(lambda *_args, **_kwargs: [object()]),
    )
    monkeypatch.setattr("igc.ds.sources.mixer.SourceMix", _FakeSourceMix)
    _FakeCorpusDataset.instances.clear()

    manifest = tmp_path / "manifest.v1.json"
    materialized = tmp_path / "materialized"
    root = materialized / "dataset" / "dell_xr8620t"
    _write_json(
        manifest,
        {
            "schema_version": 1,
            "corpora": [
                {
                    "id": "dell-xr8620t",
                    "kind": "dataset",
                    "vendor": "dell",
                    "model": "xr8620t",
                    "capture_id": "capture-1",
                    "archive": "corpora/dataset/dell_xr8620t.tar.gz",
                }
            ],
        },
    )
    _write_json(
        root / "rest_api_map.v1.json",
        {
            "url_file_mapping": {
                "/redfish/v1/Systems/1": "json_responses/_redfish_v1_Systems_1.json"
            },
            "allowed_methods_mapping": {"/redfish/v1/Systems/1": ["GET", "HEAD", "PATCH"]},
        },
    )
    _write_json(
        root / "json_responses" / "_redfish_v1_Systems_1.json",
        {
            "@odata.id": "/redfish/v1/Systems/1",
            "@odata.type": "#ComputerSystem.v1_20_0.ComputerSystem",
            "Id": "1",
        },
    )

    specs = _specs(
        tmp_path,
        corpus_manifest=str(manifest),
        corpus_root=str(materialized),
        corpus_kind="dataset",
    )
    dataset = IgcMain(specs).dataset

    assert dataset is _FakeCorpusDataset.instances[0]
    eval_dataset = _FakeCorpusDataset.instances[1]
    assert dataset.default_tokenize == "gpt2"
    assert dataset.max_len == 32
    assert dataset.objective == "legacy"
    assert eval_dataset.default_tokenize == "gpt2"
    assert eval_dataset.max_len == 32
    assert eval_dataset.input_tokenizer is dataset.tokenizer
    assert eval_dataset.objective == "legacy"
    assert dataset.eval_split_sha256 == eval_dataset.data_sha256
    examples_path = dataset.corpus_dir / "examples.jsonl"
    heldout_examples_path = eval_dataset.corpus_dir / "examples.jsonl"
    manifest_path = dataset.corpus_dir / "manifest.json"
    heldout_manifest_path = eval_dataset.corpus_dir / "manifest.json"
    assert examples_path.is_file()
    assert heldout_examples_path.is_file()
    assert manifest_path.is_file()
    assert heldout_manifest_path.is_file()

    row = json.loads(examples_path.read_text(encoding="utf-8").strip())
    heldout_row = json.loads(heldout_examples_path.read_text(encoding="utf-8").strip())
    assert row["request_or_action"] == {
        "method": "GET",
        "url": "/redfish/v1/Systems/Train",
        "body": None,
    }
    assert row["allowed_methods"] == ["GET", "HEAD", "PATCH"]
    assert row["vendor"] == "dell"
    assert row["provenance"]["corpus_id"] == "dell-xr8620t"
    assert heldout_row["request_or_action"]["url"] == "/redfish/v1/Systems/Heldout"
    assert heldout_row["allowed_methods"] == ["GET", "HEAD"]
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["eval_count"] == 1
    assert json.loads(heldout_manifest_path.read_text(encoding="utf-8"))["train_count"] == 1


def test_manifest_requires_materialized_root(tmp_path, monkeypatch):
    """A manifest without --corpus_root fails with a clear local error."""
    monkeypatch.setattr("igc.modules.igc_main.MetricLogger", _FakeMetricLogger)
    specs = _specs(tmp_path, corpus_manifest=str(tmp_path / "manifest.v1.json"))

    with pytest.raises(ValueError, match="--corpus_root"):
        _ = IgcMain(specs).dataset


def test_run_keeps_existing_dataset(tmp_path, monkeypatch):
    """run() must not replace a selected corpus dataset with MaskedJSONDataset."""
    monkeypatch.setattr("igc.modules.igc_main.MetricLogger", _FakeMetricLogger)

    def _raise_masked(*args, **kwargs):
        raise AssertionError("MaskedJSONDataset should not be constructed by run()")

    monkeypatch.setattr("igc.modules.igc_main.MaskedJSONDataset", _raise_masked)
    main = IgcMain(
        _specs(
            tmp_path,
            corpus_dir=str(tmp_path / "written-corpus"),
            corpus_eval_dir=str(tmp_path / "heldout-corpus"),
        )
    )
    sentinel = object()
    main._dataset = sentinel

    main.run()

    assert main._dataset is sentinel


# Author: Mus mbayramo@stanford.edu
