"""
Offline regression tests for IgcMain corpus selection.

Pins that a training launch with ``--corpus_dir`` initializes the written
``CorpusJSONLDataset`` path before dispatching to train(), rather than eagerly
rebuilding the legacy masked JSON dataset from raw captures.

Author:
Mus mbayramo@stanford.edu
"""

from argparse import Namespace

from igc.modules import igc_main


class _FakeMetricLogger:
    """Tiny MetricLogger stand-in; IgcMain only needs construction here."""

    def __init__(self, *_args, **_kwargs):
        pass


class _FakeCorpusDataset:
    """Records the corpus arguments used by IgcMain.dataset."""

    constructed = []

    def __init__(self, corpus_dir, default_tokenize=None, max_len=None, objective="legacy"):
        self.corpus_dir = corpus_dir
        self.default_tokenize = default_tokenize
        self.max_len = max_len
        self.objective = objective
        self.tokenizer = object()
        _FakeCorpusDataset.constructed.append(self)


def _spec(tmp_path):
    """Minimal IgcMain spec for exercising run() dispatch."""
    return Namespace(
        metric_report="tensorboard",
        json_data_dir=str(tmp_path / "json"),
        dataset_dir=str(tmp_path / "legacy_dataset"),
        corpus_dir=str(tmp_path / "written_corpus"),
        corpus_objective="phase1_pretrain",
        model_type="gpt2",
        seq_len=128,
        recreate_dataset=False,
        do_consistency_check=False,
        copy_llm=False,
        test_llm=False,
        train="llm",
    )


def test_run_with_corpus_dir_does_not_build_legacy_masked_dataset(monkeypatch, tmp_path):
    """``--corpus_dir`` reaches train() with CorpusJSONLDataset, not MaskedJSONDataset."""
    trained = {}
    _FakeCorpusDataset.constructed.clear()

    def fail_legacy_dataset(*_args, **_kwargs):
        raise AssertionError("MaskedJSONDataset must not be built when corpus_dir is set")

    def fake_train(self):
        trained["dataset"] = self.dataset

    monkeypatch.setattr(igc_main, "MetricLogger", _FakeMetricLogger)
    monkeypatch.setattr(igc_main, "MaskedJSONDataset", fail_legacy_dataset)
    monkeypatch.setattr("igc.ds.corpus_dataset.CorpusJSONLDataset", _FakeCorpusDataset)
    monkeypatch.setattr(igc_main.IgcMain, "train", fake_train)

    main = igc_main.IgcMain(_spec(tmp_path))
    main.run()

    assert trained["dataset"] is _FakeCorpusDataset.constructed[0]
    assert trained["dataset"].corpus_dir == str(tmp_path / "written_corpus")
    assert trained["dataset"].default_tokenize == "gpt2"
    assert trained["dataset"].max_len == 128
    assert trained["dataset"].objective == "phase1_pretrain"


# Author: Mus mbayramo@stanford.edu
