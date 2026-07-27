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

    def __init__(
        self,
        corpus_dir,
        default_tokenize=None,
        max_len=None,
        tokenizer=None,
        objective="legacy",
        phase1_structural_loss_profile="none",
        phase1_structural_loss_mode="train",
        phase1_structural_loss_seed=42,
    ):
        self.corpus_dir = corpus_dir
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
        self.eval_data_sha = ""
        _FakeCorpusDataset.constructed.append(self)

    def set_eval_split_sha256(self, value):
        self.eval_split_sha256 = value
        self.eval_data_sha = value


def _spec(tmp_path):
    """Minimal IgcMain spec for exercising run() dispatch."""
    return Namespace(
        metric_report="tensorboard",
        json_data_dir=str(tmp_path / "json"),
        dataset_dir=str(tmp_path / "legacy_dataset"),
        corpus_dir=str(tmp_path / "written_corpus"),
        corpus_eval_dir=str(tmp_path / "heldout_corpus"),
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
    assert main.eval_dataset is _FakeCorpusDataset.constructed[1]
    assert trained["dataset"].corpus_dir == str(tmp_path / "written_corpus")
    assert trained["dataset"].default_tokenize == "gpt2"
    assert trained["dataset"].max_len == 128
    assert trained["dataset"].objective == "phase1_pretrain"
    assert main.eval_dataset.corpus_dir == str(tmp_path / "heldout_corpus")
    assert main.eval_dataset.input_tokenizer is trained["dataset"].tokenizer
    assert main.eval_dataset.objective == "phase1_pretrain"
    assert trained["dataset"].phase1_structural_loss_mode == "train"
    assert main.eval_dataset.phase1_structural_loss_mode == "evaluation"
    assert trained["dataset"].eval_split_sha256 == main.eval_dataset.data_sha256
    assert trained["dataset"].eval_data_sha == main.eval_dataset.data_sha256


# Author: Mus mbayramo@stanford.edu
