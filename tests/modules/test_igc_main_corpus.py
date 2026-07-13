"""Offline tests for the main training dataset selection path."""

from types import SimpleNamespace

from igc.modules.igc_main import IgcMain


def test_run_preserves_corpus_dataset(monkeypatch, tmp_path):
    """IgcMain.run() must not overwrite --corpus_dir with the raw JSON dataset."""
    seen = {}

    class FakeMetricLogger:
        def __init__(self, *args, **kwargs):
            pass

    class FakeCorpusDataset:
        tokenizer = object()

        def __init__(self, corpus_dir, default_tokenize, max_len):
            seen["corpus_dir"] = corpus_dir
            seen["default_tokenize"] = default_tokenize
            seen["max_len"] = max_len

    def fake_train(self):
        seen["dataset_type"] = type(self.dataset).__name__

    monkeypatch.setattr("igc.modules.igc_main.MetricLogger", FakeMetricLogger)
    monkeypatch.setattr("igc.ds.corpus_dataset.CorpusJSONLDataset", FakeCorpusDataset)
    monkeypatch.setattr(IgcMain, "train", fake_train)

    args = SimpleNamespace(
        metric_report="tensorboard",
        json_data_dir=str(tmp_path / "json"),
        dataset_dir=str(tmp_path / "dataset"),
        corpus_dir=str(tmp_path / "corpus"),
        model_type="gpt2",
        seq_len=128,
        recreate_dataset=False,
        do_consistency_check=False,
        copy_llm=False,
        test_llm=False,
    )

    IgcMain(args).run()

    assert seen == {
        "corpus_dir": str(tmp_path / "corpus"),
        "default_tokenize": "gpt2",
        "max_len": 128,
        "dataset_type": "FakeCorpusDataset",
    }
