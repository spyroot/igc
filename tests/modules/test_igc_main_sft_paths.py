"""Offline tests for IgcMain shared SFT train/eval dataset routing."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from igc.modules import igc_main


_SHA_MODEL_X = "sha256:" + "1" * 64
_SHA_GOAL_EXTRACTOR = "sha256:" + "2" * 64


class _FakeMetricLogger:
    """MetricLogger stand-in; IgcMain only needs construction here."""

    def __init__(self, *_args, **_kwargs):
        pass


class _FakePromptCompletionDataset:
    """Capture train/eval SFT dataset construction without tokenizer downloads."""

    instances = []

    def __init__(
        self,
        path,
        *,
        renderer,
        metric_namespace,
        max_len,
        tokenizer=None,
        tokenizer_name=None,
        manifest_path=None,
        require_manifest=False,
        expected_dataset=None,
    ):
        self.path = Path(path)
        self.renderer_name = renderer.__name__
        self.metric_namespace = metric_namespace
        self.max_len = max_len
        self.tokenizer_name = tokenizer_name
        self.input_tokenizer = tokenizer
        self.manifest_path = Path(manifest_path) if manifest_path is not None else None
        self.require_manifest = require_manifest
        self.expected_dataset = expected_dataset
        self.tokenizer = tokenizer or object()
        suffix = "b" if tokenizer is not None else "a"
        self.data_sha256 = "sha256:" + suffix * 64
        self.manifest_sha256 = "sha256:" + suffix.upper() * 64
        self.eval_split_sha256 = ""
        self.eval_manifest_sha256 = ""
        self.eval_data_sha = ""
        self.bound_eval_dataset = None
        _FakePromptCompletionDataset.instances.append(self)

    def set_eval_split_sha256(self, value):
        self.eval_split_sha256 = value
        self.eval_data_sha = value

    def bind_eval_dataset(self, dataset):
        self.bound_eval_dataset = dataset
        self.eval_split_sha256 = dataset.data_sha256
        self.eval_data_sha = dataset.data_sha256
        self.eval_manifest_sha256 = dataset.manifest_sha256


class _FakeCorpusJSONLDataset:
    """Capture Phase 1 corpus dataset construction without tokenizer downloads."""

    instances = []

    def __init__(
        self,
        corpus_dir,
        *,
        default_tokenize,
        max_len,
        tokenizer=None,
        objective="legacy",
        phase1_structural_loss_profile="none",
        phase1_structural_loss_mode="train",
        phase1_structural_loss_seed=42,
    ):
        self.path = Path(corpus_dir)
        self.default_tokenize = default_tokenize
        self.max_len = max_len
        self.input_tokenizer = tokenizer
        self.objective = objective
        self.phase1_structural_loss_profile = phase1_structural_loss_profile
        self.phase1_structural_loss_mode = phase1_structural_loss_mode
        self.phase1_structural_loss_seed = phase1_structural_loss_seed
        self.phase1_structural_loss_spec_sha = "sha256:" + "c" * 64
        self.tokenizer = tokenizer or object()
        suffix = "d" if tokenizer is None else "e"
        self.data_sha256 = "sha256:" + suffix * 64
        self.eval_split_sha256 = ""
        _FakeCorpusJSONLDataset.instances.append(self)

    def set_eval_split_sha256(self, value):
        self.eval_split_sha256 = value


def _spec(
    tmp_path: Path,
    *,
    phase: int,
    eval_path: str | None = None,
    data_manifest: str | None = "train.manifest.json",
    eval_manifest: str | None = "eval.manifest.json",
) -> Namespace:
    """Minimal spec for Phase2/3 labelled SFT dataset construction."""
    if phase == 2:
        task = "text_to_rest_api_list"
        weights_role = "goal_extractor"
        parent_adapter_dir = "/models/model_x"
        parent_artifact_sha = _SHA_MODEL_X
    elif phase == 3:
        task = "text_and_rest_api_list_to_calls"
        weights_role = "argument_extractor"
        parent_adapter_dir = "/models/goal_extractor"
        parent_artifact_sha = _SHA_GOAL_EXTRACTOR
    else:  # pragma: no cover - helper is intentionally narrow
        raise AssertionError(phase)
    return Namespace(
        metric_report="none",
        json_data_dir=str(tmp_path / "json"),
        dataset_dir=str(tmp_path / "datasets"),
        corpus_dir="",
        corpus_eval_dir="",
        corpus_manifest="",
        corpus_root="",
        corpus_kind="dataset",
        corpus_objective="phase1_pretrain",
        model_type="gpt2",
        seq_len=128,
        recreate_dataset=False,
        do_consistency_check=False,
        copy_llm=False,
        test_llm=False,
        train="",
        llm=None,
        rl=None,
        device="cpu",
        sft_task=task,
        sft_data_path=str(tmp_path / f"phase{phase}-train.jsonl"),
        sft_eval_data_path="" if eval_path is None else eval_path,
        sft_data_manifest="" if data_manifest is None else str(tmp_path / data_manifest),
        sft_eval_manifest="" if eval_manifest is None else str(tmp_path / eval_manifest),
        weights_role=weights_role,
        parent_adapter_dir=parent_adapter_dir,
        parent_artifact_sha=parent_artifact_sha,
    )


def _phase1_spec(
    tmp_path: Path,
    *,
    corpus_dir: str = "",
    corpus_eval_dir: str = "",
    corpus_manifest: str = "",
    corpus_root: str = "",
) -> Namespace:
    """Minimal spec for Phase 1 shared SFT corpus routing."""
    return Namespace(
        metric_report="none",
        json_data_dir=str(tmp_path / "json"),
        dataset_dir=str(tmp_path / "datasets"),
        corpus_dir=corpus_dir,
        corpus_eval_dir=corpus_eval_dir,
        corpus_manifest=corpus_manifest,
        corpus_root=corpus_root,
        corpus_kind="dataset",
        corpus_objective="phase1_pretrain",
        model_type="gpt2",
        seq_len=128,
        recreate_dataset=False,
        do_consistency_check=False,
        copy_llm=False,
        test_llm=False,
        train="",
        llm=None,
        rl=None,
        device="cpu",
        sft_task="redfish_json_reconstruction",
        sft_data_path="",
        sft_eval_data_path="",
        sft_data_manifest="",
        sft_eval_manifest="",
        weights_role="model_x",
        parent_adapter_dir="",
        parent_artifact_sha="",
    )


@pytest.mark.parametrize(
    ("phase", "namespace", "renderer_name"),
    [
        (2, "phase2_goal_extraction", "render_phase2_sft"),
        (3, "phase3_argument_extraction", "render_phase3_sft"),
    ],
)
def test_igc_main_builds_explicit_phase2_phase3_train_eval_sft_datasets(
    monkeypatch,
    tmp_path,
    phase,
    namespace,
    renderer_name,
) -> None:
    """Phase2/3 SFT uses explicit train/eval JSONL paths and binds eval SHA."""
    monkeypatch.setattr(igc_main, "MetricLogger", _FakeMetricLogger)
    monkeypatch.setattr(
        "igc.ds.sft_dataset.PromptCompletionJSONLDataset",
        _FakePromptCompletionDataset,
    )
    _FakePromptCompletionDataset.instances.clear()
    specs = _spec(
        tmp_path,
        phase=phase,
        eval_path=str(tmp_path / f"phase{phase}-eval.jsonl"),
    )

    main = igc_main.IgcMain(specs)
    dataset = main.dataset
    eval_dataset = main.eval_dataset

    assert dataset is _FakePromptCompletionDataset.instances[0]
    assert eval_dataset is _FakePromptCompletionDataset.instances[1]
    assert dataset.path == Path(specs.sft_data_path)
    assert eval_dataset.path == Path(specs.sft_eval_data_path)
    assert dataset.manifest_path == Path(specs.sft_data_manifest)
    assert eval_dataset.manifest_path == Path(specs.sft_eval_manifest)
    assert dataset.require_manifest is True
    assert eval_dataset.require_manifest is True
    assert dataset.expected_dataset == "D1"
    assert eval_dataset.expected_dataset == "D1"
    assert dataset.metric_namespace == namespace
    assert eval_dataset.metric_namespace == namespace
    assert dataset.renderer_name == renderer_name
    assert eval_dataset.renderer_name == renderer_name
    assert dataset.tokenizer_name == "gpt2"
    assert eval_dataset.input_tokenizer is dataset.tokenizer
    assert dataset.bound_eval_dataset is eval_dataset
    assert dataset.eval_split_sha256 == eval_dataset.data_sha256
    assert dataset.eval_data_sha == eval_dataset.data_sha256
    assert dataset.eval_manifest_sha256 == eval_dataset.manifest_sha256
    assert specs.task_spec_sha.startswith("sha256:")
    assert specs.phase_number == phase


def test_igc_main_phase2_requires_explicit_sft_eval_data_path(
    monkeypatch,
    tmp_path,
) -> None:
    """Shared SFT must fail before constructing a random split for Phase 2."""
    monkeypatch.setattr(igc_main, "MetricLogger", _FakeMetricLogger)
    specs = _spec(tmp_path, phase=2, eval_path=None)

    with pytest.raises(ValueError, match="--sft_eval_data_path"):
        _ = igc_main.IgcMain(specs).dataset


@pytest.mark.parametrize(
    ("data_manifest", "eval_manifest", "message"),
    [
        (None, "eval.manifest.json", "--sft_data_manifest"),
        ("train.manifest.json", None, "--sft_eval_manifest"),
    ],
)
def test_igc_main_phase2_requires_explicit_sft_release_manifests(
    monkeypatch,
    tmp_path,
    data_manifest,
    eval_manifest,
    message,
) -> None:
    """Phase 2/3 SFT requires immutable manifests paired with train and eval JSONL."""
    monkeypatch.setattr(igc_main, "MetricLogger", _FakeMetricLogger)
    specs = _spec(
        tmp_path,
        phase=2,
        eval_path=str(tmp_path / "phase2-eval.jsonl"),
        data_manifest=data_manifest,
        eval_manifest=eval_manifest,
    )

    with pytest.raises(ValueError, match=message):
        _ = igc_main.IgcMain(specs).dataset


def test_igc_main_phase1_shared_sft_requires_canonical_registry_corpus_dirs(
    monkeypatch,
    tmp_path,
) -> None:
    """Phase 1 SFT rejects manifest rematerialization and requires train/eval D0 dirs."""
    monkeypatch.setattr(igc_main, "MetricLogger", _FakeMetricLogger)

    manifest_specs = _phase1_spec(
        tmp_path,
        corpus_manifest=str(tmp_path / "redfish_ctl-manifest.json"),
        corpus_root=str(tmp_path / "redfish_ctl-root"),
    )
    with pytest.raises(ValueError, match="build_phase1_registry_corpus.py"):
        _ = igc_main.IgcMain(manifest_specs).dataset

    train_only_specs = _phase1_spec(
        tmp_path,
        corpus_dir=str(tmp_path / "registry-train-d0"),
    )
    with pytest.raises(ValueError, match="--corpus_eval_dir is required"):
        _ = igc_main.IgcMain(train_only_specs).dataset

    monkeypatch.setattr(
        "igc.ds.corpus_dataset.CorpusJSONLDataset",
        _FakeCorpusJSONLDataset,
    )
    _FakeCorpusJSONLDataset.instances.clear()
    specs = _phase1_spec(
        tmp_path,
        corpus_dir=str(tmp_path / "registry-train-d0"),
        corpus_eval_dir=str(tmp_path / "registry-heldout-d0"),
    )

    main = igc_main.IgcMain(specs)
    dataset = main.dataset
    eval_dataset = main.eval_dataset

    assert dataset is _FakeCorpusJSONLDataset.instances[0]
    assert eval_dataset is _FakeCorpusJSONLDataset.instances[1]
    assert dataset.path == Path(specs.corpus_dir)
    assert eval_dataset.path == Path(specs.corpus_eval_dir)
    assert dataset.objective == "phase1_pretrain"
    assert eval_dataset.objective == "phase1_pretrain"
    assert dataset.phase1_structural_loss_profile == "none"
    assert dataset.phase1_structural_loss_mode == "train"
    assert eval_dataset.phase1_structural_loss_mode == "evaluation"
    assert dataset.default_tokenize == "gpt2"
    assert eval_dataset.input_tokenizer is dataset.tokenizer
    assert dataset.eval_split_sha256 == eval_dataset.data_sha256
    assert specs.task_spec_sha.startswith("sha256:")
    assert specs.phase1_structural_loss_spec_sha == "sha256:" + "c" * 64
    assert specs.phase_number == 1
