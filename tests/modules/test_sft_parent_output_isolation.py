"""Offline tests for SFT parent/output adapter isolation."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from igc.modules.train import sft
from igc.modules.train.sft import validate_parent_output_isolation


def test_parent_output_isolation_allows_phase1_empty_parent(tmp_path: Path) -> None:
    """Phase 1 has no parent adapter, so any resolved output directory is valid."""
    output = tmp_path / "phase1" / "model_x"

    validate_parent_output_isolation("", str(output))


def test_parent_output_isolation_allows_distinct_resolved_paths(tmp_path: Path) -> None:
    """Phase 2/3 may read one immutable parent and write a separate adapter."""
    parent = tmp_path / "parents" / "model_x"
    output = tmp_path / "runs" / "goal_extractor"
    parent.mkdir(parents=True)
    output.mkdir(parents=True)

    validate_parent_output_isolation(str(parent), str(output))


@pytest.mark.parametrize(
    "output_factory",
    [
        pytest.param(lambda parent: parent, id="identical-path"),
        pytest.param(lambda parent: parent / ".." / parent.name, id="dotdot-alias"),
    ],
)
def test_parent_output_isolation_rejects_same_resolved_path(
    tmp_path: Path,
    output_factory,
) -> None:
    """Identical or alias-resolved parent/output paths would overwrite the parent."""
    parent = tmp_path / "adapters" / "model_x"
    parent.mkdir(parents=True)

    with pytest.raises(ValueError, match="must differ"):
        validate_parent_output_isolation(str(parent), str(output_factory(parent)))


def test_sft_trainer_invokes_isolation_after_base_initialization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """SFTTrainer gets the resolved fine-tune output from base init before checking."""

    class StopAfterIsolation(RuntimeError):
        pass

    order: list[str] = []
    captured: dict[str, str] = {}
    parent = tmp_path / "parents" / "model_x"
    output_root = tmp_path / "runs"

    def fake_base_init(
        self,
        module_name,
        spec,
        llm_model,
        llm_tokenizer,
        ds=None,
        metric_logger=None,
        is_inference=False,
        device=None,
    ) -> None:
        _ = llm_model, llm_tokenizer, ds, metric_logger, is_inference, device
        order.append("base")
        self._trainer_args = spec
        self._module_checkpoint_dir = str(output_root / module_name)

    def fake_validate(parent_adapter_dir: str, output_adapter_dir: str) -> None:
        assert order == ["base"]
        captured["parent"] = parent_adapter_dir
        captured["output"] = output_adapter_dir
        raise StopAfterIsolation

    monkeypatch.setattr(sft.LlmModule, "__init__", fake_base_init)
    monkeypatch.setattr(sft, "validate_parent_output_isolation", fake_validate)

    with pytest.raises(StopAfterIsolation):
        sft.SFTTrainer(
            "phase2",
            Namespace(parent_adapter_dir=str(parent)),
            llm_model=object(),
            llm_tokenizer=object(),
            dataset=object(),
        )

    assert captured == {
        "parent": str(parent),
        "output": str(output_root / "phase2" / "fine_tuned"),
    }


class _TinyDataset:
    """Small trainer dataset double for config-read tests."""

    tokenizer = object()
    metric_namespace = ""

    def __len__(self) -> int:
        return 2


def _minimal_sft_spec(**overrides) -> Namespace:
    values = {
        "parent_adapter_dir": "",
        "num_train_epochs": 1,
        "per_device_train_batch_size": 1,
        "llm_mask_freq": 1,
        "num_workers": 0,
        "llm_learning_rate": 1e-4,
        "llm_optimizer": "AdamW",
        "llm_weight_decay": 0.0,
        "max_grad_norm": 1.0,
    }
    values.update(overrides)
    return Namespace(**values)


def _patch_lightweight_trainer_init(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_base_init(
        self,
        module_name,
        spec,
        llm_model,
        llm_tokenizer,
        ds=None,
        metric_logger=None,
        is_inference=False,
        device=None,
    ) -> None:
        _ = llm_tokenizer, ds, metric_logger, is_inference, device
        self._trainer_args = spec
        self._module_checkpoint_dir = str(tmp_path / "runs" / module_name)
        self.model = llm_model
        self.rank = -1
        self._overfit = False
        self.logger = SimpleNamespace(info=lambda *_args, **_kwargs: None)

    monkeypatch.setattr(sft.LlmModule, "__init__", fake_base_init)
    monkeypatch.setattr(sft, "validate_parent_output_isolation", lambda *_args: None)
    monkeypatch.setattr(
        sft.TorchBuilder,
        "create_optimizer",
        staticmethod(lambda *_args, **_kwargs: object()),
    )


@pytest.mark.parametrize(
    ("reset_lr_value", "expected"),
    [
        pytest.param(None, False, id="absent-default-false"),
        pytest.param(False, False, id="explicit-false"),
        pytest.param(True, True, id="explicit-true"),
    ],
)
def test_sft_trainer_reads_reset_lr_from_namespace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    reset_lr_value: bool | None,
    expected: bool,
) -> None:
    """reset_lr is an optional argparse.Namespace attribute, defaulting to False."""
    _patch_lightweight_trainer_init(monkeypatch, tmp_path)
    spec = (
        _minimal_sft_spec()
        if reset_lr_value is None
        else _minimal_sft_spec(reset_lr=reset_lr_value)
    )

    trainer = sft.SFTTrainer(
        "phase1",
        spec,
        llm_model=object(),
        llm_tokenizer=object(),
        dataset=_TinyDataset(),
    )

    assert trainer._reset_lr is expected


@pytest.mark.parametrize(
    ("namespace", "expect_random_sampler"),
    [
        pytest.param(Namespace(), False, id="absent-default-none"),
        pytest.param(Namespace(random_sampler_enabled=False), False, id="explicit-false"),
        pytest.param(Namespace(random_sampler_enabled=True), True, id="explicit-true"),
    ],
)
def test_sft_dataset_sampler_reads_random_sampler_enabled_from_namespace(
    namespace: Namespace,
    expect_random_sampler: bool,
) -> None:
    """dataset_sampler honors optional Namespace.random_sampler_enabled safely."""
    trainer = sft.SFTTrainer.__new__(sft.SFTTrainer)
    trainer.dataset = _TinyDataset()
    trainer._trainer_args = namespace

    sampler = trainer.dataset_sampler()

    if expect_random_sampler:
        assert isinstance(sampler, sft.RandomSampler)
        assert sampler.data_source is trainer.dataset
    else:
        assert sampler is None
