"""Offline regression: validate() tolerates an empty eval shard.

Under FSDP with drop_last + a small eval set, a rank can be handed 0 eval batches.
The accuracy computation divided correct/total unconditionally, so that rank crashed
with ZeroDivisionError mid-epoch — which, on a multi-GPU run, takes the whole fleet
down at a collective. validate() now returns 0.0 for an empty shard. (The companion
epoch-boundary barrier fix is distributed-only and is exercised on the 4-GPU run.)

Author:
Mus mbayramo@stanford.edu
"""

import types

import pytest

import igc.modules.train.sft as sft
from igc.modules.train.sft import SFTTrainer


def test_validate_empty_eval_returns_zero_not_crash():
    """0 eval batches -> 0.0 accuracy, no ZeroDivisionError."""
    trainer = SFTTrainer.__new__(SFTTrainer)
    trainer.model = types.SimpleNamespace(eval=lambda: None)

    # empty dataloader -> the batch loop never runs -> total_predictions == 0
    assert trainer.validate([]) == 0.0


class _SizedDataset:
    """Tiny dataset double carrying a metric namespace and tokenizer."""

    def __init__(self, n: int, metric_namespace: str = "phase2_goal_extraction"):
        self._n = n
        self.metric_namespace = metric_namespace
        self.tokenizer = object()

    def __len__(self):
        return self._n


def _split_trainer(train_dataset, eval_dataset=None):
    trainer = SFTTrainer.__new__(SFTTrainer)
    trainer.dataset = train_dataset
    trainer._eval_dataset = eval_dataset
    trainer._metric_namespace = getattr(train_dataset, "metric_namespace", "")
    return trainer


def test_shared_sft_split_dataset_uses_explicit_eval_without_random_split():
    """Metric-namespaced SFT tasks return the immutable train/eval datasets as-is."""
    train = _SizedDataset(3, metric_namespace="phase1_finetune")
    heldout = _SizedDataset(2, metric_namespace="phase1_finetune")
    trainer = _split_trainer(train, heldout)

    assert trainer.split_dataset(ratio=0.1) == (train, heldout)


@pytest.mark.parametrize(
    ("train_size", "eval_size", "message"),
    [
        (3, None, "explicit immutable held-out dataset"),
        (0, 2, "must be non-empty"),
        (3, 0, "must be non-empty"),
    ],
)
def test_shared_sft_split_dataset_requires_explicit_non_empty_eval(
    train_size,
    eval_size,
    message,
):
    """Shared SFT rejects absent or empty eval data instead of random splitting."""
    train = _SizedDataset(train_size)
    heldout = None if eval_size is None else _SizedDataset(eval_size)
    trainer = _split_trainer(train, heldout)

    with pytest.raises(ValueError, match=message):
        trainer.split_dataset()


def test_train_builds_eval_dataloader_without_drop_last(monkeypatch):
    """The eval DataLoader preserves short held-out shards with drop_last=False."""
    captured = []

    class StopAfterDataloaders(RuntimeError):
        pass

    class _FakeDataLoader:
        def __init__(self, dataset, **kwargs):
            self.dataset = dataset
            self.kwargs = kwargs
            captured.append(kwargs)

        def __len__(self):
            return 1

    class _FakeModel:
        device = "cpu"

        def to(self, _device):
            return self

    trainer = SFTTrainer.__new__(SFTTrainer)
    trainer.model = _FakeModel()
    trainer.dataset = _SizedDataset(3, metric_namespace="phase3_argument_extraction")
    trainer._eval_dataset = _SizedDataset(1, metric_namespace="phase3_argument_extraction")
    trainer._metric_namespace = "phase3_argument_extraction"
    trainer._trainer_args = types.SimpleNamespace(
        gradient_accumulation_steps=1,
        llm_scheduler="OneCycleLR",
        max_train_steps=0,
    )
    trainer._module_checkpoint_dir = None
    trainer._lr = 1e-4
    trainer._reset_lr = False
    trainer.rank = -1
    trainer.is_accelerator = False
    trainer.device = "cpu"
    trainer.batch_size = 2
    trainer._num_workers = 0
    trainer._is_shuffle = True
    trainer._pin_memory = False
    trainer.num_epochs = 1
    trainer._select_best_by_eval_loss = True
    trainer._best_validation_metric = float("inf")
    trainer._best_checkpoint_path = ""
    trainer.module_name = "fixture"
    trainer.logger = types.SimpleNamespace(info=lambda *_args, **_kwargs: None)
    trainer.optimizer = object()

    monkeypatch.setattr(sft, "safe_resize_token_embeddings", lambda *_args: None)
    monkeypatch.setattr(sft, "DataLoader", _FakeDataLoader)
    monkeypatch.setattr(SFTTrainer, "dataset_sampler", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        SFTTrainer,
        "load_checkpoint",
        lambda *_args, **_kwargs: types.SimpleNamespace(
            last_epoch=0,
            best_accuracy=0.0,
            best_metric=None,
            best_metric_mode=None,
            scheduler_state=None,
            batch_idx=0,
        ),
    )
    monkeypatch.setattr(
        sft.TorchBuilder,
        "create_scheduler",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(StopAfterDataloaders()),
    )

    with pytest.raises(StopAfterDataloaders):
        trainer._train()

    assert captured[0]["drop_last"] is True
    assert captured[1]["drop_last"] is False
    assert captured[1]["shuffle"] is False


# Author: Mus mbayramo@stanford.edu
