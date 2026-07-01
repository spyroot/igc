"""M1 trainer regressions for the CUDA path.

The offline test exercises the plain, non-accelerator end-of-train path so the
guarded unwrap regression is caught without a GPU. The CUDA-marked test stays
skipped on CPU-only hosts and checks the M1 device-placement and one-step loss
surface when a local GPU is available.

Author:
Mus mbayramo@stanford.edu
"""

import argparse
from types import SimpleNamespace

import pytest
import torch

from igc.modules.base.igc_base_module import CheckpointState
from igc.modules.llm_train_state_encoder import LlmEmbeddingsTrainer
from igc.shared.shared_torch_builder import TorchBuilder


class _TinyTokenizer:
    """Tiny tokenizer facade used by the trainer's embedding resize step."""

    pad_token_id = 0

    def __len__(self):
        """Return a tiny stable vocab size."""
        return 32


class _EmptyDataset:
    """Dataset facade with the tokenizer hooks used by the trainer."""

    tokenizer = _TinyTokenizer()

    def disable_masking(self):
        """No-op masking hook for the trainer."""


class _Recorder:
    """Records whether save hooks were reached."""

    def __init__(self):
        """Initialize counters for save assertions."""
        self.save_model_calls = 0
        self.save_finetuned_calls = 0

    def save_model(self, checkpoint_dir):
        """Record a save_model call."""
        self.save_model_calls += 1

    def save_finetuned(self):
        """Record a save_finetuned call."""
        self.save_finetuned_calls += 1


class _PlainModel(torch.nn.Module):
    """Minimal model with the methods the zero-epoch trainer path touches."""

    def __init__(self):
        """Create one parameter and track the current device."""
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.device = torch.device("cpu")

    def get_input_embeddings(self):
        """Expose the embedding width safe_resize_token_embeddings inspects."""
        from types import SimpleNamespace
        return SimpleNamespace(num_embeddings=getattr(self, "vocab_size", 0))

    def resize_token_embeddings(self, vocab_size):
        """Record resize calls without changing parameters."""
        self.vocab_size = vocab_size

    def to(self, device):
        """Move parameters and keep the trainer-visible device attribute current."""
        super().to(device)
        self.device = torch.device(device)
        return self


class _CudaBatchDataset(torch.utils.data.Dataset):
    """One-sample dataset for a tiny CUDA train step."""

    tokenizer = _TinyTokenizer()

    def __len__(self):
        """Return one trainable sample."""
        return 1

    def __getitem__(self, index):
        """Return a short causal-LM token sequence."""
        return {
            "input_ids": torch.tensor([1, 2, 3, 4], dtype=torch.long),
            "attention_mask": torch.tensor([1, 1, 1, 1], dtype=torch.long),
        }

    def disable_masking(self):
        """No-op masking hook for the trainer."""


def _plain_trainer(tmp_path):
    """Build an uninitialized trainer with only the attributes _train uses."""
    trainer = LlmEmbeddingsTrainer.__new__(LlmEmbeddingsTrainer)
    recorder = _Recorder()
    trainer.save_model = recorder.save_model
    trainer.save_finetuned = recorder.save_finetuned
    trainer.model = _PlainModel()
    trainer.dataset = _EmptyDataset()
    trainer.tokenizer = trainer.dataset.tokenizer
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.01)
    trainer.logger = SimpleNamespace(info=lambda *args, **kwargs: None)
    trainer.metric_logger = SimpleNamespace(log_metric=lambda *args, **kwargs: None)
    trainer.rank = -1
    trainer._accelerator = None
    trainer.is_accelerator = False
    trainer._device = torch.device("cpu")
    trainer._module_checkpoint_dir = str(tmp_path)
    trainer._trainer_args = argparse.Namespace(llm_scheduler="linear")
    trainer.num_epochs = 0
    trainer.batch_size = 1
    trainer._reset_lr = False
    trainer._lr = 0.01
    trainer._num_workers = 0
    trainer._is_shuffle = False
    trainer._pin_memory = False
    trainer._is_quantize = False
    trainer._best_validation_metric = float("-inf")
    trainer._masked_freq = 1
    trainer._current_mask_method_counter = 0
    trainer._current_mask_method_idx = 0
    trainer._num_mask_passed = 1
    trainer.on_epoch_eval = False
    trainer._eval_freq = 1
    trainer._save_freq = 1
    trainer.masking_methods = []
    trainer.dataset_sampler = lambda: None
    trainer.split_dataset = lambda: ([], [])
    trainer.load_checkpoint = lambda *args, **kwargs: CheckpointState(0, None, None, None, 0)
    return trainer, recorder


def test_plain_end_of_train_skips_accelerator_unwrap(tmp_path, monkeypatch):
    """A non-accelerator M1 train run reaches save_model with no unwrap_model access."""
    trainer, recorder = _plain_trainer(tmp_path)

    monkeypatch.setattr(
        TorchBuilder,
        "create_scheduler",
        lambda *args, **kwargs: SimpleNamespace(
            step=lambda: None,
            load_state_dict=lambda state: None,
        ),
    )

    trainer._train(mask_type=[])

    assert trainer._accelerator is None
    assert recorder.save_model_calls == 1
    assert recorder.save_finetuned_calls == 1


@pytest.mark.gpu
def test_cuda_one_step_keeps_model_on_cuda_and_loss_finite(monkeypatch, tmp_path):
    """A tiny CUDA M1 step keeps parameters on cuda and produces a finite loss."""
    if not torch.cuda.is_available():
        pytest.skip("requires a local CUDA-capable GPU")

    transformers = pytest.importorskip("transformers")

    model = transformers.GPT2LMHeadModel(
        transformers.GPT2Config(
            n_layer=2,
            n_head=2,
            n_embd=32,
            vocab_size=len(_TinyTokenizer()),
            n_positions=16,
        )
    )
    trainer, recorder = _plain_trainer(tmp_path)
    trainer.model = model
    trainer.dataset = _CudaBatchDataset()
    trainer.tokenizer = trainer.dataset.tokenizer
    trainer.optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=1e-4)
    trainer._device = torch.device("cuda")
    trainer.num_epochs = 1
    losses = []
    trainer.metric_logger = SimpleNamespace(
        log_metric=lambda name, value, step: losses.append(value)
        if name == "train/loss"
        else None
    )
    trainer.split_dataset = lambda: (trainer.dataset, trainer.dataset)
    trainer.load_checkpoint = lambda *args, **kwargs: CheckpointState(0, None, None, None, 0)
    trainer.validate = lambda eval_dataloader: 1.0

    monkeypatch.setattr(
        TorchBuilder,
        "create_scheduler",
        lambda *args, **kwargs: SimpleNamespace(
            step=lambda: None,
            load_state_dict=lambda state: None,
        ),
    )

    trainer._train(mask_type=[])

    assert all(parameter.device.type == "cuda" for parameter in trainer.model.parameters())
    assert losses and torch.isfinite(torch.tensor(losses)).all()
    assert recorder.save_model_calls == 1
    assert recorder.save_finetuned_calls == 1


# Author: Mus mbayramo@stanford.edu
