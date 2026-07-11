"""Offline regressions for sharded-run checkpointing support.

Under ZeRO-3/FSDP a rank-0-only ``unwrap_model().state_dict()`` deadlocks (the
gather is collective) or writes shards. The trainer now gathers via
``accelerator.get_state_dict`` on every rank behind a rank-0 verdict made
uniform by ``broadcast_flag``, and ``IgcModule.save_checkpoint`` /
``IgcModule.save_model`` accept the pre-gathered ``model_state_dict`` instead of
re-calling ``state_dict()``. The end-of-train save (``_save_final_checkpoint``)
runs that gather + a final barrier on EVERY rank so rank 0 never enters the
collective alone. CPU-only; single-process behavior plus these structural
guarantees are pinned here — the true multi-rank path runs on the cluster rung
(gpu-marked).

Author:
Mus mbayramo@stanford.edu
"""

import argparse

import loguru
import torch

from igc.modules.base.igc_base_module import IgcModule
from igc.modules.llm_train_state_encoder import LlmEmbeddingsTrainer
from igc.shared.shared_accelerator import broadcast_flag


class _SingleProcessAccelerator:
    num_processes = 1


def test_broadcast_flag_single_process_is_identity():
    """With one process there is nothing to synchronize."""
    accelerator = _SingleProcessAccelerator()
    assert broadcast_flag(accelerator, True) is True
    assert broadcast_flag(accelerator, False) is False


def _module(tmp_path):
    module = IgcModule.__new__(IgcModule)
    module.rank = 0
    module.module_name = "tmod"
    module.logger = loguru.logger
    module.model = torch.nn.Linear(2, 2)
    module.optimizer = None
    module.scheduler = None
    module._trainer_args = argparse.Namespace(seed=1, output_dir=str(tmp_path))
    return module


def test_save_checkpoint_uses_pregathered_state_dict(tmp_path):
    """A provided model_state_dict is saved verbatim, not model.state_dict()."""
    module = _module(tmp_path)
    gathered = {"weight": torch.ones(2, 2), "bias": torch.zeros(2)}

    path = module.save_checkpoint(
        str(tmp_path), epoch=1, model_state_dict=gathered
    )
    checkpoint = torch.load(path, weights_only=True)

    assert torch.equal(checkpoint["model_state_dict"]["weight"], torch.ones(2, 2))


def test_save_checkpoint_falls_back_to_model_state(tmp_path):
    """Without the override the model's own state dict is saved (plain path)."""
    module = _module(tmp_path)
    path = module.save_checkpoint(str(tmp_path), epoch=1)
    checkpoint = torch.load(path, weights_only=True)
    assert set(checkpoint["model_state_dict"]) == {"weight", "bias"}


def test_save_model_uses_pregathered_state_dict(tmp_path):
    """save_model writes the provided gather verbatim — no rank-0-only state_dict().

    This is the exact call the end-of-train deadlock lived in: under FSDP the
    final ``save_model`` must serialize the collective's result, not re-enter it.
    """
    module = _module(tmp_path)
    gathered = {"weight": torch.ones(2, 2), "bias": torch.zeros(2)}

    module.save_model(str(tmp_path), model_state_dict=gathered)
    checkpoint = torch.load(module._model_file(str(tmp_path)), weights_only=True)

    assert torch.equal(checkpoint["model_state_dict"]["weight"], torch.ones(2, 2))
    assert checkpoint["is_trained"] is True


def test_save_model_falls_back_to_model_state(tmp_path):
    """Without the override save_model saves the model's own state (plain path)."""
    module = _module(tmp_path)
    module.save_model(str(tmp_path))
    checkpoint = torch.load(module._model_file(str(tmp_path)), weights_only=True)
    assert torch.equal(checkpoint["model_state_dict"]["weight"], module.model.weight)


class _SpyModel(torch.nn.Linear):
    """Linear that records direct ``state_dict()`` calls.

    On the sharded end-of-train path rank 0 must NEVER call ``state_dict()``
    itself — that is the collective it would enter alone. The gather goes through
    the (mock) accelerator instead, so this counter must stay at zero.
    """

    def __init__(self):
        super().__init__(2, 2)
        self.state_dict_calls = 0

    def state_dict(self, *args, **kwargs):  # noqa: D102
        self.state_dict_calls += 1
        return super().state_dict(*args, **kwargs)


class _MockShardedAccelerator:
    """Minimal FSDP-like accelerator recording the collective + barrier calls."""

    num_processes = 4
    # A fixed "gathered" dict distinct from the model's real init, so a saved
    # checkpoint proves the gather (not a local state_dict) supplied the weights.
    SENTINEL = {"weight": torch.ones(2, 2), "bias": torch.zeros(2)}

    def __init__(self):
        self.get_state_dict_calls = 0
        self.unwrap_calls = 0
        self.wait_calls = 0

    def get_state_dict(self, model):
        self.get_state_dict_calls += 1
        return {k: v.clone() for k, v in self.SENTINEL.items()}

    def unwrap_model(self, model):
        self.unwrap_calls += 1
        return model

    def wait_for_everyone(self):
        self.wait_calls += 1


def _trainer(tmp_path, rank, accelerator):
    trainer = LlmEmbeddingsTrainer.__new__(LlmEmbeddingsTrainer)
    trainer.rank = rank
    trainer.module_name = "state_encoder"
    trainer.logger = loguru.logger
    trainer.model = _SpyModel()
    trainer.is_accelerator = True
    trainer._accelerator = accelerator
    trainer._module_checkpoint_dir = str(tmp_path)
    return trainer


def test_final_checkpoint_gathers_and_barriers_on_every_rank(tmp_path):
    """Every rank runs the collective gather + barrier; rank 0 never calls state_dict.

    The end-of-train hang was structural: ``get_state_dict`` ran only where rank 0
    reached it, so peers exited while rank 0 blocked in the gather. Assert the
    gather AND the final ``wait_for_everyone`` barrier fire on each of the four
    ranks, and that rank 0 writes the gathered weights without a rank-0-only
    ``model.state_dict()``.
    """
    for rank in (0, 1, 2, 3):
        accelerator = _MockShardedAccelerator()
        trainer = _trainer(tmp_path, rank, accelerator)
        finetuned_seen = []
        # Stub the HF-format save to record the dict it is handed (real
        # save_pretrained needs a PreTrainedModel; the contract under test is
        # that the pre-gathered dict is threaded through, not re-derived).
        trainer.save_finetuned = lambda model_state_dict=None: finetuned_seen.append(
            model_state_dict)

        trainer._save_final_checkpoint()

        # The collective and the barrier run on EVERY rank — this is the fix.
        assert accelerator.get_state_dict_calls == 1, f"rank {rank} skipped gather"
        assert accelerator.wait_calls == 1, f"rank {rank} skipped barrier"
        # Rank 0 must not re-enter the gather via a local state_dict().
        assert trainer.model.state_dict_calls == 0, f"rank {rank} called state_dict"
        # The pre-gathered dict is threaded to the HF-format writer too.
        assert len(finetuned_seen) == 1
        assert finetuned_seen[0] is not None
        assert torch.equal(finetuned_seen[0]["weight"], torch.ones(2, 2))


def test_final_checkpoint_writes_gathered_weights_on_rank_zero(tmp_path):
    """Rank 0 persists the gathered weights, not the model's untouched init."""
    accelerator = _MockShardedAccelerator()
    trainer = _trainer(tmp_path, rank=0, accelerator=accelerator)
    trainer.save_finetuned = lambda model_state_dict=None: None

    trainer._save_final_checkpoint()

    checkpoint = torch.load(
        trainer._model_file(str(tmp_path)), weights_only=True)
    assert torch.equal(checkpoint["model_state_dict"]["weight"], torch.ones(2, 2))


# Author: Mus mbayramo@stanford.edu
