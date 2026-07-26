"""Offline tests for downstream LLM stages loading the trained state encoder.

The goal/parameter/autoencoder stages consume a state encoder checkpoint produced
by the prior Phase 1/model_x run. These tests use small fakes so the contract is
covered with no GPU, network, HuggingFace download, or real checkpoint.

Author:
Mus mbayramo@stanford.edu
"""
import argparse
from types import SimpleNamespace

import pytest

from igc.modules.llm.igc_llm_module import IgcLanguageModule
from igc.shared.modules_typing import ModelType


class FakeTokenizer:
    """Tiny tokenizer double that only supports len()."""

    def __len__(self):
        return 7


class FakeDataset:
    """Dataset double exposing the tokenizer attribute used by the loader."""

    tokenizer = FakeTokenizer()


def make_module(llm_stage: str) -> IgcLanguageModule:
    """Build an IgcLanguageModule around fakes for one downstream stage."""
    spec = argparse.Namespace(
        llm=llm_stage,
        log_level="ERROR",
        device="cpu",
    )
    return IgcLanguageModule(spec, metric_logger=None, ds=FakeDataset())


def test_encoder_stage_loads_prior_state_encoder(monkeypatch):
    """The encoder stage returns the fine-tuned state encoder and marks it trained."""
    module = make_module("encoder")
    state_encoder = object()
    monkeypatch.setattr(module, "load_finetuned_state_encoder", lambda: state_encoder)

    model, tokenizer, model_state = module.load_finetuned_llm()

    assert model is state_encoder
    assert tokenizer is module._dataset.tokenizer
    assert model_state is ModelType.FINETUNED


def test_encoder_stage_fails_fast_without_state_encoder(monkeypatch):
    """An encoder stage without a Phase 1/model_x checkpoint fails fast."""
    module = make_module("encoder")
    monkeypatch.setattr(module, "load_finetuned_state_encoder", lambda: None)

    with pytest.raises(RuntimeError, match="accepted model_x checkpoint"):
        module.load_finetuned_llm()


@pytest.mark.parametrize("llm_stage", ["goal", "parameter"])
def test_legacy_goal_parameter_stages_point_to_shared_sft(llm_stage):
    """Retired standalone downstream trainers fail with the shared SFT route."""
    module = make_module(llm_stage)

    with pytest.raises(RuntimeError, match="--llm sft"):
        module.load_finetuned_llm()


def test_sft_model_loading_enables_gradient_checkpointing_and_disables_use_cache(
    monkeypatch,
) -> None:
    """SFT model loading applies memory knobs before handing the model to trainer."""
    loaded_model = SimpleNamespace(
        config=SimpleNamespace(use_cache=True),
        gradient_checkpointing_enable_calls=0,
    )

    def gradient_checkpointing_enable():
        loaded_model.gradient_checkpointing_enable_calls += 1

    loaded_model.gradient_checkpointing_enable = gradient_checkpointing_enable
    trained = {}

    class FakeSFTTrainer:
        def __init__(
            self,
            *,
            llm_model,
            llm_tokenizer,
            eval_dataset,
            **_kwargs,
        ):
            trained["model"] = llm_model
            trained["tokenizer"] = llm_tokenizer
            trained["eval_dataset"] = eval_dataset

        def train(self):
            trained["trained"] = True

        def get_model(self):
            return trained["model"]

        def get_tokenizer(self):
            return trained["tokenizer"]

    spec = argparse.Namespace(
        llm="sft",
        sft_task="redfish_json_reconstruction",
        log_level="ERROR",
        device="cpu",
        device_map=None,
        gradient_checkpointing=True,
        use_peft=False,
    )
    heldout = object()
    module = IgcLanguageModule(
        spec,
        metric_logger=None,
        ds=FakeDataset(),
        eval_ds=heldout,
        from_pretrained=lambda *_args, **_kwargs: (loaded_model, object()),
    )
    monkeypatch.setattr(
        "igc.modules.llm.igc_llm_module.safe_resize_token_embeddings",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr("igc.modules.llm.igc_llm_module.SFTTrainer", FakeSFTTrainer)

    model, tokenizer, model_state = module.load_finetuned_llm()

    assert model is loaded_model
    assert tokenizer is module._dataset.tokenizer
    assert model_state is ModelType.UNTRAINED
    assert trained["trained"] is True
    assert trained["eval_dataset"] is heldout
    assert loaded_model.gradient_checkpointing_enable_calls == 1
    assert loaded_model.config.use_cache is False


# Author: Mus mbayramo@stanford.edu
