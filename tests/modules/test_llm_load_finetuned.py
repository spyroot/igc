"""Offline tests for downstream LLM stages loading the trained state encoder.

The goal/parameter/autoencoder stages consume a state encoder checkpoint produced
by the prior M1 run. These tests use small fakes so the contract is covered with
no GPU, network, HuggingFace download, or real checkpoint.

Author:
Mus mbayramo@stanford.edu
"""
import argparse

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


@pytest.mark.parametrize("llm_stage", ["goal", "parameter", "encoder"])
def test_downstream_stages_load_prior_state_encoder(monkeypatch, llm_stage):
    """Downstream stages return the fine-tuned state encoder and mark it trained."""
    module = make_module(llm_stage)
    state_encoder = object()
    monkeypatch.setattr(module, "load_finetuned_state_encoder", lambda: state_encoder)

    model, tokenizer, model_state = module.load_finetuned_llm()

    assert model is state_encoder
    assert tokenizer is module._dataset.tokenizer
    assert model_state is ModelType.FINETUNED


@pytest.mark.parametrize("llm_stage", ["goal", "parameter", "encoder"])
def test_downstream_stages_fail_fast_without_state_encoder(monkeypatch, llm_stage):
    """A downstream stage without an M1 checkpoint raises a clear setup error."""
    module = make_module(llm_stage)
    monkeypatch.setattr(module, "load_finetuned_state_encoder", lambda: None)

    with pytest.raises(RuntimeError, match=f"train stage m1 .*--llm {llm_stage}"):
        module.load_finetuned_llm()


# Author: Mus mbayramo@stanford.edu
