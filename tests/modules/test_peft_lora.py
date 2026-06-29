"""Offline CPU tests for PEFT LoRA application.

The target-module selection is pure logic; the apply_lora path builds a tiny GPT-2
from config (random init, no download) and checks that LoRA leaves only a small
fraction of parameters trainable. Needs peft + transformers (igc-dev).

Author:
Mus mbayramo@stanford.edu
"""
import pytest

from igc.modules.llm.peft_lora import (
    apply_lora,
    default_target_modules,
    trainable_parameter_summary,
)


def test_default_targets_gpt2():
    """GPT-2 gets the Conv1D target names."""
    assert default_target_modules(model_type="gpt2") == ["c_attn", "c_fc", "c_proj"]


def test_default_targets_modern_decoder():
    """A modern decoder gets the projection target names, not the GPT-2 ones."""
    targets = default_target_modules(model_type="llama")
    assert "q_proj" in targets and "down_proj" in targets
    assert "c_attn" not in default_target_modules(model_type="qwen2")


def test_default_targets_from_model_config():
    """When no model_type is given, the model's config.model_type is used."""

    class _Cfg:
        model_type = "gpt2"

    class _Model:
        config = _Cfg()

    assert default_target_modules(model=_Model()) == ["c_attn", "c_fc", "c_proj"]


def test_apply_lora_on_tiny_gpt2_makes_few_params_trainable():
    """LoRA on a tiny gpt2 leaves only the adapters trainable (trainable << total)."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    from transformers import GPT2Config, GPT2LMHeadModel

    cfg = GPT2Config(n_layer=2, n_head=2, n_embd=32, vocab_size=128, n_positions=64)
    model = GPT2LMHeadModel(cfg)
    base_total = sum(p.numel() for p in model.parameters())

    peft_model = apply_lora(model, r=4, model_type="gpt2")
    trainable, total = trainable_parameter_summary(peft_model)

    assert 0 < trainable < total          # only adapters train
    assert trainable < base_total * 0.5   # LoRA is a small fraction of the model


# Author: Mus mbayramo@stanford.edu
