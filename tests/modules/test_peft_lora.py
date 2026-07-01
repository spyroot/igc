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
    default_save_modules,
    default_target_modules,
    trainable_parameter_summary,
)


def _tiny_gpt2():
    """A tiny random-init GPT-2 (no download) for LoRA plumbing tests."""
    from transformers import GPT2Config, GPT2LMHeadModel

    return GPT2LMHeadModel(GPT2Config(n_layer=2, n_head=2, n_embd=32, vocab_size=128, n_positions=64))


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


def test_default_save_modules_by_backbone():
    """The embedding module kept trainable is backbone-selected."""
    assert default_save_modules(model_type="gpt2") == ["wte"]
    assert default_save_modules(model_type="qwen2") == ["embed_tokens"]
    assert default_save_modules(model_type="llama") == ["embed_tokens"]


def test_apply_lora_on_tiny_gpt2_makes_few_params_trainable():
    """LoRA on a tiny gpt2 leaves adapters + embeddings trainable (still << total)."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    model = _tiny_gpt2()
    base_total = sum(p.numel() for p in model.parameters())

    peft_model = apply_lora(model, r=4, model_type="gpt2")
    trainable, total = trainable_parameter_summary(peft_model)

    assert 0 < trainable < total          # base weights stay frozen
    assert trainable < base_total * 0.5   # adapters + embedding are a small fraction


def test_embeddings_trainable_by_default_but_optional():
    """train_embeddings keeps the resized embedding trainable; opting out freezes it."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")

    with_emb, _ = trainable_parameter_summary(apply_lora(_tiny_gpt2(), r=4, model_type="gpt2"))
    without_emb, _ = trainable_parameter_summary(
        apply_lora(_tiny_gpt2(), r=4, model_type="gpt2", train_embeddings=False)
    )
    assert with_emb > without_emb  # the embedding rows add trainable params


# Author: Mus mbayramo@stanford.edu
