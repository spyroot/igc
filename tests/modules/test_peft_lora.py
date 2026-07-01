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
    """LoRA on a tiny gpt2 leaves only the adapters trainable by default (<< total)."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    model = _tiny_gpt2()
    base_total = sum(p.numel() for p in model.parameters())

    peft_model = apply_lora(model, r=4, model_type="gpt2")
    trainable, total = trainable_parameter_summary(peft_model)

    assert 0 < trainable < total          # base weights (incl. embedding) stay frozen
    assert trainable < base_total * 0.5   # adapters are a small fraction


def test_embedding_training_is_opt_in():
    """Embeddings are frozen by default; train_embeddings=True adds them (opt-in)."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")

    frozen, _ = trainable_parameter_summary(apply_lora(_tiny_gpt2(), r=4, model_type="gpt2"))
    whole, _ = trainable_parameter_summary(
        apply_lora(_tiny_gpt2(), r=4, model_type="gpt2", train_embeddings=True)
    )
    assert whole > frozen  # opting in adds the full embedding's params


def test_new_token_ids_trains_only_new_rows():
    """train_embeddings + new_token_ids trains just those rows (< whole matrix)."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")

    frozen, _ = trainable_parameter_summary(apply_lora(_tiny_gpt2(), r=4, model_type="gpt2"))
    whole, _ = trainable_parameter_summary(
        apply_lora(_tiny_gpt2(), r=4, model_type="gpt2", train_embeddings=True)
    )
    new_rows, _ = trainable_parameter_summary(
        apply_lora(_tiny_gpt2(), r=4, model_type="gpt2", train_embeddings=True, new_token_ids=[126, 127])
    )
    assert frozen < new_rows < whole  # 2 rows: more than adapters-only, far less than whole


def test_adapter_method_rslora_and_bad_method():
    """adapter_method='rslora' sets use_rslora; an unknown method raises."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    pm = apply_lora(_tiny_gpt2(), r=8, model_type="gpt2", adapter_method="rslora")
    assert pm.peft_config["default"].use_rslora is True
    with pytest.raises(ValueError):
        apply_lora(_tiny_gpt2(), model_type="gpt2", adapter_method="bogus")


# Author: Mus mbayramo@stanford.edu
