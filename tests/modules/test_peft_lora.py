"""Offline CPU tests for PEFT LoRA application.

The target-module selection is pure logic; the apply_lora path builds a tiny GPT-2
from config (random init, no download) and checks that LoRA leaves only a small
fraction of parameters trainable. Needs peft + transformers (igc-dev).

Author:
Mus mbayramo@stanford.edu
"""
from types import SimpleNamespace

import pytest

from igc.modules.llm.peft_lora import (
    apply_lora,
    default_save_modules,
    default_target_modules,
    trainable_parameter_summary,
    validate_loaded_adapter_profile,
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


def test_default_save_modules_from_model_config():
    """When no model_type is given, the model config selects the saved embedding."""

    class _Cfg:
        model_type = "gpt2"

    class _Model:
        config = _Cfg()

    assert default_save_modules(model=_Model()) == ["wte"]


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


def _loaded_adapter(
    *,
    r=32,
    alpha=64,
    dropout=0.05,
    target_modules=("q_proj", "v_proj", "o_proj"),
    use_rslora=False,
    use_dora=False,
    active_adapter="default",
):
    """Build a loaded PEFT-model double with one adapter config."""
    config = SimpleNamespace(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=set(target_modules),
        use_rslora=use_rslora,
        use_dora=use_dora,
    )
    return SimpleNamespace(
        active_adapter=active_adapter,
        peft_config={"default": config},
    )


@pytest.mark.parametrize(
    ("adapter_method", "flags"),
    [
        ("lora", {}),
        ("rslora", {"use_rslora": True}),
        ("dora", {"use_dora": True}),
    ],
)
def test_validate_loaded_adapter_profile_accepts_exact_profile_match(
    adapter_method,
    flags,
):
    """Loaded parent adapter config must match the resolved profile exactly."""
    validate_loaded_adapter_profile(
        _loaded_adapter(**flags),
        r=32,
        alpha=64,
        dropout=0.05,
        target_modules=["o_proj", "q_proj", "v_proj"],
        adapter_method=adapter_method,
    )


@pytest.mark.parametrize(
    ("adapter_overrides", "profile_overrides", "message"),
    [
        ({"r": 16}, {}, "r=16"),
        ({"alpha": 32}, {}, "lora_alpha=32"),
        ({"dropout": 0.1}, {}, "lora_dropout=0.1"),
        ({"target_modules": ("q_proj", "k_proj")}, {}, "target_modules"),
        ({}, {"target_modules": ["q_proj", "v_proj", "gate_proj"]}, "target_modules"),
        ({"use_rslora": True}, {"adapter_method": "lora"}, "adapter_method='rslora'"),
        ({"use_dora": True}, {"adapter_method": "rslora"}, "adapter_method='dora'"),
    ],
)
def test_validate_loaded_adapter_profile_rejects_parent_profile_mismatches(
    adapter_overrides,
    profile_overrides,
    message,
):
    """Parent PEFT config drift in rank, alpha, dropout, targets, or method is fatal."""
    profile = {
        "r": 32,
        "alpha": 64,
        "dropout": 0.05,
        "target_modules": ["q_proj", "v_proj", "o_proj"],
        "adapter_method": "lora",
    }
    profile.update(profile_overrides)

    with pytest.raises(RuntimeError, match=message):
        validate_loaded_adapter_profile(
            _loaded_adapter(**adapter_overrides),
            **profile,
        )


def test_validate_loaded_adapter_profile_rejects_incompatible_method_flags():
    """A persisted adapter cannot claim both rsLoRA and DoRA at once."""
    with pytest.raises(RuntimeError, match="incompatible methods"):
        validate_loaded_adapter_profile(
            _loaded_adapter(use_rslora=True, use_dora=True),
            r=32,
            alpha=64,
            dropout=0.05,
            target_modules=["q_proj", "v_proj", "o_proj"],
            adapter_method="rslora",
        )


def test_apply_lora_explicit_modules_to_save_override_default_embedding():
    """An explicit modules_to_save list is passed to PEFT instead of the default embedding."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")

    peft_model = apply_lora(
        _tiny_gpt2(),
        r=4,
        model_type="gpt2",
        modules_to_save=["lm_head"],
    )

    assert peft_model.peft_config["default"].modules_to_save == ["lm_head"]
    assert any(
        "lm_head.modules_to_save" in name and param.requires_grad
        for name, param in peft_model.named_parameters()
    )
    assert not any(
        "wte.modules_to_save" in name and param.requires_grad
        for name, param in peft_model.named_parameters()
    )


def test_apply_lora_can_disable_saved_embeddings():
    """train_embeddings=False leaves modules_to_save unset for adapter-only tuning."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")

    peft_model = apply_lora(
        _tiny_gpt2(),
        r=4,
        model_type="gpt2",
        train_embeddings=False,
    )

    assert peft_model.peft_config["default"].modules_to_save is None
    assert not any("modules_to_save" in name for name, _ in peft_model.named_parameters())


# Author: Mus mbayramo@stanford.edu
