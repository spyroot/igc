"""Offline tests for the Phase 1/2/3 training profile registry.

Pins named profiles, that the 7B rsLoRA candidates resolve to the exact LoRA
config docs/TRAINING_OPTIMIZATION_PLAN.md specifies, that full-FT profiles carry
no adapter and shard, and that env-backed SHA fields fail loudly. Pure stdlib —
no torch/peft.

Author:
Mus mbayramo@stanford.edu
"""

from pathlib import Path

import pytest
import yaml

from igc.modules.train.profiles import (
    AdapterSpec,
    apply_lora_kwargs,
    profile_names,
    resolve_profile,
)

_FOUNDATION_SHA = "sha256:" + "1" * 64
_TOKENIZER_SHA = "sha256:" + "2" * 64
_MODEL_X_SHA = "sha256:" + "3" * 64
_GOAL_EXTRACTOR_SHA = "sha256:" + "4" * 64
_PROFILE_YAML = Path("configs/training/profiles.yaml")
_PROFILE_HYPERPARAMETERS = {
    "optimizer",
    "weight_decay",
    "max_grad_norm",
    "gradient_checkpointing",
    "max_lr",
    "div_factor",
    "final_div_factor",
    "cycle_momentum",
    "anneal_strategy",
    "seed",
    "phase1_structural_loss_profile",
}

_REGISTERED = [
    "phase1_gpt2_smoke",
    "phase1_3b_lora",
    "phase1_7b_lora",
    "phase1_7b_rslora_r32",
    "phase1_7b_rslora_r32_structural_mask",
    "phase1_local",
    "phase1_3b_full",
    "phase1_7b_full_zero3",
    "phase2_7b_rslora_r32",
    "phase3_7b_rslora_r32",
]

_PROFILE_CASES = [
    (
        "phase1_gpt2_smoke", "gpt2", False, 8, 1, 5e-5, "none", 256, "no", 50,
        "phase1_finetune", "model_x", "sft", "phase1_pretrain",
    ),
    (
        "phase1_3b_lora", "Qwen/Qwen2.5-3B-Instruct", True, 8, 2,
        1e-4, "none", 1024, "bf16", None,
        "phase1_finetune", "model_x", "sft", "phase1_pretrain",
    ),
    (
        "phase1_7b_lora", "Qwen/Qwen2.5-7B-Instruct", True, 8, 4,
        1e-4, "none", 2048, "bf16", None,
        "phase1_finetune", "model_x", "sft", "phase1_pretrain",
    ),
    (
        "phase1_7b_rslora_r32", "Qwen/Qwen2.5-7B-Instruct", True, 8, 4,
        1e-4, "none", 2048, "bf16", None,
        "phase1_finetune", "model_x", "sft", "phase1_pretrain",
    ),
    (
        "phase1_7b_rslora_r32_structural_mask", "Qwen/Qwen2.5-7B-Instruct", True, 8, 4,
        1e-4, "none", 2048, "bf16", None,
        "phase1_finetune", "model_x", "sft", "phase1_pretrain",
    ),
    (
        "phase1_3b_full", "Qwen/Qwen2.5-3B-Instruct", False, 4, 8,
        2e-5, "zero3", 1024, "bf16", None,
        "phase1_finetune", "model_x", "sft", "phase1_pretrain",
    ),
    (
        "phase1_7b_full_zero3", "Qwen/Qwen2.5-7B-Instruct", False, 2, 16,
        1e-5, "zero3", 2048, "bf16", None,
        "phase1_finetune", "model_x", "sft", "phase1_pretrain",
    ),
    (
        "phase2_7b_rslora_r32", "Qwen/Qwen2.5-7B-Instruct", True, 8, 4,
        1e-4, "none", 1024, "bf16", None,
        "phase2_goal_extraction", "goal_extractor", "sft", "labelled_requests",
    ),
    (
        "phase3_7b_rslora_r32", "Qwen/Qwen2.5-7B-Instruct", True, 8, 4,
        1e-4, "none", 1536, "bf16", None,
        "phase3_argument_extraction", "argument_extractor", "sft", "labelled_calls",
    ),
]


@pytest.fixture(autouse=True)
def _profile_sha_env(monkeypatch):
    """Provide active profile SHA/adapter env values without touching the real shell."""
    monkeypatch.setenv("IGC_MODEL_DIR", "/models/foundation-local")
    monkeypatch.setenv("IGC_FOUNDATION_MODEL_SHA", _FOUNDATION_SHA)
    monkeypatch.setenv("IGC_TOKENIZER_SHA", _TOKENIZER_SHA)
    monkeypatch.setenv("IGC_MODEL_X_ADAPTER_DIR", "/models/model_x")
    monkeypatch.setenv("IGC_MODEL_X_SHA", _MODEL_X_SHA)
    monkeypatch.setenv("IGC_GOAL_EXTRACTOR_ADAPTER_DIR", "/models/goal_extractor")
    monkeypatch.setenv("IGC_GOAL_EXTRACTOR_SHA", _GOAL_EXTRACTOR_SHA)


def test_all_registered_profiles_present():
    """Every named Phase 1/2/3 profile is present in registry order."""
    names = profile_names()
    assert names == _REGISTERED


@pytest.mark.parametrize(
    (
        "name", "model", "use_peft", "batch_size", "grad_accum", "lr",
        "sharding", "seq_len", "precision", "max_steps",
        "phase", "weights_role", "llm_stage", "corpus_objective",
    ),
    _PROFILE_CASES,
)
def test_profile_matrix_matches_plan_contract(
    name,
    model,
    use_peft,
    batch_size,
    grad_accum,
    lr,
    sharding,
    seq_len,
    precision,
    max_steps,
    phase,
    weights_role,
    llm_stage,
    corpus_objective,
):
    """Each named profile keeps the executable launch contract pinned."""
    p = resolve_profile(name)
    assert p.model == model
    assert p.use_peft is use_peft
    assert p.batch_size == batch_size
    assert p.grad_accum == grad_accum
    assert p.lr == lr
    assert p.sharding == sharding
    assert p.seq_len == seq_len
    assert p.precision == precision
    assert p.max_steps == max_steps
    assert p.phase == phase
    assert p.weights_role == weights_role
    assert p.llm_stage == llm_stage
    assert p.corpus_objective == corpus_objective
    expected_structural_loss = (
        "historical_structural_mask_v1"
        if name == "phase1_7b_rslora_r32_structural_mask"
        else "none"
    )
    assert p.phase1_structural_loss_profile == expected_structural_loss
    assert p.early_stopping_patience == 3
    assert p.early_stopping_min_delta == 0.005


def test_profile_yaml_exposes_profile_driven_hyperparameters():
    """Every YAML profile owns the optimizer, scheduler, clipping, and seed knobs."""
    raw = yaml.safe_load(_PROFILE_YAML.read_text(encoding="utf-8"))

    for name, profile in raw["profiles"].items():
        missing = _PROFILE_HYPERPARAMETERS - set(profile)
        assert missing == set(), name
        assert profile["optimizer"] in {"Adam", "AdamW", "SGD"}
        assert profile["weight_decay"] >= 0.0
        assert profile["max_grad_norm"] > 0.0
        assert isinstance(profile["gradient_checkpointing"], bool)
        assert profile["max_lr"] > 0.0
        assert profile["div_factor"] > 0.0
        assert profile["final_div_factor"] > 0.0
        assert isinstance(profile["cycle_momentum"], bool)
        assert profile["anneal_strategy"] in {"cos", "linear"}
        assert isinstance(profile["seed"], int)


def test_resolved_profile_describe_includes_profile_driven_hyperparameters():
    """Resolved profiles expose the same knobs for W&B config and reports."""
    described = resolve_profile(
        "phase1_3b_lora",
        optimizer="SGD",
        weight_decay=0.02,
        max_grad_norm=0.75,
        gradient_checkpointing=False,
        max_lr=0.003,
        div_factor=8.0,
        final_div_factor=512.0,
        cycle_momentum=False,
        anneal_strategy="linear",
        seed=1234,
    ).describe()

    assert {
        key: described[key]
        for key in sorted(_PROFILE_HYPERPARAMETERS)
    } == {
        "anneal_strategy": "linear",
        "cycle_momentum": False,
        "div_factor": 8.0,
        "final_div_factor": 512.0,
        "gradient_checkpointing": False,
        "max_grad_norm": 0.75,
        "max_lr": 0.003,
        "optimizer": "SGD",
        "phase1_structural_loss_profile": "none",
        "seed": 1234,
        "weight_decay": 0.02,
    }


def test_7b_rslora_matches_plan_spec():
    """phase1_7b_rslora_r32 resolves to the plan's exact LoraConfig kwargs."""
    p = resolve_profile("phase1_7b_rslora_r32")
    assert p.model == "Qwen/Qwen2.5-7B-Instruct" and p.use_peft
    assert p.foundation_model_sha == _FOUNDATION_SHA
    assert p.tokenizer_sha == _TOKENIZER_SHA
    kw = apply_lora_kwargs(p)
    assert kw["r"] == 32 and kw["alpha"] == 64 and kw["dropout"] == 0.05
    assert kw["adapter_method"] == "rslora"
    assert kw["init_lora_weights"] is True
    assert kw["target_modules"] == [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    ]


def test_full_ft_profiles_have_no_adapter_and_shard():
    """Full fine-tune profiles disable PEFT, shard with zero3, and refuse LoRA kwargs."""
    for n in ("phase1_3b_full", "phase1_7b_full_zero3"):
        p = resolve_profile(n)
        assert not p.use_peft and p.adapter is None and p.sharding == "zero3"
        with pytest.raises(ValueError):
            apply_lora_kwargs(p)


def test_smoke_profile_is_cheap_and_capped():
    """The GPT-2 smoke is a small, step-capped full-FT launch check."""
    p = resolve_profile("phase1_gpt2_smoke")
    assert p.model == "gpt2" and p.max_steps == 50 and not p.use_peft


def test_resolve_applies_overrides_and_rejects_typos():
    """Overrides apply; an unknown field raises, and an unknown profile raises."""
    p = resolve_profile("phase1_3b_lora", batch_size=16, lr=2e-4, max_steps=200)
    assert p.batch_size == 16 and p.lr == 2e-4 and p.max_steps == 200
    with pytest.raises(ValueError):
        resolve_profile("phase1_3b_lora", btch_size=16)  # typo must not silently no-op
    with pytest.raises(KeyError):
        resolve_profile("does_not_exist")


def test_resolve_override_does_not_mutate_registered_profile():
    """Overrides return a copy and leave the registered profile unchanged."""
    base = resolve_profile("phase1_3b_lora")
    changed = resolve_profile("phase1_3b_lora", batch_size=16, lr=2e-4, max_steps=200)
    again = resolve_profile("phase1_3b_lora")
    assert changed.batch_size == 16 and changed.lr == 2e-4 and changed.max_steps == 200
    assert again is base
    assert again.batch_size == 8 and again.lr == 1e-4 and again.max_steps is None


def test_peft_false_override_disables_lora_kwargs_even_with_adapter():
    """A use_peft=False override makes a profile behave as a full fine-tune."""
    p = resolve_profile("phase1_3b_lora", use_peft=False)
    assert p.adapter is not None
    assert p.describe()["adapter"]["method"] == "full_finetune"
    with pytest.raises(ValueError):
        apply_lora_kwargs(p)


def test_adapter_init_maps():
    """Adapter init names map to PEFT init_lora_weights values."""
    assert AdapterSpec(init="pissa").init_lora_weights() == "pissa"
    assert AdapterSpec(init="eva").init_lora_weights() == "eva"
    assert AdapterSpec(init="loftq").init_lora_weights() == "loftq"
    assert AdapterSpec(init="default").init_lora_weights() is True
    with pytest.raises(ValueError):
        AdapterSpec(init="nonsense").init_lora_weights()


def test_apply_lora_kwargs_preserves_custom_adapter_targets():
    """Custom adapter init and target modules pass through as PEFT kwargs."""
    adapter = AdapterSpec(init="loftq", target_modules=("x_proj",))
    p = resolve_profile("phase1_3b_lora", adapter=adapter)
    kw = apply_lora_kwargs(p)
    assert kw["init_lora_weights"] == "loftq"
    assert kw["target_modules"] == ["x_proj"]


def test_describe_is_flat_log_safe_dict():
    """describe() yields a flat dict suitable for stdout + W&B config."""
    d = resolve_profile("phase1_7b_rslora_r32").describe()
    assert d["profile"] == "phase1_7b_rslora_r32" and d["use_peft"] is True
    assert d["phase"] == "phase1_finetune"
    assert d["weights_role"] == "model_x"
    assert d["llm_stage"] == "sft"
    assert d["corpus_objective"] == "phase1_pretrain"
    assert d["foundation_model_sha"] == _FOUNDATION_SHA
    assert d["tokenizer_sha"] == _TOKENIZER_SHA
    assert d["early_stopping_patience"] == 3
    assert d["early_stopping_min_delta"] == 0.005
    assert d["adapter"]["method"] == "rslora" and d["adapter"]["r"] == 32
    full = resolve_profile("phase1_7b_full_zero3").describe()
    assert full["adapter"]["method"] == "full_finetune" and full["sharding"] == "zero3"


def test_active_phase_profiles_expand_sha_and_parent_environment():
    """Phase2/3 active profiles resolve parent adapter and SHA env placeholders."""
    phase2 = resolve_profile("phase2_7b_rslora_r32")
    phase3 = resolve_profile("phase3_7b_rslora_r32")

    assert phase2.foundation_model_sha == _FOUNDATION_SHA
    assert phase2.tokenizer_sha == _TOKENIZER_SHA
    assert phase2.parent_adapter == "/models/model_x"
    assert phase2.parent_artifact_sha == _MODEL_X_SHA
    assert phase2.weights_role == "goal_extractor"
    assert phase3.parent_adapter == "/models/goal_extractor"
    assert phase3.parent_artifact_sha == _GOAL_EXTRACTOR_SHA
    assert phase3.weights_role == "argument_extractor"


def test_active_profile_rejects_unresolved_or_invalid_sha_env(monkeypatch):
    """SHA-backed active profile fields fail closed when env expansion is unsafe."""
    monkeypatch.delenv("IGC_MODEL_X_SHA")
    with pytest.raises(ValueError, match="references an unset environment variable"):
        resolve_profile("phase2_7b_rslora_r32")

    monkeypatch.setenv("IGC_MODEL_X_SHA", "not-a-sha")
    with pytest.raises(ValueError, match="parent_artifact_sha"):
        resolve_profile("phase2_7b_rslora_r32")


# Author: Mus mbayramo@stanford.edu
