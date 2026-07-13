"""Offline tests for the M1 training profile registry (the run contract).

Pins the six named profiles, that the 7B rsLoRA candidate resolves to the exact
LoRA config docs/TRAINING_OPTIMIZATION_PLAN.md specifies, that full-FT profiles carry
no adapter and shard, and that override typos fail loudly. Pure stdlib — no torch/peft.

Author:
Mus mbayramo@stanford.edu
"""

import pytest

from igc.modules.train.profiles import (
    AdapterSpec,
    apply_lora_kwargs,
    profile_names,
    resolve_profile,
)

_REGISTERED = [
    "m1_gpt2_smoke",
    "m1_3b_lora",
    "m1_7b_lora",
    "m1_7b_rslora_r32",
    "m1_local",
    "m1_3b_full",
    "m1_7b_full_zero3",
    "m1_cpu_smoke",
    "m1_gb300_3b_lora",
    "m1_gb300_7b_lora",
    "m1_nv72_7b_rslora_r32",
    "m2_cpu_smoke",
    "m2_gb300_autoencoder",
    "m2_nv72_autoencoder",
]

_PROFILE_CASES = [
    ("m1_gpt2_smoke", "gpt2", False, 8, 1, 5e-5, "none", 256, "no", 50),
    (
        "m1_3b_lora", "Qwen/Qwen2.5-3B-Instruct", True, 8, 2,
        1e-4, "none", 1024, "bf16", None,
    ),
    (
        "m1_7b_lora", "Qwen/Qwen2.5-7B-Instruct", True, 8, 4,
        1e-4, "none", 1024, "bf16", None,
    ),
    (
        "m1_7b_rslora_r32", "Qwen/Qwen2.5-7B-Instruct", True, 8, 4,
        1e-4, "none", 1024, "bf16", None,
    ),
    (
        "m1_3b_full", "Qwen/Qwen2.5-3B-Instruct", False, 4, 8,
        2e-5, "zero3", 1024, "bf16", None,
    ),
    (
        "m1_7b_full_zero3", "Qwen/Qwen2.5-7B-Instruct", False, 2, 16,
        1e-5, "zero3", 1024, "bf16", None,
    ),
]


def test_all_registered_profiles_present():
    """Every named profile from the plan (plus the env-driven m1_local) is present."""
    names = profile_names()
    assert names == _REGISTERED


@pytest.mark.parametrize(
    (
        "name", "model", "use_peft", "batch_size", "grad_accum", "lr",
        "sharding", "seq_len", "precision", "max_steps",
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


def test_7b_rslora_matches_plan_spec():
    """m1_7b_rslora_r32 resolves to the plan's exact LoraConfig kwargs."""
    p = resolve_profile("m1_7b_rslora_r32")
    assert p.model == "Qwen/Qwen2.5-7B-Instruct" and p.use_peft
    kw = apply_lora_kwargs(p)
    assert kw["r"] == 32 and kw["alpha"] == 64 and kw["dropout"] == 0.05
    assert kw["adapter_method"] == "rslora"
    assert kw["init_lora_weights"] is True
    assert kw["target_modules"] == [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    ]


def test_full_ft_profiles_have_no_adapter_and_shard():
    """Full fine-tune profiles disable PEFT, shard with zero3, and refuse LoRA kwargs."""
    for n in ("m1_3b_full", "m1_7b_full_zero3"):
        p = resolve_profile(n)
        assert not p.use_peft and p.adapter is None and p.sharding == "zero3"
        with pytest.raises(ValueError):
            apply_lora_kwargs(p)


def test_smoke_profile_is_cheap_and_capped():
    """The GPT-2 smoke is a small, step-capped full-FT launch check."""
    p = resolve_profile("m1_gpt2_smoke")
    assert p.model == "gpt2" and p.max_steps == 50 and not p.use_peft


def test_resolve_applies_overrides_and_rejects_typos():
    """Overrides apply; an unknown field raises, and an unknown profile raises."""
    p = resolve_profile("m1_3b_lora", batch_size=16, lr=2e-4, max_steps=200)
    assert p.batch_size == 16 and p.lr == 2e-4 and p.max_steps == 200
    with pytest.raises(ValueError):
        resolve_profile("m1_3b_lora", btch_size=16)  # typo must not silently no-op
    with pytest.raises(KeyError):
        resolve_profile("does_not_exist")


def test_resolve_override_does_not_mutate_registered_profile():
    """Overrides return a copy and leave the registered profile unchanged."""
    base = resolve_profile("m1_3b_lora")
    changed = resolve_profile("m1_3b_lora", batch_size=16, lr=2e-4, max_steps=200)
    again = resolve_profile("m1_3b_lora")
    assert changed.batch_size == 16 and changed.lr == 2e-4 and changed.max_steps == 200
    assert again is base
    assert again.batch_size == 8 and again.lr == 1e-4 and again.max_steps is None


def test_peft_false_override_disables_lora_kwargs_even_with_adapter():
    """A use_peft=False override makes a profile behave as a full fine-tune."""
    p = resolve_profile("m1_3b_lora", use_peft=False)
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
    p = resolve_profile("m1_3b_lora", adapter=adapter)
    kw = apply_lora_kwargs(p)
    assert kw["init_lora_weights"] == "loftq"
    assert kw["target_modules"] == ["x_proj"]


def test_describe_is_flat_log_safe_dict():
    """describe() yields a flat dict suitable for stdout + W&B config."""
    d = resolve_profile("m1_7b_rslora_r32").describe()
    assert d["profile"] == "m1_7b_rslora_r32" and d["use_peft"] is True
    assert d["adapter"]["method"] == "rslora" and d["adapter"]["r"] == 32
    full = resolve_profile("m1_7b_full_zero3").describe()
    assert full["adapter"]["method"] == "full_finetune" and full["sharding"] == "zero3"


# Author: Mus mbayramo@stanford.edu
