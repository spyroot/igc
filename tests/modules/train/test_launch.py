"""Offline tests for profile_to_argv (name -> igc_main.py command line).

Pins that each named Phase 1 profile maps to the right igc_main flags: the smoke caps
steps and runs full-FT, LoRA profiles carry adapter flags + epochs, the 7B rsLoRA
candidate emits --adapter_method rslora at r32/alpha64, and full-FT profiles turn on
the accelerator + sharding. Data/output dirs are intentionally NOT baked in. Pure stdlib.

Author:
Mus mbayramo@stanford.edu
"""
import json

import pytest

from igc.modules.train.launch import main, profile_to_argv
from igc.modules.train.profiles import AdapterSpec, resolve_profile


def _val(argv, flag):
    """Value following ``flag`` in the argv list, or None if the flag is absent."""
    return argv[argv.index(flag) + 1] if flag in argv else None


def test_smoke_is_step_capped_full_ft():
    """The GPT-2 smoke caps steps and does not enable PEFT."""
    argv = profile_to_argv(resolve_profile("phase1_gpt2_smoke"))
    assert _val(argv, "--profile") == "phase1_gpt2_smoke"
    assert _val(argv, "--model_type") == "gpt2"
    assert _val(argv, "--max_train_steps") == "50"
    assert _val(argv, "--corpus_objective") == "phase1_pretrain"
    assert _val(argv, "--weights_role") == "model_x"
    assert _val(argv, "--llm") == "sft"
    assert _val(argv, "--early_stopping_patience") == "3"
    assert _val(argv, "--early_stopping_min_delta") == "0.005"
    assert "--use_peft" not in argv and "--num_train_epochs" not in argv


def test_3b_lora_has_adapter_flags_and_epochs():
    """The 3B LoRA profile emits PEFT + adapter flags and epoch-based length."""
    argv = profile_to_argv(resolve_profile("phase1_3b_lora"))
    assert "--use_peft" in argv
    assert _val(argv, "--lora_r") == "16" and _val(argv, "--adapter_method") == "lora"
    assert _val(argv, "--num_train_epochs") == "3" and "--max_train_steps" not in argv


def test_7b_rslora_argv_matches_spec():
    """phase1_7b_rslora_r32 emits the plan's rsLoRA r32/alpha64 adapter flags."""
    argv = profile_to_argv(resolve_profile("phase1_7b_rslora_r32"))
    assert _val(argv, "--model_type") == "Qwen/Qwen2.5-7B-Instruct"
    assert _val(argv, "--adapter_method") == "rslora"
    assert _val(argv, "--lora_r") == "32" and _val(argv, "--lora_alpha") == "64"
    assert _val(argv, "--lora_init") == "default"


def test_profile_to_argv_forwards_every_lora_target_module_after_flag():
    """All adapter target modules must follow one --lora_target_modules flag."""
    target_modules = ("q_proj", "v_proj", "down_proj", "lm_head")
    profile = resolve_profile(
        "phase1_3b_lora",
        adapter=AdapterSpec(target_modules=target_modules),
    )

    argv = profile_to_argv(profile)

    start = argv.index("--lora_target_modules")
    assert argv[start:] == ["--lora_target_modules", *target_modules]


def test_profile_to_argv_forwards_warmup_ratio():
    """Profile warmup_ratio must reach igc_main instead of staying launcher-only."""
    argv = profile_to_argv(resolve_profile("phase1_3b_lora", warmup_ratio=0.17))

    assert _val(argv, "--warmup_ratio") == "0.17"


def test_profile_to_argv_forwards_optimizer_scheduler_and_seed_values():
    """Every profile-owned optimizer/scheduler knob reaches igc_main argv."""
    argv = profile_to_argv(
        resolve_profile(
            "phase1_3b_lora",
            optimizer="SGD",
            weight_decay=0.123,
            max_grad_norm=0.456,
            max_lr=0.002,
            warmup_ratio=0.12,
            div_factor=10.0,
            final_div_factor=100.0,
            anneal_strategy="linear",
            seed=777,
            gradient_checkpointing=True,
            cycle_momentum=True,
        )
    )

    assert _val(argv, "--llm_optimizer") == "SGD"
    assert _val(argv, "--llm_weight_decay") == "0.123"
    assert _val(argv, "--max_grad_norm") == "0.456"
    assert _val(argv, "--max_lr") == "0.002"
    assert _val(argv, "--warmup_ratio") == "0.12"
    assert _val(argv, "--div_factor") == "10.0"
    assert _val(argv, "--final_div_factor") == "100.0"
    assert _val(argv, "--anneal_strategy") == "linear"
    assert _val(argv, "--seed") == "777"
    assert "--gradient_checkpointing" in argv
    assert "--no-gradient_checkpointing" not in argv
    assert "--cycle_momentum" in argv
    assert "--no-cycle_momentum" not in argv


def test_profile_to_argv_forwards_boolean_optional_false_states():
    """BooleanOptionalAction-backed profile knobs emit explicit false flags."""
    argv = profile_to_argv(
        resolve_profile(
            "phase1_3b_lora",
            gradient_checkpointing=False,
            cycle_momentum=False,
        )
    )

    assert "--no-gradient_checkpointing" in argv
    assert "--gradient_checkpointing" not in argv
    assert "--no-cycle_momentum" in argv
    assert "--cycle_momentum" not in argv


def test_full_ft_enables_accelerator_and_sharding():
    """Full fine-tune profiles turn on the accelerator + ZeRO-3 and drop PEFT."""
    argv = profile_to_argv(resolve_profile("phase1_7b_full_zero3"))
    assert "--use_accelerator" in argv and _val(argv, "--sharding") == "zero3"
    assert _val(argv, "--mixed_precision") == "bf16" and "--use_peft" not in argv


def test_override_flows_into_argv():
    """A resolved override (e.g. batch_size) is reflected in the argv."""
    argv = profile_to_argv(resolve_profile("phase1_7b_lora", batch_size=16))
    assert _val(argv, "--per_device_train_batch_size") == "16"


def test_main_print_argv_applies_typed_set_overrides(capsys):
    """CLI --set overrides are coerced before printing the igc_main argv."""
    rc = main([
        "--profile", "phase1_3b_lora",
        "--set", "batch_size=12",
        "--set", "lr=2e-4",
        "--set", "max_steps=25",
        "--print-argv",
    ])
    argv = capsys.readouterr().out.strip().split()

    assert rc == 0
    assert _val(argv, "--per_device_train_batch_size") == "12"
    assert _val(argv, "--llm_learning_rate") == "0.0002"
    assert _val(argv, "--max_train_steps") == "25"
    assert _val(argv, "--early_stopping_patience") == "3"
    assert "--num_train_epochs" not in argv


def test_main_prints_public_safe_profile_json(capsys):
    """Without --print-argv, CLI output is a resolved profile JSON payload."""
    rc = main(["--profile", "phase1_7b_full_zero3", "--set", "num_workers=2"])
    payload = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert payload["profile"] == "phase1_7b_full_zero3"
    assert payload["num_workers"] == 2
    assert payload["phase"] == "phase1_finetune"
    assert payload["weights_role"] == "model_x"
    assert payload["corpus_objective"] == "phase1_pretrain"
    assert payload["early_stopping_patience"] == 3
    assert payload["early_stopping_min_delta"] == 0.005
    assert payload["adapter"]["method"] == "full_finetune"
    assert "json_data_dir" not in payload and "output_dir" not in payload


def test_main_rejects_unknown_set_override():
    """A typo in --set fails loudly instead of silently changing the command."""
    with pytest.raises(ValueError, match="unknown profile override"):
        main(["--profile", "phase1_3b_lora", "--set", "btch_size=16"])


def test_main_rejects_unknown_profile_with_valid_names(capsys):
    """Unknown profile errors list valid Phase 1 names instead of argparse choices noise."""
    with pytest.raises(SystemExit) as exc:
        main(["--profile", "phase9_magic"])

    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "unknown profile 'phase9_magic'" in err
    assert "phase1_gpt2_smoke" in err


# Author: Mus mbayramo@stanford.edu
