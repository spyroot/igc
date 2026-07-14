"""Offline tests for profile_to_argv (name -> igc_main.py command line).

Pins that each named profile maps to the right igc_main flags: the smoke caps steps and
runs full-FT, LoRA profiles carry adapter flags + epochs, the 7B rsLoRA candidate emits
--adapter_method rslora at r32/alpha64, and full-FT profiles turn on the accelerator +
sharding. Data/output dirs are intentionally NOT baked in. Pure stdlib.

Author:
Mus mbayramo@stanford.edu
"""
import json

import pytest

from igc.modules.train.launch import main, profile_to_argv
from igc.modules.train.launch import phase_profile_to_argv
from igc.modules.train.phase_specs import load_phase_profile
from igc.modules.train.profiles import resolve_profile


def _val(argv, flag):
    """Value following ``flag`` in the argv list, or None if the flag is absent."""
    return argv[argv.index(flag) + 1] if flag in argv else None


def test_smoke_is_step_capped_full_ft():
    """The GPT-2 smoke caps steps and does not enable PEFT."""
    argv = profile_to_argv(resolve_profile("m1_gpt2_smoke"))
    assert _val(argv, "--model_type") == "gpt2"
    assert _val(argv, "--max_train_steps") == "50"
    assert "--use_peft" not in argv and "--num_train_epochs" not in argv


def test_3b_lora_has_adapter_flags_and_epochs():
    """The 3B LoRA profile emits PEFT + adapter flags and epoch-based length."""
    argv = profile_to_argv(resolve_profile("m1_3b_lora"))
    assert "--use_peft" in argv
    assert _val(argv, "--lora_r") == "16" and _val(argv, "--adapter_method") == "lora"
    assert _val(argv, "--num_train_epochs") == "3" and "--max_train_steps" not in argv


def test_7b_rslora_argv_matches_spec():
    """m1_7b_rslora_r32 emits the plan's rsLoRA r32/alpha64 adapter flags."""
    argv = profile_to_argv(resolve_profile("m1_7b_rslora_r32"))
    assert _val(argv, "--model_type") == "Qwen/Qwen2.5-7B-Instruct"
    assert _val(argv, "--adapter_method") == "rslora"
    assert _val(argv, "--lora_r") == "32" and _val(argv, "--lora_alpha") == "64"
    assert _val(argv, "--lora_init") == "default"


def test_full_ft_enables_accelerator_and_sharding():
    """Full fine-tune profiles turn on the accelerator + ZeRO-3 and drop PEFT."""
    argv = profile_to_argv(resolve_profile("m1_7b_full_zero3"))
    assert "--use_accelerator" in argv and _val(argv, "--sharding") == "zero3"
    assert _val(argv, "--mixed_precision") == "bf16" and "--use_peft" not in argv


def test_override_flows_into_argv():
    """A resolved override (e.g. batch_size) is reflected in the argv."""
    argv = profile_to_argv(resolve_profile("m1_7b_lora", batch_size=16))
    assert _val(argv, "--per_device_train_batch_size") == "16"


def test_main_print_argv_applies_typed_set_overrides(capsys):
    """CLI --set overrides are coerced before printing the igc_main argv."""
    rc = main([
        "--profile", "m1_3b_lora",
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
    assert "--num_train_epochs" not in argv


def test_main_prints_public_safe_profile_json(capsys):
    """Without --print-argv, CLI output is a resolved profile JSON payload."""
    rc = main(["--profile", "m1_7b_full_zero3", "--set", "num_workers=2"])
    payload = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert payload["profile"] == "m1_7b_full_zero3"
    assert payload["num_workers"] == 2
    assert payload["adapter"]["method"] == "full_finetune"
    assert "json_data_dir" not in payload and "output_dir" not in payload


def test_main_rejects_unknown_set_override():
    """A typo in --set fails loudly instead of silently changing the command."""
    with pytest.raises(ValueError, match="unknown profile override"):
        main(["--profile", "m1_3b_lora", "--set", "btch_size=16"])


def test_phase2_profile_maps_to_goal_trainer_and_metadata():
    """Phase 2 profiles feed the existing goal trainer with phase metadata."""
    argv = phase_profile_to_argv(load_phase_profile("phase2_gpt2_smoke"))

    assert _val(argv, "--phase") == "phase2_goal_extract"
    assert _val(argv, "--profile") == "phase2_gpt2_smoke"
    assert _val(argv, "--objective") == "ordered_rest_goal_extraction"
    assert _val(argv, "--dataset_jsonl").endswith("D1_ordered_rest_goals.jsonl")
    assert _val(argv, "--llm") == "goal"
    assert _val(argv, "--llm_optimizer") == "AdamW"
    assert _val(argv, "--llm_scheduler") == "CosineAnnealingLR"
    assert _val(argv, "--max_train_steps") == "8"
    assert "--gradient_checkpointing" not in argv


def test_phase3_profile_maps_to_parameter_trainer():
    """Phase 3 profiles feed the existing argument/parameter trainer."""
    argv = phase_profile_to_argv(load_phase_profile("phase3_gpt2_smoke"))

    assert _val(argv, "--phase") == "phase3_argument_extract"
    assert _val(argv, "--objective") == "ordered_method_argument_extraction"
    assert _val(argv, "--llm") == "parameter"
    assert _val(argv, "--dataset_jsonl").endswith("D2_ordered_method_arguments.jsonl")


def test_phase_gpu_profile_carries_peft_dtype_and_accelerator_flags():
    """The 7B phase profile emits rsLoRA, trust_remote_code, dtype, and DDP flags."""
    argv = phase_profile_to_argv(load_phase_profile("phase1_qwen2_5_7b_rslora"))

    assert _val(argv, "--phase") == "phase1_pretrain"
    assert _val(argv, "--llm") == "latent"
    assert _val(argv, "--model_type") == "Qwen/Qwen2.5-7B-Instruct"
    assert "--trust_remote_code" in argv
    assert _val(argv, "--llm_torch_dtype") == "bfloat16"
    assert _val(argv, "--adapter_method") == "rslora"
    assert _val(argv, "--lora_r") == "32"
    assert _val(argv, "--lora_alpha") == "64"
    assert _val(argv, "--lora_init") == "default"
    assert _val(argv, "--gradient_checkpointing") == "True"
    assert "--use_accelerator" in argv
    assert _val(argv, "--sharding") == "ddp"
    assert _val(argv, "--mixed_precision") == "bf16"


def test_main_accepts_phase_profile_and_overrides(capsys):
    """The CLI accepts phase profiles, so profiles.yaml has a live launcher path."""
    rc = main([
        "--profile", "phase2_gpt2_smoke",
        "--set", "batch_size=2",
        "--set", "lr=1e-4",
        "--print-argv",
    ])
    argv = capsys.readouterr().out.strip().split()

    assert rc == 0
    assert _val(argv, "--profile") == "phase2_gpt2_smoke"
    assert _val(argv, "--per_device_train_batch_size") == "2"
    assert _val(argv, "--llm_learning_rate") == "0.0001"


def test_main_rejects_unknown_phase_set_override():
    """A typo in a phase --set fails loudly instead of silently changing the command."""
    with pytest.raises(ValueError, match="unknown phase profile override"):
        main(["--profile", "phase2_gpt2_smoke", "--set", "btch_size=16"])


# Author: Mus mbayramo@stanford.edu
