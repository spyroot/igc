"""Offline tests for the action-repr / launch-fix arg-parser changes.

Checks the new pointer-policy flags and the launch-blocker fixes: --action_repr
defaults to onehot, --raw_data_dir is defined (so the RL path stops crashing),
rl batch/buffer sizes are ints, and --model_type accepts a free-form backbone id.
Importable offline now that deepspeed is imported lazily.

Author:
Mus mbayramo@stanford.edu
"""
import sys

import pytest

from igc.shared.shared_arg_parser import shared_arg_parser


def _parse(monkeypatch, argv):
    """Build the parser with a controlled argv and return the parsed namespace."""
    monkeypatch.setattr(sys, "argv", ["igc", *argv])
    args, _sections = shared_arg_parser()
    return args


def test_action_repr_defaults_onehot(monkeypatch):
    """--action_repr defaults to the legacy onehot path (safe default)."""
    args = _parse(monkeypatch, [])
    assert args.action_repr == "onehot"
    assert args.action_emb_dim == 256


def test_action_repr_pointer_selectable(monkeypatch):
    """--action_repr pointer selects the candidate-scoring path."""
    args = _parse(monkeypatch, ["--action_repr", "pointer"])
    assert args.action_repr == "pointer"


def test_action_repr_rejects_unknown_mode(monkeypatch):
    """--action_repr rejects unimplemented modes instead of silently accepting them."""
    with pytest.raises(SystemExit):
        _parse(monkeypatch, ["--action_repr", "tool"])


def test_raw_data_dir_is_defined(monkeypatch):
    """--raw_data_dir exists with a sensible default (fixes the AttributeError)."""
    args = _parse(monkeypatch, [])
    assert args.raw_data_dir == "~/.json_responses"


def test_rl_sizes_are_ints(monkeypatch):
    """rl batch/buffer sizes parse as ints, not floats."""
    args = _parse(monkeypatch, [])
    assert isinstance(args.rl_batch_size, int) and args.rl_batch_size == 8
    assert isinstance(args.rl_buffer_size, int)


def test_model_type_is_free_form(monkeypatch):
    """--model_type accepts an arbitrary HF repo id (no gpt2-only choices)."""
    args = _parse(monkeypatch, ["--model_type", "meta-llama/Meta-Llama-3-8B"])
    assert args.model_type == "meta-llama/Meta-Llama-3-8B"


def test_large_model_loader_flags_parse(monkeypatch):
    """Large-backbone loader flags are available without importing a model."""
    defaults = _parse(monkeypatch, [])
    assert defaults.trust_remote_code is False
    assert defaults.llm_torch_dtype is None

    args = _parse(
        monkeypatch,
        [
            "--model_type",
            "/models/DeepSeek-V4-Flash",
            "--trust_remote_code",
            "--llm_torch_dtype",
            "bfloat16",
        ],
    )
    assert args.model_type == "/models/DeepSeek-V4-Flash"
    assert args.trust_remote_code is True
    assert args.llm_torch_dtype == "bfloat16"


def test_large_model_dtype_rejects_unknown_value(monkeypatch):
    """--llm_torch_dtype stays constrained to supported loader values."""
    with pytest.raises(SystemExit):
        _parse(monkeypatch, ["--llm_torch_dtype", "int8"])


def test_peft_flags(monkeypatch):
    """--use_peft defaults off; LoRA hyperparams have sane defaults and parse."""
    defaults = _parse(monkeypatch, [])
    assert defaults.use_peft is False
    assert defaults.lora_r == 16
    assert defaults.lora_alpha == 32
    assert defaults.lora_target_modules is None
    on = _parse(monkeypatch, ["--use_peft", "--lora_r", "8", "--lora_target_modules", "q_proj", "v_proj"])
    assert on.use_peft is True and on.lora_r == 8
    assert on.lora_target_modules == ["q_proj", "v_proj"]


def test_sharding_and_accelerator_flags_parse(monkeypatch):
    """Sharding and accelerator flags parse without importing DeepSpeed or Accelerate."""
    args = _parse(
        monkeypatch,
        [
            "--sharding",
            "zero3",
            "--mixed_precision",
            "bf16",
            "--use_accelerator",
        ],
    )
    assert args.sharding == "zero3"
    assert args.mixed_precision == "bf16"
    assert args.use_accelerator is True
    assert args.device is None


def test_phase_profile_metadata_flags_parse(monkeypatch):
    """Phase launcher metadata is accepted by igc_main.py."""
    args = _parse(
        monkeypatch,
        [
            "--phase",
            "phase2_goal_extract",
            "--profile",
            "phase2_gpt2_smoke",
            "--objective",
            "ordered_rest_goal_extraction",
            "--dataset_jsonl",
            "/models/igc/goal-datasets/D1_ordered_rest_goals.jsonl",
        ],
    )
    assert args.phase == "phase2_goal_extract"
    assert args.profile == "phase2_gpt2_smoke"
    assert args.objective == "ordered_rest_goal_extraction"
    assert args.dataset_jsonl.endswith("D1_ordered_rest_goals.jsonl")


def test_redfish_connection_flags_parse_without_live_mode(monkeypatch):
    """Redfish connection flags parse but do not imply live execution."""
    args = _parse(
        monkeypatch,
        [
            "--redfish-ip",
            "https://example.invalid",
            "--redfish-port",
            "8443",
            "--insecure",
            "--is-http",
            "--x-auth",
            "placeholder-token",
        ],
    )
    assert args.live is False
    assert args.redfish_ip == "https://example.invalid"
    assert args.redfish_port == 8443
    assert args.insecure is True
    assert args.is_http is True
    assert args.x_auth == "placeholder-token"


# Author: Mus mbayramo@stanford.edu
