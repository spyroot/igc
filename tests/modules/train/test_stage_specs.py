"""Offline tests for M1/M2 YAML-backed training profiles."""

from pathlib import Path

import pytest

from igc.modules.train.profiles import (
    profile_names,
    resolve_profile,
    specs_dir,
)
from igc.modules.train.stage_specs import apply_stage_spec, load_stage_spec


def test_profiles_load_from_committed_yaml_specs():
    """The M1/M2 run contract is backed by public-safe YAML specs."""
    root = specs_dir()
    assert root == Path("configs/training/profiles")
    assert (root / "m1_cpu_smoke.yaml").is_file()
    assert (root / "m2_cpu_smoke.yaml").is_file()

    names = profile_names()
    assert "m1_cpu_smoke" in names
    assert "m1_gpt2_smoke" in names
    assert "m2_cpu_smoke" in names
    assert "m2_gb300_autoencoder" in names
    assert "m2_nv72_autoencoder" in names


@pytest.mark.parametrize("name,stage,llm_stage", [
    ("m1_cpu_smoke", "m1", "latent"),
    ("m1_nv72_7b_rslora_r32", "m1", "latent"),
    ("m2_cpu_smoke", "m2", "encoder"),
    ("m2_gb300_autoencoder", "m2", "encoder"),
])
def test_profiles_carry_stage_and_cli_entrypoint(name, stage, llm_stage):
    """Profiles say which curriculum stage they train and which CLI entrypoint to call."""
    profile = resolve_profile(name)
    assert profile.stage == stage
    assert profile.llm_stage == llm_stage
    assert profile.implemented is True
    assert profile.data_contract == "captured_redfish_json"
    assert profile.live_redfish_allowed is False


def test_m2_cpu_profile_is_local_smoke_and_uses_autoencoder_knobs():
    """M2 has a cheap CPU smoke profile instead of inheriting M1's causal-LM defaults."""
    profile = resolve_profile("m2_cpu_smoke")
    assert profile.model == "gpt2"
    assert profile.use_peft is False
    assert profile.batch_size == 2
    assert profile.seq_len == 64
    assert profile.max_steps == 5
    assert profile.auto_encoder_lr == 1e-3
    assert profile.metric_prefix == "m2/state_autoencoder"
    assert profile.nccl_mnnvl_enable == "0"
    assert profile.nccl_cumem_enable == "1"


def test_apply_stage_spec_updates_trainer_namespace():
    """A resolved spec can be applied to the namespace consumed by trainers."""
    class Args:
        train = "none"
        llm = "latent"
        model_type = "gpt2"
        per_device_train_batch_size = 4
        gradient_accumulation_steps = 8
        num_workers = 1
        seq_len = 1024
        auto_encoder_train_steps = None
        metric_prefix = None

    args = Args()
    profile = load_stage_spec("m2_cpu_smoke")
    apply_stage_spec(args, profile)

    assert args.train == "llm"
    assert args.llm == "encoder"
    assert args.per_device_train_batch_size == 2
    assert args.gradient_accumulation_steps == 1
    assert args.seq_len == 64
    assert args.auto_encoder_train_steps == 5
    assert args.metric_prefix == "m2/state_autoencoder"


def test_unimplemented_or_live_default_profiles_are_rejected(tmp_path):
    """Spec validation rejects dishonest implemented flags and live-host defaults."""
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        """
version: 1
name: bad_live
stage: m2
implemented: true
llm_stage: encoder
backbone: {model: gpt2, torch_dtype: float32}
adapter: {use_peft: false}
optimizer: {auto_encoder_lr: 0.001, auto_encoder_optimizer: Adam}
dataloader: {batch_size: 1, grad_accum: 1, num_workers: 0, seq_len: 32}
training: {epochs: 1, max_steps: 1, sharding: none, precision: no, metric_prefix: m2/state_autoencoder}
data: {contract: captured_redfish_json, live_redfish_allowed: true}
runtime: {nccl_mnnvl_enable: "0", nccl_cumem_enable: "1"}
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="live Redfish"):
        resolve_profile(str(bad))
