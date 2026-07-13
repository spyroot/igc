"""Dedicated M1/M2 training spec helpers.

This module keeps stage-specific callers from depending on the profile registry
internals. Profiles are YAML-backed under ``configs/training/profiles`` and
resolve to the same ``TrainingProfile`` object the launchers already understand.
"""

from __future__ import annotations

from pathlib import Path

from igc.modules.train.profiles import TrainingProfile, resolve_profile, specs_dir


def load_stage_spec(name_or_path: str | Path) -> TrainingProfile:
    """Load one M1/M2 training spec by name or YAML path."""
    profile = resolve_profile(str(name_or_path))
    if profile.stage not in {"m1", "m2"}:
        raise ValueError(f"unsupported training stage {profile.stage!r}")
    return profile


def training_specs_dir() -> Path:
    """Return the committed public-safe training spec directory."""
    return specs_dir()


def apply_stage_spec(namespace, profile: TrainingProfile):
    """Apply a resolved M1/M2 profile to an argparse-style namespace.

    The shared trainers consume plain argparse attributes. This helper is the
    bridge from the YAML-backed profile contract to those existing attributes.
    """
    namespace.train = "llm"
    namespace.llm = profile.llm_stage
    namespace.model_type = profile.model
    namespace.llm_torch_dtype = profile.torch_dtype
    namespace.per_device_train_batch_size = profile.batch_size
    namespace.gradient_accumulation_steps = profile.grad_accum
    namespace.num_workers = profile.num_workers
    namespace.llm_learning_rate = profile.lr
    namespace.llm_scheduler = profile.scheduler
    namespace.seq_len = profile.seq_len
    namespace.num_train_epochs = profile.epochs
    namespace.metric_prefix = profile.metric_prefix
    namespace.use_peft = profile.use_peft
    namespace.sharding = profile.sharding
    namespace.mixed_precision = profile.precision
    namespace.use_accelerator = profile.sharding != "none"

    if profile.stage == "m1":
        namespace.max_train_steps = profile.max_steps
    elif profile.stage == "m2":
        namespace.auto_encoder_lr = profile.auto_encoder_lr
        namespace.auto_encoder_optimizer = profile.auto_encoder_optimizer
        namespace.auto_encoder_weight_decay = profile.auto_encoder_weight_decay
        namespace.auto_encoder_train_steps = profile.max_steps

    if profile.use_peft and profile.adapter is not None:
        namespace.lora_r = profile.adapter.r
        namespace.lora_alpha = profile.adapter.alpha
        namespace.lora_dropout = profile.adapter.dropout
        namespace.adapter_method = profile.adapter.method
        namespace.lora_init = profile.adapter.init

    return namespace
