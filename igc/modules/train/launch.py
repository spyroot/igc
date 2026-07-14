"""Resolve a named training profile into an ``igc_main.py`` command line.

So an experiment can be run by NAME rather than a long, error-prone flag list:
``python -m igc.modules.train.launch --profile m1_7b_rslora_r32 --print-argv`` prints the
exact argv, and ``scripts/run_profile.sh`` feeds it to ``igc_main.py`` with the data/output
dirs supplied from the environment (kept out of code so nothing endpoint- or path-specific
is committed). Without ``--print-argv`` it prints the resolved profile
(:meth:`~igc.modules.train.profiles.TrainingProfile.describe`) for the log.

Pure stdlib; the argv is deterministic and offline-testable.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

import argparse
import copy
import json
from typing import List

from igc.modules.train.phase_specs import load_phase_profile
from igc.modules.train.phase_specs import profile_names as phase_profile_names
from igc.modules.train.profiles import TrainingProfile, profile_names, resolve_profile


_PHASE_LLM = {
    "phase1_pretrain": "latent",
    "phase2_goal_extract": "goal",
    "phase3_argument_extract": "parameter",
}

_SCHEDULER_ALIASES = {
    "cosine": "CosineAnnealingLR",
    "onecycle": "OneCycleLR",
    "one_cycle": "OneCycleLR",
    "linear": "LambdaLR",
}

_OPTIMIZER_ALIASES = {
    "adamw_torch_fused": "AdamW",
    "adamw": "AdamW",
}


def profile_to_argv(profile: TrainingProfile) -> List[str]:
    """Map a resolved profile to the ``igc_main.py`` M1 (state-encoder) argv.

    Data/output locations are intentionally NOT included — the launcher supplies
    ``--json_data_dir`` / ``--output_dir`` from the environment so no path or endpoint is
    baked into committed code.

    :param profile: the resolved :class:`~igc.modules.train.profiles.TrainingProfile`.
    :return: the argv list (train stage, model, optimization, adapter, sharding).
    """
    argv = [
        "--train", "llm", "--llm", "latent",
        "--model_type", profile.model,
        "--llm_torch_dtype", profile.torch_dtype,
        "--per_device_train_batch_size", str(profile.batch_size),
        "--gradient_accumulation_steps", str(profile.grad_accum),
        "--num_workers", str(profile.num_workers),
        "--llm_learning_rate", str(profile.lr),
        "--llm_scheduler", profile.scheduler,
        "--seq_len", str(profile.seq_len),
    ]
    if profile.max_steps is not None:
        argv += ["--max_train_steps", str(profile.max_steps)]
    else:
        argv += ["--num_train_epochs", str(profile.epochs)]
    if profile.use_peft and profile.adapter is not None:
        a = profile.adapter
        argv += [
            "--use_peft",
            "--lora_r", str(a.r), "--lora_alpha", str(a.alpha), "--lora_dropout", str(a.dropout),
            "--adapter_method", a.method, "--lora_init", a.init,
        ]
    if profile.sharding and profile.sharding != "none":
        argv += ["--use_accelerator", "--sharding", profile.sharding,
                 "--mixed_precision", profile.precision]
    return argv


def all_profile_names() -> List[str]:
    """All profile names accepted by this launcher: M1 dataclass + phase YAML."""
    return profile_names() + phase_profile_names()


def phase_profile_to_argv(profile: dict) -> List[str]:
    """Map a resolved Phase 1/2/3 YAML profile to parser-valid ``igc_main.py`` argv.

    The phase spec remains the run contract for model/trainer/optimizer/PEFT
    choices and W&B metric namespaces. This adapter feeds that contract into the
    existing training entrypoint without requiring each launcher to reimplement
    phase-specific flag translation.

    :param profile: resolved profile from :func:`load_phase_profile`.
    :return: the argv list for ``igc_main.py``.
    """
    phase = str(profile["phase"])
    llm = _PHASE_LLM[phase]
    model = profile["model"]
    trainer = profile["trainer"]
    optimizer = profile["optimizer"]
    dataset = profile["dataset"]

    argv = [
        "--phase", phase,
        "--profile", str(profile["name"]),
        "--objective", str(profile["objective"]),
        "--dataset_jsonl", str(dataset.get("jsonl", "")),
        "--train", "llm", "--llm", llm,
        "--model_type", str(model["model_type"]),
        "--llm_torch_dtype", str(model.get("torch_dtype", "auto")),
        "--per_device_train_batch_size", str(trainer["batch_size"]),
        "--gradient_accumulation_steps", str(trainer["gradient_accumulation_steps"]),
        "--num_workers", str(trainer["dataloader_num_workers"]),
        "--llm_optimizer", _optimizer_name(str(optimizer["name"])),
        "--llm_learning_rate", str(optimizer["learning_rate"]),
        "--llm_weight_decay", str(optimizer["weight_decay"]),
        "--llm_scheduler", _scheduler_name(str(optimizer["scheduler"])),
        "--max_grad_norm", str(optimizer["max_grad_norm"]),
        "--seq_len", str(trainer["max_length"]),
        "--seed", str(trainer["seed"]),
    ]
    if model.get("trust_remote_code"):
        argv.append("--trust_remote_code")
    if trainer.get("tf32"):
        argv.append("--tf32")
    if trainer.get("gradient_checkpointing"):
        argv += ["--gradient_checkpointing", "True"]
    if trainer.get("max_steps") is not None:
        argv += ["--max_train_steps", str(trainer["max_steps"])]
    else:
        argv += ["--num_train_epochs", str(trainer["epochs"])]

    peft = profile.get("peft") or {}
    if peft.get("enabled"):
        argv += [
            "--use_peft",
            "--adapter_method", str(peft["method"]),
            "--lora_r", str(peft["r"]),
            "--lora_alpha", str(peft["lora_alpha"]),
            "--lora_dropout", str(peft["lora_dropout"]),
            "--lora_init", _lora_init(peft.get("init_lora_weights")),
        ]

    distributed = profile.get("distributed") or {}
    if int(distributed.get("nproc_per_node", 1) or 1) > 1:
        precision = _mixed_precision(str(trainer.get("precision", "fp32")))
        argv += [
            "--use_accelerator",
            "--sharding", str(distributed.get("sharding", "ddp")),
            "--mixed_precision", precision,
        ]
    return argv


def main(argv=None) -> int:
    """CLI: resolve ``--profile`` and print either its argv or its description."""
    ap = argparse.ArgumentParser(description="Resolve a training profile to a command line.")
    ap.add_argument("--profile", required=True, choices=all_profile_names(),
                    help="Named training profile from the M1 profile module or Phase YAML spec.")
    ap.add_argument("--print-argv", action="store_true",
                    help="Print the space-joined igc_main.py argv (for a launcher).")
    ap.add_argument("--set", action="append", default=[], metavar="field=value",
                    help="Override a profile field (e.g. --set batch_size=16 --set lr=2e-4).")
    args = ap.parse_args(argv)

    overrides = {}
    for kv in args.set:
        key, _, value = kv.partition("=")
        overrides[key] = _coerce(value)

    if args.profile in phase_profile_names():
        profile = _apply_phase_overrides(load_phase_profile(args.profile), overrides)
        payload = profile
        profile_argv = phase_profile_to_argv(profile)
    else:
        profile = resolve_profile(args.profile, **overrides)
        payload = profile.describe()
        profile_argv = profile_to_argv(profile)

    if args.print_argv:
        print(" ".join(profile_argv))
    else:
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


def _coerce(value: str):
    """Best-effort str -> int/float/bool for --set overrides."""
    low = value.lower()
    if low in ("true", "false"):
        return low == "true"
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return value


def _apply_phase_overrides(profile: dict, overrides: dict) -> dict:
    """Apply common ``--set`` overrides to a resolved phase profile."""
    if not overrides:
        return profile
    profile = copy.deepcopy(profile)
    mapping = {
        "model": ("model", "model_type"),
        "model_type": ("model", "model_type"),
        "batch_size": ("trainer", "batch_size"),
        "gradient_accumulation_steps": ("trainer", "gradient_accumulation_steps"),
        "grad_accum": ("trainer", "gradient_accumulation_steps"),
        "num_workers": ("trainer", "dataloader_num_workers"),
        "dataloader_num_workers": ("trainer", "dataloader_num_workers"),
        "seq_len": ("trainer", "max_length"),
        "max_length": ("trainer", "max_length"),
        "epochs": ("trainer", "epochs"),
        "max_steps": ("trainer", "max_steps"),
        "precision": ("trainer", "precision"),
        "lr": ("optimizer", "learning_rate"),
        "learning_rate": ("optimizer", "learning_rate"),
        "weight_decay": ("optimizer", "weight_decay"),
    }
    bad = sorted(set(overrides) - set(mapping))
    if bad:
        raise ValueError(f"unknown phase profile override(s) {bad}; valid: {sorted(mapping)}")
    for key, value in overrides.items():
        section, field = mapping[key]
        profile[section][field] = value
    return profile


def _scheduler_name(name: str) -> str:
    """Normalize phase-spec scheduler names to parser-supported scheduler names."""
    return _SCHEDULER_ALIASES.get(name.lower(), name)


def _optimizer_name(name: str) -> str:
    """Normalize phase-spec optimizer names to parser-supported optimizer names."""
    return _OPTIMIZER_ALIASES.get(name.lower(), name)


def _mixed_precision(precision: str) -> str:
    """Map trainer precision values to accelerate ``--mixed_precision`` choices."""
    normalized = precision.lower()
    if normalized in ("fp32", "float32", "no", "none"):
        return "no"
    if normalized in ("bf16", "bfloat16"):
        return "bf16"
    if normalized in ("fp16", "float16"):
        return "fp16"
    return normalized


def _lora_init(value) -> str:
    """Map phase PEFT init values to the parser's ``--lora_init`` choices."""
    if value is True or value in (None, "", "true", "True"):
        return "default"
    return str(value)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


# Author: Mus mbayramo@stanford.edu
