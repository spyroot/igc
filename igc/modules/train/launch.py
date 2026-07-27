"""Resolve a named Phase 1/2/3 profile into an ``igc_main.py`` command line.

So a Phase 1 Redfish JSON pretraining/fine-tune can be run by NAME rather than a long,
error-prone flag list:
``python -m igc.modules.train.launch --profile phase1_7b_rslora_r32 --print-argv`` prints
the exact argv, and ``scripts/run_profile.sh`` feeds it to ``igc_main.py`` with the
data/output dirs supplied from the environment (kept out of code so nothing endpoint- or
path-specific is committed). Without ``--print-argv`` it prints the resolved profile
(:meth:`~igc.modules.train.profiles.TrainingProfile.describe`) for the log.

Pure stdlib; the argv is deterministic and offline-testable.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

import argparse
import json
from typing import List

from igc.modules.train.profiles import TrainingProfile, profile_names, resolve_profile


def profile_to_argv(profile: TrainingProfile) -> List[str]:
    """Map a resolved profile to the shared ``igc_main.py`` SFT argv.

    Data/output locations are intentionally NOT included — the launcher supplies
    ``--json_data_dir`` / ``--output_dir`` from the environment so no path or endpoint is
    baked into committed code. The profile does include ``--corpus_objective`` so the
    resolved command states the real data objective and selects the one shared
    ``--llm sft`` trainer route.

    :param profile: the resolved :class:`~igc.modules.train.profiles.TrainingProfile`.
    :return: the argv list (train stage, model, optimization, adapter, sharding).
    """
    argv = [
        "--profile", profile.name,
        "--phase", profile.phase,
        "--weights_role", profile.weights_role,
        "--sft_task", profile.task,
        "--train", "llm", "--llm", profile.llm_stage,
        "--corpus_objective", profile.corpus_objective,
        "--phase1_structural_loss_profile", profile.phase1_structural_loss_profile,
        "--model_type", profile.model,
        "--llm_torch_dtype", profile.torch_dtype,
        "--per_device_train_batch_size", str(profile.batch_size),
        "--gradient_accumulation_steps", str(profile.grad_accum),
        "--num_workers", str(profile.num_workers),
        "--llm_optimizer", profile.optimizer,
        "--llm_learning_rate", str(profile.lr),
        "--llm_weight_decay", str(profile.weight_decay),
        "--max_grad_norm", str(profile.max_grad_norm),
        "--llm_scheduler", profile.scheduler,
        "--max_lr", str(profile.max_lr),
        "--warmup_ratio", str(profile.warmup_ratio),
        "--div_factor", str(profile.div_factor),
        "--final_div_factor", str(profile.final_div_factor),
        "--anneal_strategy", profile.anneal_strategy,
        "--seed", str(profile.seed),
        "--early_stopping_patience", str(profile.early_stopping_patience),
        "--early_stopping_min_delta", str(profile.early_stopping_min_delta),
        "--eval_steps", str(profile.eval_steps),
        "--save_steps", str(profile.save_steps),
        "--seq_len", str(profile.seq_len),
    ]
    argv.append(
        "--gradient_checkpointing"
        if profile.gradient_checkpointing
        else "--no-gradient_checkpointing"
    )
    argv.append("--cycle_momentum" if profile.cycle_momentum else "--no-cycle_momentum")
    if profile.parent_adapter:
        argv += ["--parent_adapter_dir", profile.parent_adapter]
    if profile.parent_artifact_sha:
        argv += ["--parent_artifact_sha", profile.parent_artifact_sha]
    if profile.foundation_model_sha:
        argv += ["--foundation_model_sha", profile.foundation_model_sha]
    if profile.tokenizer_sha:
        argv += ["--tokenizer_sha", profile.tokenizer_sha]
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
            "--lora_target_modules", *a.target_modules,
        ]
    if profile.sharding and profile.sharding != "none":
        argv += ["--use_accelerator", "--sharding", profile.sharding,
                 "--mixed_precision", profile.precision]
    return argv


def main(argv=None) -> int:
    """CLI: resolve ``--profile`` and print either its argv or its description."""
    ap = argparse.ArgumentParser(
        description="Resolve a Phase 1/2/3 SFT profile to a command line."
    )
    ap.add_argument("--profile", required=True,
                    help="Named training profile from igc.modules.train.profiles.")
    ap.add_argument("--print-argv", action="store_true",
                    help="Print the space-joined igc_main.py argv (for a launcher).")
    ap.add_argument("--set", action="append", default=[], metavar="field=value",
                    help="Override a profile field (e.g. --set batch_size=16 --set lr=2e-4).")
    args = ap.parse_args(argv)

    overrides = {}
    for kv in args.set:
        key, _, value = kv.partition("=")
        overrides[key] = _coerce(value)
    try:
        profile = resolve_profile(args.profile, **overrides)
    except KeyError:
        ap.error(
            f"unknown profile {args.profile!r}; valid profiles: {', '.join(profile_names())}"
        )

    if args.print_argv:
        print(" ".join(profile_to_argv(profile)))
    else:
        print(json.dumps(profile.describe(), indent=2, sort_keys=True, default=str))
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


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


# Author: Mus mbayramo@stanford.edu
