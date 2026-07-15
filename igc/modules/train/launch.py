"""Resolve a named Phase 1 profile into an ``igc_main.py`` command line.

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


_RENAMED_PROFILES = {
    "m1_gpt2_smoke": "phase1_gpt2_smoke",
    "m1_3b_lora": "phase1_3b_lora",
    "m1_7b_lora": "phase1_7b_lora",
    "m1_7b_rslora_r32": "phase1_7b_rslora_r32",
    "m1_local": "phase1_local",
    "m1_3b_full": "phase1_3b_full",
    "m1_7b_full_zero3": "phase1_7b_full_zero3",
}


def profile_to_argv(profile: TrainingProfile) -> List[str]:
    """Map a resolved profile to the ``igc_main.py`` Phase 1 argv.

    Data/output locations are intentionally NOT included — the launcher supplies
    ``--json_data_dir`` / ``--output_dir`` from the environment so no path or endpoint is
    baked into committed code. The profile does include ``--corpus_objective`` so the
    resolved command states the real data objective instead of hiding Phase 1 behind the
    internal ``--llm latent`` trainer route.

    :param profile: the resolved :class:`~igc.modules.train.profiles.TrainingProfile`.
    :return: the argv list (train stage, model, optimization, adapter, sharding).
    """
    argv = [
        "--profile", profile.name,
        "--weights_role", profile.weights_role,
        "--train", "llm", "--llm", profile.llm_stage,
        "--corpus_objective", profile.corpus_objective,
        "--model_type", profile.model,
        "--llm_torch_dtype", profile.torch_dtype,
        "--per_device_train_batch_size", str(profile.batch_size),
        "--gradient_accumulation_steps", str(profile.grad_accum),
        "--num_workers", str(profile.num_workers),
        "--llm_learning_rate", str(profile.lr),
        "--llm_scheduler", profile.scheduler,
        "--early_stopping_patience", str(profile.early_stopping_patience),
        "--early_stopping_min_delta", str(profile.early_stopping_min_delta),
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


def main(argv=None) -> int:
    """CLI: resolve ``--profile`` and print either its argv or its description."""
    ap = argparse.ArgumentParser(description="Resolve a Phase 1 training profile to a command line.")
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
        renamed = _RENAMED_PROFILES.get(args.profile)
        if renamed is not None:
            ap.error(f"profile {args.profile!r} was renamed to {renamed!r}; use phase1_* names")
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
