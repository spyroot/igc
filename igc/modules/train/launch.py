"""Resolve a named M1 profile into an ``igc_main.py`` command line (self-serve launch).

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
import json
from typing import List

from igc.modules.train.profiles import TrainingProfile, profile_names, resolve_profile


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
    ap = argparse.ArgumentParser(description="Resolve an M1 training profile to a command line.")
    ap.add_argument("--profile", required=True, choices=profile_names(),
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
    profile = resolve_profile(args.profile, **overrides)

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
