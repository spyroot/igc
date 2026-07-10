"""Build the Accelerate ``Accelerator`` with optional ZeRO / FSDP model sharding.

The legacy builder created a bare ``Accelerator()`` (plain DDP), which replicates the
whole model on every GPU and therefore cannot fit a large backbone no matter how many
GPUs are available. This wires the ``--sharding`` knob to a real DeepSpeed ZeRO or
PyTorch FSDP plugin so parameters / gradients / optimizer state are sharded across the
fleet — the prerequisite for fine-tuning a large model on the GB300s.

Sharding is opt-in: ``--sharding none`` keeps the original single-process/DDP behaviour
for the small/offline path. ``zero3`` is the recommended path for a large dense backbone.
MoE expert-parallelism (for a giant MoE like DeepSeek) and FP8 *weight* conversion are
separate, heavier tracks; ``--mixed_precision fp8`` here only sets the compute precision.

The flag→config mapping (:func:`sharding_config`) is a pure function with no heavy
imports, so the sharding logic is CPU/offline-testable; only :func:`build_accelerator`
touches accelerate / deepspeed.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

import argparse

_ZERO_STAGE = {"zero2": 2, "zero3": 3, "zero3_offload": 3}

# Sharding modes this builder knows how to wire. The CLI restricts ``--sharding`` to this
# set (:mod:`igc.shared.shared_arg_parser`), but :func:`sharding_config` also runs on
# Namespaces built directly (train profiles, config-driven launches, tests) that bypass the
# parser, so it validates too — an unknown mode must fail fast rather than silently produce a
# ``sharded`` run with ``zero_stage=None`` (a broken DeepSpeed plugin on a real GB300 job).
_VALID_SHARDING = ("none", "ddp", "zero2", "zero3", "zero3_offload", "fsdp")


def sharding_config(cmd: argparse.Namespace) -> dict:
    """Pure mapping of CLI flags to a sharding/precision config (no accelerate import).

    :param cmd: parsed args; reads ``sharding``, ``mixed_precision``,
        ``gradient_accumulation_steps``, ``device_placement``.
    :return: a config dict consumed by :func:`build_accelerator`. ``sharded`` is False for
        ``none``/``ddp`` (plain replication) and True for any ZeRO/FSDP mode; a sharded run
        defaults to ``bf16`` and disables Accelerate ``device_placement`` (DeepSpeed/FSDP
        place the model themselves).
    :raises ValueError: if ``sharding`` is not one of :data:`_VALID_SHARDING`.
    """
    sharding = getattr(cmd, "sharding", "none") or "none"
    if sharding not in _VALID_SHARDING:
        raise ValueError(
            f"Unsupported sharding mode {sharding!r}; expected one of {_VALID_SHARDING}"
        )
    mixed_precision = getattr(cmd, "mixed_precision", None)
    sharded = sharding not in ("none", "ddp")
    if mixed_precision is None and sharded:
        mixed_precision = "bf16"
    return {
        "sharding": sharding,
        "sharded": sharded,
        "mixed_precision": mixed_precision,
        "gradient_accumulation_steps": int(getattr(cmd, "gradient_accumulation_steps", 1) or 1),
        "zero_stage": _ZERO_STAGE.get(sharding),
        "offload": sharding == "zero3_offload",
        "device_placement": bool(getattr(cmd, "device_placement", True)) and not sharded,
    }


def _plugin_kwargs(cfg: dict) -> dict:
    """Construct the Accelerate plugin kwargs for the chosen sharding (lazy heavy imports).

    :param cfg: output of :func:`sharding_config`.
    :return: ``{}`` for none/ddp, ``{"deepspeed_plugin": ...}`` for ZeRO, or
        ``{"fsdp_plugin": ...}`` for FSDP.
    """
    if not cfg["sharded"]:
        return {}
    import importlib
    _utils = importlib.import_module("accelerate.utils")
    if cfg["sharding"] == "fsdp":
        # FSDP2 (torch-native): per-layer wrap via the model's _no_split_modules,
        # resharding after forward, sharded state dicts for multi-rank saves.
        return {"fsdp_plugin": _utils.FullyShardedDataParallelPlugin(
            fsdp_version=2,
            auto_wrap_policy="transformer_based_wrap",
            reshard_after_forward=True,
            state_dict_type="SHARDED_STATE_DICT",
            cpu_ram_efficient_loading=True,
        )}
    # ZeRO needs deepspeed installed; fail with a clear message BEFORE Accelerator
    # (whose failed init leaks ACCELERATE_USE_DEEPSPEED and poisons retries).
    _ds_available = getattr(_utils, "is_deepspeed_available", None)
    if _ds_available is not None and not _ds_available():
        raise ValueError(
            f"--sharding {cfg['sharding']} requires deepspeed, which is not "
            f"installed; install deepspeed or use --sharding fsdp.")
    DeepSpeedPlugin = _utils.DeepSpeedPlugin
    return {
        "deepspeed_plugin": DeepSpeedPlugin(
            zero_stage=cfg["zero_stage"],
            offload_optimizer_device="cpu" if cfg["offload"] else "none",
            offload_param_device="cpu" if cfg["offload"] else "none",
            gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        )
    }


def build_accelerator(cmd: argparse.Namespace):
    """Render an Accelerator, sharded per ``--sharding`` (ZeRO/FSDP) when requested.

    :param cmd: parsed command-line args.
    :return: the configured ``accelerate.Accelerator``.
    """
    from accelerate import Accelerator

    cfg = sharding_config(cmd)
    accelerator_args = {}
    if not cfg["sharded"]:
        accelerator_args["device_placement"] = cfg["device_placement"]
    if cfg["mixed_precision"]:
        accelerator_args["mixed_precision"] = cfg["mixed_precision"]
    if cfg["gradient_accumulation_steps"] > 1:
        accelerator_args["gradient_accumulation_steps"] = cfg["gradient_accumulation_steps"]
    accelerator_args.update(_plugin_kwargs(cfg))
    accelerator = Accelerator(**accelerator_args)
    # a sharded config in a single-process launch silently trains UNSHARDED —
    # fail loudly instead so a misconfigured GB300 job dies at startup.
    dist_type = getattr(accelerator, "distributed_type", None)
    if cfg["sharded"] and dist_type is not None and str(dist_type).endswith("NO"):
        raise RuntimeError(
            f"--sharding {cfg['sharding']} requested but this is a single-process "
            f"launch (distributed_type=NO); start via accelerate launch/torchrun "
            f"with --num_processes > 1 so sharding engages.")
    return accelerator


def broadcast_flag(accelerator, flag: bool) -> bool:
    """Make a rank-0 boolean decision uniform across ranks.

    Collective checkpoint operations (e.g. ``get_state_dict`` gathers under
    ZeRO-3/FSDP) require every rank to take the same branch; a rank-local
    verdict (like a validation-accuracy comparison) deadlocks the fleet when
    ranks disagree. Single-process accelerators pass the flag through.

    :param accelerator: the built ``accelerate.Accelerator``.
    :param flag: this rank's local decision.
    :return: rank 0's decision, uniform on every rank.
    """
    if getattr(accelerator, "num_processes", 1) <= 1:
        return flag
    import torch
    from accelerate.utils import broadcast
    verdict = torch.tensor([1 if flag else 0], device=accelerator.device)
    return bool(broadcast(verdict, from_process=0).item())


# Author: Mus mbayramo@stanford.edu
