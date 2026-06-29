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


def sharding_config(cmd: argparse.Namespace) -> dict:
    """Pure mapping of CLI flags to a sharding/precision config (no accelerate import).

    :param cmd: parsed args; reads ``sharding``, ``mixed_precision``,
        ``gradient_accumulation_steps``, ``device_placement``.
    :return: a config dict consumed by :func:`build_accelerator`. ``sharded`` is False for
        ``none``/``ddp`` (plain replication) and True for any ZeRO/FSDP mode; a sharded run
        defaults to ``bf16`` and disables Accelerate ``device_placement`` (DeepSpeed/FSDP
        place the model themselves).
    """
    sharding = getattr(cmd, "sharding", "none") or "none"
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
    if cfg["sharding"] == "fsdp":
        from accelerate.utils import FullyShardedDataParallelPlugin
        return {"fsdp_plugin": FullyShardedDataParallelPlugin()}
    from accelerate.utils import DeepSpeedPlugin
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
    return Accelerator(**accelerator_args)


# Author: Mus mbayramo@stanford.edu
