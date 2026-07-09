"""Offline tests for the --sharding -> Accelerate config mapping (no accelerate/GPU).

sharding_config is pure, so the ZeRO/FSDP selection logic is verified on CPU without
deepspeed or a multi-GPU launch. Covers the DDP default, each ZeRO stage, offload, FSDP,
the bf16 default for sharded runs, and that DeepSpeed/FSDP disable device_placement.

Author:
Mus mbayramo@stanford.edu
"""
import argparse

import pytest

from igc.shared.shared_accelerator import sharding_config


def _ns(**kw):
    kw.setdefault("device_placement", True)
    kw.setdefault("mixed_precision", None)
    kw.setdefault("gradient_accumulation_steps", 1)
    return argparse.Namespace(**kw)


def test_none_is_unsharded_ddp():
    """Default 'none' replicates (not sharded), keeps device_placement, no precision forced."""
    c = sharding_config(_ns(sharding="none"))
    assert c["sharded"] is False
    assert c["zero_stage"] is None and c["offload"] is False
    assert c["device_placement"] is True
    assert c["mixed_precision"] is None


def test_zero3_shards_defaults_bf16_and_drops_device_placement():
    """zero3 -> sharded, stage 3, bf16 by default, device_placement off (DeepSpeed places)."""
    c = sharding_config(_ns(sharding="zero3"))
    assert c["sharded"] is True and c["zero_stage"] == 3 and c["offload"] is False
    assert c["mixed_precision"] == "bf16"
    assert c["device_placement"] is False


def test_zero2_and_offload_variants():
    """zero2 -> stage 2; zero3_offload -> stage 3 with CPU offload."""
    assert sharding_config(_ns(sharding="zero2"))["zero_stage"] == 2
    off = sharding_config(_ns(sharding="zero3_offload"))
    assert off["zero_stage"] == 3 and off["offload"] is True


def test_fsdp_is_sharded_without_zero_stage():
    """fsdp -> sharded but no DeepSpeed zero stage."""
    c = sharding_config(_ns(sharding="fsdp"))
    assert c["sharded"] is True and c["zero_stage"] is None


def test_explicit_precision_and_grad_accum_preserved():
    """An explicit --mixed_precision (e.g. fp8) and grad-accum pass through unchanged."""
    c = sharding_config(_ns(sharding="zero3", mixed_precision="fp8", gradient_accumulation_steps=8))
    assert c["mixed_precision"] == "fp8"
    assert c["gradient_accumulation_steps"] == 8


def test_unsharded_run_keeps_user_precision():
    """A precision set on an unsharded run is honored (not overridden to bf16)."""
    c = sharding_config(_ns(sharding="none", mixed_precision="fp16"))
    assert c["sharded"] is False and c["mixed_precision"] == "fp16"


def test_unknown_sharding_mode_raises():
    """An unsupported mode (bypassing the CLI ``choices`` guard) fails fast, not silently.

    Without the guard ``zero4`` would map to ``sharded=True, zero_stage=None`` and build a
    broken DeepSpeed plugin at runtime; the guard rejects it with the offending value named.
    """
    with pytest.raises(ValueError, match="zero4"):
        sharding_config(_ns(sharding="zero4"))


def test_missing_sharding_attr_defaults_to_none():
    """A Namespace without ``sharding`` at all defaults to the valid 'none' (no raise)."""
    c = sharding_config(argparse.Namespace())
    assert c["sharded"] is False and c["sharding"] == "none"


# Author: Mus mbayramo@stanford.edu
