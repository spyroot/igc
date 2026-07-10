"""Offline tests for the --sharding -> Accelerate config mapping (no accelerate/GPU).

sharding_config is pure, so the ZeRO/FSDP selection logic is verified on CPU without
deepspeed or a multi-GPU launch. Covers the DDP default, each ZeRO stage, offload, FSDP,
the bf16 default for sharded runs, and that DeepSpeed/FSDP disable device_placement.

Author:
Mus mbayramo@stanford.edu
"""
import argparse
import sys
import types

import pytest

from igc.shared.shared_accelerator import build_accelerator, sharding_config


def _ns(**kw):
    kw.setdefault("device_placement", True)
    kw.setdefault("mixed_precision", None)
    kw.setdefault("gradient_accumulation_steps", 1)
    return argparse.Namespace(**kw)


def _fake_accelerate(monkeypatch):
    """Install a tiny accelerate surface so builder tests stay CPU/offline."""

    class FakeAccelerator:
        calls = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.calls.append(kwargs)

    class FakeDeepSpeedPlugin:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeFSDPPlugin:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    accelerate = types.ModuleType("accelerate")
    accelerate_utils = types.ModuleType("accelerate.utils")
    accelerate.Accelerator = FakeAccelerator
    accelerate_utils.DeepSpeedPlugin = FakeDeepSpeedPlugin
    accelerate_utils.FullyShardedDataParallelPlugin = FakeFSDPPlugin
    monkeypatch.setitem(sys.modules, "accelerate", accelerate)
    monkeypatch.setitem(sys.modules, "accelerate.utils", accelerate_utils)
    return FakeAccelerator, FakeDeepSpeedPlugin, FakeFSDPPlugin


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


def test_build_accelerator_unsharded_forwards_safe_runtime_args(monkeypatch):
    """Unsharded build forwards device placement, precision, and grad accumulation."""
    fake_accelerator, _, _ = _fake_accelerate(monkeypatch)

    accelerator = build_accelerator(
        _ns(
            sharding="none",
            device_placement=False,
            mixed_precision="fp16",
            gradient_accumulation_steps=4,
        )
    )

    assert isinstance(accelerator, fake_accelerator)
    assert fake_accelerator.calls == [
        {
            "device_placement": False,
            "mixed_precision": "fp16",
            "gradient_accumulation_steps": 4,
        }
    ]


def test_build_accelerator_zero3_offload_uses_deepspeed_plugin(monkeypatch):
    """zero3_offload builds a CPU-offload DeepSpeed plugin and lets it place models."""
    fake_accelerator, fake_deepspeed, _ = _fake_accelerate(monkeypatch)

    build_accelerator(_ns(sharding="zero3_offload", gradient_accumulation_steps=3))

    kwargs = fake_accelerator.calls[-1]
    plugin = kwargs["deepspeed_plugin"]
    assert isinstance(plugin, fake_deepspeed)
    assert "device_placement" not in kwargs
    assert kwargs["mixed_precision"] == "bf16"
    assert kwargs["gradient_accumulation_steps"] == 3
    assert plugin.kwargs == {
        "zero_stage": 3,
        "offload_optimizer_device": "cpu",
        "offload_param_device": "cpu",
        "gradient_accumulation_steps": 3,
    }


def test_build_accelerator_fsdp_uses_fsdp_plugin(monkeypatch):
    """fsdp builds the FSDP plugin without constructing a DeepSpeed plugin."""
    fake_accelerator, _, fake_fsdp = _fake_accelerate(monkeypatch)

    build_accelerator(_ns(sharding="fsdp", mixed_precision="fp8"))

    kwargs = fake_accelerator.calls[-1]
    assert isinstance(kwargs["fsdp_plugin"], fake_fsdp)
    assert "deepspeed_plugin" not in kwargs
    assert "device_placement" not in kwargs
    assert kwargs["mixed_precision"] == "fp8"


# Author: Mus mbayramo@stanford.edu


def test_fsdp_plugin_is_fsdp2_configured(monkeypatch):
    """fsdp builds an FSDP2 plugin: versioned, auto-wrapped, resharding, sharded saves."""
    fake_accelerator, _, fake_fsdp = _fake_accelerate(monkeypatch)
    build_accelerator(_ns(sharding="fsdp"))
    kwargs = fake_accelerator.calls[-1]
    plugin = kwargs["fsdp_plugin"]
    assert isinstance(plugin, fake_fsdp)
    assert plugin.kwargs["fsdp_version"] == 2
    assert plugin.kwargs["auto_wrap_policy"] == "transformer_based_wrap"
    assert plugin.kwargs["reshard_after_forward"] is True
    assert plugin.kwargs["state_dict_type"] == "SHARDED_STATE_DICT"


def test_zero_without_deepspeed_raises_before_accelerator(monkeypatch):
    """zero3 without deepspeed installed fails with a clear message, not an env leak."""
    import sys
    import types

    accelerate = types.ModuleType("accelerate")
    accelerate_utils = types.ModuleType("accelerate.utils")

    class _NeverAccelerator:
        def __init__(self, **kwargs):
            raise AssertionError("Accelerator must not be constructed")

    accelerate.Accelerator = _NeverAccelerator
    accelerate_utils.is_deepspeed_available = lambda: False
    monkeypatch.setitem(sys.modules, "accelerate", accelerate)
    monkeypatch.setitem(sys.modules, "accelerate.utils", accelerate_utils)

    with pytest.raises(ValueError, match="--sharding fsdp"):
        build_accelerator(_ns(sharding="zero3"))


def test_sharded_single_process_launch_fails_loudly(monkeypatch):
    """A sharded config resolving to distributed_type NO raises instead of training unsharded."""
    import sys
    import types

    class _NoDistAccelerator:
        calls = []

        def __init__(self, **kwargs):
            self.distributed_type = "DistributedType.NO"
            type(self).calls.append(kwargs)

    class _Plugin:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    accelerate = types.ModuleType("accelerate")
    accelerate_utils = types.ModuleType("accelerate.utils")
    accelerate.Accelerator = _NoDistAccelerator
    accelerate_utils.FullyShardedDataParallelPlugin = _Plugin
    monkeypatch.setitem(sys.modules, "accelerate", accelerate)
    monkeypatch.setitem(sys.modules, "accelerate.utils", accelerate_utils)

    with pytest.raises(RuntimeError, match="single-process"):
        build_accelerator(_ns(sharding="fsdp"))
