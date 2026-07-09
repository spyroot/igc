"""Offline tests for the GB300/Blackwell perf helpers in ``TorchBuilder``.

Covers TF32 enablement, the fused-AdamW kwarg selection, ``maybe_compile`` gating, and
that ``create_optimizer`` still builds a working AdamW on CPU (fused falls back cleanly).
All CPU/offline — the CUDA-only paths degrade to no-ops here.

Author:
Mus mbayramo@stanford.edu
"""
import torch
from torch import nn

from igc.shared.shared_torch_builder import TorchBuilder


def test_enable_perf_backends_sets_tf32_flags():
    """enable_perf_backends(True) turns on the TF32 matmul/cuDNN flags."""
    prev_matmul = torch.backends.cuda.matmul.allow_tf32
    prev_cudnn = torch.backends.cudnn.allow_tf32
    try:
        assert TorchBuilder.enable_perf_backends(True) is True
        assert torch.backends.cuda.matmul.allow_tf32 is True
        assert torch.backends.cudnn.allow_tf32 is True
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_matmul
        torch.backends.cudnn.allow_tf32 = prev_cudnn


def test_enable_perf_backends_disabled_is_noop():
    """enable_perf_backends(False) does nothing and returns False."""
    assert TorchBuilder.enable_perf_backends(False) is False


def test_fused_kwargs_only_for_adam_family_on_cuda():
    """Fused kwargs are empty off-CUDA and for non-Adam optimizers."""
    # This test runs on CPU, so cuda.is_available() is False -> always empty here.
    assert TorchBuilder._fused_optimizer_kwargs("adamw") == {}
    assert TorchBuilder._fused_optimizer_kwargs("sgd") == {}
    assert TorchBuilder._fused_optimizer_kwargs("rmsprop") == {}


def test_create_optimizer_builds_working_adamw_on_cpu():
    """create_optimizer returns a usable AdamW on CPU (fused path falls back to plain)."""
    model = nn.Linear(4, 2)
    opt = TorchBuilder.create_optimizer("AdamW", model, lr=1e-3, weight_decay=0.0)
    assert isinstance(opt, torch.optim.AdamW)
    # a step must not raise
    model(torch.zeros(1, 4)).sum().backward()
    opt.step()


def test_maybe_compile_is_noop_when_disabled_or_off_cuda():
    """maybe_compile returns the same model when disabled, and on CPU even when enabled."""
    model = nn.Linear(2, 2)
    assert TorchBuilder.maybe_compile(model, enabled=False) is model
    # enabled but no CUDA here -> unchanged
    assert TorchBuilder.maybe_compile(model, enabled=True) is model


# Author: Mus mbayramo@stanford.edu
