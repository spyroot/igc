"""
Offline smoke tests for the multi-GPU training-step profiler (``scripts/gpu_train_profile.py``).

The profiler's value is on a GB300 node under ``accelerate launch``; these CPU tests just guard that
the section decomposition, the timer, and a single-process run stay wired — so a refactor cannot
silently break the harness before it reaches the node. No GPU, no HF download (config-built backbone,
``TRANSFORMERS_OFFLINE``).

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "gpu_train_profile",
    Path(__file__).resolve().parents[2] / "scripts" / "gpu_train_profile.py",
)
gtp = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(gtp)


def test_sections_and_dtype_mapping():
    """The step decomposes into the four named sections and precision maps to a torch dtype."""
    import torch
    assert gtp.SECTIONS == ("data", "forward", "backward", "optimizer")
    assert gtp._dtype("bf16") is torch.bfloat16
    assert gtp._dtype("fp32") is torch.float32


def test_timer_records_every_section_on_cpu():
    """The CPU timer path records one sample per section entered (no CUDA required)."""
    timers = gtp._Timers(use_cuda=False)
    for s in gtp.SECTIONS:
        with timers.time(s):
            _ = sum(range(1000))
    for s in gtp.SECTIONS:
        assert len(timers.samples[s]) == 1
        assert timers.samples[s][0] >= 0.0


def test_synthetic_batch_shapes():
    """The synthetic batch has the real (batch, seq_len) shape and matching labels."""
    import torch
    batch = gtp._synthetic_batch(3, 16, vocab_size=100, device=torch.device("cpu"))
    assert batch["input_ids"].shape == (3, 16)
    assert batch["attention_mask"].shape == (3, 16)
    assert torch.equal(batch["input_ids"], batch["labels"])
    assert batch["attention_mask"].sum().item() == 3 * 16


@pytest.mark.slow
def test_single_process_run_produces_summary(tmp_path):
    """A tiny single-process CPU run returns a summary with per-section stats and throughput."""
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    args = gtp.argparse.Namespace(
        model_type="gpt2", batch_size=2, seq_len=32, precision="fp32",
        steps=2, warmup=1, trace=False, output_dir=str(tmp_path),
    )
    summary = gtp.run(args)
    assert summary is not None
    assert summary["samples_per_sec"] > 0.0
    assert set(summary["sections"]) == set(gtp.SECTIONS)
    assert (tmp_path / "profile_summary.json").exists()


# Author: Mus mbayramo@stanford.edu
