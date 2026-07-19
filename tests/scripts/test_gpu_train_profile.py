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
import sys
import types
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "gpu_train_profile",
    Path(__file__).resolve().parents[2] / "scripts" / "gpu_train_profile.py",
)
gtp = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(gtp)
_REPO_ROOT = Path(__file__).resolve().parents[2]


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


def test_profile_sweep_uses_portable_bash_shebang():
    """The node/container sweep wrapper must not depend on macOS Homebrew bash."""
    script = _REPO_ROOT / "scripts" / "gpu_train_profile.sh"
    assert script.read_text(encoding="utf-8").splitlines()[0] == "#!/usr/bin/env bash"


def test_build_backbone_loads_config_from_local_cache(monkeypatch):
    """Direct profiler invocation must not try a Hugging Face network lookup."""
    calls = []

    class FakeConfig:
        n_positions = 8
        max_position_embeddings = 8

    class FakeAutoConfig:
        @staticmethod
        def from_pretrained(model_type, **kwargs):
            calls.append((model_type, kwargs))
            return FakeConfig()

        @staticmethod
        def for_model(_model_type):
            raise AssertionError("fallback should not be used for this path")

    class FakeAutoModelForCausalLM:
        @staticmethod
        def from_config(config):
            return ("model", config)

    fake_transformers = types.SimpleNamespace(
        AutoConfig=FakeAutoConfig,
        AutoModelForCausalLM=FakeAutoModelForCausalLM,
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    model, config = gtp.build_backbone("cached-model", seq_len=16)

    assert model == "model"
    assert calls == [("cached-model", {"local_files_only": True})]
    assert config.n_positions == 16
    assert config.max_position_embeddings == 16


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
