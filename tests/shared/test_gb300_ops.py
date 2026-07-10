"""Offline tests for the GB300 ops helpers.

Covers the pure pieces of the operational path: the HF token mapping in
``shared_main`` (docs standardize on ``HUGGINGFACE_TOKEN`` but
``huggingface_hub`` reads ``HF_TOKEN``; values are never asserted into logs),
and the nccl_smoke bandwidth/size arithmetic (the collective loop itself is a
cluster-rung tool). CPU-only, no network.

Author:
Mus mbayramo@stanford.edu
"""

import importlib.util
import os
import pathlib
import sys

import pytest
import torch  # noqa: F401 — ensures the heavy import happens before shared_main

from igc.shared.shared_main import shared_main

_SCRIPTS = pathlib.Path(__file__).resolve().parents[2] / "scripts"


def _load_nccl_smoke():
    spec = importlib.util.spec_from_file_location("nccl_smoke", _SCRIPTS / "nccl_smoke.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_shared_main(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["igc_main.py", "--output_dir", str(tmp_path)])
    return shared_main(is_cuda_empty_cache=False)


def test_huggingface_token_mapped_to_hf_token(monkeypatch, tmp_path):
    """HUGGINGFACE_TOKEN is mirrored into HF_TOKEN for huggingface_hub."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "sentinel-token")
    _run_shared_main(monkeypatch, tmp_path)
    assert os.environ.get("HF_TOKEN") == "sentinel-token"


def test_existing_hf_token_not_clobbered(monkeypatch, tmp_path):
    """An explicitly-set HF_TOKEN wins over the legacy variable."""
    monkeypatch.setenv("HF_TOKEN", "explicit")
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "legacy")
    _run_shared_main(monkeypatch, tmp_path)
    assert os.environ["HF_TOKEN"] == "explicit"


def test_busbw_ring_accounting():
    """Bus bandwidth follows the ring all-reduce byte accounting."""
    smoke = _load_nccl_smoke()
    # 1e9 bytes moved in 1s -> 1 GB/s: pick numel so bytes==1e9 for world=2, iters=1
    # bytes = numel*4*1*2*(2-1)/2 = numel*4  -> numel = 2.5e8
    assert smoke.busbw_gbps(250_000_000, 1, 1.0, 2) == pytest.approx(1.0)


def test_busbw_degenerate_inputs_are_zero():
    """Zero time, one rank, or zero iters never divide by zero."""
    smoke = _load_nccl_smoke()
    assert smoke.busbw_gbps(10, 1, 0.0, 4) == 0.0
    assert smoke.busbw_gbps(10, 1, 1.0, 1) == 0.0
    assert smoke.busbw_gbps(10, 0, 1.0, 4) == 0.0


def test_tensor_megabytes():
    """Size helper reports fp32 MiB."""
    smoke = _load_nccl_smoke()
    assert smoke.tensor_megabytes(64 * 1024 * 1024) == pytest.approx(256.0)


# Author: Mus mbayramo@stanford.edu
