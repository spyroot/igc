"""Offline tests for the config-driven backbone loader (igc.modules.shared.llm_shared).

Mocks AutoModelForCausalLM/AutoTokenizer so it verifies the dispatch (local path or HF id,
trust_remote_code, dtype, pad-token handling) with no download — the change that lets
--model_type=/home/nvidia/models/DeepSeek-V4-Flash load the raw weights off the node.

Author:
Mus mbayramo@stanford.edu
"""
import argparse
import types

import pytest

from igc.modules.shared import llm_shared
from igc.modules.shared.llm_shared import _model_id, _resolve_dtype, _spec_flag


class _FakeModel:
    def __init__(self):
        self.config = types.SimpleNamespace(pad_token_id=None, pad_token=None)


class _FakeTokenizer:
    def __init__(self, pad=None):
        self.pad_token = pad
        self.pad_token_id = 0 if pad is not None else None
        self.eos_token = "<eos>"
        self.eos_token_id = 2


def _install_fakes(monkeypatch):
    captured = {}

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(model_id, **kw):
            captured["model"] = (model_id, kw)
            return _FakeModel()

    class FakeAutoTok:
        @staticmethod
        def from_pretrained(model_id, **kw):
            captured["tok"] = (model_id, kw)
            return _FakeTokenizer(pad=None)

    monkeypatch.setattr(llm_shared, "AutoModelForCausalLM", FakeAutoModel)
    monkeypatch.setattr(llm_shared, "AutoTokenizer", FakeAutoTok)
    return captured


def test_loads_local_path_with_trust_and_dtype(monkeypatch):
    """A local weights dir + trust_remote_code + bf16 routes through to from_pretrained as-is."""
    import torch

    captured = _install_fakes(monkeypatch)
    spec = argparse.Namespace(
        model_type="/home/nvidia/models/DeepSeek-V4-Flash",
        trust_remote_code=True,
        llm_torch_dtype="bfloat16",
    )
    model, tokenizer = llm_shared.from_pretrained_default(spec)
    model_id, kw = captured["model"]
    assert model_id == "/home/nvidia/models/DeepSeek-V4-Flash"
    assert kw["trust_remote_code"] is True
    assert kw["dtype"] is torch.bfloat16  # transformers 5.x kwarg (torch_dtype deprecated)
    assert captured["tok"][1]["trust_remote_code"] is True
    # pad token gets filled from eos when the tokenizer lacks one
    assert tokenizer.pad_token == "<eos>"
    assert model.config.pad_token_id == tokenizer.pad_token_id


def test_bare_string_carries_no_flags(monkeypatch):
    """A bare string model id implies no trust_remote_code and no dtype kwarg."""
    captured = _install_fakes(monkeypatch)
    llm_shared.from_pretrained_default("gpt2")
    model_id, kw = captured["model"]
    assert model_id == "gpt2"
    assert kw["trust_remote_code"] is False
    assert "dtype" not in kw and "torch_dtype" not in kw  # None dtype is not forwarded


def test_only_tokenizer_skips_model(monkeypatch):
    """only_tokenizer=True loads the tokenizer and not the model."""
    captured = _install_fakes(monkeypatch)
    model, tokenizer = llm_shared.from_pretrained_default("gpt2", only_tokenizer=True)
    assert model is None and tokenizer is not None
    assert "model" not in captured


def test_only_model_skips_tokenizer_and_padding(monkeypatch):
    """only_model=True loads no tokenizer and leaves pad config untouched."""
    captured = _install_fakes(monkeypatch)
    model, tokenizer = llm_shared.from_pretrained_default("gpt2", only_model=True)
    assert model is not None and tokenizer is None
    assert "tok" not in captured
    assert model.config.pad_token_id is None
    assert model.config.pad_token is None


def test_load_pretrained_default_forwards_loader_flags(monkeypatch):
    """Fine-tuned load forwards cache/trust/device-map/loading flags to Auto classes."""
    captured = _install_fakes(monkeypatch)
    spec = argparse.Namespace(
        llm_cache_dir="/tmp/igc-hf-cache",
        trust_remote_code=True,
        llm_ignore_mismatched_sizes=True,
        llm_output_loading_info=True,
        llm_fast_init=False,
        llm_torch_dtype="auto",
    )
    model, tokenizer = llm_shared.load_pretrained_default(
        spec,
        "/tmp/igc-model",
        device_map={"": "cpu"},
    )
    assert model is not None and tokenizer is not None
    assert captured["tok"] == (
        "/tmp/igc-model",
        {"cache_dir": "/tmp/igc-hf-cache", "trust_remote_code": True},
    )
    assert captured["model"] == (
        "/tmp/igc-model",
        {
            "cache_dir": "/tmp/igc-hf-cache",
            "trust_remote_code": True,
            "ignore_mismatched_sizes": True,
            "output_loading_info": True,
            # _fast_init is no longer forwarded: transformers 5.x ignores it.
            "dtype": "auto",
            "device_map": {"": "cpu"},
        },
    )


def test_helpers():
    """_model_id / _spec_flag / _resolve_dtype behave for str and Namespace inputs."""
    import torch

    assert _model_id("x") == "x"
    assert _model_id(argparse.Namespace(model_type="y")) == "y"
    assert _spec_flag("x", "trust_remote_code", False) is False  # str carries no flags
    assert _spec_flag(argparse.Namespace(trust_remote_code=True), "trust_remote_code", False) is True
    assert _resolve_dtype(None) is None
    assert _resolve_dtype("auto") == "auto"
    assert _resolve_dtype("bfloat16") is torch.bfloat16


if __name__ == "__main__":
    pytest.main([__file__, "-q"])


# Author: Mus mbayramo@stanford.edu
