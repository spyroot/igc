"""Offline regression: the JSONDataset saved-tokenizer loader is backbone-agnostic.

``JSONDataset.load_tokenizer`` (in ``igc/ds/redfish_dataset.py``) loads a *saved*
tokenizer directory on the resume/inference path — it is called from
``igc/modules/llm_train_state_encoder.py`` (``_load`` reload) and ``igc/ds/corpus_dataset.py``.
That directory is written by ``_load_tokenizer`` from the current ``--model_type``, so it may
hold a Qwen/Llama tokenizer, not GPT-2's. The loader must therefore go through
``AutoTokenizer`` — loading a non-GPT-2 saved tokenizer with the GPT-2-only class silently
returns the wrong tokenizer. These tests pin that on CPU with no network or download.

Author:
Mus mbayramo@stanford.edu
"""
import pytest

import igc.ds.redfish_dataset as rd
from igc.ds.redfish_dataset import JSONDataset


class _FakeTok:
    """Minimal stand-in for a tokenizer returned by ``from_pretrained``."""

    def __init__(self, name: str):
        self.name_or_path = name
        self.eos_token = "<eos>"
        self.eos_token_id = 0
        self.pad_token = None
        self.pad_token_id = None


def test_load_tokenizer_uses_autotokenizer(tmp_path, monkeypatch):
    """load_tokenizer loads the saved dir via AutoTokenizer, not a GPT-2-locked class."""
    tok_dir = tmp_path / "tokenizer"
    tok_dir.mkdir()

    seen = {}

    def fake_from_pretrained(path):
        seen["path"] = str(path)
        return _FakeTok(str(path))

    monkeypatch.setattr(rd.AutoTokenizer, "from_pretrained", fake_from_pretrained)

    tok = JSONDataset.load_tokenizer(str(tok_dir))

    assert seen["path"] == str(tok_dir)      # the given directory was loaded
    assert tok.pad_token == tok.eos_token    # pad token synthesized from eos
    assert tok.pad_token_id == tok.eos_token_id
    # Regression guard: the GPT-2-only tokenizer class must no longer be imported here.
    assert not hasattr(rd, "GPT2Tokenizer")


def test_load_tokenizer_missing_dir_raises(tmp_path):
    """A missing tokenizer directory raises a clear ValueError (contract unchanged)."""
    with pytest.raises(ValueError):
        JSONDataset.load_tokenizer(str(tmp_path / "does_not_exist"))


# Author: Mus mbayramo@stanford.edu
