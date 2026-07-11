"""Offline regression tests for JSONDataset tokenizer selection.

Author:
Mus mbayramo@stanford.edu
"""
from pathlib import Path

import pytest

from igc.ds import redfish_dataset
from igc.ds.redfish_dataset import JSONDataset


class FakeTokenizer:
    """Tiny tokenizer double with the methods JSONDataset uses at construction."""

    eos_token = "<eos>"
    eos_token_id = 2

    def __init__(self, name_or_path: str, *, pad_token=None, pad_token_id=None):
        self.name_or_path = name_or_path
        self.pad_token = pad_token
        self.pad_token_id = pad_token_id
        self.added_tokens = []
        self.special_token_maps = []
        self.vocab_size = 100

    def add_tokens(self, tokens, special_tokens: bool = False):
        """Record added tokens and return how many would have been added."""
        self.added_tokens.append((tuple(tokens), special_tokens))
        return len(tokens)

    def __len__(self):
        """Return a stable fake vocabulary size."""
        return self.vocab_size

    def add_special_tokens(self, special_tokens_dict):
        """Record special-token maps and return the number of mapped tokens."""
        self.special_token_maps.append(special_tokens_dict)
        return len(special_tokens_dict.get("additional_special_tokens", ()))

    def save_pretrained(self, tokenizer_dir: str):
        """Create the tokenizer directory, matching the HuggingFace save contract needed here."""
        Path(tokenizer_dir).mkdir(parents=True, exist_ok=True)
        return (tokenizer_dir,)


def test_default_tokenize_builds_backbone_tokenizer_without_gpt2_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-GPT2 default_tokenize path drives AutoTokenizer and is not replaced by GPT2."""
    model_id = "vendor/non-gpt2-backbone"
    auto_calls = []

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(name_or_path):
            auto_calls.append(str(name_or_path))
            pad_token = "<pad>" if Path(str(name_or_path)).name == "tokenizer" else None
            pad_token_id = 0 if pad_token is not None else None
            return FakeTokenizer(
                str(name_or_path),
                pad_token=pad_token,
                pad_token_id=pad_token_id,
            )

    monkeypatch.setattr(redfish_dataset, "AutoTokenizer", FakeAutoTokenizer)
    monkeypatch.setattr(JSONDataset, "_build_tokenizer", lambda self: None)

    dataset = JSONDataset(
        dataset_dir=str(tmp_path / "dataset"),
        raw_json_directory_path=str(tmp_path / "raw-json"),
        default_tokenize=model_id,
        skip_download=True,
        skip_creation=True,
    )

    assert auto_calls[0] == model_id
    assert auto_calls[1] == dataset.tokenizer_dir()
    assert dataset._default_tokenize_name == model_id
    assert dataset.tokenizer.name_or_path == dataset.tokenizer_dir()
    assert dataset.tokenizer.pad_token == "<pad>"


def test_injected_tokenizer_skips_backbone_factory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit tokenizer remains authoritative and avoids Auto/GPT2 factories."""
    tokenizer = FakeTokenizer("provided-tokenizer", pad_token="<pad>", pad_token_id=0)

    class FailingTokenizerFactory:
        @staticmethod
        def from_pretrained(name_or_path):
            raise AssertionError(f"unexpected tokenizer factory call for {name_or_path}")

    monkeypatch.setattr(redfish_dataset, "AutoTokenizer", FailingTokenizerFactory)

    dataset = JSONDataset(
        dataset_dir=str(tmp_path / "dataset"),
        raw_json_directory_path=str(tmp_path / "raw-json"),
        default_tokenize="vendor/non-gpt2-backbone",
        tokenizer=tokenizer,
        skip_download=True,
        skip_creation=True,
    )

    assert dataset.tokenizer is tokenizer
    assert dataset._dataset_file_name.endswith("processed_dataset_provided-tokenizer.pt")


# Author: Mus mbayramo@stanford.edu
