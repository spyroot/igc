"""Offline tests for JSONDataset chunk shape behavior.

Author:
Mus mbayramo@stanford.edu
"""

from pathlib import Path

import torch

from igc.ds.redfish_dataset import JSONDataset


class TinyTokenizer:
    """Minimal tokenizer double used to keep chunk tests offline."""

    name_or_path = "tiny"
    eos_token = "<eos>"
    eos_token_id = 99
    pad_token = "<pad>"
    pad_token_id = 99


def make_dataset(tmp_path: Path, *, max_len: int = 10, overlap: int = 2) -> JSONDataset:
    """Create a JSONDataset that skips downloads and dataset construction."""
    return JSONDataset(
        dataset_dir=str(tmp_path / "dataset"),
        raw_json_directory_path=str(tmp_path / "raw-json"),
        tokenizer=TinyTokenizer(),
        max_len=max_len,
        overlap=overlap,
        skip_download=True,
        skip_creation=True,
    )


def test_create_chunks_returns_short_input_unchanged(tmp_path: Path) -> None:
    """Inputs at or below max_len remain a single unpadded chunk."""
    dataset = make_dataset(tmp_path)
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    attention_mask = torch.ones_like(input_ids)

    chunks = dataset.create_chunks(input_ids, attention_mask)

    assert len(chunks) == 1
    chunk_ids, chunk_mask = chunks[0]
    assert torch.equal(chunk_ids, input_ids)
    assert torch.equal(chunk_mask, attention_mask)


def test_create_chunks_pads_final_overlapping_chunk(tmp_path: Path) -> None:
    """The final partial window pads ids and masks without changing prior chunks."""
    dataset = make_dataset(tmp_path)
    input_ids = torch.arange(1, 16).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)

    chunks = dataset.create_chunks(input_ids, attention_mask)

    assert len(chunks) == 2
    assert torch.equal(chunks[0][0], torch.arange(1, 11).unsqueeze(0))
    assert torch.equal(chunks[0][1], torch.ones((1, 10), dtype=torch.int64))
    assert torch.equal(chunks[1][0], torch.tensor([[9, 10, 11, 12, 13, 14, 15, 99, 99, 99]]))
    assert torch.equal(chunks[1][1], torch.tensor([[1, 1, 1, 1, 1, 1, 1, 0, 0, 0]]))


def test_create_chunks_does_not_duplicate_tail_after_exact_stride_cover(
    tmp_path: Path,
) -> None:
    """A full final window should stop without emitting a redundant padded tail."""
    dataset = make_dataset(tmp_path, max_len=10, overlap=2)
    input_ids = torch.arange(1, 19).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)

    chunks = dataset.create_chunks(input_ids, attention_mask)

    assert len(chunks) == 2
    assert torch.equal(chunks[0][0], torch.arange(1, 11).unsqueeze(0))
    assert torch.equal(chunks[1][0], torch.arange(9, 19).unsqueeze(0))
    assert torch.equal(chunks[1][1], torch.ones((1, 10), dtype=torch.int64))


# Author: Mus mbayramo@stanford.edu
