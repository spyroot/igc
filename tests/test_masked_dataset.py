"""Offline shape tests for masked Redfish dataset helpers."""

import torch

from igc.ds.redfish_masked_dataset import MaskedJSONDataset


class _TinyTokenizer:
    """Tokenizer double exposing only the fields used by new-token masking."""

    vocab_size = 5

    def __len__(self) -> int:
        return 8


def test_mask_tensor_ids_json_kv_span_masks_target_value_and_end_token() -> None:
    """A matched JSON key span keeps the input shape and masks through terminator."""
    input_ids = torch.tensor([[1, 10, 20, 30, 31, 99, 2]])
    attention_mask = torch.ones_like(input_ids)

    mask = MaskedJSONDataset.mask_tensor_ids_json_kv_span(
        input_ids,
        attention_mask,
        target_ids=[10, 20],
        end_toks_ids=[[99]],
    )

    assert mask.shape == attention_mask.shape
    assert torch.equal(mask, torch.tensor([[0, 1, 1, 1, 1, 1, 0]]))


def test_mask_tensor_ids_json_kv_span_returns_original_when_missing() -> None:
    """return_original preserves an all-ones mask when the key is absent."""
    input_ids = torch.tensor([[1, 2, 3, 4]])
    attention_mask = torch.ones_like(input_ids)

    mask = MaskedJSONDataset.mask_tensor_ids_json_kv_span(
        input_ids,
        attention_mask,
        target_ids=[10],
        end_toks_ids=[[99]],
        return_original=True,
    )

    assert mask.shape == attention_mask.shape
    assert torch.equal(mask, attention_mask)


def test_mask_tensor_ids_json_section_expands_one_dimensional_attention_mask() -> None:
    """Section masking normalizes a 1-D attention mask to one batch row."""
    input_ids = torch.tensor([[0, 1, 2, 1, 3, 9, 4, 9, 5]])
    attention_mask = torch.ones(input_ids.size(1), dtype=torch.long)

    mask = MaskedJSONDataset.mask_tensor_ids_json_section(
        input_ids,
        attention_mask,
        target_token=1,
        start_token=1,
        end_token=9,
    )

    assert mask.shape == input_ids.shape
    assert torch.equal(mask, torch.tensor([[0, 1, 1, 1, 1, 1, 1, 1, 0]]))


def test_mask_all_new_tokens_marks_only_added_token_ids() -> None:
    """New-token masking keeps shape and selects ids added past base vocab."""
    dataset = MaskedJSONDataset.__new__(MaskedJSONDataset)
    dataset.tokenizer = _TinyTokenizer()
    input_ids = torch.tensor([[0, 5, 6, 4, 7]])
    attention_mask = torch.ones(input_ids.size(1), dtype=torch.long)

    mask = dataset.mask_all_new_tokens(input_ids, attention_mask)

    assert mask.shape == input_ids.shape
    assert torch.equal(mask, torch.tensor([[0, 1, 1, 0, 1]]))
