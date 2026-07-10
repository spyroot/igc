"""Edge coverage for dataset tokenizer provenance checks."""

import pytest

from igc.ds.redfish_dataset import JSONDataset


def test_string_token_counts_are_compared_as_integers():
    """String token counts from JSON metadata compare equal to live counts."""
    JSONDataset.check_tokenizer_provenance(
        {"num_tokens": "53147", "model_type": "gpt2"},
        "53147",
        "gpt2",
    )


def test_none_provenance_fields_are_skipped():
    """Explicit null tokenizer fields behave like absent legacy fields."""
    JSONDataset.check_tokenizer_provenance(
        {"num_tokens": None, "model_type": None},
        53147,
        "gpt2",
    )


def test_unknown_live_model_type_skips_model_comparison():
    """A missing live model id still allows token-count-only validation."""
    JSONDataset.check_tokenizer_provenance(
        {"num_tokens": 53147, "model_type": "gpt2"},
        53147,
        None,
    )


def test_empty_recorded_model_type_is_an_explicit_mismatch():
    """An empty recorded model id is not treated as absent provenance."""
    with pytest.raises(ValueError) as excinfo:
        JSONDataset.check_tokenizer_provenance(
            {"num_tokens": 53147, "model_type": ""},
            53147,
            "gpt2",
        )

    message = str(excinfo.value)
    assert "--model_type ''" in message
    assert "current run uses 'gpt2'" in message
    assert "--recreate_dataset" in message


def test_smaller_live_vocab_error_pins_counts_and_rebuild_hint():
    """A cache that can contain out-of-vocab ids reports both vocab sizes."""
    with pytest.raises(ValueError) as excinfo:
        JSONDataset.check_tokenizer_provenance(
            {"num_tokens": "151936"},
            "53147",
            "Qwen/Qwen2.5-7B-Instruct",
        )

    message = str(excinfo.value)
    assert "151936-token tokenizer" in message
    assert "only 53147 tokens" in message
    assert "--recreate_dataset" in message


def test_grown_live_vocab_warning_pins_growth_delta():
    """A grown tokenizer warns with the exact positive delta."""
    with pytest.warns(UserWarning) as warnings:
        JSONDataset.check_tokenizer_provenance({"num_tokens": "53140"}, 53147)

    assert len(warnings) == 1
    message = str(warnings[0].message)
    assert "53140-token tokenizer" in message
    assert "the loaded one has 53147" in message
    assert "grew by 7" in message
    assert "--recreate_dataset" in message
