"""Offline regression for the autoencoder dtype cast (bf16 backbone path).

A bf16 backbone emits bf16 hidden states while the autoencoder stays fp32, so
feeding them directly raised 'Input type (BFloat16) and bias type (float)
should be the same' — seen live on the GB300 with --llm_torch_dtype bfloat16.
_to_autoencoder_dtype casts to the autoencoder's parameter dtype. CPU-only.

Author:
Mus mbayramo@stanford.edu
"""

import torch

from igc.modules.igc_train_auto_state_encoder import AutoencoderTrainer


def _trainer_with_fp32_autoencoder():
    t = AutoencoderTrainer.__new__(AutoencoderTrainer)
    t.model_autoencoder = torch.nn.Linear(4, 4)  # fp32 by default
    return t


def test_bf16_hidden_states_cast_to_autoencoder_dtype():
    """A bf16 tensor is cast to the fp32 autoencoder dtype (no runtime mismatch)."""
    t = _trainer_with_fp32_autoencoder()
    out = t._to_autoencoder_dtype(torch.zeros(2, 4, dtype=torch.bfloat16))
    assert out.dtype == torch.float32
    # and it now feeds the autoencoder without a dtype error
    t.model_autoencoder(out)


def test_matching_dtype_is_untouched():
    """An fp32 tensor stays fp32 (no needless copy path errors)."""
    t = _trainer_with_fp32_autoencoder()
    out = t._to_autoencoder_dtype(torch.zeros(2, 4, dtype=torch.float32))
    assert out.dtype == torch.float32
