"""Offline regression: AutoStateEncoder adapts to non-GPT-2 backbones.

The decoder's final Linear and the encoder's first Linear were hardcoded to
GPT-2's hidden 768 / seq_len 1024 (the ``1026 = 1024 + 2`` conv-output magic), so
a modern backbone like Qwen2.5-3B (hidden 2048) crashed with a
'2097152 must match 786432' reconstruction-shape mismatch. AutoStateEncoder now
takes ``hidden_dim`` / ``seq_len`` and derives the encoder input from
Conv1DLatent's real output width. CPU-only.

Author:
Mus mbayramo@stanford.edu
"""

import torch

from igc.modules.llm.igc_autoencoder import AutoStateEncoder


def _make(input_shape, seq_len, hidden_dim):
    """Build an AutoStateEncoder and a zero hidden-state batch for it."""
    ae = AutoStateEncoder(input_shape=input_shape, seq_len=seq_len, hidden_dim=hidden_dim)
    x = torch.zeros(2, seq_len, hidden_dim)  # (batch, seq_len, hidden_dim)
    return ae, x


def test_reconstruct_matches_flattened_input_hidden_2048():
    """A 2048-hidden backbone reconstructs to the flattened input shape (the crash case)."""
    seq_len, hidden_dim = 1024, 2048
    ae, x = _make((seq_len, hidden_dim), seq_len, hidden_dim)
    recon = ae(x)
    # decoder reconstructs the flattened hidden state (batch, seq_len * hidden_dim)
    assert recon.shape == (x.shape[0], seq_len * hidden_dim)
    # the trainer's mse target lines up element-for-element (no broadcast mismatch)
    flat = x.view(x.shape[0], -1)
    assert flat.shape == recon.shape
    torch.nn.functional.mse_loss(flat, recon)  # would raise on a shape mismatch


def test_reconstruct_gpt2_dims_unchanged():
    """The legacy GPT-2 geometry (768 / 1024) still round-trips (no regression)."""
    seq_len, hidden_dim = 1024, 768
    ae, x = _make((seq_len, hidden_dim), seq_len, hidden_dim)
    recon = ae(x)
    assert recon.shape == (x.shape[0], seq_len * hidden_dim)


def test_encoder_input_tracks_seq_len_not_hardcoded_1026():
    """Encoder first Linear derives from Conv1DLatent output, not the 1026 magic."""
    # conv output width is seq_len + 2, so a non-1024 seq_len must NOT stay 1026.
    seq_len, hidden_dim = 512, 2048
    ae, x = _make((seq_len, hidden_dim), seq_len, hidden_dim)
    assert ae.encoder[0].in_features == seq_len + 2  # 514, proving 1026 is gone
    recon = ae(x)
    assert recon.shape == (x.shape[0], seq_len * hidden_dim)


def test_default_dims_are_gpt2():
    """Unspecified hidden_dim/seq_len keep the GPT-2 defaults (backward compatible)."""
    ae = AutoStateEncoder(input_shape=(1024, 768))
    assert ae.decoder[-1].out_features == 1024 * 768
    assert ae.encoder[0].in_features == 1024 + 2


# Author: Mus mbayramo@stanford.edu
