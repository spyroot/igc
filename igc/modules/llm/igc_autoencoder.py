import torch
import torch.nn as nn

from igc.modules.encoders.base_encoder import Conv1DLatent


class AutoStateEncoder(nn.Module):
    """Reconstruct backbone hidden states through a compact latent bottleneck.

    ``Conv1DLatent`` pools ``(batch, seq_len, hidden_dim)`` hidden states into a
    per-sequence feature vector, the encoder squeezes it to ``latent_dim``, and
    the decoder expands the latent back to the flattened ``seq_len * hidden_dim``
    hidden state. ``hidden_dim`` and ``seq_len`` must match the backbone that
    produced the states — the backbone hidden size and the dataset chunk length.
    They used to be hardcoded to GPT-2's 768 / 1024, so any other backbone (e.g.
    Qwen2.5-3B, hidden 2048) crashed with a reconstruction-shape mismatch.
    """

    def __init__(self, input_shape=None, seq_len=1024, hidden_dim=768, latent_dim=64):
        """
        :param input_shape: ``(positions, hidden)`` backbone shape driving Conv1DLatent.
        :param seq_len: sequence length of the hidden states to reconstruct — the
            dataset chunk length (``max_len``), not the GPT-2 default.
        :param hidden_dim: backbone hidden size ``H`` of the states to reconstruct.
        :param latent_dim: bottleneck width.
        """
        super(AutoStateEncoder, self).__init__()

        self.conv1d = Conv1DLatent(input_shape)

        # Conv1DLatent maps (batch, seq_len, H) -> (batch, conv_out). Measure that
        # width from a dummy forward instead of hardcoding GPT-2's 1026 (= 1024 + 2),
        # so a different seq_len or conv geometry stays consistent.
        with torch.no_grad():
            probe = torch.zeros(1, seq_len, input_shape[1])
            conv_out_dim = self.conv1d(probe).shape[-1]

        self.encoder = nn.Sequential(
            nn.Linear(conv_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, seq_len * hidden_dim),
        )

    def forward(self, x):
        """
        :param x: hidden states ``(batch, seq_len, hidden_dim)``.
        :return: flattened reconstruction ``(batch, seq_len * hidden_dim)``.
        """
        x = self.conv1d(x)
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed
