import torch.nn as nn

from igc.modules.encoders.base_encoder import Conv1DLatent


class AutoStateEncoder(nn.Module):
    """
    """
    def __init__(self, input_shape=None, seq_len=1024, hidden_dim=768, latent_dim=64):
        """
        :param seq_len:
        :param hidden_dim:
        :param latent_dim:
        """
        super(AutoStateEncoder, self).__init__()

        self.conv1d = Conv1DLatent(input_shape)

        self.encoder = nn.Sequential(
            nn.Linear(1026, hidden_dim),
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
        :param x:
        :return:
        """
        x = self.conv1d(x)
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed


