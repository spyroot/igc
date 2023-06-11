import torch.nn as nn


class AutoStateEncoder(nn.Module):
    """
    """
    def __init__(self, seq_len=1023, hidden_dim=768, latent_dim=256):
        """
        :param seq_len:
        :param hidden_dim:
        :param latent_dim:
        """
        super(AutoStateEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(seq_len, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x):
        """
        :param x:
        :return:
        """
        # flatten
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed


