import torch
from torch import nn
import torch.nn.functional as F


class Igc_QNetwork(nn.Module):
    def __init__(self, input_dim, num_actions, hidden_dim=128):
        """

        :param input_dim:
        :param num_actions:
        :param hidden_dim:
        """
        super().__init__()
        self.dense1 = nn.Linear(input_dim, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, hidden_dim)
        self.dense3 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, num_actions)

    def forward(self, inputs: torch.Tensor):
        """

        :param inputs:
        :return:
        """
        x = self.dense1(inputs)
        x = F.relu(x)
        x = self.dense2(x)
        x = F.relu(x)
        x = self.dense3(x)
        x = F.relu(x)
        output = self.out(x)
        return output
