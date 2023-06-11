import torch
from torch import nn
import torch.nn.functional as F


class Igc_QNetwork(nn.Module):
    def __init__(self, input_dim, num_actions, hidden_dim=128):
        super().__init__()
        print(f"QNetwork input_dim {input_dim} output_dim {num_actions}")
        self.dense1 = nn.Linear(input_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, num_actions)

    def forward(self, inputs: torch.Tensor):
        x = self.dense1(inputs)
        x = F.relu(x)
        # x = self.dense2(x)
        # print("x shape after dense 2", x.shape)
        # x = F.relu(x)
        # print("x shape x", x.shape)
        # x = x.squeeze()
        output = self.out(x)
        return output


