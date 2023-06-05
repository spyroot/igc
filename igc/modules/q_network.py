from torch import nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=768):
        super().__init__()
        print(f"QNetwork input {input_dim}")
        self.dense1 = nn.Linear(input_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, inputs):
        x = self.dense1(inputs)
        x = F.relu(x)
        output = self.out(x)
        return output
