"""
Just random data for testing
Mus mbayramo@stanford.edu
"""


import torch
from torch.utils.data import Dataset


class RandomDataset(Dataset):
    """Random dataset for basic testing.
    """
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len
