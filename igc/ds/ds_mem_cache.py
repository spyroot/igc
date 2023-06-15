"""
Mem cache wrapper around any dataset.
Mus mbayramo@stanford.edu
"""
import torch
from torch.utils.data.dataset import random_split


class IgcMemoryCache(torch.utils.data.Dataset):
    """
    """
    def __init__(self, dataset):
        """

        :param dataset:
        """
        self.dataset = dataset
        self._cache = [None] * len(dataset)

    def __getitem__(self, n):
        if self._cache[n] is not None:
            return self._cache[n]

        item = self.dataset[n]
        self._cache[n] = item

        return item

    def __len__(self):
        return len(self.dataset)
