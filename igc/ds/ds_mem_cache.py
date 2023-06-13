import torch
from torch.utils.data.dataset import random_split


class IgcMemoryCache(torch.utils.data.Dataset):
    """THis wraps around a dataset so that, whenever a new is returned,
    it is saved to memory cache, meanwhile it immediately IGC dataset
    hence all method caller can call.

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
