from torch.utils.data import Sampler


class MaskedSampler(Sampler[int]):
    def __init__(self, data_source) -> None:
        super().__init__()
        self.data_source = data_source
        self.max_samples = 4

    def __iter__(self):
        return iter(self.data_source.sample_masked_iter())

    def __len__(self):
        return len(self.data_source._masked_data['train_data'])
