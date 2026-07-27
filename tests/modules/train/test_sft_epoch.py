"""Focused tests for epoch-aware dynamic SFT datasets."""

from igc.modules.train.sft import select_dataset_epoch


class _EpochDataset:
    def __init__(self) -> None:
        self.epochs = []

    def set_epoch(self, epoch: int) -> None:
        self.epochs.append(epoch)


def test_select_dataset_epoch_calls_optional_dataset_hook() -> None:
    dataset = _EpochDataset()

    select_dataset_epoch(dataset, 4)

    assert dataset.epochs == [4]


def test_select_dataset_epoch_accepts_static_dataset() -> None:
    select_dataset_epoch(object(), 4)
