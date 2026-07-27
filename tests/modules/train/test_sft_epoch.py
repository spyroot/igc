"""Focused tests for epoch-aware dynamic SFT datasets."""

from types import SimpleNamespace

from igc.modules.train.sft import select_dataset_epoch, select_training_epoch


class _EpochDataset:
    def __init__(self) -> None:
        self.epochs = []

    def set_epoch(self, epoch: int) -> None:
        self.epochs.append(epoch)


class _EpochSurface:
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


def test_select_training_epoch_advances_prepared_loader() -> None:
    dataset = _EpochDataset()
    dataloader = _EpochSurface()

    select_training_epoch(dataset, dataloader, 5)

    assert dataset.epochs == [5]
    assert dataloader.epochs == [5]


def test_select_training_epoch_advances_plain_loader_sampler() -> None:
    dataset = _EpochDataset()
    sampler = _EpochSurface()
    dataloader = SimpleNamespace(sampler=sampler)

    select_training_epoch(dataset, dataloader, 6)

    assert dataset.epochs == [6]
    assert sampler.epochs == [6]
