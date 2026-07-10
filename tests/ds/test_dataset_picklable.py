"""Offline regressions for JSONDataset picklability (DataLoader workers).

``pickle.dumps(dataset)`` previously failed with ``cannot pickle
'_io.TextIOWrapper'`` because ``self.logger`` (loguru stdout + file sinks) and
the build-time ``RestTrajectory`` live in ``__dict__`` — breaking every
``num_workers > 0`` DataLoader under the spawn start method (macOS default;
the GB300 profile sets ``num_workers=8``). ``__getstate__`` drops both and
``__setstate__`` rebuilds a process-local logger. Tested on a ``__new__``
instance with a real open file handle standing in for the sink. CPU-only.

Author:
Mus mbayramo@stanford.edu
"""

import pickle

import torch

from igc.ds.redfish_dataset import JSONDataset


def _dataset_like(tmp_path):
    """A minimal JSONDataset (bypasses __init__) with worker-relevant state."""
    ds = JSONDataset.__new__(JSONDataset)
    ds.logger = open(tmp_path / "sink.log", "w")  # unpicklable stand-in
    ds._rest_trajectories = open(tmp_path / "traj.log", "w")
    ds._data = {"train_data": [torch.tensor([1]), torch.tensor([2])]}
    ds._masked_data = None
    return ds


def test_pickle_round_trip_drops_handles_and_keeps_data(tmp_path):
    """dumps/loads succeeds; data survives; handles are gone."""
    ds = _dataset_like(tmp_path)
    clone = pickle.loads(pickle.dumps(ds))

    assert len(clone) == 2
    assert torch.equal(clone[0], torch.tensor([1]))
    assert clone._rest_trajectories is None


def test_unpickled_instance_has_working_logger(tmp_path):
    """__setstate__ rebuilds a process-local logger (not None, callable)."""
    clone = pickle.loads(pickle.dumps(_dataset_like(tmp_path)))
    assert clone.logger is not None
    clone.logger.info("worker logger alive")  # must not raise


def test_original_instance_unchanged_by_getstate(tmp_path):
    """__getstate__ copies state; the live instance keeps its logger."""
    ds = _dataset_like(tmp_path)
    pickle.dumps(ds)
    assert ds.logger is not None
    assert hasattr(ds.logger, "write")  # still the original handle


# Author: Mus mbayramo@stanford.edu
