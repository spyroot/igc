"""Regression: the autoencoder trainer checkpoints the model it trains.

``AutoencoderTrainer``'s ``self.model`` is the FROZEN backbone; the trained
network is ``self.model_autoencoder``. The base save/load hardwired
``self.model``, so a full run wrote the untouched backbone and downstream RL
loaded an untrained autoencoder. The base now accepts an optional target model;
these pin that save_model/save_checkpoint/load_checkpoint persist and restore
the passed model, not self.model. CPU-only.

Author:
Mus mbayramo@stanford.edu
"""

import argparse

import loguru
import torch

from igc.modules.base.igc_base_module import IgcModule


def _module(tmp_path):
    m = IgcModule.__new__(IgcModule)
    m.rank = 0
    m.module_name = "state_autoencoder"
    m.logger = loguru.logger
    m.model = torch.nn.Linear(4, 4)          # stand-in frozen backbone
    m.optimizer = None
    m.scheduler = None
    m._trainer_args = argparse.Namespace(seed=1, output_dir=str(tmp_path))
    m._module_checkpoint_dir = str(tmp_path)
    return m


def _distinct_autoencoder():
    ae = torch.nn.Linear(4, 4)
    with torch.no_grad():
        ae.weight.fill_(7.0)  # a value the backbone never has
    return ae


def test_save_model_persists_the_passed_model(tmp_path):
    """save_model(model=ae) writes the autoencoder weights, not the backbone."""
    m = _module(tmp_path)
    ae = _distinct_autoencoder()
    m.save_model(str(tmp_path), model=ae)
    saved = torch.load(m._model_file(str(tmp_path)), weights_only=True)
    assert torch.equal(saved["model_state_dict"]["weight"], ae.weight)


def test_save_and_load_checkpoint_round_trip_targets_model(tmp_path):
    """save_checkpoint(model=ae) then load_checkpoint(model=ae2) restores ae."""
    m = _module(tmp_path)
    ae = _distinct_autoencoder()
    m.optimizer = torch.optim.SGD(ae.parameters(), lr=0.1)
    m.save_checkpoint(str(tmp_path), epoch=1, model=ae, optimizer=m.optimizer)

    fresh = torch.nn.Linear(4, 4)
    m.load_checkpoint(str(tmp_path), model=fresh)
    assert torch.equal(fresh.weight, ae.weight)
    # the frozen backbone was never overwritten by the restore
    assert not torch.equal(m.model.weight, ae.weight)


def test_default_still_targets_self_model(tmp_path):
    """Without model=, behavior is unchanged (saves/loads self.model)."""
    m = _module(tmp_path)
    m.save_model(str(tmp_path))
    saved = torch.load(m._model_file(str(tmp_path)), weights_only=True)
    assert torch.equal(saved["model_state_dict"]["weight"], m.model.weight)


# Author: Mus mbayramo@stanford.edu
