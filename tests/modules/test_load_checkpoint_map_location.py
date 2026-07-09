"""Offline regression for device mapping in ``IgcModule.load_checkpoint``.

``load_checkpoint`` computes ``map_to`` (the caller's ``map_location`` or the
``{'cuda:1': 'cuda:0'}`` default) but historically dropped it on the floor: the
underlying ``torch.load`` was called without ``map_location``, so a checkpoint
saved on one device could fail or mis-map when resumed on another (e.g. cuda->cpu).

These tests pin two things on CPU with no GPU required:
  * ``map_location`` is actually forwarded to ``torch.load`` (spy), including the
    ``{'cuda:1': 'cuda:0'}`` remap when the caller passes nothing; and
  * an end-to-end ``map_location='cpu'`` load restores weights onto CPU.

Author:
Mus mbayramo@stanford.edu
"""
import logging

import torch
import torch.nn as nn

from igc.modules.base.igc_base_module import IgcModule


def _bypass_module(model, optimizer):
    """A minimally-populated IgcModule that bypasses the heavy __init__.

    Only the attributes ``load_checkpoint`` touches on the resume path are set.
    """
    module = IgcModule.__new__(IgcModule)
    module._module_checkpoint_dir = "present"  # non-None so we don't early-return
    module.logger = logging.getLogger("test_load_checkpoint_map_location")
    module.model = model
    module.optimizer = optimizer
    module.scheduler = None
    module.module_name = "test-module"
    module.rank = 0
    return module


def _write_checkpoint(path, model, optimizer, epoch=3):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
        },
        path,
    )


def test_load_checkpoint_loads_onto_cpu(tmp_path):
    """map_location='cpu' loads weights onto CPU without error."""
    src = nn.Linear(4, 2)
    src_opt = torch.optim.SGD(src.parameters(), lr=0.1)
    ckpt = tmp_path / "chk.pt"
    _write_checkpoint(ckpt, src, src_opt, epoch=3)

    dst = nn.Linear(4, 2)
    dst_opt = torch.optim.SGD(dst.parameters(), lr=0.1)
    module = _bypass_module(dst, dst_opt)

    state = module.load_checkpoint(str(ckpt), map_location="cpu")

    assert state.last_epoch == 3
    for param in dst.parameters():
        assert param.device.type == "cpu"
    # weights were actually copied from the checkpoint, not left at init.
    assert torch.allclose(dst.weight, src.weight)
    assert torch.allclose(dst.bias, src.bias)


def test_load_checkpoint_forwards_map_location(tmp_path, monkeypatch):
    """The caller's map_location reaches torch.load (regression for the dropped kwarg)."""
    src = nn.Linear(4, 2)
    src_opt = torch.optim.SGD(src.parameters(), lr=0.1)
    ckpt = tmp_path / "chk.pt"
    _write_checkpoint(ckpt, src, src_opt)

    dst = nn.Linear(4, 2)
    dst_opt = torch.optim.SGD(dst.parameters(), lr=0.1)
    module = _bypass_module(dst, dst_opt)

    captured = {}
    real_load = torch.load

    def spy(f, *args, **kwargs):
        captured["map_location"] = kwargs.get("map_location")
        return real_load(f, *args, **kwargs)

    monkeypatch.setattr(torch, "load", spy)

    module.load_checkpoint(str(ckpt), map_location="cpu")
    assert captured["map_location"] == "cpu"


def test_load_checkpoint_default_maps_cuda1_to_cuda0(tmp_path, monkeypatch):
    """With no map_location, the cuda:1->cuda:0 default remap is forwarded to torch.load."""
    src = nn.Linear(4, 2)
    src_opt = torch.optim.SGD(src.parameters(), lr=0.1)
    ckpt = tmp_path / "chk.pt"
    _write_checkpoint(ckpt, src, src_opt)

    dst = nn.Linear(4, 2)
    dst_opt = torch.optim.SGD(dst.parameters(), lr=0.1)
    module = _bypass_module(dst, dst_opt)

    captured = {}
    real_load = torch.load

    def spy(f, *args, **kwargs):
        captured["map_location"] = kwargs.get("map_location")
        return real_load(f, *args, **kwargs)

    monkeypatch.setattr(torch, "load", spy)

    module.load_checkpoint(str(ckpt))  # map_location defaults to None
    assert captured["map_location"] == {"cuda:1": "cuda:0"}


# Author: Mus mbayramo@stanford.edu
