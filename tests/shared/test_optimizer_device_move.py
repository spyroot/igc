"""Offline regression for moving resumed optimizer state across devices.

On resume the optimizer's momentum buffers load on the checkpoint's device
(cpu) while the model params are on the trainer device, so the first
``optimizer.step()`` mixes cpu and cuda tensors (fused Adam raises). The
``meta`` device stands in for a distinct target so the move is verifiable with
no GPU. CPU-only.

Author:
Mus mbayramo@stanford.edu
"""

import torch

from igc.shared.shared_torch_utils import move_optimizer_state_to_device


def _stepped_optimizer():
    """An SGD-with-momentum whose state holds a real cpu tensor after a step."""
    model = torch.nn.Linear(3, 3)
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    model(torch.ones(2, 3)).sum().backward()
    opt.step()  # populates momentum_buffer state tensors
    return opt


def test_state_tensors_move_to_target_device():
    """Every state tensor is relocated to the requested device."""
    opt = _stepped_optimizer()
    assert any(
        isinstance(v, torch.Tensor) and v.device.type == "cpu"
        for s in opt.state.values() for v in s.values()
    )

    move_optimizer_state_to_device(opt, torch.device("meta"))

    for state in opt.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor):
                assert value.device.type == "meta"


def test_no_state_is_a_noop():
    """A fresh optimizer with no state does not raise."""
    model = torch.nn.Linear(2, 2)
    opt = torch.optim.Adam(model.parameters())
    move_optimizer_state_to_device(opt, torch.device("cpu"))  # must not raise
