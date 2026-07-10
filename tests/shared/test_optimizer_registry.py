"""Offline regressions for the optimizer registry in ``TorchBuilder``.

``AdamW2`` aliased ``transformers.AdamW``, which was removed in transformers
5.x — selecting it failed at trainer construction. The registry no longer
advertises it (the ``--llm_optimizer`` CLI choices derive from this list), and
every advertised name must resolve to a real ``torch.optim`` class. CPU-only.

Author:
Mus mbayramo@stanford.edu
"""

import pytest
import torch

from igc.shared.shared_torch_builder import TorchBuilder


def test_adamw2_no_longer_advertised():
    """The removed transformers.AdamW alias is gone from the registry."""
    assert "AdamW2" not in TorchBuilder.get_supported_optimizers()


@pytest.mark.parametrize("name", TorchBuilder.get_supported_optimizers())
def test_every_advertised_optimizer_constructs(name):
    """Each advertised optimizer builds against torch.optim on a tiny model."""
    model = torch.nn.Linear(2, 2)
    optimizer = TorchBuilder.create_optimizer(
        name, model, lr=0.01, weight_decay=0.0
    )
    assert isinstance(optimizer, torch.optim.Optimizer)


def test_unknown_optimizer_raises():
    """A stale/unknown name fails with a clear error, not an AttributeError."""
    with pytest.raises(ValueError, match="not recognized"):
        TorchBuilder.create_optimizer(
            "AdamW2", torch.nn.Linear(2, 2), lr=0.01, weight_decay=0.0
        )


# Author: Mus mbayramo@stanford.edu
