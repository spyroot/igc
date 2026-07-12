"""Offline regression: the trainer resolves the real hidden-state seq_len.

``AutoencoderTrainer`` used to build ``AutoStateEncoder`` without a seq_len /
hidden_dim, defaulting to GPT-2's 1024 / 768 and crashing on any other backbone.
``_resolve_seq_len`` now derives the sequence length from the dataset chunk
length (``max_len``), falling back to the CLI ``seq_len`` and then the positional
dimension. CPU-only; uses ``__new__`` to skip the heavy backbone constructor.

Author:
Mus mbayramo@stanford.edu
"""

import argparse
from types import SimpleNamespace

from igc.modules.igc_train_auto_state_encoder import AutoencoderTrainer


def _trainer():
    return AutoencoderTrainer.__new__(AutoencoderTrainer)


def test_prefers_dataset_max_len():
    """The dataset's real max_len wins over the CLI seq_len and positions."""
    t = _trainer()
    t.dataset = SimpleNamespace(_max_len=777)
    spec = argparse.Namespace(seq_len=1024)
    assert t._resolve_seq_len(spec, (2048, 4096)) == 777


def test_falls_back_to_spec_seq_len_without_dataset():
    """No usable dataset max_len -> use the CLI seq_len."""
    t = _trainer()  # no self.dataset attribute at all (inference path)
    spec = argparse.Namespace(seq_len=512)
    assert t._resolve_seq_len(spec, (2048, 4096)) == 512


def test_falls_back_to_positions_last():
    """No dataset and no seq_len -> the positional dimension of input_shape."""
    t = _trainer()
    t.dataset = None
    spec = argparse.Namespace()  # no seq_len
    assert t._resolve_seq_len(spec, (2048, 4096)) == 2048


def test_ignores_non_positive_max_len():
    """A zero/None max_len is skipped, not returned as a bogus seq_len."""
    t = _trainer()
    t.dataset = SimpleNamespace(_max_len=0)
    spec = argparse.Namespace(seq_len=333)
    assert t._resolve_seq_len(spec, (2048, 4096)) == 333


# Author: Mus mbayramo@stanford.edu
