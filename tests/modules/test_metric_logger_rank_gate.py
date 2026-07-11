"""Offline regression: metric backend is rank-gated to one run per job.

Under multi-GPU every rank builds the MetricLogger, so a W&B backend would open
one run per GPU. create_logger returns a no-op NullLogger on non-main ranks so
a 4-GPU job produces exactly one W&B run (driven by rank 0). CPU-only; the rank
is faked via the RANK/LOCAL_RANK env vars accelerate/torchrun set.

Author:
Mus mbayramo@stanford.edu
"""

import pytest

from igc.modules.base.metric_factory import NullLogger, create_logger


def test_non_main_rank_gets_null_logger(monkeypatch):
    """RANK != 0 yields a no-op logger regardless of the requested backend."""
    monkeypatch.setenv("RANK", "2")
    logger = create_logger("wandb")
    assert isinstance(logger, NullLogger)
    logger.log_scalar("train/loss", 1.0, 0)  # must be a silent no-op


def test_local_rank_also_gates(monkeypatch):
    """LOCAL_RANK is honored when RANK is absent."""
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.setenv("LOCAL_RANK", "1")
    assert isinstance(create_logger("wandb"), NullLogger)


def test_main_rank_builds_real_backend(monkeypatch, tmp_path):
    """Rank 0 builds the real backend (tensorboard here — no network/creds)."""
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    logger = create_logger("tensorboard", output_dir=str(tmp_path))
    assert not isinstance(logger, NullLogger)
    assert logger is not None


def test_single_process_is_main(monkeypatch, tmp_path):
    """No RANK env (single-process run) counts as main."""
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    assert not isinstance(
        create_logger("tensorboard", output_dir=str(tmp_path)), NullLogger)


# Author: Mus mbayramo@stanford.edu
