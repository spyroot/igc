"""Offline test that the wandb MetricLogger backend builds and forwards metrics.

Regression for the bug where create_logger('wandb') returned None when the parsed spec
carried no project/entity, silently dropping every training metric. Uses a fake wandb
module — no network, no real run.

Author:
Mus mbayramo@stanford.edu
"""
import sys


class _FakeRun:
    def __init__(self):
        self.logged = []

    def log(self, data, step=None):
        self.logged.append((data, step))


class _FakeWandb:
    def __init__(self):
        self.last_init = None
        self.run = _FakeRun()

    def init(self, project=None, entity=None, **kwargs):
        # Real wandb.init also accepts name/group/job_type/tags/config; accept and ignore
        # them here so the fake matches how WandbLogger calls wandb.init.
        self.last_init = {"project": project, "entity": entity}
        return self.run


def _install_fake_wandb(monkeypatch):
    fake = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    return fake


def test_wandb_logger_builds_and_logs(monkeypatch):
    """create_logger('wandb', project, entity) builds a logger that forwards metrics."""
    fake = _install_fake_wandb(monkeypatch)
    from igc.modules.base.igc_metric_logger import MetricLogger

    ml = MetricLogger("wandb", project="igc", entity="spyroot")
    assert ml.logger is not None  # the bug made this None -> nothing tracked
    assert fake.last_init == {"project": "igc", "entity": "spyroot"}
    ml.log_metric("llm_emb_epoch_loss", 0.5, 1)
    assert fake.run.logged == [({"llm_emb_epoch_loss": 0.5}, 1)]


def test_wandb_logger_falls_back_to_env(monkeypatch):
    """With no project/entity in the spec, the logger still builds, reading the env."""
    fake = _install_fake_wandb(monkeypatch)
    monkeypatch.setenv("WANDB_PROJECT", "igc")
    monkeypatch.setenv("WANDB_ENTITY", "spyroot")
    from igc.modules.base.igc_metric_logger import MetricLogger

    ml = MetricLogger("wandb")  # no project/entity kwargs
    assert ml.logger is not None
    assert fake.last_init == {"project": "igc", "entity": "spyroot"}


# Author: Mus mbayramo@stanford.edu
