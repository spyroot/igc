"""Offline regressions for the env-driven ``m1_local`` profile and seq_len argv.

``m1_local`` takes its weights dir from ``$IGC_MODEL_DIR`` at resolve time (no
node-local path is committed), failing loudly when the variable is unset. The
launcher argv must emit ``--seq_len`` — the profile field was previously dead,
freezing chunk length at the build-time default regardless of the profile.

Author:
Mus mbayramo@stanford.edu
"""

import pytest

from igc.modules.train.launch import profile_to_argv
from igc.modules.train.profiles import resolve_profile


def test_m1_local_resolves_model_from_env(monkeypatch):
    """$IGC_MODEL_DIR expands into the profile's model at resolve time."""
    monkeypatch.setenv("IGC_MODEL_DIR", "/nvme/models/DeepSeek-V4-Flash")
    profile = resolve_profile("m1_local")
    assert profile.model == "/nvme/models/DeepSeek-V4-Flash"


def test_m1_local_unset_env_raises(monkeypatch):
    """An unset IGC_MODEL_DIR is a loud error, not a literal '$IGC_MODEL_DIR'."""
    monkeypatch.delenv("IGC_MODEL_DIR", raising=False)
    with pytest.raises(ValueError, match="IGC_MODEL_DIR"):
        resolve_profile("m1_local")


def test_env_expansion_applies_after_overrides(monkeypatch):
    """A model override wins and needs no env var."""
    monkeypatch.delenv("IGC_MODEL_DIR", raising=False)
    profile = resolve_profile("m1_local", model="gpt2")
    assert profile.model == "gpt2"


def test_argv_emits_profile_seq_len():
    """profile_to_argv forwards seq_len (256 for the gpt2 smoke profile)."""
    argv = profile_to_argv(resolve_profile("m1_gpt2_smoke"))
    assert "--seq_len" in argv
    assert argv[argv.index("--seq_len") + 1] == "256"


# Author: Mus mbayramo@stanford.edu
