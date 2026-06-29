"""Regression: is_inference must default to the bool False, not the truthy string "False".

The string "False" is truthy AND fails the isinstance(..., bool) guard in IgcModule, so a
default-constructed trainer silently ran in inference mode or raised a TypeError. Checks the
signature default of every trainer that exposes the param — no construction needed.

Author:
Mus mbayramo@stanford.edu
"""
import importlib
import inspect

import pytest


@pytest.mark.parametrize(
    "modpath,clsname",
    [
        ("igc.modules.base.igc_rl_base_module", "RlBaseModule"),
        ("igc.modules.igc_train_agent", "IgcAgentTrainer"),
        ("igc.modules.igc_train_auto_state_encoder", "AutoencoderTrainer"),
    ],
)
def test_is_inference_defaults_to_bool_false(modpath, clsname):
    """The is_inference default is the bool False (not the string 'False')."""
    klass = getattr(importlib.import_module(modpath), clsname)
    default = inspect.signature(klass.__init__).parameters["is_inference"].default
    assert default is False, f"{clsname}.is_inference default is {default!r}, expected bool False"


# Author: Mus mbayramo@stanford.edu
