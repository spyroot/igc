"""Executable evidence for review issue 1: Phase 1 input noise is absent.

One invariant, one test: rendering always sorts keys canonically (so upstream
key-order noise never reaches the model), the branch's only shuffle is the
mask-span candidate shuffle, and ``build_phase1_structural_loss_view`` hard
requires ``x.json == y_true.json`` so content noise added later is rejected
by the existing guard. Masking landed; shuffling didn't. When per-epoch input
noise lands, this demonstration is expected to be replaced by the real noise
contract tests.
"""

from __future__ import annotations

import copy
import inspect

import pytest

import igc.ds.corpus_dataset as corpus_dataset_module
import igc.ds.phase1_render as phase1_render_module
import igc.ds.phase1_structural_loss as structural_loss_module
from igc.ds.phase1_render import build_phase1_row, render_phase1_prompt
from igc.ds.phase1_structural_loss import (
    build_phase1_structural_loss_view,
    load_phase1_structural_loss_profile,
)


def _row() -> dict:
    """One Phase 1 row with enough sibling keys for order to be observable."""
    body = {
        "@odata.id": "/redfish/v1/Systems/1",
        "Id": "1",
        "Name": "System One",
        "PowerState": "On",
        "SystemType": "Physical",
        "Actions": {
            "#ComputerSystem.Reset": {
                "target": "/redfish/v1/Systems/1/Actions/ComputerSystem.Reset",
                "ResetType@Redfish.AllowableValues": ["On", "ForceOff"],
            }
        },
    }
    return build_phase1_row(
        rest_api="/redfish/v1/Systems/1",
        allowed_methods=["GET", "PATCH"],
        input_json=body,
        target_json=body,
    )


def _reversed_key_order(value):
    """Deep copy with every object's key insertion order reversed."""
    if isinstance(value, dict):
        return {
            key: _reversed_key_order(child)
            for key, child in reversed(list(value.items()))
        }
    if isinstance(value, list):
        return [_reversed_key_order(child) for child in value]
    return value


def test_phase1_input_noise_is_impossible_without_code_changes() -> None:
    """Key/order/input noise is absent and cannot be added from data alone.

    Three legs of the same invariant:

    1. canonical rendering — deep-reversing the key insertion order of
       ``x.json`` produces a byte-identical prompt, so key-order noise is
       erased before tokenization;
    2. the only randomness in the view pipeline is the span-candidate
       shuffle inside the structural-loss module — neither the renderer nor
       the dataset shuffles anything;
    3. the ``x.json == y_true.json`` guard rejects content noise (key
       dropout) outright, so input noise added upstream is refused today.
    """
    # Leg 1: rendering canonicalizes any upstream key order.
    source = _row()
    reordered = copy.deepcopy(source)
    reordered["x"]["json"] = _reversed_key_order(reordered["x"]["json"])
    assert list(reordered["x"]["json"]) != list(source["x"]["json"])
    assert reordered["x"]["json"] == source["x"]["json"]
    original_prompt, _ = render_phase1_prompt(source)
    reordered_prompt, _ = render_phase1_prompt(reordered)
    assert reordered_prompt == original_prompt

    # Leg 2: the branch's sole shuffle is the mask-span candidate shuffle.
    structural_source = inspect.getsource(structural_loss_module)
    assert structural_source.count(".shuffle(") == 1
    assert "rng.shuffle(candidates)" in structural_source
    assert ".shuffle(" not in inspect.getsource(corpus_dataset_module)
    render_source = inspect.getsource(phase1_render_module)
    assert ".shuffle(" not in render_source
    assert "sorted(current.items()" in render_source

    # Leg 3: the equality guard rejects content noise added upstream.
    profile = load_phase1_structural_loss_profile("historical_structural_mask_v1")
    dropped = _row()
    dropped["x"]["json"] = {
        key: value
        for key, value in dropped["x"]["json"].items()
        if key != "PowerState"
    }
    with pytest.raises(ValueError, match="x.json == y_true.json"):
        build_phase1_structural_loss_view(
            dropped,
            profile=profile,
            mode="train",
            run_seed=31,
            epoch=0,
            row_index=0,
        )
