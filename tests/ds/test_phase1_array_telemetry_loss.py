"""Executable evidence for review break 3: array loss memorizes raw telemetry.

The objective lists arrays among the loss targets, but the ``json_arrays``
family is indiscriminate: it cannot tell a semantic array (an
``@Redfish.AllowableValues`` action vocabulary) from a pure-telemetry array
(firmware key indices). On the same real GB300 ERoT ImageSlot capture used
by the root-mask pin, the epoch that lands on ``json_arrays`` deterministically
draws BOTH integer arrays — 64 raw firmware key indices — hides them from the
input, and puts loss on every digit, making the model memorize device
telemetry byte for byte while "ordinary copied telemetry tokens contribute no
loss" is the stated objective. No guard distinguishes array kinds. When
value-aware array selection (or a semantic whitelist) lands, this
demonstration is expected to be replaced by the real array contract tests.
"""

from __future__ import annotations

import json
from pathlib import Path

from igc.ds.phase1_render import (
    build_phase1_row,
    phase1_json_dumps,
    render_phase1_prompt,
    validate_phase1_row,
)
from igc.ds.phase1_structural_loss import (
    build_phase1_structural_loss_view,
    load_phase1_structural_loss_profile,
)


JSON_ARRAYS_FAMILY_INDEX = 4  # position of json_arrays in the v1 profile
FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "jsons"
    / "supermicro_gb300_flat_image_slot.json"
)


def test_array_family_puts_loss_on_every_raw_telemetry_integer() -> None:
    """Both telemetry arrays are drawn, hidden, and fully loss-covered.

    Three legs of the same break:

    1. reachability — epoch 4 / row 0 starts the family cycle at
       ``json_arrays``; the document has exactly two arrays and
       ``max_spans: 2`` takes every candidate, so the draw is deterministic
       regardless of shuffle order;
    2. telemetry memorization — both arrays hold nothing but raw firmware
       key indices; the input replaces them with the mask token while the
       loss spans parse back to the exact integer lists, so every digit of
       device telemetry carries loss;
    3. no guard — nothing distinguishes semantic arrays from telemetry
       arrays, and the degenerate view validates clean.
    """
    document = json.loads(FIXTURE.read_text())

    # Fixture preconditions: exactly two flat integer arrays. A swapped
    # fixture must fail loudly because leg 1 relies on both facts.
    arrays = {key: value for key, value in document.items() if isinstance(value, list)}
    assert sorted(arrays) == ["AllowedKeyIndices", "RevokedKeyIndices"]
    assert all(isinstance(item, int) for value in arrays.values() for item in value)

    source = build_phase1_row(
        rest_api=document["@odata.id"],
        allowed_methods=["GET"],
        input_json=document,
        target_json=document,
    )
    profile = load_phase1_structural_loss_profile("historical_structural_mask_v1")

    view = build_phase1_structural_loss_view(
        source,
        profile=profile,
        mode="train",
        run_seed=31,
        epoch=JSON_ARRAYS_FAMILY_INDEX,
        row_index=0,
    )

    # Leg 1: the deterministic double-array draw happened.
    assert view.family == "json_arrays"
    assert sorted(view.operations) == [
        "mask:json_arrays:/AllowedKeyIndices",
        "mask:json_arrays:/RevokedKeyIndices",
    ]

    # Leg 2a: the input no longer contains the telemetry values.
    assert view.row["x"]["json"]["AllowedKeyIndices"] == profile.mask_token
    assert view.row["x"]["json"]["RevokedKeyIndices"] == profile.mask_token
    prompt, _ = render_phase1_prompt(view.row)
    assert "RevokedKeyIndices" in prompt        # the key survives as a hint
    assert "\n    16,\n" not in prompt          # the values do not

    # Leg 2b: each loss span parses back to the exact hidden integer list,
    # so every telemetry digit carries loss. Loss this epoch is precisely
    # the two arrays — the scalar outside them stays uncovered.
    rendered = phase1_json_dumps(source["y_true"]["json"])
    span_values = sorted(
        json.loads(rendered[start:end]) for start, end in view.completion_spans
    )
    assert span_values == sorted(arrays.values())
    for start, end in view.completion_spans:
        assert str(document["FirmwareComparisonNumber"]) not in rendered[start:end]

    # Leg 3: the telemetry-memorization view validates clean.
    validate_phase1_row(view.row)
