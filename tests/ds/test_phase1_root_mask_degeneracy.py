"""Regression coverage for the flat-resource root masking failure.

Grounded on real data: the fixture is an actual GB300 HGX ERoT firmware
ImageSlot document from the public ``redfish_ctl`` Supermicro corpus, where
19% of the 1,886 captured resources are flat like it (a private full-crawl
measured 27.3% of 7,329). The ``json_objects`` family must never select the
root span: flat documents fall back to one bounded root field so the model
cannot be trained to regenerate an entire firmware document from its URL.
"""

from __future__ import annotations

import json
from pathlib import Path

from igc.ds.phase1_render import (
    build_phase1_row,
    phase1_json_dumps,
    render_phase1_completion,
    render_phase1_prompt,
    validate_phase1_row,
)
from igc.ds.phase1_structural_loss import (
    build_phase1_structural_loss_view,
    load_phase1_structural_loss_profile,
)


JSON_OBJECTS_FAMILY_INDEX = 3  # position of json_objects in the v1 profile
FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "jsons"
    / "supermicro_gb300_flat_image_slot.json"
)


def _nested_object_count(value) -> int:
    """Objects strictly below the root — 0 means the root span is unique."""
    count = 0
    children = value.values() if isinstance(value, dict) else value
    if isinstance(value, (dict, list)):
        for child in children:
            if isinstance(child, dict):
                count += 1
            count += _nested_object_count(child)
    return count


def test_root_object_span_is_rejected_for_flat_real_corpus_document() -> None:
    """A real flat capture receives one bounded field mask, never a root mask."""
    document = json.loads(FIXTURE.read_text())

    # If the fixture becomes nested, this no longer exercises the flat-row
    # fallback and must fail instead of silently weakening the regression.
    assert document["@odata.id"].startswith("/redfish/v1/")
    assert _nested_object_count(document) == 0
    assert "FirmwareComparisonNumber" in document

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
        epoch=JSON_OBJECTS_FAMILY_INDEX,
        row_index=0,
    )

    assert view.family == "json_objects"
    assert len(view.operations) == 1
    assert view.operations != ("mask:json_objects:/",)

    # The root survives and exactly one field is hidden. The supervised span
    # is therefore smaller than the whole canonical document.
    assert view.row["x"]["json"] != {profile.mask_token: True}
    assert len(view.row["x"]["json"]) == len(document)
    assert list(view.row["x"]["json"].values()).count(profile.mask_token) == 1
    completion = render_phase1_completion(source["y_true"]["json"])
    rendered = phase1_json_dumps(source["y_true"]["json"])
    assert completion == rendered + "\n"
    assert all(
        0 <= start < end <= len(rendered) and end - start < len(rendered)
        for start, end in view.completion_spans
    )

    prompt, target_json = render_phase1_prompt(view.row)
    assert profile.mask_token in prompt
    assert document["@odata.id"] in prompt
    assert phase1_json_dumps(document) not in prompt
    assert "FirmwareComparisonNumber" in completion
    assert target_json == source["y_true"]["json"]

    # The bounded view remains a valid Phase 1 row.
    validate_phase1_row(view.row)
