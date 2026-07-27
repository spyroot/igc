"""Executable evidence for review break 2: root masking recreates memorization.

Grounded on real data: the fixture is an actual GB300 HGX ERoT firmware
ImageSlot document from the public ``redfish_ctl`` Supermicro corpus, where
19% of the 1,886 captured resources are flat like it (a private full-crawl
measured 27.3% of 7,329). The ``json_objects`` family indexes every object
span including the ROOT; for a flat document the root is the only candidate,
so the draw is deterministic: the whole input context collapses to a single
mask token while the loss span covers the entire completion — the model must
reproduce real firmware states and version numbers from the URL alone, which
is exactly the memorize-from-URL objective Phase 1 was redesigned to
eliminate. No existing guard rejects the degenerate view; acceptance is the
defect this test pins. When root exclusion (or a subtree-size cap) lands,
this demonstration is expected to be replaced by the real coverage contract
tests.
"""

from __future__ import annotations

import json
from pathlib import Path

from igc.ds.phase1_render import (
    build_phase1_row,
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


def test_root_mask_degenerates_to_memorization_on_real_corpus_document() -> None:
    """Root draw is certain for a real flat capture, and no guard rejects it.

    Three legs of the same break:

    1. reachability — ``row_epoch_cycle`` starts at family index
       ``(row_index + epoch) % 7``, so epoch 3 / row 0 selects
       ``json_objects``; the real document is flat, so the root span is the
       only candidate and the draw is certain, no randomness involved;
    2. degeneracy — the masked input is exactly ``{mask_token: True}`` (all
       document content gone) while the loss span covers the entire
       completion, so every token of the real firmware document carries loss
       and the only remaining signal is the URL in the prompt;
    3. no guard — ``build_phase1_structural_loss_view`` returns the view and
       ``validate_phase1_row`` accepts it; nothing in the pipeline refuses
       the memorization task.
    """
    document = json.loads(FIXTURE.read_text())

    # Fixture preconditions: a real, flat Redfish resource. If the fixture
    # is ever swapped for a nested document this test must scream, because
    # leg 1 would no longer be deterministic.
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

    # Leg 1: the deterministic root draw happened on the real document.
    assert view.family == "json_objects"
    assert view.operations == ("mask:json_objects:/",)

    # Leg 2: input context is a bare mask token; loss covers the whole doc.
    assert view.row["x"]["json"] == {profile.mask_token: True}
    completion = render_phase1_completion(source["y_true"]["json"])
    assert view.completion_spans == ((0, len(completion)),)

    prompt, target_json = render_phase1_prompt(view.row)
    assert profile.mask_token in prompt
    assert document["@odata.id"] in prompt      # the URL survives...
    assert "FirmwareComparisonNumber" not in prompt  # ...the document does not
    assert str(document["FirmwareComparisonNumber"]) not in prompt
    assert "FirmwareComparisonNumber" in completion  # yet all of it carries loss
    assert target_json == source["y_true"]["json"]

    # Leg 3: every existing guard accepts the degenerate view.
    validate_phase1_row(view.row)
