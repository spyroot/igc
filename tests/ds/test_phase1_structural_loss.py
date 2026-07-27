"""Contract tests for the historical Phase 1 selective structural loss."""

from __future__ import annotations

import copy
import json

import pytest

from igc.ds.phase1_render import (
    build_phase1_row,
    phase1_json_with_spans,
    render_phase1_completion,
)
from igc.ds.phase1_structural_loss import (
    build_phase1_structural_loss_view,
    load_phase1_structural_loss_profile,
)


EXPECTED_FAMILIES = (
    "odata_id",
    "action_targets",
    "target_keys",
    "json_objects",
    "json_arrays",
    "allowable_values",
    "api_prefixes",
)


def _row() -> dict:
    body = {
        "@odata.id": "/redfish/v1/Systems/1",
        "Actions": {
            "#ComputerSystem.Reset": {
                "target": "/redfish/v1/Systems/1/Actions/ComputerSystem.Reset",
                "ResetType@Redfish.AllowableValues": [
                    "On",
                    "ForceOff",
                ],
            }
        },
        "Boot": {"BootSourceOverrideTarget": "Pxe"},
    }
    return build_phase1_row(
        rest_api="/redfish/v1/Systems/1",
        allowed_methods=["GET", "PATCH"],
        input_json=body,
        target_json=body,
    )


def test_structural_loss_profile_locks_historical_family_order() -> None:
    """The first baseline contains exactly the seven proven historical families."""
    profile = load_phase1_structural_loss_profile(
        "historical_structural_mask_v1"
    )

    assert profile.enabled is True
    assert profile.selection == "row_epoch_cycle"
    assert tuple(family.name for family in profile.families) == EXPECTED_FAMILIES
    assert profile.families[-1].max_spans is None
    assert profile.spec_sha256.startswith("sha256:")


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"empty_array": [], "empty_object": {}},
        {"nested": [{"value": 1}, [True, None, "text"]]},
        _row()["y_true"]["json"],
    ],
)
def test_span_renderer_is_byte_identical_to_canonical_json(body: dict) -> None:
    """Adding span indexes must not change the existing Phase 1 target bytes."""
    rendered, spans = phase1_json_with_spans(body)

    assert rendered == json.dumps(body, indent=2, sort_keys=True)
    assert spans
    assert all(0 <= span.start < span.end <= len(rendered) for span in spans)


def test_train_epochs_cover_each_structural_family_without_mutating_target() -> None:
    """The curriculum rotates families while D0 and canonical y_true stay immutable."""
    profile = load_phase1_structural_loss_profile(
        "historical_structural_mask_v1"
    )
    source = _row()
    original = copy.deepcopy(source)
    completion = render_phase1_completion(source["y_true"]["json"])
    observed = []
    snippets_by_family = {}

    for epoch in range(len(EXPECTED_FAMILIES)):
        result = build_phase1_structural_loss_view(
            source,
            profile=profile,
            mode="train",
            run_seed=31,
            epoch=epoch,
            row_index=0,
        )
        observed.append(result.family)
        snippets_by_family[result.family] = tuple(
            completion[start:end]
            for start, end in result.completion_spans
        )
        assert result.row["y_true"] == original["y_true"]
        assert result.row["x"] != original["x"]
        assert result.operations
        assert result.completion_spans
        assert all(
            completion[start:end]
            for start, end in result.completion_spans
        )

    assert tuple(observed) == EXPECTED_FAMILIES
    assert all("\"@odata.id\"" in text for text in snippets_by_family["odata_id"])
    assert all("\"target\"" in text for text in snippets_by_family["action_targets"])
    assert snippets_by_family["target_keys"] == ('"target"',)
    assert all(
        text.startswith("{") and text.endswith("}")
        for text in snippets_by_family["json_objects"]
    )
    assert all(
        text.startswith("[") and text.endswith("]")
        for text in snippets_by_family["json_arrays"]
    )
    assert all(
        "@Redfish.AllowableValues" in text
        for text in snippets_by_family["allowable_values"]
    )
    assert set(snippets_by_family["api_prefixes"]) == {"/redfish/v1/"}
    assert source == original


def test_train_family_start_depends_on_row_and_falls_forward_when_absent() -> None:
    """Rows start at different families and missing families use the next available one."""
    profile = load_phase1_structural_loss_profile(
        "historical_structural_mask_v1"
    )
    source = _row()

    first = build_phase1_structural_loss_view(
        source,
        profile=profile,
        mode="train",
        run_seed=31,
        epoch=0,
        row_index=0,
    )
    second = build_phase1_structural_loss_view(
        source,
        profile=profile,
        mode="train",
        run_seed=31,
        epoch=0,
        row_index=1,
    )
    sparse = build_phase1_row(
        rest_api="/redfish/v1/Systems/1",
        allowed_methods=["GET"],
        input_json={"Id": "1"},
        target_json={"Id": "1"},
    )
    fallback = build_phase1_structural_loss_view(
        sparse,
        profile=profile,
        mode="train",
        run_seed=31,
        epoch=0,
        row_index=1,
    )

    assert first.family == "odata_id"
    assert second.family == "action_targets"
    assert fallback.family == "json_objects"


def test_structural_loss_is_reproducible_and_heldout_is_epoch_fixed() -> None:
    """Train sampling is seeded and evaluation ignores mutable epoch state."""
    profile = load_phase1_structural_loss_profile(
        "historical_structural_mask_v1"
    )
    source = _row()
    kwargs = {
        "profile": profile,
        "run_seed": 97,
        "row_index": 3,
    }

    train_a = build_phase1_structural_loss_view(
        source,
        mode="train",
        epoch=4,
        **kwargs,
    )
    train_b = build_phase1_structural_loss_view(
        source,
        mode="train",
        epoch=4,
        **kwargs,
    )
    heldout_a = build_phase1_structural_loss_view(
        source,
        mode="evaluation",
        epoch=0,
        **kwargs,
    )
    heldout_b = build_phase1_structural_loss_view(
        source,
        mode="evaluation",
        epoch=999,
        **kwargs,
    )

    assert train_a == train_b
    assert heldout_a == heldout_b
    assert heldout_a.family != "full_completion"
    assert heldout_a.completion_spans
    assert heldout_a.row["x"] != source["x"]


def test_api_prefix_loss_covers_every_prefix_removed_from_input() -> None:
    """Substring masking and completion supervision have identical cardinality."""
    body = {
        f"Uri{index:02d}": f"/redfish/v1/Systems/{index}"
        for index in range(20)
    }
    source = build_phase1_row(
        rest_api="/redfish/v1/Systems",
        allowed_methods=["GET"],
        input_json=body,
        target_json=body,
    )
    profile = load_phase1_structural_loss_profile(
        "historical_structural_mask_v1"
    )
    result = build_phase1_structural_loss_view(
        source,
        profile=profile,
        mode="train",
        run_seed=31,
        epoch=6,
        row_index=0,
    )
    completion = render_phase1_completion(source["y_true"]["json"])

    assert result.family == "api_prefixes"
    assert len(result.completion_spans) == 20
    assert all(
        completion[start:end] == "/redfish/v1/"
        for start, end in result.completion_spans
    )
    assert "/redfish/v1/" not in result.row["x"]["rest_api"]
    assert "/redfish/v1/" not in json.dumps(result.row["x"]["json"])


def test_disabled_profile_preserves_full_completion_contract() -> None:
    """Existing Phase 1 profiles retain their prior full-completion labels."""
    source = _row()
    result = build_phase1_structural_loss_view(
        source,
        profile=load_phase1_structural_loss_profile("none"),
        mode="train",
        run_seed=1,
        epoch=4,
        row_index=2,
    )

    assert result.row == source
    assert result.row is not source
    assert result.family == "full_completion"
    assert result.completion_spans == ()
    assert result.operations == ()


def test_structural_loss_rejects_non_reconstruction_source_rows() -> None:
    """Selective masking requires the materialized D0 input/target parity gate."""
    source = _row()
    source["x"]["json"] = {"different": True}

    with pytest.raises(ValueError, match="x.json == y_true.json"):
        build_phase1_structural_loss_view(
            source,
            profile=load_phase1_structural_loss_profile(
                "historical_structural_mask_v1"
            ),
            mode="train",
            run_seed=1,
            epoch=0,
            row_index=0,
        )
