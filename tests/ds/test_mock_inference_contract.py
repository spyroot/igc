"""Offline tests for the Phase 2/3 mock inference dataset contract.

The real ``D1`` rows wait for Phase 1 model weights; these tests pin only the tiny
mock rows and deterministic helpers that downstream builders can replace with trained
predictions later. Pure stdlib — no torch, no W&B, no live Redfish host.

Author:
Mus mbayramo@stanford.edu
"""

from copy import deepcopy

from igc.ds.mock_inference_contract import (
    D0,
    D1,
    build_phase3_calls,
    predict_phase2_rest_api_list,
    render_model_x,
    render_y_true,
    validate_phase2_row,
    validate_phase3_row,
)
from igc.modules.base.metric_keys import PHASE23_WANDB_METRIC_KEYS


def test_D0_phase2_rows_preserve_operator_uri_order() -> None:
    """Phase 2 maps operator text plus current context to an ordered rest_api_list."""
    row = D0[0]

    y_pred = {"rest_api_list": predict_phase2_rest_api_list(row)}

    assert set(row) == {"model_x", "x", "y_true", "y_pred"}
    assert row["model_x"] == "model_x"
    assert validate_phase2_row(row) == []
    assert y_pred == row["y_true"]


def test_D0_context_uses_phase1_renderer_shape() -> None:
    """D0 keeps the operator sentence and REST context under x, not a separate island."""
    row = D0[0]

    assert set(row["x"]) == {"text", "rest_api", "method", "json", "allowed_methods"}
    rendered = render_model_x(row)
    completion = render_y_true(row)

    assert "### REST API\n/redfish/v1" in rendered
    assert "### Allowed Methods\nGET, HEAD" in rendered
    assert "### Operator Sentence\nRead /redfish/v1/Managers/1 first" in rendered
    assert '"rest_api_list": [' in completion
    assert completion not in rendered


def test_phase2_ordering_uses_operator_sentence_not_context_order() -> None:
    """The URI order in x wins over the order of @odata.id links in json."""
    row = {
        "model_x": "model_x",
        "x": {
            "text": "Read /redfish/v1/Managers/1 first, then /redfish/v1/Systems/1.",
            "rest_api": "/redfish/v1",
            "method": "GET",
            "json": [
                {
                    "@odata.id": "/redfish/v1",
                    "Links": [
                        {"@odata.id": "/redfish/v1/Systems/1"},
                        {"@odata.id": "/redfish/v1/Managers/1"},
                    ],
                },
            ],
            "allowed_methods": {"/redfish/v1": ["GET", "HEAD"]},
        },
        "y_true": {
            "rest_api_list": [
                "/redfish/v1/Managers/1",
                "/redfish/v1/Systems/1",
            ],
        },
        "y_pred": {"rest_api_list": []},
    }

    assert predict_phase2_rest_api_list(row) == row["y_true"]["rest_api_list"]


def test_D1_phase3_rows_build_ordered_calls_with_get_arguments_empty() -> None:
    """Phase 3 keeps rest_api_list order and emits {} arguments for read-only GET calls."""
    row = D1[0]

    y_pred = {"calls": build_phase3_calls(row)}

    assert set(row) == {"model_x", "x", "y_true", "y_pred"}
    assert set(row["x"]) == {
        "text",
        "rest_api",
        "method",
        "json",
        "allowed_methods",
        "rest_api_list",
    }
    assert validate_phase3_row(row) == []
    assert y_pred == row["y_true"]
    assert y_pred["calls"][0]["method"] == "GET"
    assert y_pred["calls"][0]["arguments"] == {}


def test_phase3_post_arguments_are_selected_from_json_allowable_values() -> None:
    """A mutating action picks the mentioned allowable argument from current json."""
    row = {
        "model_x": "model_x",
        "x": {
            "text": "Read /redfish/v1/Systems/1, then reset with GracefulRestart at "
                    "/redfish/v1/Systems/1/Actions/ComputerSystem.Reset.",
            "rest_api": "/redfish/v1/Systems/1",
            "method": "GET",
            "json": [
                {
                    "@odata.id": "/redfish/v1/Systems/1",
                    "Actions": {
                        "#ComputerSystem.Reset": {
                            "target": "/redfish/v1/Systems/1/Actions/ComputerSystem.Reset",
                            "ResetType@Redfish.AllowableValues": [
                                "On",
                                "GracefulRestart",
                            ],
                        },
                    },
                },
            ],
            "rest_api_list": [
                "/redfish/v1/Systems/1",
                "/redfish/v1/Systems/1/Actions/ComputerSystem.Reset",
            ],
            "allowed_methods": {
                "/redfish/v1/Systems/1": ["GET", "HEAD"],
                "/redfish/v1/Systems/1/Actions/ComputerSystem.Reset": ["POST"],
            },
        },
        "y_true": {
            "calls": [
                {
                    "rest_api": "/redfish/v1/Systems/1",
                    "allowed_methods": ["GET", "HEAD"],
                    "method": "GET",
                    "arguments": {},
                },
                {
                    "rest_api": "/redfish/v1/Systems/1/Actions/ComputerSystem.Reset",
                    "allowed_methods": ["POST"],
                    "method": "POST",
                    "arguments": {"ResetType": "GracefulRestart"},
                },
            ],
        },
        "y_pred": {"calls": []},
    }

    assert build_phase3_calls(row) == row["y_true"]["calls"]


def test_phase23_metric_keys_are_constants_not_live_wandb_calls() -> None:
    """Metric keys are documented strings and do not import or initialize W&B."""
    assert "phase2_goal_extract/eval/ordered_exact_match_rate" in PHASE23_WANDB_METRIC_KEYS
    assert "phase2_goal_extract/eval/set_match_rate" in PHASE23_WANDB_METRIC_KEYS
    assert "phase3_argument_extract/eval/call_ordered_exact_match_rate" in PHASE23_WANDB_METRIC_KEYS
    assert "phase3_argument_extract/eval/readonly_empty_arguments_rate" in PHASE23_WANDB_METRIC_KEYS


def test_phase2_validator_reports_locked_key_drift() -> None:
    """Malformed D0 rows fail loudly when the locked top-level contract drifts."""
    row = deepcopy(D0[0])
    del row["model_x"]

    assert validate_phase2_row(row) == [
        "missing keys: ['model_x']",
        "model_x must be the locked model_x value",
    ]


def test_phase2_validator_rejects_committed_predictions() -> None:
    """Committed mock rows keep y_pred empty so training data remains ground truth only."""
    row = deepcopy(D0[0])
    row["y_pred"] = {"rest_api_list": ["/redfish/v1/Managers/1"]}

    assert validate_phase2_row(row) == [
        "y_pred.rest_api_list must be empty for committed mock rows",
    ]


def test_phase3_validator_rejects_get_arguments() -> None:
    """Read-only Phase 3 GET rows must keep arguments as an empty JSON object."""
    row = deepcopy(D1[0])
    row["y_true"]["calls"][0]["arguments"] = {"unexpected": True}

    assert validate_phase3_row(row) == ["call 0 GET arguments must be {}"]


# Author: Mus mbayramo@stanford.edu
