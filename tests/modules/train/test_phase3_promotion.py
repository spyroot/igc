"""Unit tests for strict Phase 3 call-set promotion evidence."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from igc.modules.train.phase3_promotion import (
    REQUIRED_ARGUMENT_CLASSES,
    Phase3PromotionError,
    evaluate_phase3_promotion,
    evaluate_phase3_rows,
    validate_phase_views,
)


GET_API = "/redfish/v1/Systems/1"
HEAD_API = "/redfish/v1/Managers/1"
PATCH_SCALAR_API = "/redfish/v1/Systems/1/Actions/ComputerSystem.Reset"
PATCH_NESTED_API = "/redfish/v1/Systems/1/Bios/Settings"
POST_ZERO_API = "/redfish/v1/SessionService/Sessions"
POST_ONE_API = "/redfish/v1/Managers/1/Actions/Manager.Reset"
POST_MULTI_API = "/redfish/v1/UpdateService/Actions/UpdateService.SimpleUpdate"
DELETE_API = "/redfish/v1/SessionService/Sessions/1"
EXTRA_API = "/redfish/v1/Chassis/1"
DIGEST = "sha256:" + "b" * 64
PARENT_DIGEST = "sha256:" + "c" * 64
SOURCE_FULL_MANIFEST_DIGEST = "sha256:" + "d" * 64
TRAIN_MANIFEST_DIGEST = "sha256:" + "9" * 64
SOURCE_FULL_JSONL_DIGEST = "sha256:" + "6" * 64
TRAIN_JSONL_DIGEST = "sha256:" + "7" * 64
HELDOUT_MANIFEST_DIGEST = "sha256:" + "e" * 64
HELDOUT_JSONL_DIGEST = "sha256:" + "f" * 64
SPLIT_RELEASE_DIGEST = "sha256:" + "0" * 64
COMMIT_SHA = "0123456789abcdef0123456789abcdef01234567"
FOUNDATION_DIGEST = "sha256:" + "1" * 64
TOKENIZER_DIGEST = "sha256:" + "2" * 64
TASK_SPEC_DIGEST = "sha256:" + "3" * 64


CONTEXTS = {
    GET_API: {
        "rest_api": GET_API,
        "allowed_methods": ["GET"],
        "operation_names": [],
        "argument_schema": {},
        "json": {"@odata.id": GET_API, "Id": "1"},
    },
    HEAD_API: {
        "rest_api": HEAD_API,
        "allowed_methods": ["HEAD"],
        "operation_names": ["HeadManager"],
        "argument_schema": {},
        "json": {"@odata.id": HEAD_API, "Id": "1"},
    },
    PATCH_SCALAR_API: {
        "rest_api": PATCH_SCALAR_API,
        "allowed_methods": ["PATCH"],
        "operation_names": ["Reset"],
        "argument_schema": {
            "properties": {"ResetType": {"type": "string"}},
            "required": ["ResetType"],
        },
        "json": {"@odata.id": PATCH_SCALAR_API, "target": PATCH_SCALAR_API},
    },
    PATCH_NESTED_API: {
        "rest_api": PATCH_NESTED_API,
        "allowed_methods": ["PATCH"],
        "operation_names": ["UpdateBiosSettings"],
        "argument_schema": {
            "properties": {
                "Attributes": {
                    "type": "object",
                    "properties": {"BootMode": {"type": "string"}},
                    "required": ["BootMode"],
                },
            },
            "required": ["Attributes"],
        },
        "json": {"@odata.id": PATCH_NESTED_API, "Attributes": {"BootMode": "Legacy"}},
    },
    POST_ZERO_API: {
        "rest_api": POST_ZERO_API,
        "allowed_methods": ["POST"],
        "operation_names": ["CreateSessionWithDefaults"],
        "argument_schema": {},
        "json": {"@odata.id": POST_ZERO_API, "Members": []},
    },
    POST_ONE_API: {
        "rest_api": POST_ONE_API,
        "allowed_methods": ["POST"],
        "operation_names": ["ResetManager"],
        "argument_schema": {
            "properties": {"ResetType": {"type": "string"}},
            "required": ["ResetType"],
        },
        "json": {"@odata.id": POST_ONE_API, "target": POST_ONE_API},
    },
    POST_MULTI_API: {
        "rest_api": POST_MULTI_API,
        "allowed_methods": ["POST"],
        "operation_names": ["SimpleUpdate"],
        "argument_schema": {
            "properties": {
                "ImageURI": {"type": "string"},
                "TransferProtocol": {"type": "string"},
            },
            "required": ["ImageURI", "TransferProtocol"],
        },
        "json": {"@odata.id": POST_MULTI_API, "target": POST_MULTI_API},
    },
    DELETE_API: {
        "rest_api": DELETE_API,
        "allowed_methods": ["DELETE"],
        "operation_names": ["DeleteSession"],
        "argument_schema": {},
        "json": {"@odata.id": DELETE_API, "Id": "1"},
    },
    EXTRA_API: {
        "rest_api": EXTRA_API,
        "allowed_methods": ["GET"],
        "operation_names": ["GetChassis"],
        "argument_schema": {},
        "json": {"@odata.id": EXTRA_API, "Id": "1"},
    },
}


CALLS = {
    "get": {
        "rest_api": GET_API,
        "http_method": "GET",
        "operation_name": None,
        "arguments": {},
    },
    "head": {
        "rest_api": HEAD_API,
        "http_method": "HEAD",
        "operation_name": "HeadManager",
        "arguments": {},
    },
    "patch_scalar": {
        "rest_api": PATCH_SCALAR_API,
        "http_method": "PATCH",
        "operation_name": "Reset",
        "arguments": {"ResetType": "GracefulRestart"},
    },
    "patch_nested": {
        "rest_api": PATCH_NESTED_API,
        "http_method": "PATCH",
        "operation_name": "UpdateBiosSettings",
        "arguments": {"Attributes": {"BootMode": "Uefi"}},
    },
    "post_zero": {
        "rest_api": POST_ZERO_API,
        "http_method": "POST",
        "operation_name": "CreateSessionWithDefaults",
        "arguments": {},
    },
    "post_one": {
        "rest_api": POST_ONE_API,
        "http_method": "POST",
        "operation_name": "ResetManager",
        "arguments": {"ResetType": "ForceRestart"},
    },
    "post_multiple": {
        "rest_api": POST_MULTI_API,
        "http_method": "POST",
        "operation_name": "SimpleUpdate",
        "arguments": {
            "ImageURI": "https://updates.example.invalid/firmware.bin",
            "TransferProtocol": "HTTPS",
        },
    },
    "delete": {
        "rest_api": DELETE_API,
        "http_method": "DELETE",
        "operation_name": "DeleteSession",
        "arguments": {},
    },
    "extra": {
        "rest_api": EXTRA_API,
        "http_method": "GET",
        "operation_name": "GetChassis",
        "arguments": {},
    },
}


def _copy_calls(names: list[str]) -> list[dict]:
    return [deepcopy(CALLS[name]) for name in names]


def _phase3_row(
    calls: list[dict],
    *,
    y_pred_calls: list[dict] | None = None,
    y_pred: object | None = None,
    grounding_sources: list[str] | None = None,
    grounded: bool = True,
    row_id: str = "d1-row-1",
) -> dict:
    selected = [call["rest_api"] for call in calls]
    return {
        "phase": 3,
        "source_dataset": "D1",
        "task": "text_and_rest_api_list_to_calls",
        "target_semantics": "unordered_unique_call_set",
        "x": {
            "text": "Execute the selected Redfish calls.",
            "rest_api_list": selected,
            "api_context": [deepcopy(CONTEXTS[api]) for api in selected],
        },
        "y_true": {"calls": deepcopy(calls)},
        "y_pred": (
            y_pred
            if y_pred is not None
            else {
                "calls": deepcopy(
                    list(reversed(calls)) if y_pred_calls is None else y_pred_calls,
                ),
            }
        ),
        "label_evidence": {
            "argument_value_grounding_by_api": {
                api: {
                    "grounded": grounded,
                    "sources": list(
                        grounding_sources or ["operator_text", "argument_schema"]
                    ),
                }
                for api in selected
            },
        },
        "metadata": {"row_id": row_id, "sample_width_k": len(selected)},
    }


def _phase2_view(
    rest_apis: list[str],
    *,
    row_id: str = "d1-row-1",
    text: str = "Execute the selected Redfish calls.",
    api_context: list[dict] | None = None,
) -> dict[str, object]:
    return {
        "x": {
            "text": text,
            "api_context": (
                deepcopy(api_context)
                if api_context is not None
                else [deepcopy(CONTEXTS[api]) for api in rest_apis]
            ),
        },
        "y_true": {"rest_api_list": list(rest_apis)},
        "metadata": {"row_id": row_id},
    }


def _rows_all_argument_classes() -> list[dict]:
    return [
        _phase3_row(_copy_calls(["get"])),
        _phase3_row(_copy_calls(["head", "patch_scalar"])),
        _phase3_row(_copy_calls(["patch_nested", "post_zero", "delete"])),
        _phase3_row(_copy_calls(["post_one", "post_multiple"])),
    ]


def _observed_evidence(**overrides: object) -> dict[str, object]:
    evidence = {
        "observed_source_full_manifest_sha": SOURCE_FULL_MANIFEST_DIGEST,
        "observed_source_full_sha": SOURCE_FULL_JSONL_DIGEST,
        "observed_train_manifest_sha": TRAIN_MANIFEST_DIGEST,
        "observed_train_sha": TRAIN_JSONL_DIGEST,
        "observed_heldout_manifest_sha": HELDOUT_MANIFEST_DIGEST,
        "observed_heldout_sha": HELDOUT_JSONL_DIGEST,
        "observed_split_release_sha": SPLIT_RELEASE_DIGEST,
    }
    evidence.update(overrides)
    return evidence


def _artifact_evidence(*, heldout_rows: int = 4, **overrides: object) -> dict[str, object]:
    evidence: dict[str, object] = {
        "immutable_full_manifest": {
            "immutable": True,
            "complete": True,
            "rows": 4,
            "sha256": SOURCE_FULL_MANIFEST_DIGEST,
            "artifact_sha": SOURCE_FULL_JSONL_DIGEST,
        },
        "immutable_train_manifest": {
            "immutable": True,
            "complete": True,
            "rows": 4,
            "manifest_sha": TRAIN_MANIFEST_DIGEST,
            "artifact_sha": TRAIN_JSONL_DIGEST,
        },
        "disjoint_split_release": {
            "immutable": True,
            "complete": True,
            "disjoint": True,
            "sha256": SPLIT_RELEASE_DIGEST,
            "train_manifest_sha": TRAIN_MANIFEST_DIGEST,
            "heldout_manifest_sha": HELDOUT_MANIFEST_DIGEST,
        },
        "real_promoted_parent_checkpoint": {
            "role": "goal_extractor",
            "promotion_status": "pass",
            "artifact_sha": PARENT_DIGEST,
        },
        "real_heldout_data": {
            "real": True,
            "split": "heldout",
            "rows": heldout_rows,
            "manifest_sha": HELDOUT_MANIFEST_DIGEST,
            "artifact_sha": HELDOUT_JSONL_DIGEST,
        },
        "artifact_sha": DIGEST,
        "checkpoint_reload": {
            "status": "pass",
            "artifact_sha": DIGEST,
            "adapter_dir": "/promoted/argument_extractor",
        },
        "inference_smoke": {"status": "pass", "artifact_sha": DIGEST},
    }
    for key, value in overrides.items():
        if key == "immutable_full_manifest" and value is False:
            evidence["immutable_full_manifest"] = {
                **evidence["immutable_full_manifest"],
                "complete": False,
            }
        elif key == "immutable_train_manifest" and value is False:
            evidence["immutable_train_manifest"] = {
                **evidence["immutable_train_manifest"],
                "complete": False,
            }
        elif key == "disjoint_split_release" and value is False:
            evidence["disjoint_split_release"] = {
                **evidence["disjoint_split_release"],
                "disjoint": False,
            }
        elif key == "real_promoted_parent_checkpoint" and value is False:
            evidence["real_promoted_parent_checkpoint"] = {
                **evidence["real_promoted_parent_checkpoint"],
                "promotion_status": "fail",
            }
        elif key == "real_heldout_data" and value is False:
            evidence["real_heldout_data"] = {
                **evidence["real_heldout_data"],
                "real": False,
            }
        elif key == "artifact_sha" and value is False:
            evidence["artifact_sha"] = "not-a-sha256"
        elif key == "checkpoint_reload_succeeded" and value is False:
            evidence["checkpoint_reload"] = {
                **evidence["checkpoint_reload"],
                "status": "fail",
            }
        elif key == "inference_smoke_succeeded" and value is False:
            evidence["inference_smoke"] = {
                **evidence["inference_smoke"],
                "status": "fail",
            }
        else:
            evidence[key] = value
    return evidence


def _run_report(**manifest_overrides: object) -> dict[str, object]:
    manifest: dict[str, object] = {
        "phase": "phase3_argument_extraction",
        "task": "text_and_rest_api_list_to_calls",
        "parent_role": "goal_extractor",
        "parent_artifact_sha": PARENT_DIGEST,
        "output_role": "argument_extractor",
        "task_spec_sha": TASK_SPEC_DIGEST,
        "foundation_model_sha": FOUNDATION_DIGEST,
        "tokenizer_sha": TOKENIZER_DIGEST,
        "data_manifest": TRAIN_MANIFEST_DIGEST,
        "source_manifest_sha": SOURCE_FULL_MANIFEST_DIGEST,
        "eval_manifest_sha": HELDOUT_MANIFEST_DIGEST,
        "eval_split": HELDOUT_JSONL_DIGEST,
        "eval_data_sha": HELDOUT_JSONL_DIGEST,
        "train_data_sha": TRAIN_JSONL_DIGEST,
        "git_commit": COMMIT_SHA,
        "checkpoint_path": "/runs/phase3/checkpoints/argument_extractor_epoch_best.pt",
        "promoted_artifact_path": "/promoted/argument_extractor",
        "promotion_source": "best_checkpoint",
        "training": {"optimizer_steps": 100, "train_loss": 0.1},
    }
    manifest.update(manifest_overrides)
    return {"manifest": manifest, "metrics": {"eval_loss": 0.1, "eval_accuracy": 1.0}}


def _thresholds(**overrides: object) -> dict[str, object]:
    thresholds: dict[str, object] = {
        "min_json_parse_rate": 1.0,
        "min_call_set_exact_match_rate": 1.0,
        "min_rest_api_coverage_exact": 1.0,
        "min_method_exact_match_rate": 1.0,
        "min_operation_name_exact_match_rate": 1.0,
        "min_argument_schema_valid_rate": 1.0,
        "min_argument_value_grounding_rate": 1.0,
        "min_arguments_exact_match_rate": 1.0,
        "min_readonly_empty_arguments_rate": 1.0,
        "max_invalid_method_rate": 0.0,
        "max_duplicate_call_rate": 0.0,
        "max_missing_call_rate": 0.0,
        "max_extra_call_rate": 0.0,
    }
    thresholds.update(overrides)
    return thresholds


def test_phase3_rows_cover_argument_classes_widths_nullable_operation_and_order() -> None:
    """Phase 3 metrics are keyed by REST API, not prediction list position."""
    metrics = evaluate_phase3_rows(_rows_all_argument_classes())

    assert metrics["json_parse_rate"] == 1.0
    assert metrics["call_set_exact_match_rate"] == 1.0
    assert metrics["rest_api_coverage_exact_rate"] == 1.0
    assert metrics["operation_name_exact_match_rate"] == 1.0
    assert metrics["readonly_empty_arguments_rate"] == 1.0
    assert metrics["argument_schema_valid_rate"] == 1.0
    assert metrics["argument_value_grounding_rate"] == 1.0
    assert metrics["call_set_exact_match_by_width"] == {
        "1": 1.0,
        "2": 1.0,
        "3": 1.0,
    }
    assert metrics["arguments_exact_match_by_class"] == {
        name: 1.0 for name in sorted(REQUIRED_ARGUMENT_CLASSES)
    }


@pytest.mark.parametrize(
    ("prediction", "expected_metric", "expected_value"),
    [
        (_copy_calls(["get"]), "missing_call_rate", 0.5),
        (_copy_calls(["get", "head", "extra"]), "extra_call_rate", 0.5),
        (_copy_calls(["get", "get"]), "duplicate_call_rate", 0.5),
    ],
)
def test_phase3_rows_count_missing_extra_and_duplicate_calls(
    prediction: list[dict],
    expected_metric: str,
    expected_value: float,
) -> None:
    """Call-set coverage defects are visible in dedicated aggregate metrics."""
    metrics = evaluate_phase3_rows([
        _phase3_row(_copy_calls(["get", "head"]), y_pred_calls=prediction),
    ])

    assert metrics["call_set_exact_match_rate"] == 0.0
    assert metrics["rest_api_coverage_exact_rate"] == 0.0
    assert metrics[expected_metric] == expected_value


def test_phase3_zero_prediction_counts_all_missing_without_invalid_or_duplicate_rates() -> None:
    """An empty parsed call list misses everything but has no bad-method duplicates."""
    metrics = evaluate_phase3_rows([
        _phase3_row(_copy_calls(["get", "head"]), y_pred_calls=[]),
    ])

    assert metrics["json_parse_rate"] == 1.0
    assert metrics["rest_api_coverage_exact_rate"] == 0.0
    assert metrics["missing_call_rate"] == 1.0
    assert metrics["invalid_method_rate"] == 0.0
    assert metrics["duplicate_call_rate"] == 0.0
    assert metrics["extra_call_rate"] == 0.0


@pytest.mark.parametrize(
    "prediction",
    [
        pytest.param(
            [{**CALLS["get"], "http_method": "get"}],
            id="lowercase-method",
        ),
        pytest.param(
            [{**CALLS["get"], "http_method": "TRACE"}],
            id="unsupported-method",
        ),
        pytest.param(
            [{**CALLS["get"], "http_method": "POST"}],
            id="disallowed-by-context",
        ),
        pytest.param(
            _copy_calls(["extra"]),
            id="missing-api-context",
        ),
    ],
)
def test_phase3_rows_count_invalid_methods_independently_from_json_parse(
    prediction: list[dict],
) -> None:
    """Lowercase, unsupported, disallowed, and contextless methods are method defects."""
    metrics = evaluate_phase3_rows([
        _phase3_row(_copy_calls(["get"]), y_pred_calls=prediction),
    ])

    assert metrics["json_parse_rate"] == 1.0
    assert metrics["invalid_method_rate"] == 1.0
    assert metrics["call_set_exact_match_rate"] == 0.0


def test_phase3_rows_keep_valid_json_parse_separate_from_call_shape() -> None:
    """A syntactic JSON object can still be an invalid Phase 3 call contract."""
    metrics = evaluate_phase3_rows([
        _phase3_row(_copy_calls(["get"]), y_pred={"calls": {"not": "a-list"}}),
    ])

    assert metrics["json_parse_rate"] == 1.0
    assert metrics["call_set_exact_match_rate"] == 0.0
    assert metrics["rest_api_coverage_exact_rate"] == 0.0
    assert metrics["invalid_method_rate"] == 0.0


def test_phase3_rows_reject_empty_api_context_for_width_zero() -> None:
    """Phase 3 k=0 rows still require non-empty api_context public context."""
    row = _phase3_row([])

    assert row["metadata"]["sample_width_k"] == 0
    assert row["x"]["rest_api_list"] == []
    assert row["x"]["api_context"] == []
    assert row["y_true"]["calls"] == []
    with pytest.raises(Phase3PromotionError, match="api_context must be a non-empty list"):
        evaluate_phase3_rows([row])


@pytest.mark.parametrize(
    ("mutate_context", "message"),
    [
        (
            lambda context: context.update({"private_selected": True}),
            "api_context fields",
        ),
        (
            lambda context: context.pop("operation_names"),
            "api_context fields",
        ),
        (
            lambda context: context.update({"allowed_methods": ["get"]}),
            "allowed_methods must be uppercase",
        ),
        (
            lambda context: context.update({"allowed_methods": ["GET", "GET"]}),
            "allowed_methods must be unique",
        ),
        (
            lambda context: context.update({"allowed_methods": ["TRACE"]}),
            "unsupported methods",
        ),
        (
            lambda context: context.update({"operation_names": ["Reset", "Reset"]}),
            "operation_names must be unique",
        ),
        (
            lambda context: context.update({"operation_names": ["Reset", 7]}),
            "operation_names must be list",
        ),
        (
            lambda context: context.update({"argument_schema": []}),
            "argument_schema must be an object",
        ),
        (
            lambda context: context.update({"json": []}),
            "context json must be an object",
        ),
    ],
)
def test_phase3_rows_require_exact_context_fields_and_types(
    mutate_context,
    message: str,
) -> None:
    """Phase 3 public API context is exact: no private fields and typed lists only."""
    row = _phase3_row(_copy_calls(["get"]))
    mutate_context(row["x"]["api_context"][0])

    with pytest.raises(Phase3PromotionError, match=message):
        evaluate_phase3_rows([row])


@pytest.mark.parametrize(
    "target_semantics",
    ["ordered_call_list", None],
)
def test_phase3_promotion_rows_reject_missing_or_wrong_target_semantics(
    target_semantics,
) -> None:
    """Phase 3 promotion rows must declare unordered_unique_call_set semantics."""
    row = _phase3_row(_copy_calls(["get"]))
    if target_semantics is None:
        row.pop("target_semantics")
    else:
        row["target_semantics"] = target_semantics

    with pytest.raises(Phase3PromotionError, match="target semantics"):
        evaluate_phase3_rows([row])


def test_phase3_rows_report_schema_invalid_predictions_without_accepting_them() -> None:
    """A predicted call can cover the API while still failing argument schema checks."""
    bad_schema = _copy_calls(["patch_scalar"])
    bad_schema[0]["arguments"] = {"ResetType": 7}

    metrics = evaluate_phase3_rows([
        _phase3_row(_copy_calls(["patch_scalar"]), y_pred_calls=bad_schema),
    ])

    assert metrics["rest_api_coverage_exact_rate"] == 1.0
    assert metrics["call_set_exact_match_rate"] == 0.0
    assert metrics["argument_schema_valid_rate"] == 0.0
    assert metrics["arguments_exact_match_rate"] == 0.0


def test_phase3_promotion_rejects_wrong_operation_name_with_matching_api_method_and_args() -> None:
    """Wrong operation names cannot promote even when API, method, and args match."""
    rows = _rows_all_argument_classes()
    mutated = deepcopy(rows)
    predicted_calls = mutated[1]["y_pred"]["calls"]
    patch_call = next(
        call for call in predicted_calls
        if call["rest_api"] == PATCH_SCALAR_API
    )
    patch_call["operation_name"] = "ResetManager"

    metrics = evaluate_phase3_rows(mutated)
    result = evaluate_phase3_promotion(
        rows=mutated,
        thresholds=_thresholds(),
        artifact_evidence=_artifact_evidence(heldout_rows=len(mutated)),
        run_report=_run_report(),
        view_pairs=[(_phase2_view([GET_API]), mutated[0])],
        **_observed_evidence(),
    )

    assert metrics["rest_api_coverage_exact_rate"] == 1.0
    assert metrics["http_method_exact_match_rate"] == 1.0
    assert metrics["arguments_exact_match_rate"] == 1.0
    assert metrics["argument_schema_valid_rate"] == 1.0
    assert metrics["operation_name_exact_match_rate"] < 1.0
    assert metrics["call_set_exact_match_rate"] < 1.0
    assert result["status"] == "fail"
    assert {
        "min_call_set_exact_match_rate",
        "min_operation_name_exact_match_rate",
    } <= {failure["name"] for failure in result["failures"]}


def test_phase3_rows_reject_current_json_as_argument_grounding() -> None:
    """Arbitrary current JSON evidence is not accepted as value grounding."""
    with pytest.raises(Phase3PromotionError, match="unsupported argument-grounding source"):
        evaluate_phase3_rows([
            _phase3_row(
                _copy_calls(["patch_scalar"]),
                grounding_sources=["current_json"],
            ),
        ])


def test_phase3_rows_reject_unknown_argument_grounding_source() -> None:
    """Grounding evidence is closed to the canonical source vocabulary."""
    with pytest.raises(Phase3PromotionError, match="unsupported argument-grounding source"):
        evaluate_phase3_rows([
            _phase3_row(
                _copy_calls(["patch_scalar"]),
                grounding_sources=["operator_text", "fixture"],
            ),
        ])


def test_phase3_rows_reject_mutation_grounding_without_schema_or_action_source() -> None:
    """Mutation values need operator text plus schema or operation-definition evidence."""
    with pytest.raises(Phase3PromotionError, match="schema/action grounding"):
        evaluate_phase3_rows([
            _phase3_row(
                _copy_calls(["patch_scalar"]),
                grounding_sources=["operator_text"],
            ),
        ])


def test_phase3_rows_reject_ungrounded_non_empty_arguments() -> None:
    """Non-empty expected arguments require explicit non-current-json grounding."""
    with pytest.raises(Phase3PromotionError, match="not grounded"):
        evaluate_phase3_rows([
            _phase3_row(_copy_calls(["patch_nested"]), grounded=False),
        ])


def test_phase3_view_consistency_matches_phase2_rest_api_set() -> None:
    """The D1 Phase 2 and Phase 3 views must select the same REST API set."""
    phase3_row = _phase3_row(_copy_calls(["get", "head"]))
    phase2_row = _phase2_view([GET_API, HEAD_API])
    mismatched_phase2 = _phase2_view(
        [GET_API, PATCH_SCALAR_API],
        api_context=phase3_row["x"]["api_context"],
    )

    validate_phase_views(phase2_row, phase3_row)
    with pytest.raises(Phase3PromotionError, match="select different"):
        validate_phase_views(mismatched_phase2, phase3_row)


def test_phase3_view_consistency_rejects_changed_text() -> None:
    """Phase 2 and Phase 3 D1 views must share the exact natural-language text."""
    phase3_row = _phase3_row(_copy_calls(["get"]))
    phase2_row = _phase2_view([GET_API], text="Reset the selected Redfish resource.")

    with pytest.raises(Phase3PromotionError, match="different text labels"):
        validate_phase_views(phase2_row, phase3_row)


def test_phase3_view_consistency_rejects_changed_api_context() -> None:
    """Phase 2 and Phase 3 D1 views must share the exact public API context."""
    phase3_row = _phase3_row(_copy_calls(["get"]))
    changed_context = deepcopy(phase3_row["x"]["api_context"])
    changed_context[0]["json"] = {"@odata.id": GET_API, "Id": "changed"}
    phase2_row = _phase2_view([GET_API], api_context=changed_context)

    with pytest.raises(Phase3PromotionError, match="different API context"):
        validate_phase_views(phase2_row, phase3_row)


def test_phase3_view_consistency_rejects_phase3_input_set_disagreement() -> None:
    """Phase 3 x.rest_api_list must equal the Phase 2 target set."""
    phase3_row = _phase3_row(_copy_calls(["get", "head"]))
    phase3_row["x"]["rest_api_list"] = [GET_API, PATCH_SCALAR_API]
    phase2_row = _phase2_view([GET_API, HEAD_API])

    with pytest.raises(Phase3PromotionError, match="input REST API set disagrees"):
        validate_phase_views(phase2_row, phase3_row)


def test_phase3_view_consistency_requires_same_d1_row_id() -> None:
    """D1 Phase 2/3 view pairs must share the exact master-row identity."""
    phase3_row = _phase3_row(_copy_calls(["get"]), row_id="d1-row-a")
    phase2_row = _phase2_view([GET_API], row_id="d1-row-b")

    with pytest.raises(Phase3PromotionError, match="row identities"):
        validate_phase_views(phase2_row, phase3_row)


def test_phase3_promotion_requires_d1_view_consistency_and_artifact_evidence() -> None:
    """Promotion fails closed when view-pair or real artifact evidence is absent."""
    rows = _rows_all_argument_classes()
    good_view_pair = (_phase2_view([GET_API]), rows[0])
    bad_view_pair = (_phase2_view([PATCH_SCALAR_API]), rows[0])
    wrong_row_id_pair = (_phase2_view([GET_API], row_id="other-row"), rows[0])

    passing = evaluate_phase3_promotion(
        rows=rows,
        thresholds=_thresholds(),
        artifact_evidence=_artifact_evidence(heldout_rows=len(rows)),
        run_report=_run_report(),
        view_pairs=[good_view_pair],
        **_observed_evidence(),
    )
    inconsistent = evaluate_phase3_promotion(
        rows=rows,
        thresholds=_thresholds(),
        artifact_evidence=_artifact_evidence(heldout_rows=len(rows)),
        run_report=_run_report(),
        view_pairs=[bad_view_pair],
        **_observed_evidence(),
    )
    row_id_mismatch = evaluate_phase3_promotion(
        rows=rows,
        thresholds=_thresholds(),
        artifact_evidence=_artifact_evidence(heldout_rows=len(rows)),
        run_report=_run_report(),
        view_pairs=[wrong_row_id_pair],
        **_observed_evidence(),
    )
    missing_smoke = evaluate_phase3_promotion(
        rows=rows,
        thresholds=_thresholds(),
        artifact_evidence=_artifact_evidence(
            heldout_rows=len(rows),
            inference_smoke_succeeded=False,
        ),
        run_report=_run_report(),
        view_pairs=[good_view_pair],
        **_observed_evidence(),
    )

    assert passing["status"] == "pass"
    assert passing["role"] == "argument_extractor"
    assert passing["artifact_sha"] == DIGEST
    assert inconsistent["status"] == "fail"
    assert any(
        failure["name"] == "d1_phase2_phase3_view_consistency"
        for failure in inconsistent["failures"]
    )
    assert row_id_mismatch["status"] == "fail"
    assert any(
        failure["name"] == "d1_phase2_phase3_view_consistency"
        for failure in row_id_mismatch["failures"]
    )
    assert missing_smoke["status"] == "fail"
    assert any(
        failure["name"] == "inference_smoke_succeeded"
        for failure in missing_smoke["failures"]
    )


def test_phase3_promotion_rejects_heldout_suite_missing_head_method() -> None:
    """Held-out evidence must include HEAD, not just another read-only GET row."""
    rows = [
        _phase3_row(_copy_calls(["get"])),
        _phase3_row(_copy_calls(["patch_scalar"])),
        _phase3_row(_copy_calls(["patch_nested", "post_zero", "delete"])),
        _phase3_row(_copy_calls(["post_one", "post_multiple"])),
    ]

    result = evaluate_phase3_promotion(
        rows=rows,
        thresholds=_thresholds(),
        artifact_evidence=_artifact_evidence(heldout_rows=len(rows)),
        run_report=_run_report(),
        view_pairs=[(_phase2_view([GET_API]), rows[0])],
        **_observed_evidence(),
    )

    assert result["status"] == "fail"
    assert result["metrics"]["expected_calls_by_http_method"] == {
        "DELETE": 1,
        "GET": 1,
        "PATCH": 2,
        "POST": 3,
    }
    assert any(
        failure["name"] == "http_method_head_present"
        for failure in result["failures"]
    )


def test_phase3_promotion_records_and_passes_all_required_http_method_checks() -> None:
    """A complete held-out suite records passing checks for every required method."""
    rows = _rows_all_argument_classes()

    result = evaluate_phase3_promotion(
        rows=rows,
        thresholds=_thresholds(),
        artifact_evidence=_artifact_evidence(heldout_rows=len(rows)),
        run_report=_run_report(),
        view_pairs=[(_phase2_view([GET_API]), rows[0])],
        **_observed_evidence(),
    )
    method_checks = {
        check["name"]: check
        for check in result["checks"]
        if check["name"].startswith("http_method_")
        and check["name"].endswith("_present")
    }

    assert result["status"] == "pass"
    assert result["metrics"]["expected_calls_by_http_method"] == {
        "DELETE": 1,
        "GET": 1,
        "HEAD": 1,
        "PATCH": 2,
        "POST": 3,
    }
    assert {
        "http_method_delete_present",
        "http_method_get_present",
        "http_method_head_present",
        "http_method_patch_present",
        "http_method_post_present",
    } == set(method_checks)
    assert all(check["passed"] is True for check in method_checks.values())


@pytest.mark.parametrize(
    ("missing_key", "failure_name"),
    [
        ("immutable_full_manifest", "immutable_full_manifest"),
        ("immutable_train_manifest", "immutable_train_manifest"),
        ("disjoint_split_release", "disjoint_train_heldout_release"),
        ("real_promoted_parent_checkpoint", "real_promoted_parent_checkpoint"),
        ("real_heldout_data", "real_heldout_data"),
        ("artifact_sha", "artifact_sha"),
        ("checkpoint_reload_succeeded", "checkpoint_reload_succeeded"),
        ("inference_smoke_succeeded", "inference_smoke_succeeded"),
    ],
)
def test_phase3_promotion_requires_real_artifact_evidence(
    missing_key: str,
    failure_name: str,
) -> None:
    """Promotion remains hard-gated on manifest, parent, held-out, and smoke evidence."""
    rows = _rows_all_argument_classes()

    result = evaluate_phase3_promotion(
        rows=rows,
        thresholds=_thresholds(),
        artifact_evidence=_artifact_evidence(
            heldout_rows=len(rows),
            **{missing_key: False},
        ),
        run_report=_run_report(),
        view_pairs=[(_phase2_view([GET_API]), rows[0])],
        **_observed_evidence(),
    )

    assert result["status"] == "fail"
    assert any(failure["name"] == failure_name for failure in result["failures"])


@pytest.mark.parametrize(
    ("observed_overrides", "failure_name"),
    [
        (
            {"observed_source_full_manifest_sha": "sha256:" + "1" * 64},
            "full_manifest_sha_matches_file",
        ),
        (
            {"observed_source_full_sha": "sha256:" + "8" * 64},
            "immutable_full_manifest",
        ),
        (
            {"observed_train_manifest_sha": "sha256:" + "5" * 64},
            "immutable_train_manifest",
        ),
        (
            {"observed_train_sha": "sha256:" + "9" * 64},
            "immutable_train_manifest",
        ),
        (
            {"observed_heldout_manifest_sha": "sha256:" + "2" * 64},
            "heldout_manifest_sha_matches_file",
        ),
        (
            {"observed_heldout_sha": "sha256:" + "3" * 64},
            "heldout_artifact_sha_matches_file",
        ),
        (
            {"observed_split_release_sha": "sha256:" + "4" * 64},
            "disjoint_train_heldout_release",
        ),
    ],
)
def test_phase3_promotion_rejects_observed_manifest_or_heldout_digest_mismatch(
    observed_overrides: dict[str, object],
    failure_name: str,
) -> None:
    """Promotion evidence must match the actual full/heldout manifest and JSONL digests."""
    rows = _rows_all_argument_classes()

    result = evaluate_phase3_promotion(
        rows=rows,
        thresholds=_thresholds(),
        artifact_evidence=_artifact_evidence(heldout_rows=len(rows)),
        run_report=_run_report(),
        view_pairs=[(_phase2_view([GET_API]), rows[0])],
        **_observed_evidence(**observed_overrides),
    )

    assert result["status"] == "fail"
    assert any(failure["name"] == failure_name for failure in result["failures"])


@pytest.mark.parametrize(
    "split_release_update",
    [
        {"disjoint": False},
        {"train_manifest_sha": "sha256:" + "5" * 64},
        {"heldout_manifest_sha": "sha256:" + "6" * 64},
        {"sha256": "sha256:" + "7" * 64},
    ],
)
def test_phase3_promotion_rejects_split_release_lineage_mismatch(
    split_release_update: dict[str, object],
) -> None:
    """Promotion binds train/held-out child manifests to one disjoint split release."""
    rows = _rows_all_argument_classes()
    evidence = _artifact_evidence(heldout_rows=len(rows))
    evidence["disjoint_split_release"] = {
        **evidence["disjoint_split_release"],
        **split_release_update,
    }

    result = evaluate_phase3_promotion(
        rows=rows,
        thresholds=_thresholds(),
        artifact_evidence=evidence,
        run_report=_run_report(),
        view_pairs=[(_phase2_view([GET_API]), rows[0])],
        **_observed_evidence(),
    )

    assert result["status"] == "fail"
    assert any(
        failure["name"] == "disjoint_train_heldout_release"
        for failure in result["failures"]
    )


@pytest.mark.parametrize(
    ("manifest_update", "failure_name"),
    [
        ({"data_manifest": "sha256:" + "8" * 64}, "run_train_manifest_matches_file"),
        ({"train_data_sha": "sha256:" + "9" * 64}, "run_train_data_matches_file"),
    ],
)
def test_phase3_promotion_rejects_run_report_train_lineage_mismatch(
    manifest_update: dict[str, object],
    failure_name: str,
) -> None:
    """Run report data_manifest is the train manifest and train_data_sha is train JSONL."""
    rows = _rows_all_argument_classes()
    run_report = deepcopy(_run_report())
    run_report["manifest"].update(manifest_update)

    result = evaluate_phase3_promotion(
        rows=rows,
        thresholds=_thresholds(),
        artifact_evidence=_artifact_evidence(heldout_rows=len(rows)),
        run_report=run_report,
        view_pairs=[(_phase2_view([GET_API]), rows[0])],
        **_observed_evidence(),
    )

    assert result["status"] == "fail"
    assert any(failure["name"] == failure_name for failure in result["failures"])


def test_phase3_promotion_rejects_heldout_row_count_mismatch() -> None:
    """real_heldout_data.rows must match the actual held-out JSONL row count."""
    rows = _rows_all_argument_classes()

    result = evaluate_phase3_promotion(
        rows=rows,
        thresholds=_thresholds(),
        artifact_evidence=_artifact_evidence(heldout_rows=len(rows) + 1),
        run_report=_run_report(),
        view_pairs=[(_phase2_view([GET_API]), rows[0])],
        **_observed_evidence(),
    )

    assert result["status"] == "fail"
    assert any(
        failure["name"] == "heldout_row_count_matches_file"
        for failure in result["failures"]
    )


@pytest.mark.parametrize(
    ("field", "value", "failure_name"),
    [
        ("phase", "phase3_wrong", "run_role_and_task_lineage"),
        ("task", "call_ordered", "run_role_and_task_lineage"),
        ("parent_role", "model_x", "run_role_and_task_lineage"),
        (
            "parent_artifact_sha",
            "sha256:" + "9" * 64,
            "run_role_and_task_lineage",
        ),
        ("output_role", "planner", "run_role_and_task_lineage"),
        ("data_manifest", "sha256:" + "8" * 64, "run_train_manifest_matches_file"),
        (
            "source_manifest_sha",
            "sha256:" + "8" * 64,
            "run_source_full_manifest_matches_file",
        ),
        (
            "eval_manifest_sha",
            "sha256:" + "8" * 64,
            "run_eval_manifest_matches_file",
        ),
        ("eval_split", "sha256:" + "8" * 64, "run_eval_data_matches_file"),
        ("eval_data_sha", "sha256:" + "8" * 64, "run_eval_data_matches_file"),
        ("train_data_sha", "sha256:" + "8" * 64, "run_train_data_matches_file"),
        ("git_commit", "abcdef1", "training_code_commit_exists"),
        ("foundation_model_sha", "not-a-sha", "foundation_model_sha_exists"),
        ("tokenizer_sha", "not-a-sha", "tokenizer_sha_exists"),
        ("task_spec_sha", "not-a-sha", "task_spec_sha_exists"),
    ],
)
def test_phase3_promotion_requires_run_report_lineage_fields(
    field: str,
    value: object,
    failure_name: str,
) -> None:
    """Promotion requires exact training, model, task, train-data, and eval lineage."""
    rows = _rows_all_argument_classes()
    run_report = deepcopy(_run_report())
    run_report["manifest"][field] = value

    result = evaluate_phase3_promotion(
        rows=rows,
        thresholds=_thresholds(),
        artifact_evidence=_artifact_evidence(heldout_rows=len(rows)),
        run_report=run_report,
        view_pairs=[(_phase2_view([GET_API]), rows[0])],
        **_observed_evidence(),
    )

    assert result["status"] == "fail"
    assert any(failure["name"] == failure_name for failure in result["failures"])


@pytest.mark.parametrize(
    "manifest_overrides",
    [
        {"checkpoint_path": ""},
        {"promoted_artifact_path": ""},
        {"promotion_source": "final_checkpoint"},
    ],
)
def test_phase3_promotion_requires_promoted_best_checkpoint(
    manifest_overrides: dict[str, object],
) -> None:
    """The promoted argument_extractor artifact must come from the best checkpoint."""
    rows = _rows_all_argument_classes()

    result = evaluate_phase3_promotion(
        rows=rows,
        thresholds=_thresholds(),
        artifact_evidence=_artifact_evidence(heldout_rows=len(rows)),
        run_report=_run_report(**manifest_overrides),
        view_pairs=[(_phase2_view([GET_API]), rows[0])],
        **_observed_evidence(),
    )

    assert result["status"] == "fail"
    assert any(
        failure["name"] == "best_checkpoint_promoted"
        for failure in result["failures"]
    )


@pytest.mark.parametrize(
    "missing_threshold",
    [
        "min_call_set_exact_match_rate",
        "min_operation_name_exact_match_rate",
        "max_extra_call_rate",
    ],
)
def test_phase3_promotion_requires_non_null_thresholds(missing_threshold: str) -> None:
    """Missing thresholds are rejected before a promotion verdict is emitted."""
    thresholds = _thresholds(**{missing_threshold: None})
    rows = _rows_all_argument_classes()

    with pytest.raises(Phase3PromotionError, match="missing non-null thresholds"):
        evaluate_phase3_promotion(
            rows=rows,
            thresholds=thresholds,
            artifact_evidence=_artifact_evidence(),
            run_report=_run_report(),
            view_pairs=[(_phase2_view([GET_API]), rows[0])],
            **_observed_evidence(),
        )


def test_phase3_promotion_config_declares_call_set_and_operation_name_floors() -> None:
    """The checked-in promotion profile owns strict call-set and operation floors."""
    spec = yaml.safe_load(
        Path("configs/inference/phase3_argument_extractor_promotion.yaml").read_text(
            encoding="utf-8",
        ),
    )

    assert spec["thresholds"]["min_call_set_exact_match_rate"] == 0.90
    assert spec["thresholds"]["min_operation_name_exact_match_rate"] == 0.98
