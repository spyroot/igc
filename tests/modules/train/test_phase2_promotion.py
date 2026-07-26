"""Unit tests for strict Phase 2 promotion evidence."""

from __future__ import annotations

from copy import deepcopy

import pytest

from igc.modules.train.phase2_promotion import (
    REQUIRED_ROBUSTNESS_VARIANTS,
    Phase2PromotionError,
    evaluate_phase2_promotion,
    evaluate_phase2_rows,
)


API_A = "/redfish/v1/Systems/1"
API_B = "/redfish/v1/Managers/1"
API_C = "/redfish/v1/Chassis/1"
API_D = "/redfish/v1/Systems/1/Bios"
DISTRACTORS = (
    "/redfish/v1/TaskService",
    "/redfish/v1/EventService",
    "/redfish/v1/AccountService",
    "/redfish/v1/UpdateService",
)
DIGEST = "sha256:" + "a" * 64
PARENT_DIGEST = "sha256:" + "b" * 64
SOURCE_FULL_MANIFEST_DIGEST = "sha256:" + "c" * 64
TRAIN_MANIFEST_DIGEST = "sha256:" + "0" * 64
SOURCE_FULL_JSONL_DIGEST = "sha256:" + "6" * 64
TRAIN_JSONL_DIGEST = "sha256:" + "7" * 64
HELDOUT_MANIFEST_DIGEST = "sha256:" + "d" * 64
HELDOUT_JSONL_DIGEST = "sha256:" + "e" * 64
SPLIT_RELEASE_DIGEST = "sha256:" + "f" * 64
COMMIT_SHA = "0123456789abcdef0123456789abcdef01234567"
FOUNDATION_DIGEST = "sha256:" + "1" * 64
TOKENIZER_DIGEST = "sha256:" + "2" * 64
TASK_SPEC_DIGEST = "sha256:" + "3" * 64


def _context(api: str) -> dict[str, object]:
    return {
        "rest_api": api,
        "allowed_methods": ["GET", "HEAD"],
        "operation_names": ["get_resource"],
        "argument_schema": {},
        "json": {"@odata.id": api, "Name": api.rsplit("/", 1)[-1]},
    }


def _phase2_row(
    targets: list[str],
    *,
    predicted: list[str] | None = None,
    group: str = "VendorA/Model1",
    width: int | None = None,
    robustness_variant: str | None = None,
    semantic_case_id: str = "semantic-case",
    extra_context: tuple[str, ...] = (),
) -> dict:
    context_apis = list(dict.fromkeys([*targets, *extra_context, *DISTRACTORS]))
    metadata = {
        "sample_width_k": len(targets) if width is None else width,
        "heldout_vendor_or_model": group,
    }
    if robustness_variant is not None:
        metadata["semantic_case_id"] = semantic_case_id
        metadata["robustness_variant"] = robustness_variant
    return {
        "phase": 2,
        "dataset": "D1",
        "source_dataset": "D0",
        "task": "text_to_rest_api_list",
        "target_semantics": "unordered_unique_rest_api_set",
        "x": {
            "text": "Inspect the selected Redfish resources.",
            "api_context": [_context(api) for api in context_apis],
        },
        "y_true": {"rest_api_list": list(targets)},
        "y_pred": {"rest_api_list": list(targets if predicted is None else predicted)},
        "metadata": metadata,
    }


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


def _artifact_evidence(*, heldout_rows: int = 8, **overrides: object) -> dict[str, object]:
    evidence: dict[str, object] = {
        "immutable_full_manifest": {
            "immutable": True,
            "complete": True,
            "rows": 8,
            "sha256": SOURCE_FULL_MANIFEST_DIGEST,
            "artifact_sha": SOURCE_FULL_JSONL_DIGEST,
        },
        "immutable_train_manifest": {
            "immutable": True,
            "complete": True,
            "rows": 8,
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
            "role": "model_x",
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
            "adapter_dir": "/promoted/goal_extractor",
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
        "phase": "phase2_goal_extraction",
        "task": "text_to_rest_api_list",
        "parent_role": "model_x",
        "parent_artifact_sha": PARENT_DIGEST,
        "output_role": "goal_extractor",
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
        "checkpoint_path": "/runs/phase2/checkpoints/goal_extractor_epoch_best.pt",
        "promoted_artifact_path": "/promoted/goal_extractor",
        "promotion_source": "best_checkpoint",
        "training": {"optimizer_steps": 100, "train_loss": 0.1},
    }
    manifest.update(manifest_overrides)
    return {"manifest": manifest, "metrics": {"eval_loss": 0.1, "eval_accuracy": 1.0}}


def _thresholds(**overrides: object) -> dict[str, object]:
    thresholds: dict[str, object] = {
        "min_json_parse_rate": 1.0,
        "min_set_exact_match_rate": 1.0,
        "min_set_exact_match_by_width": {"1": 1.0, "2": 1.0, "3": 1.0},
        "max_duplicate_api_rate": 0.0,
        "max_invalid_api_rate": 0.0,
        "min_heldout_vendor_or_model_set_exact": 1.0,
        "min_heldout_rows_per_vendor_or_model": 1,
        "min_empty_set_exact_match_rate": 1.0,
    }
    thresholds.update(overrides)
    return thresholds


def _passing_rows() -> list[dict]:
    robustness_rows = [
        _phase2_row(
            [API_A],
            robustness_variant=variant,
            semantic_case_id="case-all-variants",
        )
        for variant in sorted(REQUIRED_ROBUSTNESS_VARIANTS)
    ]
    return [
        *robustness_rows,
        _phase2_row([API_A, API_B], group="VendorB/Model2"),
        _phase2_row([API_A, API_B, API_C], group="VendorC/Model3"),
        _phase2_row([], predicted=[], group="VendorEmpty/Model0", width=0),
    ]


def test_phase2_rows_cover_k_1_2_3_empty_set_and_all_robustness_variants() -> None:
    """Promotion metrics keep API targets unordered and robustness variant complete."""
    rows = _passing_rows()

    metrics = evaluate_phase2_rows(rows)

    assert metrics["rows"] == len(rows)
    assert metrics["json_parse_rate"] == 1.0
    assert metrics["set_exact_match_rate"] == 1.0
    assert metrics["set_exact_match_by_width"] == {
        "1": 1.0,
        "2": 1.0,
        "3": 1.0,
    }
    assert metrics["empty_set_rows"] == 1
    assert metrics["empty_set_exact_match_rate"] == 1.0
    assert metrics["robustness"] == {
        variant: True for variant in REQUIRED_ROBUSTNESS_VARIANTS
    }


def test_phase2_rows_penalize_duplicate_predictions_and_invalid_apis() -> None:
    """Duplicate and out-of-catalog predictions cannot be hidden by set scoring."""
    metrics = evaluate_phase2_rows([
        _phase2_row([API_A], predicted=[API_A, API_A], group="VendorDup/Model"),
        _phase2_row([API_B], predicted=[API_B, API_D], group="VendorInvalid/Model"),
    ])

    assert metrics["json_parse_rate"] == 1.0
    assert metrics["set_exact_match_rate"] == 0.0
    assert metrics["duplicate_api_rate"] == 0.5
    assert metrics["invalid_api_rate"] == 0.5
    assert metrics["precision"] == 0.75
    assert metrics["recall"] == 1.0


def test_phase2_rows_reject_duplicate_targets_before_promotion_metrics() -> None:
    """The held-out label itself must be a unique REST API set."""
    with pytest.raises(Phase2PromotionError, match="must be unique"):
        evaluate_phase2_rows([
            _phase2_row([API_A, API_A], predicted=[API_A]),
        ])


def test_phase2_rows_reject_identity_context_leakage_and_width_mismatch() -> None:
    """Held-out rows must be exact Phase 2/D1 rows with hidden target membership."""
    wrong_identity = _phase2_row([API_A])
    wrong_identity["dataset"] = "phase2_labelled_requests"
    with pytest.raises(Phase2PromotionError, match="Phase 2 D1 identity"):
        evaluate_phase2_rows([wrong_identity])

    wrong_semantics = _phase2_row([API_A])
    wrong_semantics["target_semantics"] = "ordered_rest_api_list"
    with pytest.raises(Phase2PromotionError, match="Phase 2 D1 identity"):
        evaluate_phase2_rows([wrong_semantics])

    missing_semantics = _phase2_row([API_A])
    missing_semantics.pop("target_semantics")
    with pytest.raises(Phase2PromotionError, match="Phase 2 D1 identity"):
        evaluate_phase2_rows([missing_semantics])

    leaked_target = _phase2_row([API_A])
    leaked_target["x"]["rest_api_list"] = [API_A]
    with pytest.raises(Phase2PromotionError, match="text and api_context"):
        evaluate_phase2_rows([leaked_target])

    leaked_membership = _phase2_row([API_A])
    leaked_membership["x"]["api_context"][0]["selected"] = True
    with pytest.raises(Phase2PromotionError, match="public context contract"):
        evaluate_phase2_rows([leaked_membership])

    too_few_distractors = _phase2_row([API_A])
    too_few_distractors["x"]["api_context"] = too_few_distractors["x"]["api_context"][:-1]
    with pytest.raises(Phase2PromotionError, match="at least 4 distractors"):
        evaluate_phase2_rows([too_few_distractors])

    width_mismatch = _phase2_row([API_A], width=2)
    with pytest.raises(Phase2PromotionError, match="target cardinality"):
        evaluate_phase2_rows([width_mismatch])


def test_phase2_promotion_enforces_vendor_or_model_floor() -> None:
    """A weak held-out vendor/model bucket fails even when global floors are relaxed."""
    rows = [
        *_passing_rows(),
        _phase2_row(
            [API_A, API_B, API_C],
            predicted=[API_A, API_B],
            group="VendorWeak/ModelX",
        ),
    ]

    result = evaluate_phase2_promotion(
        rows=rows,
        thresholds=_thresholds(
            min_set_exact_match_rate=0.0,
            min_set_exact_match_by_width={"1": 0.0, "2": 0.0, "3": 0.0},
        ),
        artifact_evidence=_artifact_evidence(heldout_rows=len(rows)),
        run_report=_run_report(),
        **_observed_evidence(),
    )

    assert result["status"] == "fail"
    assert any(
        failure["name"] == "min_heldout_vendor_or_model_set_exact"
        for failure in result["failures"]
    )


def test_phase2_promotion_requires_empty_set_and_min_rows_per_vendor_or_model() -> None:
    """Promotion requires empty-set evidence and enough rows per held-out bucket."""
    no_empty_rows = [
        row for row in _passing_rows() if row["y_true"]["rest_api_list"]
    ]

    no_empty_result = evaluate_phase2_promotion(
        rows=no_empty_rows,
        thresholds=_thresholds(
            min_set_exact_match_rate=0.0,
            min_set_exact_match_by_width={"1": 0.0, "2": 0.0, "3": 0.0},
        ),
        artifact_evidence=_artifact_evidence(heldout_rows=len(no_empty_rows)),
        run_report=_run_report(),
        **_observed_evidence(),
    )
    min_rows_result = evaluate_phase2_promotion(
        rows=_passing_rows(),
        thresholds=_thresholds(min_heldout_rows_per_vendor_or_model=2),
        artifact_evidence=_artifact_evidence(heldout_rows=len(_passing_rows())),
        run_report=_run_report(),
        **_observed_evidence(),
    )

    assert no_empty_result["status"] == "fail"
    assert any(
        failure["name"] == "empty_set_heldout_present"
        for failure in no_empty_result["failures"]
    )
    assert min_rows_result["status"] == "fail"
    assert any(
        failure["name"] == "min_heldout_rows_per_vendor_or_model"
        for failure in min_rows_result["failures"]
    )


def test_phase2_promotion_passes_with_exact_run_report_lineage() -> None:
    """Passing Phase 2 promotion returns the promoted goal_extractor identity."""
    rows = _passing_rows()

    result = evaluate_phase2_promotion(
        rows=rows,
        thresholds=_thresholds(),
        artifact_evidence=_artifact_evidence(heldout_rows=len(rows)),
        run_report=_run_report(),
        **_observed_evidence(),
    )

    assert result["status"] == "pass"
    assert result["role"] == "goal_extractor"
    assert result["artifact_sha"] == DIGEST


@pytest.mark.parametrize(
    "missing_key",
    [
        "immutable_full_manifest",
        "immutable_train_manifest",
        "disjoint_split_release",
        "real_promoted_parent_checkpoint",
        "real_heldout_data",
        "artifact_sha",
        "checkpoint_reload_succeeded",
        "inference_smoke_succeeded",
    ],
)
def test_phase2_promotion_requires_real_artifact_evidence(missing_key: str) -> None:
    """Promotion remains hard-gated on manifest, parent, held-out, and smoke evidence."""
    rows = _passing_rows()
    result = evaluate_phase2_promotion(
        rows=rows,
        thresholds=_thresholds(),
        artifact_evidence=_artifact_evidence(
            heldout_rows=len(rows),
            **{missing_key: False},
        ),
        run_report=_run_report(),
        **_observed_evidence(),
    )

    assert result["status"] == "fail"
    assert any(failure["name"] == missing_key for failure in result["failures"])


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
def test_phase2_promotion_rejects_observed_manifest_or_heldout_digest_mismatch(
    observed_overrides: dict[str, object],
    failure_name: str,
) -> None:
    """Promotion evidence must match the actual full/heldout manifest and JSONL digests."""
    rows = _passing_rows()

    result = evaluate_phase2_promotion(
        rows=rows,
        thresholds=_thresholds(),
        artifact_evidence=_artifact_evidence(heldout_rows=len(rows)),
        run_report=_run_report(),
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
def test_phase2_promotion_rejects_split_release_lineage_mismatch(
    split_release_update: dict[str, object],
) -> None:
    """Promotion binds train/held-out child manifests to one disjoint split release."""
    rows = _passing_rows()
    evidence = _artifact_evidence(heldout_rows=len(rows))
    evidence["disjoint_split_release"] = {
        **evidence["disjoint_split_release"],
        **split_release_update,
    }

    result = evaluate_phase2_promotion(
        rows=rows,
        thresholds=_thresholds(),
        artifact_evidence=evidence,
        run_report=_run_report(),
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
def test_phase2_promotion_rejects_run_report_train_lineage_mismatch(
    manifest_update: dict[str, object],
    failure_name: str,
) -> None:
    """Run report data_manifest is the train manifest and train_data_sha is train JSONL."""
    rows = _passing_rows()
    run_report = deepcopy(_run_report())
    run_report["manifest"].update(manifest_update)

    result = evaluate_phase2_promotion(
        rows=rows,
        thresholds=_thresholds(),
        artifact_evidence=_artifact_evidence(heldout_rows=len(rows)),
        run_report=run_report,
        **_observed_evidence(),
    )

    assert result["status"] == "fail"
    assert any(failure["name"] == failure_name for failure in result["failures"])


def test_phase2_promotion_rejects_heldout_row_count_mismatch() -> None:
    """real_heldout_data.rows must match the actual held-out JSONL row count."""
    rows = _passing_rows()

    result = evaluate_phase2_promotion(
        rows=rows,
        thresholds=_thresholds(),
        artifact_evidence=_artifact_evidence(heldout_rows=len(rows) + 1),
        run_report=_run_report(),
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
        ("phase", "phase2_wrong", "run_role_and_task_lineage"),
        ("task", "ordered_goals", "run_role_and_task_lineage"),
        ("parent_role", "foundation", "run_role_and_task_lineage"),
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
def test_phase2_promotion_requires_run_report_lineage_fields(
    field: str,
    value: object,
    failure_name: str,
) -> None:
    """Promotion requires exact training, model, task, train-data, and eval lineage."""
    rows = _passing_rows()
    run_report = deepcopy(_run_report())
    run_report["manifest"][field] = value

    result = evaluate_phase2_promotion(
        rows=rows,
        thresholds=_thresholds(),
        artifact_evidence=_artifact_evidence(heldout_rows=len(rows)),
        run_report=run_report,
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
def test_phase2_promotion_requires_promoted_best_checkpoint(
    manifest_overrides: dict[str, object],
) -> None:
    """The promoted goal_extractor artifact must come from the best checkpoint."""
    rows = _passing_rows()

    result = evaluate_phase2_promotion(
        rows=rows,
        thresholds=_thresholds(),
        artifact_evidence=_artifact_evidence(heldout_rows=len(rows)),
        run_report=_run_report(**manifest_overrides),
        **_observed_evidence(),
    )

    assert result["status"] == "fail"
    assert any(
        failure["name"] == "best_checkpoint_promoted"
        for failure in result["failures"]
    )


def test_phase2_promotion_requires_non_null_thresholds() -> None:
    """A missing floor is a configuration error, not a silent pass."""
    thresholds = _thresholds(min_set_exact_match_rate=None)

    with pytest.raises(Phase2PromotionError, match="missing non-null thresholds"):
        evaluate_phase2_promotion(
            rows=_passing_rows(),
            thresholds=thresholds,
            artifact_evidence=_artifact_evidence(),
            run_report=_run_report(),
            **_observed_evidence(),
        )
