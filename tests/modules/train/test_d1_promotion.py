"""Unit tests for D1 promotion gates."""

from __future__ import annotations

import pytest

from igc.modules.base.metric_keys import PHASE2_LABELLED_REQUESTS, phase_metric
from igc.modules.train.d1_promotion import D1PromotionError, evaluate_d1_promotion


DIGEST = "sha256:" + "c" * 64
PARENT_DIGEST = "sha256:" + "d" * 64
MANIFEST_DIGEST = "sha256:" + "e" * 64
HELDOUT_MANIFEST_DIGEST = "sha256:" + "f" * 64
HELDOUT_JSONL_DIGEST = "sha256:" + "a" * 64
HELDOUT_ROWS = 3
EMPTY_SET_ACCEPTED_ROWS = 100
D1_RELEASE_ROWS = EMPTY_SET_ACCEPTED_ROWS + 3


def _build_metrics(**overrides: float) -> dict[str, object]:
    width_metrics = {
        phase_metric(PHASE2_LABELLED_REQUESTS, "pro_accept_rate"): 0.95,
        phase_metric(PHASE2_LABELLED_REQUESTS, "rest_api_set_match_rate"): 0.99,
        phase_metric(PHASE2_LABELLED_REQUESTS, "nonsense_rate"): 0.0,
        phase_metric(PHASE2_LABELLED_REQUESTS, "invalid_json_rate"): 0.0,
    }
    metrics = {
        **width_metrics,
        "by_sample_width": {
            "0": dict(width_metrics),
            "1": dict(width_metrics),
            "2": dict(width_metrics),
            "3": dict(width_metrics),
        },
    }
    for key, value in overrides.items():
        if key == "by_sample_width":
            metrics[key] = value
        else:
            metrics[key] = value
            for sample_width in ("0", "1", "2", "3"):
                metrics["by_sample_width"][sample_width][key] = value
    return metrics


def _calibration_metrics(**overrides: float) -> dict[str, float]:
    metrics = {
        "precision": 0.995,
        "recall": 0.92,
        "false_accept_rate": 0.005,
        "positive_examples": 5,
        "negative_examples": 5,
        "examples": 10,
    }
    metrics.update(overrides)
    return metrics


def _release_manifest(**overrides: object) -> dict[str, object]:
    manifest: dict[str, object] = {
        "schema_version": "d1_release.v1",
        "dataset": "D1",
        "artifact_sha256": DIGEST,
        "rows": D1_RELEASE_ROWS,
        "sample_width_counts": {
            "0": EMPTY_SET_ACCEPTED_ROWS,
            "1": 1,
            "2": 1,
            "3": 1,
        },
        "balance_valid": True,
        "judge_evidence_valid": True,
        "immutable": True,
        "complete": True,
        "draft_provider_adapter": "openai-compatible",
        "judge_provider_adapter": "openai-compatible",
        "judge_route": "private-pro-route",
        "judge_model": "deepseek-v4-pro",
        "judge_profile": "phase2-d1-hardening",
        "model_x_artifact_sha": PARENT_DIGEST,
    }
    manifest.update(overrides)
    return manifest


def _observed_evidence(**overrides: object) -> dict[str, object]:
    evidence = {
        "observed_full_manifest_sha": MANIFEST_DIGEST,
        "observed_heldout_manifest_sha": HELDOUT_MANIFEST_DIGEST,
        "observed_heldout_sha": HELDOUT_JSONL_DIGEST,
        "observed_heldout_rows": HELDOUT_ROWS,
    }
    evidence.update(overrides)
    return evidence


def _artifact_evidence(**overrides: object) -> dict[str, object]:
    evidence: dict[str, object] = {
        "immutable_full_manifest": {
            "immutable": True,
            "complete": True,
            "rows": D1_RELEASE_ROWS,
            "sha256": MANIFEST_DIGEST,
            "artifact_sha": DIGEST,
        },
        "real_promoted_parent_checkpoint": {
            "role": "model_x",
            "promotion_status": "pass",
            "artifact_sha": PARENT_DIGEST,
        },
        "real_heldout_data": {
            "real": True,
            "split": "heldout",
            "rows": HELDOUT_ROWS,
            "manifest_sha": HELDOUT_MANIFEST_DIGEST,
            "artifact_sha": HELDOUT_JSONL_DIGEST,
        },
        "artifact_sha": DIGEST,
        "checkpoint_reload": {"status": "pass", "artifact_sha": PARENT_DIGEST},
        "inference_smoke": {"status": "pass", "artifact_sha": PARENT_DIGEST},
    }
    for key, value in overrides.items():
        if key == "immutable_full_manifest" and value is False:
            evidence["immutable_full_manifest"] = {
                **evidence["immutable_full_manifest"],
                "complete": False,
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


def _build_thresholds(**overrides: object) -> dict[str, object]:
    thresholds: dict[str, object] = {
        "min_pro_accept_rate": 0.9,
        "min_rest_api_set_match_rate": 0.98,
        "max_nonsense_rate": 0.01,
        "max_invalid_json_rate": 0.01,
        "min_empty_set_accepted_rows": EMPTY_SET_ACCEPTED_ROWS,
    }
    thresholds.update(overrides)
    return thresholds


def _calibration_thresholds(**overrides: object) -> dict[str, object]:
    thresholds: dict[str, object] = {
        "min_precision": 0.99,
        "min_recall": 0.90,
        "max_false_accept_rate": 0.01,
    }
    thresholds.update(overrides)
    return thresholds


def test_d1_promotion_passes_with_release_lineage_quality_and_smoke_evidence() -> None:
    """A D1 release promotes only with real providers, matching sha, and smoke evidence."""
    result = evaluate_d1_promotion(
        release_manifest=_release_manifest(),
        observed_artifact_sha=DIGEST,
        build_metrics=_build_metrics(),
        calibration_metrics=_calibration_metrics(),
        build_thresholds=_build_thresholds(),
        calibration_thresholds=_calibration_thresholds(),
        artifact_evidence=_artifact_evidence(),
        **_observed_evidence(),
    )

    assert result["schema_version"] == "d1_promotion.v1"
    assert result["status"] == "pass"
    assert result["failures"] == []
    assert any(
        check["name"] == "calibration_has_positive_examples"
        and check["observed"] is True
        for check in result["checks"]
    )
    assert any(
        check["name"] == "calibration_has_negative_examples"
        and check["observed"] is True
        for check in result["checks"]
    )
    assert any(
        check["name"] == "release_contains_balanced_k1_k2_k3"
        and check["observed"] is True
        for check in result["checks"]
    )
    assert any(
        check["name"] == "release_contains_bounded_empty_set_negatives"
        and check["observed"] == EMPTY_SET_ACCEPTED_ROWS
        and check["threshold"] == EMPTY_SET_ACCEPTED_ROWS
        for check in result["checks"]
    )
    assert any(
        check["name"] == "min_pro_accept_rate_k0"
        and check["passed"] is True
        for check in result["checks"]
    )


@pytest.mark.parametrize(
    ("sample_width_counts", "observed"),
    [
        ({"1": 1, "2": 1, "3": 1}, None),
        (
            {"0": EMPTY_SET_ACCEPTED_ROWS - 1, "1": 1, "2": 1, "3": 1},
            EMPTY_SET_ACCEPTED_ROWS - 1,
        ),
    ],
)
def test_d1_promotion_rejects_missing_or_insufficient_empty_set_width(
    sample_width_counts: dict[str, int],
    observed: int | None,
) -> None:
    """D1 promotion requires k=0 rows to meet the configured empty-set floor."""
    result = evaluate_d1_promotion(
        release_manifest=_release_manifest(sample_width_counts=sample_width_counts),
        observed_artifact_sha=DIGEST,
        build_metrics=_build_metrics(),
        calibration_metrics=_calibration_metrics(),
        build_thresholds=_build_thresholds(),
        calibration_thresholds=_calibration_thresholds(),
        artifact_evidence=_artifact_evidence(),
        **_observed_evidence(),
    )

    assert result["status"] == "fail"
    failure = next(
        item for item in result["failures"]
        if item["name"] == "release_contains_bounded_empty_set_negatives"
    )
    assert failure["observed"] == observed
    assert failure["threshold"] == EMPTY_SET_ACCEPTED_ROWS


def test_d1_promotion_balances_positive_widths_independently_from_k0() -> None:
    """A passing k0 floor does not mask imbalance across positive widths 1/2/3."""
    result = evaluate_d1_promotion(
        release_manifest=_release_manifest(
            sample_width_counts={
                "0": EMPTY_SET_ACCEPTED_ROWS,
                "1": 3,
                "2": 1,
                "3": 1,
            },
        ),
        observed_artifact_sha=DIGEST,
        build_metrics=_build_metrics(),
        calibration_metrics=_calibration_metrics(),
        build_thresholds=_build_thresholds(),
        calibration_thresholds=_calibration_thresholds(),
        artifact_evidence=_artifact_evidence(),
        **_observed_evidence(),
    )

    assert result["status"] == "fail"
    assert any(
        failure["name"] == "release_contains_balanced_k1_k2_k3"
        for failure in result["failures"]
    )
    assert not any(
        failure["name"] == "release_contains_bounded_empty_set_negatives"
        for failure in result["failures"]
    )


def test_d1_promotion_requires_by_sample_width_metrics_for_k0_k1_k2_k3() -> None:
    """Canonical build metrics must report all four sample widths, including k=0."""
    metrics = _build_metrics()
    del metrics["by_sample_width"]["0"]

    with pytest.raises(D1PromotionError, match="by_sample_width for k=0,1,2,3"):
        evaluate_d1_promotion(
            release_manifest=_release_manifest(),
            observed_artifact_sha=DIGEST,
            build_metrics=metrics,
            calibration_metrics=_calibration_metrics(),
            build_thresholds=_build_thresholds(),
            calibration_thresholds=_calibration_thresholds(),
            artifact_evidence=_artifact_evidence(),
            **_observed_evidence(),
        )


@pytest.mark.parametrize(
    ("manifest_overrides", "failure_name"),
    [
        (
            {"draft_provider_adapter": "mock"},
            "release_used_real_model_x_provider",
        ),
        (
            {"draft_provider_adapter": "file"},
            "release_used_real_model_x_provider",
        ),
        (
            {"judge_provider_adapter": "mock"},
            "release_used_real_judge_provider",
        ),
        (
            {"judge_provider_adapter": "file"},
            "release_used_real_judge_provider",
        ),
        (
            {"judge_route": "${PHASE2_JUDGE_ROUTE}"},
            "release_has_resolved_judge_identity",
        ),
        (
            {"judge_model": "${IGC_D1_JUDGE_MODEL}"},
            "release_has_resolved_judge_identity",
        ),
        (
            {"judge_profile": ""},
            "release_has_resolved_judge_identity",
        ),
        (
            {"model_x_artifact_sha": "sha256:" + "9" * 64},
            "release_model_x_matches_promoted_parent",
        ),
    ],
)
def test_d1_promotion_rejects_hardened_release_provenance_failures(
    manifest_overrides: dict[str, object],
    failure_name: str,
) -> None:
    """D1 promotion rejects mock/file providers, unresolved judge IDs, and parent SHA drift."""
    result = evaluate_d1_promotion(
        release_manifest=_release_manifest(**manifest_overrides),
        observed_artifact_sha=DIGEST,
        build_metrics=_build_metrics(),
        calibration_metrics=_calibration_metrics(),
        build_thresholds=_build_thresholds(),
        calibration_thresholds=_calibration_thresholds(),
        artifact_evidence=_artifact_evidence(),
        **_observed_evidence(),
    )

    assert result["status"] == "fail"
    assert any(failure["name"] == failure_name for failure in result["failures"])


def test_d1_promotion_rejects_missing_judge_route() -> None:
    """D1 release manifests must carry the resolved private judge route."""
    manifest = _release_manifest()
    del manifest["judge_route"]

    result = evaluate_d1_promotion(
        release_manifest=manifest,
        observed_artifact_sha=DIGEST,
        build_metrics=_build_metrics(),
        calibration_metrics=_calibration_metrics(),
        build_thresholds=_build_thresholds(),
        calibration_thresholds=_calibration_thresholds(),
        artifact_evidence=_artifact_evidence(),
        **_observed_evidence(),
    )

    assert result["status"] == "fail"
    assert any(
        failure["name"] == "release_has_resolved_judge_identity"
        for failure in result["failures"]
    )


@pytest.mark.parametrize(
    "evidence_key",
    [
        "immutable_full_manifest",
        "real_promoted_parent_checkpoint",
        "real_heldout_data",
        "artifact_sha",
        "checkpoint_reload_succeeded",
        "inference_smoke_succeeded",
    ],
)
def test_d1_promotion_requires_real_artifact_and_checkpoint_evidence(
    evidence_key: str,
) -> None:
    """Manifest, parent checkpoint, held-out data, reload, and smoke gates are hard."""
    result = evaluate_d1_promotion(
        release_manifest=_release_manifest(),
        observed_artifact_sha=DIGEST,
        build_metrics=_build_metrics(),
        calibration_metrics=_calibration_metrics(),
        build_thresholds=_build_thresholds(),
        calibration_thresholds=_calibration_thresholds(),
        artifact_evidence=_artifact_evidence(**{evidence_key: False}),
        **_observed_evidence(),
    )

    assert result["status"] == "fail"
    assert any(failure["name"] == evidence_key for failure in result["failures"])


@pytest.mark.parametrize(
    ("observed_overrides", "failure_name"),
    [
        (
            {"observed_full_manifest_sha": "sha256:" + "1" * 64},
            "full_manifest_sha_matches_file",
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
            {"observed_heldout_rows": HELDOUT_ROWS + 1},
            "heldout_row_count_matches_file",
        ),
    ],
)
def test_d1_promotion_rejects_observed_manifest_or_heldout_mismatch(
    observed_overrides: dict[str, object],
    failure_name: str,
) -> None:
    """D1 promotion evidence must match observed manifest/heldout files exactly."""
    result = evaluate_d1_promotion(
        release_manifest=_release_manifest(),
        observed_artifact_sha=DIGEST,
        build_metrics=_build_metrics(),
        calibration_metrics=_calibration_metrics(),
        build_thresholds=_build_thresholds(),
        calibration_thresholds=_calibration_thresholds(),
        artifact_evidence=_artifact_evidence(),
        **_observed_evidence(**observed_overrides),
    )

    assert result["status"] == "fail"
    assert any(failure["name"] == failure_name for failure in result["failures"])


def test_d1_promotion_rejects_non_matching_or_non_digest_release_artifacts() -> None:
    """The observed JSONL digest must exactly match the immutable release manifest."""
    mismatch = evaluate_d1_promotion(
        release_manifest=_release_manifest(),
        observed_artifact_sha="sha256:" + "e" * 64,
        build_metrics=_build_metrics(),
        calibration_metrics=_calibration_metrics(),
        build_thresholds=_build_thresholds(),
        calibration_thresholds=_calibration_thresholds(),
        artifact_evidence=_artifact_evidence(),
        **_observed_evidence(),
    )
    invalid_manifest = evaluate_d1_promotion(
        release_manifest=_release_manifest(artifact_sha256="not-a-digest"),
        observed_artifact_sha=DIGEST,
        build_metrics=_build_metrics(),
        calibration_metrics=_calibration_metrics(),
        build_thresholds=_build_thresholds(),
        calibration_thresholds=_calibration_thresholds(),
        artifact_evidence=_artifact_evidence(),
        **_observed_evidence(),
    )

    assert mismatch["status"] == "fail"
    assert invalid_manifest["status"] == "fail"
    assert any(
        failure["name"] == "artifact_sha_matches_release"
        for failure in mismatch["failures"]
    )
    assert any(
        failure["name"] == "artifact_sha_matches_release"
        for failure in invalid_manifest["failures"]
    )


@pytest.mark.parametrize(
    "build_thresholds, calibration_thresholds",
    [
        (_build_thresholds(min_pro_accept_rate=None), _calibration_thresholds()),
        (
            _build_thresholds(min_empty_set_accepted_rows=None),
            _calibration_thresholds(),
        ),
        (_build_thresholds(), _calibration_thresholds(min_precision=None)),
    ],
)
def test_d1_promotion_requires_non_null_build_and_calibration_thresholds(
    build_thresholds: dict[str, object],
    calibration_thresholds: dict[str, object],
) -> None:
    """D1 promotion blocks missing quality floors instead of interpreting them."""
    with pytest.raises(D1PromotionError, match="missing non-null quality thresholds"):
        evaluate_d1_promotion(
            release_manifest=_release_manifest(),
            observed_artifact_sha=DIGEST,
            build_metrics=_build_metrics(),
            calibration_metrics=_calibration_metrics(),
            build_thresholds=build_thresholds,
            calibration_thresholds=calibration_thresholds,
            artifact_evidence=_artifact_evidence(),
            **_observed_evidence(),
        )


@pytest.mark.parametrize(
    ("metric_overrides", "calibration_overrides", "failure_name"),
    [
        (
            {phase_metric(PHASE2_LABELLED_REQUESTS, "pro_accept_rate"): 0.89},
            {},
            "min_pro_accept_rate_k0",
        ),
        (
            {phase_metric(PHASE2_LABELLED_REQUESTS, "rest_api_set_match_rate"): 0.97},
            {},
            "min_rest_api_set_match_rate_k0",
        ),
        ({}, {"precision": 0.98}, "min_precision"),
        ({}, {"false_accept_rate": 0.02}, "max_false_accept_rate"),
        (
            {},
            {"positive_examples": 0, "examples": 5},
            "calibration_has_positive_examples",
        ),
        (
            {},
            {"negative_examples": 0, "examples": 5},
            "calibration_has_negative_examples",
        ),
    ],
)
def test_d1_promotion_applies_build_and_judge_calibration_floors(
    metric_overrides: dict[str, float],
    calibration_overrides: dict[str, float],
    failure_name: str,
) -> None:
    """Build metrics and judge calibration metrics are both promotion blockers."""
    build_metrics = _build_metrics()
    build_metrics.update(metric_overrides)
    for width_metrics in build_metrics["by_sample_width"].values():
        width_metrics.update(metric_overrides)
    calibration_metrics = _calibration_metrics()
    calibration_metrics.update(calibration_overrides)

    result = evaluate_d1_promotion(
        release_manifest=_release_manifest(),
        observed_artifact_sha=DIGEST,
        build_metrics=build_metrics,
        calibration_metrics=calibration_metrics,
        build_thresholds=_build_thresholds(),
        calibration_thresholds=_calibration_thresholds(),
        artifact_evidence=_artifact_evidence(),
        **_observed_evidence(),
    )

    assert result["status"] == "fail"
    assert any(failure["name"] == failure_name for failure in result["failures"])
