"""Unit tests for Phase 1 model_x promotion gates."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from igc.modules.base.metric_keys import PHASE1_FINETUNE, phase_metric
from igc.modules.train.phase1_promotion import (
    Phase1PromotionError,
    evaluate_phase1_promotion,
)


SHA_A = "sha256:" + "a" * 64
SHA_B = "sha256:" + "b" * 64
SHA_C = "sha256:" + "c" * 64
SHA_D = "sha256:" + "d" * 64
SHA_E = "sha256:" + "e" * 64
SHA_F = "sha256:" + "f" * 64
SHA_1 = "sha256:" + "1" * 64
SHA_2 = "sha256:" + "2" * 64
SHA_3 = "sha256:" + "3" * 64
SHA_4 = "sha256:" + "4" * 64
MODEL_X_ARTIFACT_SHA = "sha256:" + "9" * 64
COMMIT_SHA = "0123456789abcdef0123456789abcdef01234567"
ROW_IDS = [SHA_1, SHA_2]
PARSE_KEY = phase_metric(PHASE1_FINETUNE, "eval", "json_parse_rate")
EXACT_KEY = phase_metric(PHASE1_FINETUNE, "eval", "json_exact_match_rate")
IDENTITY_KEY = phase_metric(PHASE1_FINETUNE, "eval", "odata_id_match_rate")
TARGET_P95_KEY = phase_metric(
    PHASE1_FINETUNE,
    "data",
    "target_completion_tokens_p95",
)
BUDGET_KEY = phase_metric(PHASE1_FINETUNE, "eval", "min_generation_budget")


def _thresholds(**overrides: object) -> dict[str, object]:
    thresholds: dict[str, object] = {
        "require_complete_heldout_manifest": True,
        "min_model_json_parse_rate": 0.995,
        "min_model_resource_identity_match_rate": 0.995,
        "min_exact_match_delta_vs_foundation": 0.02,
        "max_instruction_judge_accept_rate_drop": 0.03,
        "generation_target_token_margin": 64,
        "min_heldout_rows_per_corpus": 100,
        "small_corpus_policy": "require_all_available_rows",
        "deterministic_golden_json_parse_rate": 1.0,
        "deterministic_golden_resource_identity_match_rate": 1.0,
    }
    thresholds.update(overrides)
    return thresholds


def _metrics(*, min_generation_budget: int = 160) -> dict[str, object]:
    baseline_metrics = {
        PARSE_KEY: 1.0,
        IDENTITY_KEY: 1.0,
        TARGET_P95_KEY: 96.0,
        BUDGET_KEY: min_generation_budget,
    }
    model_metrics = {
        PARSE_KEY: 1.0,
        IDENTITY_KEY: 1.0,
        TARGET_P95_KEY: 96.0,
        BUDGET_KEY: min_generation_budget,
    }
    return {
        "metrics": {
            "baseline": baseline_metrics,
            "model_x": model_metrics,
        },
        "comparison": {"delta": {EXACT_KEY: 0.05}},
    }


def _evidence(
    *,
    corpus_b_rows: int = 100,
    include_corpus_b: bool = True,
    rows: int = 200,
    row_keys: list[str] | None = None,
    min_generation_budget: int = 160,
) -> dict:
    row_keys = list(ROW_IDS if row_keys is None else row_keys)
    metrics_payload = _metrics(min_generation_budget=min_generation_budget)
    by_corpus = {
        "corpus_a": {
            "rows": 100,
            "json_parse_rate": 1.0,
            "json_exact_match_rate": 0.95,
            "resource_identity_match_rate": 1.0,
        },
    }
    if include_corpus_b:
        by_corpus["corpus_b"] = {
            "rows": corpus_b_rows,
            "json_parse_rate": 1.0,
            "json_exact_match_rate": 0.90,
            "resource_identity_match_rate": 1.0,
        }
    return {
        "counts": {
            "model_x": {
                "rows": rows,
                "missing_target_rows": 0,
                "missing_prediction_rows": 0,
                "target_token_rows": rows,
                "generation_budget_rows": rows,
            },
        },
        "baseline": {
            "metrics": metrics_payload["metrics"]["baseline"],
            "by_corpus": {
                "corpus_a": {**by_corpus["corpus_a"], "json_exact_match_rate": 0.90},
                "corpus_b": {
                    "rows": 100,
                    "json_parse_rate": 1.0,
                    "json_exact_match_rate": 0.85,
                    "resource_identity_match_rate": 1.0,
                },
            },
            "row_keys": row_keys,
        },
        "model_x": {
            "metrics": metrics_payload["metrics"]["model_x"],
            "by_corpus": by_corpus,
            "row_keys": row_keys,
        },
        "comparison": metrics_payload["comparison"],
    }


def _row_ids(count: int) -> list[str]:
    return [f"sha256:{index:064x}" for index in range(count)]


def _single_corpus_evidence(
    *,
    corpus: str,
    rows: int,
    row_keys: list[str],
    baseline_rows: int | None = None,
    model_rows: int | None = None,
) -> dict[str, object]:
    metrics_payload = _metrics()
    baseline_count = rows if baseline_rows is None else baseline_rows
    model_count = rows if model_rows is None else model_rows
    return {
        "counts": {
            "model_x": {
                "rows": rows,
                "missing_target_rows": 0,
                "missing_prediction_rows": 0,
                "target_token_rows": rows,
                "generation_budget_rows": rows,
            },
        },
        "baseline": {
            "metrics": metrics_payload["metrics"]["baseline"],
            "by_corpus": {
                corpus: {
                    "rows": baseline_count,
                    "json_parse_rate": 1.0,
                    "json_exact_match_rate": 0.90,
                    "resource_identity_match_rate": 1.0,
                },
            },
            "row_keys": list(row_keys),
        },
        "model_x": {
            "metrics": metrics_payload["metrics"]["model_x"],
            "by_corpus": {
                corpus: {
                    "rows": model_count,
                    "json_parse_rate": 1.0,
                    "json_exact_match_rate": 0.95,
                    "resource_identity_match_rate": 1.0,
                },
            },
            "row_keys": list(row_keys),
        },
        "comparison": metrics_payload["comparison"],
    }


def _run_report() -> dict[str, object]:
    return {
        "manifest": {
            "phase": "phase1_finetune",
            "task": "redfish_json_reconstruction",
            "parent_role": "foundation_instruct",
            "output_role": "model_x",
            "data_manifest": SHA_A,
            "eval_split": SHA_D,
            "train_data_sha": SHA_E,
            "eval_data_sha": SHA_D,
            "source_manifest_sha": SHA_F,
            "source_registry_sha": SHA_1,
            "source_artifact_manifest_shas": {"corpus_a": SHA_2, "corpus_b": SHA_3},
            "foundation_model_sha": SHA_B,
            "tokenizer_sha": SHA_C,
            "git_commit": COMMIT_SHA,
            "checkpoint_path": "/runs/model_epoch_best.pt",
            "promoted_artifact_path": "/promoted/model_x",
            "promotion_source": "best_checkpoint",
            "training": {
                "optimizer_steps": 100,
                "train_loss": 0.12,
                "learning_rate": 0.0001,
            },
        },
        "metrics": {
            "eval_loss": 0.08,
            "eval_json_parse_rate": 1.0,
        },
    }


def _payload_overrides(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "thresholds": _thresholds(),
        "full_metrics": _metrics(),
        "full_evidence": _evidence(),
        "golden_metrics": _metrics(),
        "golden_evidence": _evidence(rows=10),
        "retention_evidence": {"comparison": {"delta": {"judge_acceptance_rate": -0.01}}},
        "run_report": _run_report(),
        "heldout_manifest": {
            "approved_heldout_rows": 200,
            "required_corpora": ["corpus_a", "corpus_b"],
            "rows_by_corpus": {"corpus_a": 100, "corpus_b": 100},
            "full_rows_by_corpus": {"corpus_a": 100, "corpus_b": 100},
            "row_ids": ROW_IDS,
            "artifact_sha256": SHA_D,
            "full_corpus_manifest_sha256": SHA_F,
            "source_registry_sha256": SHA_1,
            "source_manifest_shas": {"corpus_a": SHA_2, "corpus_b": SHA_3},
        },
        "load_evidence": {
            "status": "pass",
            "role": "model_x",
            "artifact_sha": MODEL_X_ARTIFACT_SHA,
            "adapter_dir": "/promoted/model_x",
        },
    }
    payload.update(overrides)
    return payload


def _evaluate(**overrides: object) -> dict[str, object]:
    return evaluate_phase1_promotion(**_payload_overrides(**overrides))


def _small_corpus_overrides(
    *,
    heldout_rows: int,
    complete_rows: int = 64,
    evidence_rows: int | None = None,
    baseline_rows: int | None = None,
    model_rows: int | None = None,
) -> dict[str, object]:
    corpus = "corpus_small"
    rows = heldout_rows if evidence_rows is None else evidence_rows
    row_ids = _row_ids(heldout_rows)
    run_report = _run_report()
    run_report["manifest"]["source_artifact_manifest_shas"] = {corpus: SHA_2}
    return {
        "full_evidence": _single_corpus_evidence(
            corpus=corpus,
            rows=rows,
            row_keys=row_ids,
            baseline_rows=baseline_rows,
            model_rows=model_rows,
        ),
        "heldout_manifest": {
            "approved_heldout_rows": heldout_rows,
            "required_corpora": [corpus],
            "rows_by_corpus": {corpus: heldout_rows},
            "full_rows_by_corpus": {corpus: complete_rows},
            "row_ids": row_ids,
            "artifact_sha256": SHA_D,
            "full_corpus_manifest_sha256": SHA_F,
            "source_registry_sha256": SHA_1,
            "source_manifest_shas": {corpus: SHA_2},
        },
        "run_report": run_report,
    }


def test_phase1_promotion_passes_required_corpora_min_n_and_generation_budget() -> None:
    """A promotion passes with all required corpora and p95+margin token budget."""
    result = _evaluate()

    assert result["status"] == "pass"
    assert result["role"] == "model_x"
    assert result["artifact_sha"] == MODEL_X_ARTIFACT_SHA
    assert result["failures"] == []


@pytest.mark.parametrize(
    "load_evidence",
    [
        {
            "status": "pass",
            "artifact_sha": MODEL_X_ARTIFACT_SHA,
            "adapter_dir": "/promoted/model_x",
        },
        {
            "status": "pass",
            "role": "baseline",
            "artifact_sha": MODEL_X_ARTIFACT_SHA,
            "adapter_dir": "/promoted/model_x",
        },
        {
            "status": "pass",
            "role": "model_x",
            "adapter_dir": "/promoted/model_x",
        },
        {
            "status": "pass",
            "role": "model_x",
            "artifact_sha": "not-a-sha",
            "adapter_dir": "/promoted/model_x",
        },
    ],
)
def test_phase1_promotion_requires_model_x_role_and_artifact_sha(
    load_evidence: dict[str, object],
) -> None:
    """Reload evidence must identify the promoted model_x artifact by canonical SHA."""
    result = _evaluate(load_evidence=load_evidence)

    assert result["status"] == "fail"
    assert any(
        failure["name"] == "checkpoint_reload_succeeded"
        for failure in result["failures"]
    )


def test_phase1_promotion_fails_when_required_corpus_is_missing() -> None:
    """The held-out manifest controls which corpus buckets must be present."""
    result = _evaluate(full_evidence=_evidence(include_corpus_b=False))

    assert result["status"] == "fail"
    assert any(
        failure["name"] == "per_corpus_breakdown_complete"
        for failure in result["failures"]
    )


def test_phase1_promotion_fails_min_rows_per_required_corpus() -> None:
    """Each required corpus bucket must meet the configured minimum row count."""
    result = _evaluate(full_evidence=_evidence(corpus_b_rows=99))

    assert result["status"] == "fail"
    assert any(
        failure["name"] == "min_heldout_rows_corpus_b"
        for failure in result["failures"]
    )


def test_phase1_promotion_allows_complete_small_corpus_below_configured_floor() -> None:
    """A complete 64-row corpus can satisfy a 100-row configured floor."""
    result = _evaluate(**_small_corpus_overrides(heldout_rows=64))
    floor_check = next(
        check
        for check in result["checks"]
        if check["name"] == "min_heldout_rows_corpus_small"
    )

    assert result["status"] == "pass"
    assert result["failures"] == []
    assert floor_check["observed"] == 64
    assert floor_check["threshold"] == 64


def test_phase1_promotion_rejects_small_corpus_heldout_subset() -> None:
    """A 10-row heldout subset of a 64-row corpus does not satisfy all-rows policy."""
    result = _evaluate(**_small_corpus_overrides(heldout_rows=10))

    assert result["status"] == "fail"
    assert any(
        failure["name"] == "min_heldout_rows_corpus_small"
        and failure["observed"] == 10
        and failure["threshold"] == 64
        for failure in result["failures"]
    )


def test_phase1_promotion_rejects_per_corpus_evidence_row_count_mismatch() -> None:
    """Per-corpus evidence rows must equal heldout manifest rows_by_corpus exactly."""
    overrides = _small_corpus_overrides(heldout_rows=64, model_rows=63)

    result = _evaluate(**overrides)

    assert result["status"] == "fail"
    assert any(
        failure["name"] == "per_corpus_row_counts_match_manifest"
        and failure["observed"] == {"corpus_small": 63}
        and failure["threshold"] == {"corpus_small": 64}
        for failure in result["failures"]
    )


def test_phase1_promotion_generation_budget_uses_p95_plus_margin() -> None:
    """The minimum generation budget must cover target-token p95 plus margin."""
    failing = _evaluate(full_metrics=_metrics(min_generation_budget=159))
    passing = _evaluate(full_metrics=_metrics(min_generation_budget=160))

    assert failing["status"] == "fail"
    assert any(
        failure["name"] == "generation_budget_covers_target_p95"
        and failure["observed"] == 159
        and failure["threshold"] == 160
        for failure in failing["failures"]
    )
    assert passing["status"] == "pass"


def test_phase1_promotion_requires_complete_target_and_budget_row_evidence() -> None:
    """Every held-out row needs target token and generation-budget evidence."""
    evidence = deepcopy(_evidence())
    evidence["counts"]["model_x"]["target_token_rows"] = 199
    evidence["counts"]["model_x"]["generation_budget_rows"] = 199

    result = _evaluate(full_evidence=evidence)

    assert result["status"] == "fail"
    assert any(
        failure["name"] == "complete_target_token_evidence"
        for failure in result["failures"]
    )
    assert any(
        failure["name"] == "complete_generation_budget_evidence"
        for failure in result["failures"]
    )


@pytest.mark.parametrize(
    "git_commit",
    [
        "abcdef1",
        "0123456789abcdef0123456789abcdef0123456",
        "0123456789abcdef0123456789abcdef0123456z",
        "",
    ],
)
def test_phase1_promotion_requires_exact_40_hex_git_commit(git_commit: str) -> None:
    """Promotion rejects abbreviated, short, non-hex, and missing git commits."""
    run_report = deepcopy(_run_report())
    run_report["manifest"]["git_commit"] = git_commit

    result = _evaluate(run_report=run_report)

    assert result["status"] == "fail"
    assert any(
        failure["name"] == "training_code_commit_exists"
        for failure in result["failures"]
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("phase", "phase1_pretraining"),
        ("task", "redfish_instruction_following"),
        ("parent_role", "model_x"),
        ("output_role", "planner"),
    ],
)
def test_phase1_promotion_requires_phase1_run_report_role_task_lineage(
    field: str,
    value: str,
) -> None:
    """Run reports must identify foundation_instruct -> model_x Phase 1 training."""
    run_report = deepcopy(_run_report())
    run_report["manifest"][field] = value

    result = _evaluate(run_report=run_report)

    assert result["status"] == "fail"
    assert any(
        failure["name"] == "run_role_and_task_lineage"
        for failure in result["failures"]
    )


def test_phase1_promotion_requires_positive_optimizer_steps() -> None:
    """A promoted Phase 1 run must report real optimizer progress."""
    run_report = deepcopy(_run_report())
    run_report["manifest"]["training"]["optimizer_steps"] = 0

    result = _evaluate(run_report=run_report)

    assert result["status"] == "fail"
    assert any(
        failure["name"] == "optimizer_steps_positive"
        for failure in result["failures"]
    )


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("training", "train_loss", float("nan")),
        ("metrics", "eval_loss", float("inf")),
    ],
)
def test_phase1_promotion_requires_finite_training_and_report_metrics(
    section: str,
    key: str,
    value: float,
) -> None:
    """Training and report metric payloads cannot contain NaN or infinity."""
    run_report = deepcopy(_run_report())
    if section == "training":
        run_report["manifest"]["training"][key] = value
    else:
        run_report["metrics"][key] = value

    result = _evaluate(run_report=run_report)

    assert result["status"] == "fail"
    assert any(
        failure["name"] == "finite_run_metrics"
        for failure in result["failures"]
    )


@pytest.mark.parametrize(
    ("field", "failure_name"),
    [
        ("data_manifest", "dataset_manifest_sha_exists"),
        ("eval_split", "eval_split_sha_exists"),
        ("train_data_sha", "train_data_sha_exists"),
        ("eval_data_sha", "eval_data_sha_exists"),
        ("source_manifest_sha", "source_manifest_sha_exists"),
        ("source_registry_sha", "source_registry_matches_heldout_approval"),
    ],
)
def test_phase1_promotion_requires_exact_run_report_lineage(
    field: str,
    failure_name: str,
) -> None:
    """Promotion requires exact semantic, train, eval, source, and registry digests."""
    run_report = deepcopy(_run_report())
    run_report["manifest"][field] = ""

    result = _evaluate(run_report=run_report)

    assert result["status"] == "fail"
    assert any(failure["name"] == failure_name for failure in result["failures"])


@pytest.mark.parametrize(
    ("manifest_overrides", "failure_name"),
    [
        ({"artifact_sha256": SHA_3}, "heldout_artifact_matches_run_report"),
        (
            {"full_corpus_manifest_sha256": SHA_3},
            "source_manifest_matches_heldout_approval",
        ),
        ({"source_registry_sha256": SHA_3}, "source_registry_matches_heldout_approval"),
        (
            {"source_manifest_shas": {"corpus_a": SHA_4}},
            "source_artifact_manifests_match_heldout_approval",
        ),
    ],
)
def test_phase1_promotion_requires_heldout_manifest_lineage_to_match_run_report(
    manifest_overrides: dict[str, object],
    failure_name: str,
) -> None:
    """Held-out approval digests must match the report generated by the train run."""
    heldout_manifest = dict(_payload_overrides()["heldout_manifest"])
    heldout_manifest.update(manifest_overrides)

    result = _evaluate(heldout_manifest=heldout_manifest)

    assert result["status"] == "fail"
    assert any(failure["name"] == failure_name for failure in result["failures"])


def test_phase1_promotion_rejects_negative_generation_margin() -> None:
    """Promotion thresholds must be finite non-negative numbers."""
    with pytest.raises(Phase1PromotionError, match="generation_target_token_margin"):
        _evaluate(thresholds=_thresholds(generation_target_token_margin=-1))


def test_phase1_golden_acceptance_config_declares_new_run_report_hard_checks() -> None:
    """The checked-in golden acceptance spec names the new run-report gates."""
    spec = yaml.safe_load(
        Path("configs/inference/phase1_golden_acceptance.yaml").read_text(
            encoding="utf-8",
        ),
    )

    assert {
        "run_role_and_task_lineage",
        "optimizer_steps_positive",
        "finite_run_metrics",
    } <= set(spec["hard_checks"])
