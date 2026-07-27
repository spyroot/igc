"""Hard promotion checks for one accepted immutable Phase 1 model_x artifact."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

from igc.modules.base.metric_keys import PHASE1_FINETUNE, phase_metric


class Phase1PromotionError(ValueError):
    """Raised when promotion evidence is malformed."""


def evaluate_phase1_promotion(
    *,
    thresholds: Mapping[str, Any],
    full_metrics: Mapping[str, Any],
    full_evidence: Mapping[str, Any],
    golden_metrics: Mapping[str, Any],
    golden_evidence: Mapping[str, Any],
    retention_evidence: Mapping[str, Any],
    run_report: Mapping[str, Any],
    heldout_manifest: Mapping[str, Any],
    load_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate hard lineage/reload checks and config-driven quality floors."""
    checks: list[dict[str, Any]] = []
    full_model = _mapping(_mapping(full_metrics, "metrics"), "model_x")
    golden_model = _mapping(_mapping(golden_metrics, "metrics"), "model_x")
    full_comparison = _mapping(full_metrics, "comparison")
    full_delta = _mapping(full_comparison, "delta")
    retention_comparison = _mapping(retention_evidence, "comparison")
    retention_delta = _mapping(retention_comparison, "delta")
    report_manifest = _mapping(run_report, "manifest")
    report_training = _mapping(report_manifest, "training")
    report_metrics = _mapping(run_report, "metrics")

    full_counts = _model_counts(full_evidence)
    golden_counts = _model_counts(golden_evidence)
    full_baseline_evidence = _mapping(full_evidence, "baseline")
    full_model_evidence = _mapping(full_evidence, "model_x")
    golden_baseline_evidence = _mapping(golden_evidence, "baseline")
    golden_model_evidence = _mapping(golden_evidence, "model_x")
    baseline_by_corpus = _mapping(full_baseline_evidence, "by_corpus")
    model_by_corpus = _mapping(full_model_evidence, "by_corpus")
    required_corpora = _required_corpora(heldout_manifest)
    heldout_rows_by_corpus = _corpus_counts(heldout_manifest, "rows_by_corpus")
    full_rows_by_corpus = _corpus_counts(heldout_manifest, "full_rows_by_corpus")
    if not set(required_corpora) <= set(heldout_rows_by_corpus):
        raise Phase1PromotionError(
            "held-out manifest rows_by_corpus is missing a required corpus"
        )
    if not set(required_corpora) <= set(full_rows_by_corpus):
        raise Phase1PromotionError(
            "held-out manifest full_rows_by_corpus is missing a required corpus"
        )
    minimum_rows_per_corpus = _nonnegative_threshold(
        thresholds,
        "min_heldout_rows_per_corpus",
    )
    if thresholds.get("small_corpus_policy") != "require_all_available_rows":
        raise Phase1PromotionError(
            "small_corpus_policy must equal require_all_available_rows"
        )
    generation_margin = _nonnegative_threshold(
        thresholds,
        "generation_target_token_margin",
    )
    _hard(
        checks,
        "checkpoint_reload_succeeded",
        load_evidence.get("status") == "pass"
        and load_evidence.get("role") == "model_x"
        and _is_digest(load_evidence.get("artifact_sha")),
    )
    _hard(
        checks,
        "no_missing_target_rows",
        full_counts.get("missing_target_rows") == 0
        and golden_counts.get("missing_target_rows") == 0,
    )
    _hard(
        checks,
        "no_missing_prediction_rows",
        full_counts.get("missing_prediction_rows") == 0
        and golden_counts.get("missing_prediction_rows") == 0,
    )
    _hard(
        checks,
        "complete_target_token_evidence",
        full_counts.get("target_token_rows") == full_counts.get("rows"),
        observed=full_counts.get("target_token_rows"),
        threshold=full_counts.get("rows"),
    )
    _hard(
        checks,
        "complete_generation_budget_evidence",
        full_counts.get("generation_budget_rows") == full_counts.get("rows"),
        observed=full_counts.get("generation_budget_rows"),
        threshold=full_counts.get("rows"),
    )
    _hard(checks, "run_report_exists", bool(report_manifest))
    _hard(
        checks,
        "run_role_and_task_lineage",
        report_manifest.get("phase") == "phase1_finetune"
        and report_manifest.get("task") == "redfish_json_reconstruction"
        and report_manifest.get("parent_role") == "foundation_instruct"
        and report_manifest.get("output_role") == "model_x",
    )
    _hard(
        checks,
        "dataset_manifest_sha_exists",
        _is_digest(report_manifest.get("data_manifest")),
    )
    _hard(
        checks,
        "eval_split_sha_exists",
        _is_digest(report_manifest.get("eval_split")),
    )
    _hard(
        checks,
        "train_data_sha_exists",
        _is_digest(report_manifest.get("train_data_sha")),
    )
    _hard(
        checks,
        "eval_data_sha_exists",
        _is_digest(report_manifest.get("eval_data_sha")),
    )
    _hard(
        checks,
        "source_manifest_sha_exists",
        _is_digest(report_manifest.get("source_manifest_sha")),
    )
    _hard(
        checks,
        "eval_split_matches_eval_data",
        report_manifest.get("eval_split") == report_manifest.get("eval_data_sha"),
    )
    _hard(
        checks,
        "heldout_artifact_matches_run_report",
        heldout_manifest.get("artifact_sha256")
        == report_manifest.get("eval_data_sha"),
    )
    _hard(
        checks,
        "source_manifest_matches_heldout_approval",
        heldout_manifest.get("full_corpus_manifest_sha256")
        == report_manifest.get("source_manifest_sha"),
    )
    _hard(
        checks,
        "source_registry_matches_heldout_approval",
        _is_digest(report_manifest.get("source_registry_sha"))
        and heldout_manifest.get("source_registry_sha256")
        == report_manifest.get("source_registry_sha"),
    )
    _hard(
        checks,
        "source_artifact_manifests_match_heldout_approval",
        heldout_manifest.get("source_manifest_shas")
        == report_manifest.get("source_artifact_manifest_shas"),
    )
    _hard(
        checks,
        "full_metrics_match_evidence",
        _mapping(_mapping(full_metrics, "metrics"), "baseline")
        == _mapping(full_baseline_evidence, "metrics")
        and full_model == _mapping(full_model_evidence, "metrics")
        and full_comparison == _mapping(full_evidence, "comparison"),
    )
    _hard(
        checks,
        "golden_metrics_match_evidence",
        _mapping(_mapping(golden_metrics, "metrics"), "baseline")
        == _mapping(golden_baseline_evidence, "metrics")
        and golden_model == _mapping(golden_model_evidence, "metrics")
        and _mapping(golden_metrics, "comparison")
        == _mapping(golden_evidence, "comparison"),
    )
    _hard(
        checks,
        "foundation_model_sha_exists",
        _is_digest(report_manifest.get("foundation_model_sha")),
    )
    _hard(
        checks,
        "tokenizer_sha_exists",
        _is_digest(report_manifest.get("tokenizer_sha")),
    )
    _hard(
        checks,
        "training_code_commit_exists",
        _is_commit(report_manifest.get("git_commit")),
    )
    _hard(
        checks,
        "optimizer_steps_positive",
        isinstance(report_training.get("optimizer_steps"), int)
        and not isinstance(report_training.get("optimizer_steps"), bool)
        and report_training["optimizer_steps"] > 0,
    )
    _hard(
        checks,
        "finite_run_metrics",
        _all_finite_metrics(report_training)
        and _all_finite_metrics(report_metrics),
    )
    _hard(
        checks,
        "best_checkpoint_promoted",
        _best_checkpoint_promoted(report_manifest, load_evidence),
    )

    expected_rows = _heldout_rows(heldout_manifest)
    expected_row_ids = _row_ids(heldout_manifest)
    _hard(
        checks,
        "full_prediction_rows_match_heldout_manifest",
        full_baseline_evidence.get("row_keys") == expected_row_ids
        and full_model_evidence.get("row_keys") == expected_row_ids,
    )
    _hard(
        checks,
        "complete_heldout_manifest",
        not bool(thresholds.get("require_complete_heldout_manifest", False))
        or full_counts.get("rows") == expected_rows,
        observed=full_counts.get("rows"),
        threshold=expected_rows,
    )
    _hard(
        checks,
        "per_corpus_breakdown_complete",
        set(required_corpora) <= set(baseline_by_corpus)
        and set(required_corpora) <= set(model_by_corpus),
        observed=sorted(set(model_by_corpus)),
        threshold=list(required_corpora),
    )
    _hard(
        checks,
        "per_corpus_row_counts_match_manifest",
        all(
            baseline_by_corpus.get(corpus, {}).get("rows")
            == heldout_rows_by_corpus.get(corpus)
            and model_by_corpus.get(corpus, {}).get("rows")
            == heldout_rows_by_corpus.get(corpus)
            for corpus in required_corpora
        ),
        observed={
            corpus: model_by_corpus.get(corpus, {}).get("rows")
            for corpus in required_corpora
        },
        threshold={
            corpus: heldout_rows_by_corpus[corpus]
            for corpus in required_corpora
        },
    )

    parse_key = phase_metric(PHASE1_FINETUNE, "eval", "json_parse_rate")
    exact_key = phase_metric(PHASE1_FINETUNE, "eval", "json_exact_match_rate")
    identity_key = phase_metric(PHASE1_FINETUNE, "eval", "odata_id_match_rate")
    target_p95_key = phase_metric(
        PHASE1_FINETUNE,
        "data",
        "target_completion_tokens_p95",
    )
    generation_budget_key = phase_metric(
        PHASE1_FINETUNE,
        "eval",
        "min_generation_budget",
    )
    target_tokens_p95 = full_model.get(target_p95_key)
    minimum_generation_budget = full_model.get(generation_budget_key)
    finite_values = (
        full_model.get(parse_key),
        full_model.get(identity_key),
        full_delta.get(exact_key),
        golden_model.get(parse_key),
        golden_model.get(identity_key),
        retention_delta.get("judge_acceptance_rate"),
        target_tokens_p95,
        minimum_generation_budget,
    )
    _hard(
        checks,
        "finite_promotion_metrics",
        all(_finite(value) for value in finite_values),
    )
    required_generation_budget = (
        math.ceil(float(target_tokens_p95) + generation_margin)
        if _finite(target_tokens_p95)
        else None
    )
    _hard(
        checks,
        "generation_budget_covers_target_p95",
        _finite(minimum_generation_budget)
        and required_generation_budget is not None
        and float(minimum_generation_budget) >= required_generation_budget,
        observed=minimum_generation_budget,
        threshold=required_generation_budget,
    )
    _minimum(
        checks,
        "min_model_json_parse_rate",
        full_model.get(parse_key),
        thresholds.get("min_model_json_parse_rate"),
    )
    _minimum(
        checks,
        "min_model_resource_identity_match_rate",
        full_model.get(identity_key),
        thresholds.get("min_model_resource_identity_match_rate"),
    )
    _minimum(
        checks,
        "min_exact_match_delta_vs_foundation",
        full_delta.get(exact_key),
        thresholds.get("min_exact_match_delta_vs_foundation"),
    )
    observed_drop = -float(retention_delta.get("judge_acceptance_rate", float("nan")))
    _maximum(
        checks,
        "max_instruction_judge_accept_rate_drop",
        observed_drop,
        thresholds.get("max_instruction_judge_accept_rate_drop"),
    )
    _minimum(
        checks,
        "deterministic_golden_json_parse_rate",
        golden_model.get(parse_key),
        thresholds.get("deterministic_golden_json_parse_rate"),
    )
    _minimum(
        checks,
        "deterministic_golden_resource_identity_match_rate",
        golden_model.get(identity_key),
        thresholds.get("deterministic_golden_resource_identity_match_rate"),
    )
    for corpus in required_corpora:
        baseline_corpus = baseline_by_corpus.get(corpus, {})
        model_corpus = model_by_corpus.get(corpus, {})
        corpus_row_floor = min(
            minimum_rows_per_corpus,
            full_rows_by_corpus[corpus],
        )
        _minimum_value(
            checks,
            f"min_heldout_rows_{corpus}",
            model_corpus.get("rows"),
            corpus_row_floor,
        )
        _minimum_value(
            checks,
            f"min_json_parse_rate_{corpus}",
            model_corpus.get("json_parse_rate"),
            thresholds.get("min_model_json_parse_rate"),
        )
        _minimum_value(
            checks,
            f"min_resource_identity_match_rate_{corpus}",
            model_corpus.get("resource_identity_match_rate"),
            thresholds.get("min_model_resource_identity_match_rate"),
        )
        baseline_exact = baseline_corpus.get("json_exact_match_rate")
        model_exact = model_corpus.get("json_exact_match_rate")
        exact_delta = (
            float(model_exact) - float(baseline_exact)
            if _finite(model_exact) and _finite(baseline_exact)
            else None
        )
        _minimum_value(
            checks,
            f"min_exact_match_delta_{corpus}",
            exact_delta,
            thresholds.get("min_exact_match_delta_vs_foundation"),
        )
    failures = [check for check in checks if not check["passed"]]
    return {
        "schema_version": "phase1_promotion.v1",
        "status": "pass" if not failures else "fail",
        "role": "model_x",
        "artifact_sha": load_evidence.get("artifact_sha"),
        "checks": checks,
        "failures": failures,
    }


def _mapping(source: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = source.get(key)
    if not isinstance(value, Mapping):
        raise Phase1PromotionError(f"{key} must be an object")
    return value


def _model_counts(evidence: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(_mapping(evidence, "counts"), "model_x")


def _heldout_rows(manifest: Mapping[str, Any]) -> int:
    for key in ("approved_heldout_rows", "eval_count", "rows"):
        value = manifest.get(key)
        if isinstance(value, int) and value > 0:
            return value
    row_ids = manifest.get("row_ids")
    if isinstance(row_ids, list) and row_ids:
        return len(row_ids)
    raise Phase1PromotionError("held-out manifest has no approved row count")


def _required_corpora(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    value = manifest.get("required_corpora")
    if not isinstance(value, list) or not value:
        raise Phase1PromotionError("held-out manifest requires non-empty required_corpora")
    if not all(isinstance(item, str) and item.strip() for item in value):
        raise Phase1PromotionError("held-out manifest required_corpora must be list[str]")
    normalized = tuple(item.strip() for item in value)
    if len(normalized) != len(set(normalized)):
        raise Phase1PromotionError("held-out manifest required_corpora must be unique")
    return normalized


def _corpus_counts(manifest: Mapping[str, Any], key: str) -> dict[str, int]:
    value = manifest.get(key)
    if not isinstance(value, Mapping) or not value:
        raise Phase1PromotionError(f"held-out manifest requires non-empty {key}")
    if not all(
        isinstance(name, str)
        and name.strip()
        and isinstance(count, int)
        and not isinstance(count, bool)
        and count > 0
        for name, count in value.items()
    ):
        raise Phase1PromotionError(
            f"held-out manifest {key} must map corpora to positive integer counts"
        )
    return {str(name): int(count) for name, count in value.items()}


def _row_ids(manifest: Mapping[str, Any]) -> list[str]:
    value = manifest.get("row_ids")
    if not isinstance(value, list) or not value:
        raise Phase1PromotionError("held-out manifest requires non-empty row_ids")
    if not all(_is_digest(item) for item in value):
        raise Phase1PromotionError("held-out manifest row_ids must be SHA-256 digests")
    if len(value) != len(set(value)):
        raise Phase1PromotionError("held-out manifest row_ids must be unique")
    return value


def _is_digest(value: Any) -> bool:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        return False
    digest = value.removeprefix("sha256:")
    return len(digest) == 64 and all(
        char in "0123456789abcdef" for char in digest.lower()
    )


def _is_commit(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    commit = value.strip()
    return len(commit) == 40 and all(
        char in "0123456789abcdef" for char in commit.lower()
    )


def _nonnegative_threshold(thresholds: Mapping[str, Any], key: str) -> float:
    value = thresholds.get(key)
    if not _finite(value) or float(value) < 0:
        raise Phase1PromotionError(f"{key} must be a finite non-negative number")
    return float(value)


def _best_checkpoint_promoted(
    manifest: Mapping[str, Any],
    load_evidence: Mapping[str, Any],
) -> bool:
    checkpoint = str(manifest.get("checkpoint_path", ""))
    promoted = str(manifest.get("promoted_artifact_path", ""))
    loaded = str(load_evidence.get("adapter_dir", ""))
    return (
        manifest.get("promotion_source") == "best_checkpoint"
        and checkpoint.endswith("_epoch_best.pt")
        and bool(promoted)
        and Path(promoted) == Path(loaded)
    )


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _all_finite_metrics(value: Any) -> bool:
    if isinstance(value, Mapping):
        return bool(value) and all(
            _all_finite_metrics(item) for item in value.values()
        )
    return _finite(value)


def _hard(
    checks: list[dict[str, Any]],
    name: str,
    passed: bool,
    *,
    observed: Any = None,
    threshold: Any = None,
) -> None:
    checks.append({
        "name": name,
        "operator": "required",
        "observed": observed,
        "threshold": threshold,
        "passed": bool(passed),
    })


def _minimum(checks, name, observed, threshold) -> None:
    if threshold is None:
        raise Phase1PromotionError(f"missing promotion threshold {name}")
    checks.append({
        "name": name,
        "operator": ">=",
        "observed": observed,
        "threshold": threshold,
        "passed": _finite(observed) and float(observed) >= float(threshold),
    })


def _minimum_value(checks, name, observed, threshold) -> None:
    if threshold is None:
        raise Phase1PromotionError(f"missing promotion threshold {name}")
    checks.append({
        "name": name,
        "operator": ">=",
        "observed": observed,
        "threshold": threshold,
        "passed": _finite(observed) and float(observed) >= float(threshold),
    })


def _maximum(checks, name, observed, threshold) -> None:
    if threshold is None:
        raise Phase1PromotionError(f"missing promotion threshold {name}")
    checks.append({
        "name": name,
        "operator": "<=",
        "observed": observed,
        "threshold": threshold,
        "passed": _finite(observed) and float(observed) <= float(threshold),
    })
