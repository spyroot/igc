"""Shared, fail-closed artifact evidence checks for model and dataset promotion."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping


class PromotionEvidenceError(ValueError):
    """Raised when promotion evidence does not follow the shared contract."""


def is_sha256(value: Any) -> bool:
    """Return whether ``value`` is a canonical lower- or upper-case SHA-256 id."""
    if not isinstance(value, str) or not value.startswith("sha256:"):
        return False
    digest = value.removeprefix("sha256:")
    return len(digest) == 64 and all(
        char in "0123456789abcdef" for char in digest.lower()
    )


def validate_configured_hard_checks(
    checks: Any,
    required_names: Any,
) -> None:
    """Require every YAML-declared hard check to exist exactly once.

    Evaluators may emit additional threshold checks, but a configured hard check
    must never disappear because of a source rename or an incomplete refactor.

    :param checks: Result ``checks`` sequence emitted by a promotion evaluator.
    :param required_names: YAML ``hard_checks`` sequence.
    :raises PromotionEvidenceError: for malformed, duplicate, or missing checks.
    """
    if (
        not isinstance(required_names, list)
        or not required_names
        or not all(isinstance(name, str) and name for name in required_names)
        or len(required_names) != len(set(required_names))
    ):
        raise PromotionEvidenceError(
            "promotion hard_checks must be a unique non-empty list[str]"
        )
    if not isinstance(checks, list):
        raise PromotionEvidenceError("promotion result checks must be a list")
    observed: list[str] = []
    for index, check in enumerate(checks):
        if not isinstance(check, Mapping):
            raise PromotionEvidenceError(
                f"promotion result check {index} must be an object"
            )
        name = check.get("name")
        if not isinstance(name, str) or not name:
            raise PromotionEvidenceError(
                f"promotion result check {index} has no valid name"
            )
        observed.append(name)
    duplicates = sorted({name for name in observed if observed.count(name) > 1})
    if duplicates:
        raise PromotionEvidenceError(
            f"promotion result has duplicate checks: {duplicates}"
        )
    missing = sorted(set(required_names) - set(observed))
    if missing:
        raise PromotionEvidenceError(
            f"promotion result is missing configured hard checks: {missing}"
        )


def validate_promotion_evidence(
    evidence: Mapping[str, Any],
    *,
    expected_parent_role: str,
    reload_target: str,
    observed_source_full_manifest_sha: str,
    observed_source_full_sha: str,
    observed_heldout_manifest_sha: str,
    observed_heldout_sha: str,
    observed_heldout_rows: int,
    observed_train_manifest_sha: str | None = None,
    observed_train_sha: str | None = None,
    observed_split_release_sha: str | None = None,
) -> list[dict[str, Any]]:
    """Validate immutable lineage and load evidence with cross-checked digests.

    ``reload_target`` is ``artifact`` for a promoted model checkpoint and
    ``parent`` for a dataset produced by a promoted model.
    """
    if reload_target not in {"artifact", "parent"}:
        raise PromotionEvidenceError("reload_target must be artifact or parent")

    full_manifest = _mapping(evidence, "immutable_full_manifest")
    parent = _mapping(evidence, "real_promoted_parent_checkpoint")
    heldout = _mapping(evidence, "real_heldout_data")
    reload_evidence = _mapping(evidence, "checkpoint_reload")
    smoke_evidence = _mapping(evidence, "inference_smoke")
    artifact_sha = evidence.get("artifact_sha")
    parent_sha = parent.get("artifact_sha")
    expected_runtime_sha = artifact_sha if reload_target == "artifact" else parent_sha

    checks: list[dict[str, Any]] = []
    _required(
        checks,
        "immutable_full_manifest",
        full_manifest.get("immutable") is True
        and full_manifest.get("complete") is True
        and _positive_int(full_manifest.get("rows"))
        and is_sha256(full_manifest.get("sha256"))
        and is_sha256(observed_source_full_sha)
        and full_manifest.get("artifact_sha") == observed_source_full_sha,
    )
    _required(
        checks,
        "full_manifest_sha_matches_file",
        is_sha256(observed_source_full_manifest_sha)
        and full_manifest.get("sha256") == observed_source_full_manifest_sha,
    )
    if observed_split_release_sha is not None:
        if observed_train_manifest_sha is None or observed_train_sha is None:
            raise PromotionEvidenceError(
                "split promotion requires exact train manifest and artifact SHAs"
            )
        split_release = _mapping(evidence, "disjoint_split_release")
        train_manifest = _mapping(evidence, "immutable_train_manifest")
        _required(
            checks,
            "immutable_train_manifest",
            train_manifest.get("immutable") is True
            and train_manifest.get("complete") is True
            and _positive_int(train_manifest.get("rows"))
            and train_manifest.get("manifest_sha")
            == observed_train_manifest_sha
            and is_sha256(observed_train_sha)
            and train_manifest.get("artifact_sha") == observed_train_sha,
        )
        _required(
            checks,
            "disjoint_train_heldout_release",
            split_release.get("immutable") is True
            and split_release.get("complete") is True
            and split_release.get("disjoint") is True
            and is_sha256(observed_split_release_sha)
            and split_release.get("sha256") == observed_split_release_sha
            and split_release.get("train_manifest_sha")
            == observed_train_manifest_sha
            and split_release.get("heldout_manifest_sha")
            == observed_heldout_manifest_sha,
        )
    _required(
        checks,
        "real_promoted_parent_checkpoint",
        parent.get("role") == expected_parent_role
        and parent.get("promotion_status") == "pass"
        and is_sha256(parent_sha),
    )
    _required(
        checks,
        "real_heldout_data",
        heldout.get("real") is True
        and heldout.get("split") == "heldout"
        and _positive_int(heldout.get("rows"))
        and is_sha256(heldout.get("manifest_sha"))
        and is_sha256(heldout.get("artifact_sha")),
    )
    _required(
        checks,
        "heldout_manifest_sha_matches_file",
        is_sha256(observed_heldout_manifest_sha)
        and heldout.get("manifest_sha") == observed_heldout_manifest_sha,
    )
    _required(
        checks,
        "heldout_artifact_sha_matches_file",
        is_sha256(observed_heldout_sha)
        and heldout.get("artifact_sha") == observed_heldout_sha,
    )
    _required(
        checks,
        "heldout_row_count_matches_file",
        _positive_int(observed_heldout_rows)
        and heldout.get("rows") == observed_heldout_rows,
    )
    _required(checks, "artifact_sha", is_sha256(artifact_sha))
    _required(
        checks,
        "checkpoint_reload_succeeded",
        reload_evidence.get("status") == "pass"
        and is_sha256(reload_evidence.get("artifact_sha"))
        and reload_evidence.get("artifact_sha") == expected_runtime_sha,
    )
    _required(
        checks,
        "inference_smoke_succeeded",
        smoke_evidence.get("status") == "pass"
        and is_sha256(smoke_evidence.get("artifact_sha"))
        and smoke_evidence.get("artifact_sha") == expected_runtime_sha,
    )
    return checks


def validate_training_run_evidence(
    run_report: Mapping[str, Any],
    *,
    expected_phase: str,
    expected_task: str,
    expected_parent_role: str,
    expected_parent_artifact_sha: str,
    expected_output_role: str,
    observed_source_full_manifest_sha: str,
    observed_train_manifest_sha: str,
    observed_train_sha: str,
    observed_heldout_manifest_sha: str,
    observed_heldout_sha: str,
    reload_evidence: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Cross-check one SFT run report against exact promotion inputs."""
    manifest = _mapping(run_report, "manifest")
    training = _mapping(manifest, "training")
    metrics = _mapping(run_report, "metrics")
    checks: list[dict[str, Any]] = []
    _required(checks, "run_report_exists", bool(manifest))
    _required(
        checks,
        "run_role_and_task_lineage",
        manifest.get("phase") == expected_phase
        and manifest.get("task") == expected_task
        and manifest.get("parent_role") == expected_parent_role
        and manifest.get("parent_artifact_sha") == expected_parent_artifact_sha
        and manifest.get("output_role") == expected_output_role,
    )
    _required(
        checks,
        "run_train_manifest_matches_file",
        is_sha256(observed_train_manifest_sha)
        and manifest.get("data_manifest") == observed_train_manifest_sha,
    )
    _required(
        checks,
        "run_source_full_manifest_matches_file",
        is_sha256(observed_source_full_manifest_sha)
        and manifest.get("source_manifest_sha")
        == observed_source_full_manifest_sha,
    )
    _required(
        checks,
        "run_eval_manifest_matches_file",
        is_sha256(observed_heldout_manifest_sha)
        and manifest.get("eval_manifest_sha") == observed_heldout_manifest_sha,
    )
    _required(
        checks,
        "run_eval_data_matches_file",
        is_sha256(observed_heldout_sha)
        and manifest.get("eval_split") == observed_heldout_sha
        and manifest.get("eval_data_sha") == observed_heldout_sha,
    )
    _required(
        checks,
        "run_train_data_matches_file",
        is_sha256(observed_train_sha)
        and manifest.get("train_data_sha") == observed_train_sha,
    )
    _required(
        checks,
        "foundation_model_sha_exists",
        is_sha256(manifest.get("foundation_model_sha")),
    )
    _required(
        checks,
        "tokenizer_sha_exists",
        is_sha256(manifest.get("tokenizer_sha")),
    )
    _required(
        checks,
        "task_spec_sha_exists",
        is_sha256(manifest.get("task_spec_sha")),
    )
    _required(
        checks,
        "training_code_commit_exists",
        _is_commit(manifest.get("git_commit")),
    )
    _required(
        checks,
        "optimizer_steps_positive",
        _positive_int(training.get("optimizer_steps")),
    )
    _required(
        checks,
        "finite_run_metrics",
        _all_finite(training) and _all_finite(metrics),
    )
    checkpoint = str(manifest.get("checkpoint_path", ""))
    promoted = str(manifest.get("promoted_artifact_path", ""))
    loaded = str(reload_evidence.get("adapter_dir", ""))
    _required(
        checks,
        "best_checkpoint_promoted",
        manifest.get("promotion_source") == "best_checkpoint"
        and checkpoint.endswith("_epoch_best.pt")
        and bool(promoted)
        and bool(loaded)
        and Path(promoted) == Path(loaded),
    )
    return checks


def _mapping(source: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = source.get(key)
    if not isinstance(value, Mapping):
        raise PromotionEvidenceError(f"artifact evidence {key} must be an object")
    return value


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_commit(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(char in "0123456789abcdef" for char in value.lower())
    )


def _all_finite(value: Any) -> bool:
    if isinstance(value, Mapping):
        return bool(value) and all(_all_finite(item) for item in value.values())
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    return False


def _required(checks: list[dict[str, Any]], name: str, observed: Any) -> None:
    checks.append({
        "name": name,
        "operator": "required",
        "observed": bool(observed),
        "passed": bool(observed),
    })


__all__ = (
    "PromotionEvidenceError",
    "is_sha256",
    "validate_configured_hard_checks",
    "validate_promotion_evidence",
    "validate_training_run_evidence",
)
