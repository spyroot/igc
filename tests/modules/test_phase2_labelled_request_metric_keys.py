"""Metric keys for Phase 2 labelled request dataset generation.

Author:
Mus mbayramo@stanford.edu
"""
from igc.modules.base.metric_keys import (
    PHASE2_LABELLED_REQUESTS,
    PHASE2_LABELLED_REQUESTS_WANDB_METRIC_KEYS,
    phase_metric,
)


def test_phase2_labelled_request_registry_keeps_required_metric_names() -> None:
    """The offline builder has stable W&B key names for counters and metadata."""
    keys = set(PHASE2_LABELLED_REQUESTS_WANDB_METRIC_KEYS)

    assert phase_metric(PHASE2_LABELLED_REQUESTS, "build", "draft_total") in keys
    assert phase_metric(PHASE2_LABELLED_REQUESTS, "build", "accepted_total") in keys
    assert phase_metric(PHASE2_LABELLED_REQUESTS, "build", "rejected_total") in keys
    assert phase_metric(PHASE2_LABELLED_REQUESTS, "eval", "nonsense_rate") in keys
    assert phase_metric(PHASE2_LABELLED_REQUESTS, "eval", "invalid_json_rate") in keys
    assert phase_metric(PHASE2_LABELLED_REQUESTS, "eval", "pro_accept_rate") in keys
    assert phase_metric(PHASE2_LABELLED_REQUESTS, "eval", "rest_api_set_match_rate") in keys
    assert phase_metric(PHASE2_LABELLED_REQUESTS, "eval", "empty_set_match_rate") in keys
    assert phase_metric(PHASE2_LABELLED_REQUESTS, "sample_width", "k") in keys
    assert phase_metric(PHASE2_LABELLED_REQUESTS, "vendor", "source_corpus") in keys
    assert phase_metric(PHASE2_LABELLED_REQUESTS, "spec", "prompt_spec_version") in keys
    assert phase_metric(PHASE2_LABELLED_REQUESTS, "model", "model_x_artifact_sha") in keys
    assert phase_metric(PHASE2_LABELLED_REQUESTS, "judge", "model") in keys
    assert phase_metric(PHASE2_LABELLED_REQUESTS, "judge", "profile") in keys


def test_phase2_labelled_request_registry_uses_canonical_namespace() -> None:
    """New Phase 2 labelled request keys use the canonical namespace only."""
    assert PHASE2_LABELLED_REQUESTS == "phase2_labelled_requests"
    assert all(
        key.startswith("phase2_labelled_requests/")
        for key in PHASE2_LABELLED_REQUESTS_WANDB_METRIC_KEYS
    )
