"""Metric-key registry tests for Phase 2 labelled-request generation.

Author:
Mus mbayramo@stanford.edu
"""
from igc.modules.base.metric_keys import (
    PHASE2_LABELLED_REQUESTS,
    PHASE2_LABELLED_REQUESTS_WANDB_METRIC_KEYS,
    phase_metric,
)


def test_phase2_labelled_request_registry_keeps_required_metric_names() -> None:
    """Offline builder metrics keep the documented W&B key names."""
    keys = set(PHASE2_LABELLED_REQUESTS_WANDB_METRIC_KEYS)

    assert phase_metric(PHASE2_LABELLED_REQUESTS, "draft_total") in keys
    assert phase_metric(PHASE2_LABELLED_REQUESTS, "accepted_total") in keys
    assert phase_metric(PHASE2_LABELLED_REQUESTS, "rejected_total") in keys
    assert phase_metric(PHASE2_LABELLED_REQUESTS, "nonsense_rate") in keys
    assert phase_metric(PHASE2_LABELLED_REQUESTS, "invalid_json_rate") in keys
    assert phase_metric(PHASE2_LABELLED_REQUESTS, "pro_accept_rate") in keys
    assert phase_metric(PHASE2_LABELLED_REQUESTS, "rest_api_set_match_rate") in keys
    assert phase_metric(PHASE2_LABELLED_REQUESTS, "empty_set_match_rate") in keys
    assert phase_metric(PHASE2_LABELLED_REQUESTS, "sample_width", "k") in keys
    assert phase_metric(PHASE2_LABELLED_REQUESTS, "vendor", "source_corpus") in keys
    assert phase_metric(PHASE2_LABELLED_REQUESTS, "prompt_spec_version") in keys
    assert phase_metric(PHASE2_LABELLED_REQUESTS, "model_x", "artifact_sha") in keys
    assert phase_metric(PHASE2_LABELLED_REQUESTS, "judge", "model") in keys
    assert phase_metric(PHASE2_LABELLED_REQUESTS, "judge", "profile") in keys


def test_phase2_labelled_request_registry_uses_canonical_namespace() -> None:
    """New Phase 2 labelled-request keys use only the canonical namespace."""
    assert PHASE2_LABELLED_REQUESTS == "phase2_labelled_requests"
    assert all(
        key.startswith("phase2_labelled_requests/")
        for key in PHASE2_LABELLED_REQUESTS_WANDB_METRIC_KEYS
    )
