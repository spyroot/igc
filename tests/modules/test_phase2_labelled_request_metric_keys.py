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
    expected = (
        phase_metric(PHASE2_LABELLED_REQUESTS, "draft_total"),
        phase_metric(PHASE2_LABELLED_REQUESTS, "accepted_total"),
        phase_metric(PHASE2_LABELLED_REQUESTS, "rejected_total"),
        phase_metric(PHASE2_LABELLED_REQUESTS, "nonsense_rate"),
        phase_metric(PHASE2_LABELLED_REQUESTS, "invalid_json_rate"),
        phase_metric(PHASE2_LABELLED_REQUESTS, "pro_accept_rate"),
        phase_metric(PHASE2_LABELLED_REQUESTS, "rest_api_set_match_rate"),
        phase_metric(PHASE2_LABELLED_REQUESTS, "empty_set_match_rate"),
        phase_metric(PHASE2_LABELLED_REQUESTS, "sample_width", "k"),
        phase_metric(PHASE2_LABELLED_REQUESTS, "vendor", "source_corpus"),
        phase_metric(PHASE2_LABELLED_REQUESTS, "prompt_spec_version"),
        phase_metric(PHASE2_LABELLED_REQUESTS, "model_x", "artifact_sha"),
        phase_metric(PHASE2_LABELLED_REQUESTS, "judge", "model"),
        phase_metric(PHASE2_LABELLED_REQUESTS, "judge", "profile"),
    )

    assert PHASE2_LABELLED_REQUESTS_WANDB_METRIC_KEYS == expected
    assert len(PHASE2_LABELLED_REQUESTS_WANDB_METRIC_KEYS) == len(set(expected))


def test_phase2_labelled_request_registry_uses_canonical_namespace() -> None:
    """New Phase 2 labelled-request keys use only the canonical namespace."""
    assert PHASE2_LABELLED_REQUESTS == "phase2_labelled_requests"
    assert all(
        key.startswith("phase2_labelled_requests/")
        for key in PHASE2_LABELLED_REQUESTS_WANDB_METRIC_KEYS
    )
