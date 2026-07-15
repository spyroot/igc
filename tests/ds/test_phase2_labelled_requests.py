"""Offline tests for Phase 2 labelled request dataset plumbing.

The tests pin the dataset-builder contract without model calls, W&B writes, GPU
work, or live Redfish traffic.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pytest

from igc.ds.phase2_labelled_requests import (
    Phase2LabelledRequestCounters,
    Phase2RestApiRecord,
    load_phase2_labelled_requests_spec,
    parse_pro_judge_result,
    render_phase2_prompt,
    rest_api_sets_equal,
    sample_rest_api_records,
)
from igc.modules.base.metric_keys import phase_metric


CONFIG_PATH = Path(__file__).parents[2] / "configs" / "phase2_labelled_requests.yaml"


def _record(index: int, vendor: str = "dell") -> Phase2RestApiRecord:
    """Build a small public-safe Redfish record fixture."""
    rest_api = f"/redfish/v1/Systems/{index}"
    return Phase2RestApiRecord(
        rest_api=rest_api,
        allowed_methods=("GET", "HEAD"),
        json_body={
            "@odata.id": rest_api,
            "@odata.type": "#ComputerSystem.v1_20_0.ComputerSystem",
            "Name": f"System {index}",
        },
        vendor=vendor,
        source_corpus="unit_fixture",
    )


def test_prompt_spec_loading_uses_yaml_contract() -> None:
    """The runtime spec comes from the YAML file under configs."""
    spec = load_phase2_labelled_requests_spec(CONFIG_PATH)

    assert spec.dataset_name == "phase2_labelled_requests"
    assert spec.sample_widths == (1, 2, 3)
    assert spec.wandb["namespace"] == "phase2_labelled_requests"
    assert spec.model_x["artifact_sha"] == "fixture-model-x-sha256"
    assert spec.judge["profile"] == "max_reasoning"
    assert "draft_human_request" in spec.prompts
    assert "pro_judge_set_match" in spec.prompts
    assert spec.acceptance_thresholds["rest_api_set_match_rate_min"] == 0.98

    sample = sample_rest_api_records(
        [_record(1), _record(2)],
        k=2,
        rng=random.Random(3),
        allowed_widths=spec.sample_widths,
    )
    prompt = render_phase2_prompt(spec, "draft_human_request", sample)

    assert "/redfish/v1/Systems/1" in prompt
    assert "/redfish/v1/Systems/2" in prompt
    assert "fixture-model-x-sha256" not in prompt


def test_sampling_supports_only_configured_widths_one_two_and_three() -> None:
    """The builder samples one, two, or three REST API records without replacement."""
    spec = load_phase2_labelled_requests_spec(CONFIG_PATH)
    records = [_record(1), _record(2), _record(3), _record(4)]

    for width in spec.sample_widths:
        sample = sample_rest_api_records(
            records,
            k=width,
            rng=random.Random(width),
            allowed_widths=spec.sample_widths,
        )
        assert len(sample.records) == width
        assert len(set(sample.rest_api_list)) == width
        assert sample.sample_width == width

    with pytest.raises(ValueError, match="sample width"):
        sample_rest_api_records(
            records,
            k=4,
            rng=random.Random(4),
            allowed_widths=spec.sample_widths,
        )


def test_rest_api_set_comparison_is_unordered_and_empty_sets_match() -> None:
    """Phase 2 judges REST API set equality, with empty set equal to empty set."""
    assert rest_api_sets_equal(
        ["/redfish/v1/Systems", "/redfish/v1/TaskService/Tasks"],
        ["/redfish/v1/TaskService/Tasks", "/redfish/v1/Systems"],
    )
    assert rest_api_sets_equal([], [])
    assert not rest_api_sets_equal(["/redfish/v1/Systems"], [])
    assert not rest_api_sets_equal(["/redfish/v1/Systems"], ["/redfish/v1/Chassis"])


def test_pro_judge_result_parser_handles_accept_reject_empty_and_invalid_json() -> None:
    """Private judge output parsing is deterministic and side-effect free."""
    accepted = parse_pro_judge_result(
        json.dumps({
            "accepted": True,
            "rest_api_list": ["/redfish/v1/Systems", "/redfish/v1/Managers"],
            "nonsense": False,
            "reason": "The request names both resources.",
        }),
        expected_rest_apis=["/redfish/v1/Managers", "/redfish/v1/Systems"],
    )
    assert accepted.valid_json
    assert accepted.accepted
    assert accepted.pro_accept
    assert accepted.rest_api_set_match
    assert not accepted.empty_set_match

    empty = parse_pro_judge_result(
        json.dumps({
            "accepted": True,
            "rest_api_list": [],
            "nonsense": False,
        }),
        expected_rest_apis=[],
    )
    assert empty.empty_set_match
    assert empty.rest_api_set_match

    nonsense = parse_pro_judge_result(
        json.dumps({
            "accepted": False,
            "rest_api_list": [],
            "nonsense": True,
            "reason": "Draft text is not a request.",
        }),
        expected_rest_apis=["/redfish/v1/Systems"],
    )
    assert nonsense.valid_json
    assert nonsense.nonsense
    assert not nonsense.accepted

    invalid = parse_pro_judge_result("{not json", expected_rest_apis=["/redfish/v1/Systems"])
    assert invalid.invalid_json
    assert not invalid.accepted
    assert not invalid.pro_accept


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    (
        ("rest_api_list", "/redfish/v1/Systems"),
        ("rest_api_list", ["/redfish/v1/Systems", 7]),
        ("rest_api_list", None),
        ("rest_api_set", "/redfish/v1/Systems"),
        ("rest_api_set", ["/redfish/v1/Systems", 7]),
        ("rest_api_set", None),
    ),
)
def test_pro_judge_result_parser_counts_malformed_rest_api_fields_as_invalid(
    field_name: str,
    field_value: Any,
) -> None:
    """Malformed judge fields are counted as invalid output, not raised errors."""
    result = parse_pro_judge_result(
        json.dumps({
            "accepted": True,
            field_name: field_value,
            "nonsense": False,
        }),
        expected_rest_apis=[],
    )
    assert not result.valid_json
    assert result.invalid_json
    assert not result.accepted
    assert not result.pro_accept
    assert not result.rest_api_set_match
    assert result.rest_api_list == ()
    assert field_name in result.reason


def test_nonsense_invalid_and_acceptance_counters_emit_phase2_metrics() -> None:
    """Counters produce W&B-safe keys without opening a live W&B run."""
    spec = load_phase2_labelled_requests_spec(CONFIG_PATH)
    counters = Phase2LabelledRequestCounters()
    accepted = parse_pro_judge_result(
        json.dumps({
            "accepted": True,
            "rest_api_list": ["/redfish/v1/Systems"],
            "nonsense": False,
        }),
        expected_rest_apis=["/redfish/v1/Systems"],
    )
    nonsense = parse_pro_judge_result(
        json.dumps({
            "accepted": False,
            "rest_api_list": [],
            "nonsense": True,
        }),
        expected_rest_apis=["/redfish/v1/Managers"],
    )
    invalid = parse_pro_judge_result("{bad", expected_rest_apis=["/redfish/v1/Chassis"])

    counters.record_outcome(
        accepted,
        sample_width=1,
        vendor="dell",
        source_corpus="unit_fixture",
        spec=spec,
    )
    counters.record_outcome(
        nonsense,
        sample_width=2,
        vendor="hpe",
        source_corpus="unit_fixture",
        spec=spec,
    )
    counters.record_outcome(
        invalid,
        sample_width=3,
        vendor="supermicro",
        source_corpus="unit_fixture",
        spec=spec,
    )

    metrics = counters.to_wandb_metrics(spec)
    namespace = spec.wandb["namespace"]

    assert metrics[phase_metric(namespace, "build", "draft_total")] == 3
    assert metrics[phase_metric(namespace, "build", "accepted_total")] == 1
    assert metrics[phase_metric(namespace, "build", "rejected_total")] == 2
    assert metrics[phase_metric(namespace, "eval", "nonsense_rate")] == pytest.approx(1 / 3)
    assert metrics[phase_metric(namespace, "eval", "invalid_json_rate")] == pytest.approx(1 / 3)
    assert metrics[phase_metric(namespace, "eval", "pro_accept_rate")] == pytest.approx(1 / 3)
    assert metrics[phase_metric(namespace, "eval", "rest_api_set_match_rate")] == pytest.approx(1 / 3)
    assert metrics[phase_metric(namespace, "sample_width", "k")] == 3
    assert metrics[phase_metric(namespace, "vendor", "source_corpus")] == (
        "supermicro/unit_fixture"
    )
    assert metrics[phase_metric(namespace, "spec", "prompt_spec_version")] == spec.version
    assert metrics[phase_metric(namespace, "model", "model_x_artifact_sha")] == (
        "fixture-model-x-sha256"
    )
    assert metrics[phase_metric(namespace, "judge", "profile")] == "max_reasoning"
