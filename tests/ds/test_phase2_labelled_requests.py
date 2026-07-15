"""Offline tests for the Phase 2 labelled-request dataset plumbing.

The module under test must keep prompts, model identifiers, judge routing,
generation knobs, W&B keys, and acceptance thresholds in YAML/config values.
Tests use tiny Redfish-shaped records and injected fake providers; they never
call a model, W&B, a GPU, a Redfish host, or the network.

Author:
Mus mbayramo@stanford.edu
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from igc.ds.phase2_labelled_requests import (
    PHASE2_LABELLED_REQUESTS,
    Phase2LabelledRequestBuilder,
    Phase2LabelledRequestCounters,
    Phase2LabelledRequestsSpecError,
    RestApiRecord,
    compare_rest_api_sets,
    empty_set_matches,
    load_phase2_labelled_requests_spec,
    parse_pro_judge_result,
    render_model_x_prompt,
    render_pro_judge_prompt,
    sample_phase2_contexts,
    to_minimal_phase3_input,
)
from igc.modules.base.metric_keys import (
    PHASE2_LABELLED_REQUESTS_WANDB_METRIC_KEYS,
    phase_metric,
)


def _record(index: int, methods: tuple[str, ...] = ("GET", "HEAD")) -> RestApiRecord:
    """Build a tiny Redfish REST API record with source metadata."""
    return RestApiRecord(
        rest_api=f"/redfish/v1/Systems/{index}",
        allowed_methods=methods,
        json_body={
            "@odata.id": f"/redfish/v1/Systems/{index}",
            "@odata.type": "#ComputerSystem.v1_20_0.ComputerSystem",
            "Name": f"System {index}",
        },
        vendor="fixture_vendor",
        source_corpus="fixture_corpus",
    )


def _write_spec(path: Path) -> Path:
    """Write a complete test YAML spec with distinctive literal values."""
    path.write_text(
        """
dataset:
  name: phase2_labelled_requests
  prompt_spec_version: phase2-labelled-requests-test-v1
sampling:
  sample_widths: [1, 2, 3]
model_x:
  model_id: ${PHASE1_MODEL_X_MODEL_ID}
  artifact_sha: ${PHASE1_MODEL_X_ARTIFACT_SHA}
judge:
  route: private_pro
  model_id: ${PHASE2_JUDGE_MODEL_ID}
  profile: ${PHASE2_JUDGE_PROFILE}
generation:
  max_new_tokens: 96
  temperature: 0.2
  top_p: 0.95
prompts:
  model_x_draft:
    system: phase2 test model-x system prompt from YAML
    template: |
      Draft one operator request for these records:
      {records_json}
  pro_judge:
    system: phase2 test pro judge system prompt from YAML
    template: |
      Judge whether this request maps to the same unordered REST API set.
      Records:
      {records_json}
      Draft:
      {draft_text}
wandb:
  namespace: phase2_labelled_requests
  metric_keys:
    - phase2_labelled_requests/draft_total
    - phase2_labelled_requests/accepted_total
    - phase2_labelled_requests/rejected_total
    - phase2_labelled_requests/nonsense_rate
    - phase2_labelled_requests/invalid_json_rate
    - phase2_labelled_requests/pro_accept_rate
    - phase2_labelled_requests/rest_api_set_match_rate
    - phase2_labelled_requests/empty_set_match_rate
    - phase2_labelled_requests/sample_width/k
    - phase2_labelled_requests/vendor/source_corpus
    - phase2_labelled_requests/prompt_spec_version
    - phase2_labelled_requests/model_x/artifact_sha
    - phase2_labelled_requests/judge/model
    - phase2_labelled_requests/judge/profile
acceptance:
  min_pro_accept_rate: 0.9
  min_rest_api_set_match_rate: 0.98
  max_nonsense_rate: 0.01
  max_invalid_json_rate: 0.01
""",
        encoding="utf-8",
    )
    return path


def _judge_json(
    *,
    accepted: bool = True,
    rest_api_list: list[str] | None = None,
    nonsense: bool = False,
    order_evidence: str = "none",
) -> str:
    """Return a compact fake Pro judge JSON response."""
    return json.dumps(
        {
            "accepted": accepted,
            "rest_api_list": [] if rest_api_list is None else rest_api_list,
            "nonsense": nonsense,
            "reason": "fixture",
            "order_evidence": order_evidence,
        },
    )


def _phase2_metric(name: str) -> str:
    """Return a required Phase 2 labelled-request metric key."""
    return phase_metric(PHASE2_LABELLED_REQUESTS, name)


def test_loads_prompt_model_judge_generation_and_thresholds_from_yaml(tmp_path: Path) -> None:
    """The Phase 2 spec loader keeps every runtime knob in YAML."""
    spec = load_phase2_labelled_requests_spec(_write_spec(tmp_path / "phase2.yaml"))

    assert spec.dataset_name == PHASE2_LABELLED_REQUESTS
    assert spec.prompt_spec_version == "phase2-labelled-requests-test-v1"
    assert spec.sample_widths == (1, 2, 3)
    assert spec.model_x.model_id == "${PHASE1_MODEL_X_MODEL_ID}"
    assert spec.judge.route == "private_pro"
    assert spec.judge.profile == "${PHASE2_JUDGE_PROFILE}"
    assert spec.generation == {"max_new_tokens": 96, "temperature": 0.2, "top_p": 0.95}
    assert spec.wandb_namespace == PHASE2_LABELLED_REQUESTS
    assert spec.metric_keys == PHASE2_LABELLED_REQUESTS_WANDB_METRIC_KEYS
    assert spec.acceptance_thresholds["min_pro_accept_rate"] == 0.9


def test_spec_loader_rejects_malformed_phase2_specs(tmp_path: Path) -> None:
    """Spec validation fails closed for bad YAML shape and contract drift."""
    not_mapping = tmp_path / "not-mapping.yaml"
    not_mapping.write_text("- phase2_labelled_requests\n", encoding="utf-8")
    with pytest.raises(Phase2LabelledRequestsSpecError, match="must be a mapping"):
        load_phase2_labelled_requests_spec(not_mapping)

    wrong_dataset = _write_spec(tmp_path / "wrong-dataset.yaml")
    wrong_dataset.write_text(
        wrong_dataset.read_text(encoding="utf-8").replace(
            "name: phase2_labelled_requests",
            "name: legacy_dataset",
        ),
        encoding="utf-8",
    )
    with pytest.raises(Phase2LabelledRequestsSpecError, match="dataset.name"):
        load_phase2_labelled_requests_spec(wrong_dataset)

    wrong_metrics = _write_spec(tmp_path / "wrong-metrics.yaml")
    wrong_metrics.write_text(
        wrong_metrics.read_text(encoding="utf-8").replace(
            "phase2_labelled_requests/draft_total",
            "phase2_labelled_requests/draft_count",
        ),
        encoding="utf-8",
    )
    with pytest.raises(Phase2LabelledRequestsSpecError, match="metric_keys"):
        load_phase2_labelled_requests_spec(wrong_metrics)


def test_committed_phase2_labelled_requests_config_loads() -> None:
    """The checked-in builder spec stays aligned with the metric registry."""
    spec = load_phase2_labelled_requests_spec("configs/phase2_labelled_requests.yaml")

    assert spec.dataset_name == PHASE2_LABELLED_REQUESTS
    assert spec.metric_keys == PHASE2_LABELLED_REQUESTS_WANDB_METRIC_KEYS
    assert spec.sample_widths == (1, 2, 3)


def test_prompt_rendering_uses_yaml_templates_not_runtime_literals(tmp_path: Path) -> None:
    """Prompt text comes from the loaded spec and can be changed without code edits."""
    spec = load_phase2_labelled_requests_spec(_write_spec(tmp_path / "phase2.yaml"))
    records = (_record(1), _record(2))

    model_prompt = render_model_x_prompt(spec, records)
    judge_prompt = render_pro_judge_prompt(spec, records, "show both systems")

    assert "phase2 test model-x system prompt from YAML" in model_prompt
    assert "phase2 test pro judge system prompt from YAML" in judge_prompt
    assert "/redfish/v1/Systems/1" in model_prompt
    assert "show both systems" in judge_prompt

    source = Path("igc/ds/phase2_labelled_requests.py").read_text(encoding="utf-8")
    assert "phase2 test model-x system prompt from YAML" not in source
    assert "${PHASE1_MODEL_X_MODEL_ID}" not in source
    assert "${PHASE2_JUDGE_MODEL_ID}" not in source


def test_sampling_accepts_only_k_1_2_3_and_preserves_record_payloads() -> None:
    """The builder samples one, two, or three REST records with deterministic RNG."""
    records = tuple(_record(index) for index in range(5))

    for width in (1, 2, 3):
        sampled = sample_phase2_contexts(records, k=width, rng=random.Random(7))
        assert len(sampled) == width
        assert all(record.rest_api.startswith("/redfish/v1/Systems/") for record in sampled)
        assert all(record.allowed_methods == ("GET", "HEAD") for record in sampled)
        assert all(record.json_body["@odata.id"] == record.rest_api for record in sampled)

    first = sample_phase2_contexts(records, k=3, rng=random.Random(13))
    second = sample_phase2_contexts(records, k=3, rng=random.Random(13))
    assert [record.rest_api for record in first] == [record.rest_api for record in second]

    with pytest.raises(ValueError, match="sample width"):
        sample_phase2_contexts(records, k=0, rng=random.Random(1))
    with pytest.raises(ValueError, match="sample width"):
        sample_phase2_contexts(records, k=4, rng=random.Random(1))


def test_unordered_set_comparison_and_empty_set_equality() -> None:
    """API-set correctness ignores order unless a separate order signal says otherwise."""
    expected = ["/redfish/v1/A", "/redfish/v1/B"]
    predicted = ["/redfish/v1/B", "/redfish/v1/A"]

    assert compare_rest_api_sets(expected, predicted)
    assert not compare_rest_api_sets(expected, ["/redfish/v1/A"])
    assert compare_rest_api_sets([], [])
    assert empty_set_matches([], [])
    assert not empty_set_matches([], ["/redfish/v1/A"])
    assert not empty_set_matches(["/redfish/v1/A"], [])


def test_pro_judge_result_parsing_accepts_plain_and_wrapped_json() -> None:
    """The parser accepts expected judge JSON shapes and rejects malformed JSON safely."""
    plain = parse_pro_judge_result(
        _judge_json(
            rest_api_list=["/redfish/v1/B", "/redfish/v1/A"],
            order_evidence="explicit_then",
        ),
    )
    assert plain.accepted is True
    assert plain.rest_api_list == ("/redfish/v1/B", "/redfish/v1/A")
    assert plain.order_evidence == "explicit_then"
    assert plain.invalid_json is False

    wrapped = parse_pro_judge_result(json.dumps({"y_pred": json.loads(_judge_json())}))
    assert wrapped.accepted is True
    assert wrapped.invalid_json is False

    invalid = parse_pro_judge_result("{not json")
    assert invalid.accepted is False
    assert invalid.invalid_json is True
    assert invalid.rest_api_list == ()


def test_builder_uses_injected_providers_and_counts_rejections(tmp_path: Path) -> None:
    """Offline build plumbing calls injected providers and summarizes draft quality."""
    spec = load_phase2_labelled_requests_spec(_write_spec(tmp_path / "phase2.yaml"))
    records = tuple(_record(index) for index in range(4))
    seen: dict[str, dict] = {}

    def draft_provider(request: dict) -> str:
        """Return a valid human label and capture model routing details."""
        seen["draft"] = request
        return "show both sampled systems"

    def judge_provider(request: dict) -> str:
        """Accept exactly the sampled REST API set in a different order."""
        seen["judge"] = request
        expected = list(reversed(request["expected_rest_api_list"]))
        return _judge_json(rest_api_list=expected)

    builder = Phase2LabelledRequestBuilder(
        spec,
        draft_provider=draft_provider,
        judge_provider=judge_provider,
    )
    row, counters = builder.build_one(records, k=2, rng=random.Random(3))

    assert row is not None
    data = row.to_dict()
    assert data["dataset"] == PHASE2_LABELLED_REQUESTS
    assert data["task"] == "text_to_rest_api_set"
    assert data["x"]["text"] == "show both sampled systems"
    assert data["validation"]["set_coverage_preserved"] is True
    assert data["validation"]["review_judged"] is True
    assert counters.summary()[_phase2_metric("draft_total")] == 1
    assert counters.summary()[_phase2_metric("accepted_total")] == 1
    assert counters.summary()[_phase2_metric("rest_api_set_match_rate")] == 1.0
    assert seen["draft"]["model_id"] == "${PHASE1_MODEL_X_MODEL_ID}"
    assert seen["draft"]["generation"]["max_new_tokens"] == 96
    assert seen["judge"]["model_id"] == "${PHASE2_JUDGE_MODEL_ID}"
    assert seen["judge"]["profile"] == "${PHASE2_JUDGE_PROFILE}"


def test_builder_returns_none_and_counts_rejection_on_set_mismatch(tmp_path: Path) -> None:
    """Rejected rows still produce namespaced counters for offline accounting."""
    spec = load_phase2_labelled_requests_spec(_write_spec(tmp_path / "phase2.yaml"))
    builder = Phase2LabelledRequestBuilder(
        spec,
        draft_provider=lambda request: "show the sampled system",
        judge_provider=lambda request: _judge_json(
            accepted=True,
            rest_api_list=["/redfish/v1/Systems/not-sampled"],
        ),
    )

    row, counters = builder.build_one((_record(1),), k=1, rng=random.Random(1))
    summary = counters.summary()

    assert row is None
    assert summary[_phase2_metric("draft_total")] == 1
    assert summary[_phase2_metric("accepted_total")] == 0
    assert summary[_phase2_metric("rejected_total")] == 1
    assert summary[_phase2_metric("rest_api_set_match_rate")] == 0.0


def test_counters_track_nonsense_invalid_json_and_empty_set_matches() -> None:
    """Counters expose the generation-quality metrics required for W&B."""
    counters = Phase2LabelledRequestCounters()
    counters.observe_draft("valid request")
    counters.observe_judge(
        parse_pro_judge_result(_judge_json(accepted=True, rest_api_list=[])),
        expected_rest_api_list=(),
    )
    counters.observe_draft("???")
    counters.observe_judge(
        parse_pro_judge_result(_judge_json(accepted=False, nonsense=True)),
        expected_rest_api_list=("/redfish/v1/A",),
    )
    counters.observe_draft("valid but bad judge")
    counters.observe_judge(parse_pro_judge_result("not-json"), expected_rest_api_list=())

    summary = counters.summary()
    assert set(summary) <= set(PHASE2_LABELLED_REQUESTS_WANDB_METRIC_KEYS)
    assert summary[_phase2_metric("draft_total")] == 3
    assert summary[_phase2_metric("accepted_total")] == 1
    assert summary[_phase2_metric("rejected_total")] == 2
    assert summary[_phase2_metric("nonsense_rate")] == pytest.approx(1 / 3)
    assert summary[_phase2_metric("invalid_json_rate")] == pytest.approx(1 / 3)
    assert summary[_phase2_metric("pro_accept_rate")] == pytest.approx(1 / 3)
    assert summary[_phase2_metric("rest_api_set_match_rate")] == pytest.approx(1 / 3)
    assert summary[_phase2_metric("empty_set_match_rate")] == 1.0


def test_phase2_labelled_request_wandb_keys_have_required_namespace_shape() -> None:
    """Every Phase 2 labelled-request metric key stays under one namespace."""
    keys = set(PHASE2_LABELLED_REQUESTS_WANDB_METRIC_KEYS)
    expected = {
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
    }

    assert expected <= keys
    assert all(key.startswith("phase2_labelled_requests/") for key in keys)


def test_minimal_phase3_fixture_keeps_phase3_arguments_separate(tmp_path: Path) -> None:
    """Phase 2 can hand text and APIs to Phase 3 without inventing call labels."""
    spec = load_phase2_labelled_requests_spec(_write_spec(tmp_path / "phase2.yaml"))
    records = (_record(1),)
    builder = Phase2LabelledRequestBuilder(
        spec,
        draft_provider=lambda request: "show system one",
        judge_provider=lambda request: _judge_json(rest_api_list=request["expected_rest_api_list"]),
    )
    row, _ = builder.build_one(records, k=1, rng=random.Random(1))

    phase3_input = to_minimal_phase3_input(row)

    assert phase3_input["text"] == "show system one"
    assert phase3_input["rest_api_list"] == ["/redfish/v1/Systems/1"]
    assert phase3_input["allowed_methods"]["/redfish/v1/Systems/1"] == ["GET", "HEAD"]
    assert "calls" not in phase3_input
    encoded = json.dumps(phase3_input)
    assert '"method":' not in encoded
    assert '"arguments":' not in encoded


def test_minimal_phase3_fixture_rejects_missing_phase2_row() -> None:
    """Phase 3 compatibility helpers fail before fabricating labels."""
    with pytest.raises(ValueError, match="phase2 row is required"):
        to_minimal_phase3_input(None)


# Author: Mus mbayramo@stanford.edu
