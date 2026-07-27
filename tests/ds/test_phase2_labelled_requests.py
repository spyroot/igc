"""Offline tests for the Phase 2 labelled-request dataset plumbing.

The module under test must keep prompts, model identifiers, judge routing,
generation knobs, W&B keys, and acceptance thresholds in YAML/config values.
Tests use tiny Redfish-shaped records and injected fake providers; they never
call a model, W&B, a GPU, a Redfish host, or the network.

Author:
Mus mbayramo@stanford.edu
"""

from __future__ import annotations

import ast
import json
import random
from pathlib import Path
from typing import Any

import pytest

from igc.ds.phase2_labelled_requests import (
    D1SamplingBudget,
    D1_DATASET,
    PHASE2_LABELLED_REQUESTS,
    Phase2LabelledRequestBuilder,
    Phase2LabelledRequestCounters,
    Phase2LabelledRequestRow,
    Phase2LabelledRequestsSpecError,
    RestApiRecord,
    compare_rest_api_sets,
    empty_set_matches,
    evaluate_judge_calibration,
    judge_calibration_passes,
    judge_result_is_accepted,
    load_phase2_labelled_requests_spec,
    parse_pro_judge_result,
    phase2_acceptance_thresholds_pass,
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


def _action_record(index: int) -> RestApiRecord:
    """Build a REST API record with operation and argument metadata."""
    return RestApiRecord(
        rest_api=f"/redfish/v1/Systems/{index}/Actions/ComputerSystem.Reset",
        allowed_methods=("POST",),
        operation_names=("Reset",),
        argument_schema={
            "properties": {"ResetType": {"type": "string"}},
            "required": ["ResetType"],
        },
        json_body={
            "@odata.id": f"/redfish/v1/Systems/{index}/Actions/ComputerSystem.Reset",
            "target": f"/redfish/v1/Systems/{index}/Actions/ComputerSystem.Reset",
        },
        vendor="fixture_vendor",
        source_corpus="fixture_corpus",
    )


def _write_spec(path: Path) -> Path:
    """Write a complete test YAML spec with distinctive literal values."""
    path.write_text(
        """
dataset:
  name: D1
  prompt_spec_version: phase2-labelled-requests-test-v1
sampling:
  sample_widths: [1, 2, 3]
  context_distractors: 4
  empty_set_candidates: 3
  max_accepted_rows: 7
  max_candidates: 11
  max_accepted_per_combination: 2
  max_attempts_per_combination: 3
  max_accepted_per_api: 5
model_x:
  model_id: ${PHASE1_MODEL_X_MODEL_ID}
  artifact_sha: ${PHASE1_MODEL_X_ARTIFACT_SHA}
judge:
  route: private_pro
  model_id: ${PHASE2_JUDGE_MODEL_ID}
  profile: ${PHASE2_JUDGE_PROFILE}
safety:
  live_without_gate_max_candidates: 2
providers:
  draft:
    adapter: mock
    base_url_env: PHASE2_MODEL_X_BASE_URL
    api_key_env: PHASE2_MODEL_X_API_KEY
    endpoint_path: /v1/chat/completions
    timeout_seconds: 10
    response_text_path: choices.0.message.content
  judge:
    adapter: mock
    base_url_env: PHASE2_JUDGE_BASE_URL
    api_key_env: PHASE2_JUDGE_API_KEY
    endpoint_path: /v1/chat/completions
    timeout_seconds: 10
    response_text_path: choices.0.message.content
    payload_request_fields:
      - route
      - profile
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
  model_x_empty_set_draft:
    system: phase2 test model-x empty-set system prompt from YAML
    template: |
      Draft one operator request that matches none of these records:
      {records_json}
  pro_judge:
    system: phase2 test pro judge system prompt from YAML
    template: |
      Judge whether this request maps to the same unordered REST API set.
      Records:
      {records_json}
      Draft:
      {draft_text}
  pro_judge_empty_set:
    system: phase2 test pro judge empty-set system prompt from YAML
    template: |
      Judge whether this request correctly matches none of these records.
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
    - phase2_labelled_requests/natural_command_rate
    - phase2_labelled_requests/ambiguous_rate
    - phase2_labelled_requests/duplicate_intent_rate
    - phase2_labelled_requests/extra_intent_rate
    - phase2_labelled_requests/method_semantics_valid_rate
    - phase2_labelled_requests/empty_set_match_rate
    - phase2_labelled_requests/empty_set_expected_total
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
judge_calibration:
  min_precision: 0.99
  min_recall: 0.90
  max_false_accept_rate: 0.01
""",
        encoding="utf-8",
    )
    return path


def _judge_json(
    *,
    accepted: bool = True,
    rest_api_list: list[str] | None = None,
    natural: bool = True,
    nonsense: bool = False,
    ambiguous: bool = False,
    duplicate_intent: bool = False,
    extra_intents: bool = False,
    method_semantics_valid: bool = True,
    order_evidence: str = "none",
) -> str:
    """Return a compact fake Pro judge JSON response."""
    _ = order_evidence
    return json.dumps(
        {
            "accepted": accepted,
            "natural": natural,
            "nonsense": nonsense,
            "ambiguous": ambiguous,
            "duplicate_intent": duplicate_intent,
            "extra_intents": extra_intents,
            "method_semantics_valid": method_semantics_valid,
            "covered_api_set": [] if rest_api_list is None else rest_api_list,
            "reason": "fixture",
        },
    )


def _phase2_metric(group: str, name: str | None = None) -> str:
    """Return a required Phase 2 labelled-request metric key."""
    return phase_metric(PHASE2_LABELLED_REQUESTS, group, name)


def test_loads_prompt_model_judge_generation_and_thresholds_from_yaml(tmp_path: Path) -> None:
    """The Phase 2 spec loader keeps every runtime knob in YAML."""
    spec = load_phase2_labelled_requests_spec(_write_spec(tmp_path / "phase2.yaml"))

    assert spec.dataset_name == D1_DATASET
    assert spec.prompt_spec_version == "phase2-labelled-requests-test-v1"
    assert spec.sample_widths == (1, 2, 3)
    assert spec.context_distractors == 4
    assert spec.empty_set_candidates == 3
    assert spec.max_accepted_rows == 7
    assert spec.max_candidates == 11
    assert spec.max_accepted_per_combination == 2
    assert spec.max_attempts_per_combination == 3
    assert spec.max_accepted_per_api == 5
    assert spec.model_x.model_id == "${PHASE1_MODEL_X_MODEL_ID}"
    assert spec.model_x.artifact_sha == "${PHASE1_MODEL_X_ARTIFACT_SHA}"
    assert spec.empty_set_model_x_system == (
        "phase2 test model-x empty-set system prompt from YAML"
    )
    assert "matches none" in spec.empty_set_model_x_template
    assert spec.judge.route == "private_pro"
    assert spec.judge.model_id == "${PHASE2_JUDGE_MODEL_ID}"
    assert spec.judge.profile == "${PHASE2_JUDGE_PROFILE}"
    assert spec.live_without_gate_max_candidates == 2
    assert spec.draft_provider.adapter == "mock"
    assert spec.draft_provider.base_url_env == "PHASE2_MODEL_X_BASE_URL"
    assert spec.draft_provider.endpoint_path == "/v1/chat/completions"
    assert spec.draft_provider.response_text_path == "choices.0.message.content"
    assert spec.judge_provider.adapter == "mock"
    assert spec.judge_provider.payload_request_fields == ("route", "profile")
    assert spec.generation == {"max_new_tokens": 96, "temperature": 0.2, "top_p": 0.95}
    assert spec.wandb_namespace == PHASE2_LABELLED_REQUESTS
    assert spec.metric_keys == PHASE2_LABELLED_REQUESTS_WANDB_METRIC_KEYS
    assert spec.acceptance_thresholds["min_pro_accept_rate"] == 0.9
    assert spec.judge_calibration_thresholds == {
        "min_precision": 0.99,
        "min_recall": 0.90,
        "max_false_accept_rate": 0.01,
    }


def test_spec_loader_rejects_malformed_phase2_specs(tmp_path: Path) -> None:
    """Spec validation fails closed for bad YAML shape and contract drift."""
    not_mapping = tmp_path / "not-mapping.yaml"
    not_mapping.write_text("- D1\n", encoding="utf-8")
    with pytest.raises(Phase2LabelledRequestsSpecError, match="must be a mapping"):
        load_phase2_labelled_requests_spec(not_mapping)

    wrong_dataset = _write_spec(tmp_path / "wrong-dataset.yaml")
    wrong_dataset.write_text(
        wrong_dataset.read_text(encoding="utf-8").replace(
            "name: D1",
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

    wrong_widths = _write_spec(tmp_path / "wrong-widths.yaml")
    wrong_widths.write_text(
        wrong_widths.read_text(encoding="utf-8").replace(
            "sample_widths: [1, 2, 3]",
            "sample_widths: [1, 2]",
        ),
        encoding="utf-8",
    )
    with pytest.raises(Phase2LabelledRequestsSpecError, match="sample_widths"):
        load_phase2_labelled_requests_spec(wrong_widths)

    non_integer_widths = _write_spec(tmp_path / "non-integer-widths.yaml")
    non_integer_widths.write_text(
        non_integer_widths.read_text(encoding="utf-8").replace(
            "sample_widths: [1, 2, 3]",
            "sample_widths: [1, two, 3]",
        ),
        encoding="utf-8",
    )
    with pytest.raises(Phase2LabelledRequestsSpecError, match="sample_widths"):
        load_phase2_labelled_requests_spec(non_integer_widths)

    too_few_distractors = _write_spec(tmp_path / "too-few-distractors.yaml")
    too_few_distractors.write_text(
        too_few_distractors.read_text(encoding="utf-8").replace(
            "  context_distractors: 4",
            "  context_distractors: 3",
        ),
        encoding="utf-8",
    )
    with pytest.raises(Phase2LabelledRequestsSpecError, match="context_distractors"):
        load_phase2_labelled_requests_spec(too_few_distractors)

    missing_finite_control = _write_spec(tmp_path / "missing-finite-control.yaml")
    missing_finite_control.write_text(
        missing_finite_control.read_text(encoding="utf-8").replace(
            "  max_candidates: 11\n",
            "",
        ),
        encoding="utf-8",
    )
    with pytest.raises(Phase2LabelledRequestsSpecError, match="max_candidates"):
        load_phase2_labelled_requests_spec(missing_finite_control)

    zero_finite_control = _write_spec(tmp_path / "zero-finite-control.yaml")
    zero_finite_control.write_text(
        zero_finite_control.read_text(encoding="utf-8").replace(
            "  max_accepted_per_api: 5",
            "  max_accepted_per_api: 0",
        ),
        encoding="utf-8",
    )
    with pytest.raises(Phase2LabelledRequestsSpecError, match="max_accepted_per_api"):
        load_phase2_labelled_requests_spec(zero_finite_control)

    invalid_budget_relationship = _write_spec(tmp_path / "invalid-budget.yaml")
    invalid_budget_relationship.write_text(
        invalid_budget_relationship.read_text(encoding="utf-8").replace(
            "  max_accepted_per_combination: 2",
            "  max_accepted_per_combination: 4",
        ),
        encoding="utf-8",
    )
    with pytest.raises(Phase2LabelledRequestsSpecError, match="cannot exceed"):
        load_phase2_labelled_requests_spec(invalid_budget_relationship)

    wrong_namespace = _write_spec(tmp_path / "wrong-namespace.yaml")
    wrong_namespace.write_text(
        wrong_namespace.read_text(encoding="utf-8").replace(
            "namespace: phase2_labelled_requests",
            "namespace: wrong_phase2_namespace",
        ),
        encoding="utf-8",
    )
    with pytest.raises(Phase2LabelledRequestsSpecError, match="wandb.namespace"):
        load_phase2_labelled_requests_spec(wrong_namespace)

    missing_judge_route = _write_spec(tmp_path / "missing-judge-route.yaml")
    missing_judge_route.write_text(
        missing_judge_route.read_text(encoding="utf-8").replace("  route: private_pro\n", ""),
        encoding="utf-8",
    )
    with pytest.raises(Phase2LabelledRequestsSpecError, match="judge.route"):
        load_phase2_labelled_requests_spec(missing_judge_route)

    missing_providers = _write_spec(tmp_path / "missing-providers.yaml")
    missing_providers.write_text(
        missing_providers.read_text(encoding="utf-8").replace(
            """providers:
  draft:
    adapter: mock
    base_url_env: PHASE2_MODEL_X_BASE_URL
    api_key_env: PHASE2_MODEL_X_API_KEY
    endpoint_path: /v1/chat/completions
    timeout_seconds: 10
    response_text_path: choices.0.message.content
  judge:
    adapter: mock
    base_url_env: PHASE2_JUDGE_BASE_URL
    api_key_env: PHASE2_JUDGE_API_KEY
    endpoint_path: /v1/chat/completions
    timeout_seconds: 10
    response_text_path: choices.0.message.content
    payload_request_fields:
      - route
      - profile
""",
            "",
        ),
        encoding="utf-8",
    )
    with pytest.raises(Phase2LabelledRequestsSpecError, match="providers"):
        load_phase2_labelled_requests_spec(missing_providers)

    bad_provider = _write_spec(tmp_path / "bad-provider.yaml")
    bad_provider.write_text(
        bad_provider.read_text(encoding="utf-8").replace(
            "    adapter: mock\n",
            "    adapter: unknown\n",
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(Phase2LabelledRequestsSpecError, match="providers.draft.adapter"):
        load_phase2_labelled_requests_spec(bad_provider)

    live_without_base_url = _write_spec(tmp_path / "live-without-base-url.yaml")
    live_without_base_url.write_text(
        live_without_base_url.read_text(encoding="utf-8")
        .replace("    adapter: mock\n", "    adapter: openai-compatible\n", 1)
        .replace("    base_url_env: PHASE2_MODEL_X_BASE_URL\n", "", 1),
        encoding="utf-8",
    )
    with pytest.raises(Phase2LabelledRequestsSpecError, match="providers.draft.base_url_env"):
        load_phase2_labelled_requests_spec(live_without_base_url)

    malformed_payload_fields = _write_spec(tmp_path / "malformed-payload-fields.yaml")
    malformed_payload_fields.write_text(
        malformed_payload_fields.read_text(encoding="utf-8").replace(
            "      - route\n      - profile",
            "      - route\n      - 7",
        ),
        encoding="utf-8",
    )
    with pytest.raises(Phase2LabelledRequestsSpecError, match="payload_request_fields"):
        load_phase2_labelled_requests_spec(malformed_payload_fields)

    malformed_live_gate = _write_spec(tmp_path / "malformed-live-gate.yaml")
    malformed_live_gate.write_text(
        malformed_live_gate.read_text(encoding="utf-8").replace(
            "  live_without_gate_max_candidates: 2",
            "  live_without_gate_max_candidates: many",
        ),
        encoding="utf-8",
    )
    with pytest.raises(Phase2LabelledRequestsSpecError, match="live_without_gate_max_candidates"):
        load_phase2_labelled_requests_spec(malformed_live_gate)

    missing_prompt_section = _write_spec(tmp_path / "missing-prompt-section.yaml")
    missing_prompt_section.write_text(
        missing_prompt_section.read_text(encoding="utf-8").replace(
            "  model_x_draft:",
            "  model_x_missing:",
        ),
        encoding="utf-8",
    )
    with pytest.raises(Phase2LabelledRequestsSpecError, match="model_x_draft"):
        load_phase2_labelled_requests_spec(missing_prompt_section)

    missing_model_prompt_field = _write_spec(tmp_path / "missing-model-prompt-field.yaml")
    missing_model_prompt_field.write_text(
        missing_model_prompt_field.read_text(encoding="utf-8").replace(
            "      {records_json}",
            "      no record placeholder",
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(Phase2LabelledRequestsSpecError, match="records_json"):
        load_phase2_labelled_requests_spec(missing_model_prompt_field)

    unknown_judge_prompt_field = _write_spec(tmp_path / "unknown-judge-prompt-field.yaml")
    unknown_judge_prompt_field.write_text(
        unknown_judge_prompt_field.read_text(encoding="utf-8").replace(
            "      {draft_text}",
            "      {draft_text}\n      {unknown_field}",
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(Phase2LabelledRequestsSpecError, match="unknown_field"):
        load_phase2_labelled_requests_spec(unknown_judge_prompt_field)

    malformed_model_prompt_field = _write_spec(tmp_path / "malformed-model-prompt-field.yaml")
    malformed_model_prompt_field.write_text(
        malformed_model_prompt_field.read_text(encoding="utf-8").replace(
            "      {records_json}",
            "      {records_json",
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(Phase2LabelledRequestsSpecError, match="malformed format fields"):
        load_phase2_labelled_requests_spec(malformed_model_prompt_field)

    unnamed_model_prompt_field = _write_spec(tmp_path / "unnamed-model-prompt-field.yaml")
    unnamed_model_prompt_field.write_text(
        unnamed_model_prompt_field.read_text(encoding="utf-8").replace(
            "      {records_json}",
            "      {records_json}\n      {}",
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(Phase2LabelledRequestsSpecError, match="unnamed format fields"):
        load_phase2_labelled_requests_spec(unnamed_model_prompt_field)

    malformed_generation = _write_spec(tmp_path / "malformed-generation.yaml")
    malformed_generation.write_text(
        malformed_generation.read_text(encoding="utf-8").replace(
            "generation:\n  max_new_tokens: 96\n  temperature: 0.2\n  top_p: 0.95\n",
            "generation: []\n",
        ),
        encoding="utf-8",
    )
    with pytest.raises(Phase2LabelledRequestsSpecError, match="generation"):
        load_phase2_labelled_requests_spec(malformed_generation)

    missing_threshold = _write_spec(tmp_path / "missing-threshold.yaml")
    missing_threshold.write_text(
        missing_threshold.read_text(encoding="utf-8").replace(
            "  max_invalid_json_rate: 0.01\n",
            "",
        ),
        encoding="utf-8",
    )
    with pytest.raises(Phase2LabelledRequestsSpecError, match="acceptance missing"):
        load_phase2_labelled_requests_spec(missing_threshold)

    malformed_threshold = _write_spec(tmp_path / "malformed-threshold.yaml")
    malformed_threshold.write_text(
        malformed_threshold.read_text(encoding="utf-8").replace(
            "  min_pro_accept_rate: 0.9\n",
            "  min_pro_accept_rate: high\n",
        ),
        encoding="utf-8",
    )
    with pytest.raises(Phase2LabelledRequestsSpecError, match="must be numeric"):
        load_phase2_labelled_requests_spec(malformed_threshold)

    missing_calibration = _write_spec(tmp_path / "missing-calibration.yaml")
    missing_calibration.write_text(
        missing_calibration.read_text(encoding="utf-8").replace(
            "  max_false_accept_rate: 0.01\n",
            "",
        ),
        encoding="utf-8",
    )
    with pytest.raises(Phase2LabelledRequestsSpecError, match="judge_calibration missing"):
        load_phase2_labelled_requests_spec(missing_calibration)


def test_committed_phase2_labelled_requests_config_loads() -> None:
    """The checked-in builder spec stays aligned with the metric registry."""
    spec = load_phase2_labelled_requests_spec("configs/phase2_labelled_requests.yaml")

    assert spec.dataset_name == D1_DATASET
    assert spec.metric_keys == PHASE2_LABELLED_REQUESTS_WANDB_METRIC_KEYS
    assert spec.sample_widths == (1, 2, 3)
    assert spec.context_distractors == 4
    assert spec.max_accepted_rows == 100000
    assert spec.max_candidates == 300000
    assert spec.max_accepted_per_combination == 8
    assert spec.max_attempts_per_combination == 24
    assert spec.max_accepted_per_api == 200
    assert spec.model_x.model_id == "${PHASE1_MODEL_X_MODEL_ID}"
    assert spec.model_x.artifact_sha == "${PHASE1_MODEL_X_ARTIFACT_SHA}"
    assert spec.judge.route == "${PHASE2_JUDGE_ROUTE}"
    assert spec.judge.model_id == "${PHASE2_JUDGE_MODEL_ID}"
    assert spec.judge.profile == "${PHASE2_JUDGE_PROFILE}"
    assert spec.live_without_gate_max_candidates == 3
    assert spec.draft_provider.adapter == "mock"
    assert spec.draft_provider.base_url_env == "PHASE2_MODEL_X_BASE_URL"
    assert spec.judge_provider.adapter == "mock"
    assert spec.judge_provider.base_url_env == "PHASE2_JUDGE_BASE_URL"
    assert spec.judge_provider.payload_request_fields == ("route", "profile")
    assert spec.judge_calibration_thresholds == {
        "min_precision": 0.99,
        "min_recall": 0.90,
        "max_false_accept_rate": 0.01,
    }


def test_committed_phase2_labelled_requests_prompts_render_from_yaml() -> None:
    """The checked-in prompt spec renders without leaving template placeholders."""
    spec = load_phase2_labelled_requests_spec("configs/phase2_labelled_requests.yaml")
    records = (_record(1), _record(2))

    model_prompt = render_model_x_prompt(spec, records)
    judge_prompt = render_pro_judge_prompt(spec, records, "show both fixture systems")

    assert spec.model_x_system in model_prompt
    assert spec.judge_system in judge_prompt
    assert "/redfish/v1/Systems/1" in model_prompt
    assert "/redfish/v1/Systems/2" in judge_prompt
    assert "show both fixture systems" in judge_prompt
    assert "{records_json}" not in model_prompt
    assert "{draft_text}" not in judge_prompt


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

    runtime_sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            Path("igc/ds/phase2_labelled_requests.py"),
            Path("scripts/build_phase2_labelled_requests.py"),
        )
    )
    forbidden_literals = (
        "phase2 test model-x system prompt from YAML",
        "${PHASE1_MODEL_X_MODEL_ID}",
        "${PHASE2_JUDGE_MODEL_ID}",
        "Qwen/Qwen2.5",
        "deepseek",
        "You draft one concise",
        "Return JSON with accepted",
    )
    for literal in forbidden_literals:
        assert literal not in runtime_sources

    lowercase_runtime_sources = runtime_sources.lower()
    forbidden_lowercase_literals = (
        "phase2 test model-x system prompt from yaml",
        "qwen/qwen2.5",
        "deepseek-v4",
        "return json with accepted",
    )
    for literal in forbidden_lowercase_literals:
        assert literal not in lowercase_runtime_sources


def test_rest_api_record_preserves_operation_metadata_in_prompt_and_context() -> None:
    """Operation names and argument schemas survive prompt and stored D1 context paths."""
    record = _action_record(1)

    prompt_dict = record.to_prompt_dict()
    phase2_row = Phase2LabelledRequestRow(
        text="reset the selected system",
        records=(record, *tuple(_record(index) for index in range(2, 6))),
        rest_api_list=(record.rest_api,),
        prompt_spec_version="phase2-labelled-requests-test-v1",
        sample_width_k=1,
        validation={
            "valid_json": True,
            "accepted": True,
            "natural": True,
            "nonsense": False,
            "ambiguous": False,
            "duplicate_intent": False,
            "extra_intents": False,
            "method_semantics_valid": True,
            "covered_api_set": [record.rest_api],
        },
    )
    row = phase2_row.to_dict()
    context = next(
        item for item in row["x"]["api_context"]
        if item["rest_api"] == record.rest_api
    )
    phase3_input = to_minimal_phase3_input(phase2_row)
    phase3_context = next(
        item for item in phase3_input["api_context"]
        if item["rest_api"] == record.rest_api
    )

    assert prompt_dict["operation_names"] == ["Reset"]
    assert prompt_dict["argument_schema"] == {
        "properties": {"ResetType": {"type": "string"}},
        "required": ["ResetType"],
    }
    assert context["operation_names"] == ["Reset"]
    assert context["argument_schema"] == prompt_dict["argument_schema"]
    assert phase3_context["operation_names"] == ["Reset"]
    assert phase3_context["argument_schema"] == prompt_dict["argument_schema"]


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
    with pytest.raises(ValueError, match="not enough records"):
        sample_phase2_contexts(records[:1], k=2, rng=random.Random(1))


@pytest.mark.parametrize(
    "ceiling",
    [
        "max_candidates",
        "max_attempts_per_combination",
        "max_accepted_per_combination",
        "max_accepted_per_api",
        "max_accepted_rows",
    ],
)
def test_d1_sampling_budget_stops_provider_calls_before_exhausted_ceilings(
    tmp_path: Path,
    ceiling: str,
) -> None:
    """No draft or judge provider call is made after any finite D1 budget is hit."""
    spec = load_phase2_labelled_requests_spec(_write_spec(tmp_path / "phase2.yaml"))
    records = tuple(_record(index) for index in range(1, 6))
    combinations = [(record.rest_api,) for record in records]
    budget = D1SamplingBudget(
        max_accepted_rows=1,
        max_candidates=1,
        max_accepted_per_combination=1,
        max_attempts_per_combination=1,
        max_accepted_per_api=1,
    )
    if ceiling == "max_candidates":
        budget.attempts_total = 1
    elif ceiling == "max_attempts_per_combination":
        for combination in combinations:
            budget.attempts_by_combination[combination] = 1
    elif ceiling == "max_accepted_per_combination":
        for combination in combinations:
            budget.accepted_by_combination[combination] = 1
    elif ceiling == "max_accepted_per_api":
        for record in records:
            budget.accepted_by_api[record.rest_api] = 1
    elif ceiling == "max_accepted_rows":
        budget.accepted_total = 1
    calls = {"draft": 0, "judge": 0}
    builder = Phase2LabelledRequestBuilder(
        spec,
        draft_provider=lambda _request: calls.__setitem__("draft", calls["draft"] + 1)
        or "read the system",
        judge_provider=lambda request: calls.__setitem__("judge", calls["judge"] + 1)
        or _judge_json(rest_api_list=request["expected_rest_api_list"]),
        sampling_budget=budget,
    )

    row, counters = builder.build_one(records, k=1, rng=random.Random(1))

    assert row is None
    assert calls == {"draft": 0, "judge": 0}
    assert counters.summary()[_phase2_metric("draft_total")] == 0


def test_builder_rejects_insufficient_unique_source_pool_before_budget_or_providers(
    tmp_path: Path,
) -> None:
    """Positive k requires k plus configured distractors before generation starts."""
    spec = load_phase2_labelled_requests_spec(_write_spec(tmp_path / "phase2.yaml"))
    budget = D1SamplingBudget.from_spec(spec)
    calls = {"draft": 0, "judge": 0}

    def fail_draft(_request):
        calls["draft"] += 1
        raise AssertionError("draft provider must not be called")

    def fail_judge(_request):
        calls["judge"] += 1
        raise AssertionError("judge provider must not be called")

    builder = Phase2LabelledRequestBuilder(
        spec,
        draft_provider=fail_draft,
        judge_provider=fail_judge,
        sampling_budget=budget,
    )

    with pytest.raises(ValueError, match="targets plus distractors"):
        builder.build_one(
            tuple(_record(index) for index in range(spec.context_distractors)),
            k=1,
            rng=random.Random(1),
        )

    assert calls == {"draft": 0, "judge": 0}
    assert budget.summary()["observed"] == {
        "attempts_total": 0,
        "accepted_total": 0,
        "empty_set_attempts_total": 0,
        "empty_set_accepted_total": 0,
        "unique_combinations_attempted": 0,
    }


def test_builder_rejects_duplicate_rest_api_source_pool_before_budget_or_providers(
    tmp_path: Path,
) -> None:
    """Duplicate source records fail closed before draft/judge calls or budget use."""
    spec = load_phase2_labelled_requests_spec(_write_spec(tmp_path / "phase2.yaml"))
    duplicate = _record(1)
    records = (duplicate, duplicate, *tuple(_record(index) for index in range(2, 7)))
    budget = D1SamplingBudget.from_spec(spec)
    calls = {"draft": 0, "judge": 0}

    builder = Phase2LabelledRequestBuilder(
        spec,
        draft_provider=lambda _request: calls.__setitem__("draft", calls["draft"] + 1)
        or "must not draft",
        judge_provider=lambda _request: calls.__setitem__("judge", calls["judge"] + 1)
        or _judge_json(rest_api_list=[]),
        sampling_budget=budget,
    )

    with pytest.raises(ValueError, match="unique rest_api"):
        builder.build_one(records, k=1, rng=random.Random(1))

    assert calls == {"draft": 0, "judge": 0}
    assert budget.summary()["observed"] == {
        "attempts_total": 0,
        "accepted_total": 0,
        "empty_set_attempts_total": 0,
        "empty_set_accepted_total": 0,
        "unique_combinations_attempted": 0,
    }


def test_d1_sampling_budget_summary_exposes_limits_and_observed_counts(
    tmp_path: Path,
) -> None:
    """Budget summaries are bounded counters only: limits and observed counts."""
    spec = load_phase2_labelled_requests_spec(_write_spec(tmp_path / "phase2.yaml"))
    budget = D1SamplingBudget.from_spec(spec)
    builder = Phase2LabelledRequestBuilder(
        spec,
        draft_provider=lambda _request: "read the selected system",
        judge_provider=lambda request: _judge_json(
            rest_api_list=request["expected_rest_api_list"],
        ),
        sampling_budget=budget,
    )

    row, _counters = builder.build_one(
        tuple(_record(index) for index in range(8)),
        k=1,
        rng=random.Random(1),
    )

    assert row is not None
    assert budget.summary() == {
        "limits": {
            "max_accepted_rows": 7,
            "max_candidates": 11,
            "max_accepted_per_combination": 2,
            "max_attempts_per_combination": 3,
            "max_accepted_per_api": 5,
            "max_empty_set_candidates": 3,
        },
        "observed": {
            "attempts_total": 1,
            "accepted_total": 1,
            "empty_set_attempts_total": 0,
            "empty_set_accepted_total": 0,
            "unique_combinations_attempted": 1,
        },
    }


def test_unordered_set_comparison_and_empty_set_equality() -> None:
    """API-set correctness ignores order unless a separate order signal says otherwise."""
    expected = ["/redfish/v1/A", "/redfish/v1/B"]
    predicted = ["/redfish/v1/B", "/redfish/v1/A"]

    assert compare_rest_api_sets(expected, predicted)
    assert not compare_rest_api_sets(expected, ["/redfish/v1/A"])
    assert not compare_rest_api_sets(expected, expected + ["/redfish/v1/C"])
    assert compare_rest_api_sets([], [])
    assert empty_set_matches([], [])
    assert not empty_set_matches([], ["/redfish/v1/A"])
    assert not empty_set_matches(["/redfish/v1/A"], [])
    assert not empty_set_matches(["/redfish/v1/A"], ["/redfish/v1/A"])


def test_pro_judge_result_parsing_accepts_plain_and_wrapped_json() -> None:
    """The parser accepts expected judge JSON shapes and rejects malformed JSON safely."""
    plain = parse_pro_judge_result(
        _judge_json(
            rest_api_list=["/redfish/v1/B", "/redfish/v1/A"],
        ),
    )
    assert plain.accepted is True
    assert plain.covered_api_set == ("/redfish/v1/B", "/redfish/v1/A")
    assert plain.natural is True
    assert plain.ambiguous is False
    assert plain.duplicate_intent is False
    assert plain.extra_intents is False
    assert plain.method_semantics_valid is True
    assert plain.invalid_json is False

    wrapped = parse_pro_judge_result(json.dumps({"y_pred": json.loads(_judge_json())}))
    assert wrapped.accepted is True
    assert wrapped.invalid_json is False
    assert wrapped.nonsense is False
    assert wrapped.covered_api_set == ()

    rejected = parse_pro_judge_result(
        _judge_json(
            accepted=False,
            rest_api_list=["/redfish/v1/A"],
            nonsense=False,
        ),
    )
    assert rejected.accepted is False
    assert rejected.invalid_json is False
    assert rejected.covered_api_set == ("/redfish/v1/A",)
    assert rejected.nonsense is False

    invalid = parse_pro_judge_result("{not json")
    assert invalid.accepted is False
    assert invalid.invalid_json is True
    assert invalid.covered_api_set == ()

    non_object = parse_pro_judge_result(json.dumps(["not", "a", "mapping"]))
    assert non_object.accepted is False
    assert non_object.invalid_json is True
    assert non_object.covered_api_set == ()
    assert "not a mapping" in non_object.reason


def test_pro_judge_result_parsing_requires_nonsense_boolean() -> None:
    """Judge output must carry an explicit nonsense verdict."""
    missing = parse_pro_judge_result(
        json.dumps({
            "accepted": True,
            "natural": True,
            "ambiguous": False,
            "duplicate_intent": False,
            "extra_intents": False,
            "method_semantics_valid": True,
            "covered_api_set": ["/redfish/v1/Systems/1"],
            "reason": "fixture",
        }),
    )
    assert missing.accepted is False
    assert missing.invalid_json is True
    assert "nonsense" in missing.reason


def test_pro_judge_result_parsing_requires_accepted_boolean() -> None:
    """Judge output must carry an explicit accepted verdict."""
    missing = parse_pro_judge_result(
        json.dumps({
            "natural": True,
            "nonsense": False,
            "ambiguous": False,
            "duplicate_intent": False,
            "extra_intents": False,
            "method_semantics_valid": True,
            "covered_api_set": ["/redfish/v1/Systems/1"],
            "reason": "fixture",
        }),
    )
    assert missing.accepted is False
    assert missing.invalid_json is True
    assert "accepted" in missing.reason

    payload = json.loads(_judge_json(rest_api_list=["/redfish/v1/Systems/1"]))
    payload["accepted"] = "yes"
    malformed = parse_pro_judge_result(json.dumps(payload))
    assert malformed.accepted is False
    assert malformed.invalid_json is True
    assert "accepted" in malformed.reason
    assert "boolean" in malformed.reason


def test_pro_judge_result_parsing_rejects_extra_order_evidence_field() -> None:
    """Order evidence is not part of the strict Phase 2 judge verdict."""
    payload = json.loads(_judge_json(rest_api_list=["/redfish/v1/Systems/1"]))
    payload["order_evidence"] = "explicit_then"
    result = parse_pro_judge_result(json.dumps(payload))
    assert result.accepted is False
    assert result.invalid_json is True
    assert "exactly" in result.reason


def test_pro_judge_result_parsing_rejects_rest_api_set_alias() -> None:
    """The strict judge parser rejects legacy REST API aliases."""
    result = parse_pro_judge_result(
        json.dumps({
            "accepted": True,
            "rest_api_set": ["/redfish/v1/Systems/2", "/redfish/v1/Systems/1"],
            "natural": True,
            "nonsense": False,
            "ambiguous": False,
            "duplicate_intent": False,
            "extra_intents": False,
            "method_semantics_valid": True,
            "reason": "fixture",
        }),
    )

    assert result.accepted is False
    assert result.invalid_json is True
    assert result.covered_api_set == ()


def test_pro_judge_result_parsing_rejects_accept_boolean_alias() -> None:
    """The strict judge parser rejects the legacy accept boolean alias."""
    result = parse_pro_judge_result(
        json.dumps({
            "accept": True,
            "natural": True,
            "nonsense": False,
            "ambiguous": False,
            "duplicate_intent": False,
            "extra_intents": False,
            "method_semantics_valid": True,
            "covered_api_set": ["/redfish/v1/Systems/1"],
            "reason": "fixture",
        }),
    )

    assert result.accepted is False
    assert result.invalid_json is True
    assert result.covered_api_set == ()


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    (
        ("covered_api_set", "/redfish/v1/Systems"),
        ("covered_api_set", ["/redfish/v1/Systems", 7]),
        ("covered_api_set", None),
    ),
)
def test_pro_judge_result_parsing_counts_malformed_rest_api_fields_as_invalid(
    field_name: str,
    field_value: Any,
) -> None:
    """Malformed judge REST API fields are invalid output, not exceptions."""
    result = parse_pro_judge_result(
        json.dumps({
            "accepted": True,
            "natural": True,
            field_name: field_value,
            "nonsense": False,
            "ambiguous": False,
            "duplicate_intent": False,
            "extra_intents": False,
            "method_semantics_valid": True,
            "reason": "fixture",
        }),
    )

    assert result.accepted is False
    assert result.invalid_json is True
    assert result.covered_api_set == ()
    assert field_name in result.reason


def test_pro_judge_result_parsing_requires_rest_api_field() -> None:
    """A bare accepted judge result is malformed, not an accepted empty set."""
    result = parse_pro_judge_result(
        json.dumps({
            "accepted": True,
            "natural": True,
            "nonsense": False,
            "ambiguous": False,
            "duplicate_intent": False,
            "extra_intents": False,
            "method_semantics_valid": True,
            "reason": "fixture",
        }),
    )

    assert result.accepted is False
    assert result.invalid_json is True
    assert result.covered_api_set == ()
    assert "covered_api_set" in result.reason


def test_pro_judge_result_parsing_requires_acceptance_boolean() -> None:
    """A judge result without accepted or accept is malformed output."""
    result = parse_pro_judge_result(
        json.dumps({
            "natural": True,
            "nonsense": False,
            "ambiguous": False,
            "duplicate_intent": False,
            "extra_intents": False,
            "method_semantics_valid": True,
            "covered_api_set": ["/redfish/v1/Systems"],
            "reason": "fixture",
        }),
    )

    assert result.accepted is False
    assert result.invalid_json is True
    assert result.covered_api_set == ()
    assert "accepted" in result.reason


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    (
        ("accepted", "true"),
        ("accepted", 1),
        ("accepted", None),
        ("natural", "true"),
        ("natural", 1),
        ("natural", None),
        ("nonsense", "false"),
        ("nonsense", 0),
        ("nonsense", None),
        ("ambiguous", "false"),
        ("ambiguous", 0),
        ("duplicate_intent", "false"),
        ("duplicate_intent", 0),
        ("extra_intents", "false"),
        ("extra_intents", 0),
        ("method_semantics_valid", "true"),
        ("method_semantics_valid", 1),
    ),
)
def test_pro_judge_result_parsing_counts_malformed_booleans_as_invalid(
    field_name: str,
    field_value: Any,
) -> None:
    """Malformed judge booleans are invalid output, not truthy/falsy coercions."""
    payload: dict[str, Any] = json.loads(
        _judge_json(rest_api_list=["/redfish/v1/Systems"]),
    )
    payload[field_name] = field_value

    result = parse_pro_judge_result(
        json.dumps(payload),
    )

    assert result.accepted is False
    assert result.invalid_json is True
    assert result.covered_api_set == ()
    assert field_name in result.reason


def test_strict_judge_acceptance_requires_every_semantic_flag() -> None:
    """A row is accepted only when every judge flag and the API set match."""
    selected = ("/redfish/v1/Systems/1", "/redfish/v1/Managers/1")
    accepted = parse_pro_judge_result(_judge_json(rest_api_list=list(reversed(selected))))

    assert judge_result_is_accepted(accepted, selected_api_set=selected)

    cases = {
        "valid_json": "not-json",
        "accepted": _judge_json(accepted=False, rest_api_list=list(selected)),
        "natural": _judge_json(natural=False, rest_api_list=list(selected)),
        "nonsense": _judge_json(nonsense=True, rest_api_list=list(selected)),
        "ambiguous": _judge_json(ambiguous=True, rest_api_list=list(selected)),
        "duplicate_intent": _judge_json(duplicate_intent=True, rest_api_list=list(selected)),
        "extra_intents": _judge_json(extra_intents=True, rest_api_list=list(selected)),
        "method_semantics_valid": _judge_json(
            method_semantics_valid=False,
            rest_api_list=list(selected),
        ),
        "covered_api_set_missing": _judge_json(rest_api_list=[selected[0]]),
        "covered_api_set_extra": _judge_json(rest_api_list=[*selected, "/redfish/v1/Chassis/1"]),
    }

    for label, raw in cases.items():
        verdict = parse_pro_judge_result(raw)
        assert not judge_result_is_accepted(verdict, selected_api_set=selected), label


def test_judge_calibration_floors_require_precision_recall_and_false_accept_rate(
    tmp_path: Path,
) -> None:
    """Calibration must pass all YAML floors: .99 precision, .90 recall, .01 false accept."""
    spec = load_phase2_labelled_requests_spec(_write_spec(tmp_path / "phase2.yaml"))
    target = ("/redfish/v1/Systems/1",)
    strict_accept = parse_pro_judge_result(_judge_json(rest_api_list=list(target)))
    strict_reject = parse_pro_judge_result(_judge_json(accepted=False, rest_api_list=list(target)))

    passing = evaluate_judge_calibration([
        (strict_accept, target, True),
        (strict_accept, target, True),
        (strict_reject, target, False),
        (strict_reject, target, False),
    ])
    low_precision = evaluate_judge_calibration([
        (strict_accept, target, True),
        (strict_accept, target, False),
    ])
    low_recall = evaluate_judge_calibration([
        (strict_accept, target, True),
        (strict_reject, target, True),
        (strict_reject, target, False),
    ])
    false_accept = evaluate_judge_calibration([
        (strict_accept, target, True),
        (strict_accept, target, False),
        (strict_reject, target, False),
    ])

    assert passing["precision"] == 1.0
    assert passing["recall"] == 1.0
    assert passing["false_accept_rate"] == 0.0
    assert passing["positive_examples"] == 2
    assert passing["negative_examples"] == 2
    assert passing["examples"] == 4
    assert judge_calibration_passes(spec, passing)
    assert not judge_calibration_passes(spec, low_precision)
    assert not judge_calibration_passes(spec, low_recall)
    assert not judge_calibration_passes(spec, false_accept)


@pytest.mark.parametrize("human_accept", [True, False])
def test_judge_calibration_requires_positive_and_negative_human_examples(
    human_accept: bool,
) -> None:
    """Calibration evidence must include both human-accepted and human-rejected rows."""
    target = ("/redfish/v1/Systems/1",)
    verdict = parse_pro_judge_result(_judge_json(rest_api_list=list(target)))

    with pytest.raises(ValueError, match="human-accepted and human-rejected"):
        evaluate_judge_calibration([
            (verdict, target, human_accept),
            (verdict, target, human_accept),
        ])


def test_builder_uses_injected_providers_and_counts_rejections(tmp_path: Path) -> None:
    """Offline build plumbing calls injected providers and summarizes draft quality."""
    spec = load_phase2_labelled_requests_spec(_write_spec(tmp_path / "phase2.yaml"))
    records = tuple(_record(index) for index in range(8))
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
    summary = counters.summary()

    assert row is not None
    data = row.to_dict()
    assert data["phase"] == 2
    assert data["dataset"] == "D1"
    assert data["source_dataset"] == "D0"
    assert data["task"] == "text_to_rest_api_list"
    assert data["x"]["text"] == "show both sampled systems"
    assert "records" not in data["x"]
    assert "rest_api_list" not in data["x"]
    assert len(data["x"]["api_context"]) == 6
    context_apis = {context["rest_api"] for context in data["x"]["api_context"]}
    assert set(data["y_true"]["rest_api_list"]) <= context_apis
    assert len(context_apis - set(data["y_true"]["rest_api_list"])) >= 4
    assert all("selected" not in context for context in data["x"]["api_context"])
    assert data["validation"]["set_coverage_preserved"] is True
    assert data["validation"]["review_judged"] is True
    assert data["validation"]["covered_api_set"] == data["y_true"]["rest_api_list"]
    assert set(summary) == set(PHASE2_LABELLED_REQUESTS_WANDB_METRIC_KEYS)
    assert summary[_phase2_metric("draft_total")] == 1
    assert summary[_phase2_metric("accepted_total")] == 1
    assert summary[_phase2_metric("rest_api_set_match_rate")] == 1.0
    assert summary[_phase2_metric("natural_command_rate")] == 1.0
    assert summary[_phase2_metric("ambiguous_rate")] == 0.0
    assert summary[_phase2_metric("duplicate_intent_rate")] == 0.0
    assert summary[_phase2_metric("extra_intent_rate")] == 0.0
    assert summary[_phase2_metric("method_semantics_valid_rate")] == 1.0
    assert summary[_phase2_metric("sample_width", "k")] == 2
    assert summary[_phase2_metric("vendor", "source_corpus")] == "fixture_vendor:fixture_corpus"
    assert summary[_phase2_metric("prompt_spec_version")] == "phase2-labelled-requests-test-v1"
    assert summary[_phase2_metric("model_x", "artifact_sha")] == "${PHASE1_MODEL_X_ARTIFACT_SHA}"
    assert summary[_phase2_metric("judge", "model")] == "${PHASE2_JUDGE_MODEL_ID}"
    assert summary[_phase2_metric("judge", "profile")] == "${PHASE2_JUDGE_PROFILE}"
    assert seen["draft"]["model_id"] == "${PHASE1_MODEL_X_MODEL_ID}"
    assert seen["draft"]["generation"]["max_new_tokens"] == 96
    assert seen["judge"]["model_id"] == "${PHASE2_JUDGE_MODEL_ID}"
    assert seen["judge"]["profile"] == "${PHASE2_JUDGE_PROFILE}"
    assert seen["judge"]["route"] == "private_pro"


def test_builder_k0_produces_judged_empty_set_negative_row(tmp_path: Path) -> None:
    """k=0 is a real judged negative row with context but no selected target set."""
    spec = load_phase2_labelled_requests_spec(_write_spec(tmp_path / "phase2.yaml"))
    records = tuple(_record(index) for index in range(8))
    seen: dict[str, dict] = {}

    def draft_provider(request: dict) -> str:
        seen["draft"] = request
        return "show a Redfish request that intentionally matches no listed resource"

    def judge_provider(request: dict) -> str:
        seen["judge"] = request
        return _judge_json(rest_api_list=request["expected_rest_api_list"])

    row, counters = Phase2LabelledRequestBuilder(
        spec,
        draft_provider=draft_provider,
        judge_provider=judge_provider,
    ).build_one(records, k=0, rng=random.Random(17))

    assert row is not None
    data = row.to_dict()
    assert data["x"]["text"] == (
        "show a Redfish request that intentionally matches no listed resource"
    )
    assert data["y_true"]["rest_api_list"] == []
    assert data["metadata"]["sample_width_k"] == 0
    assert len(data["x"]["api_context"]) == spec.context_distractors
    assert {context["rest_api"] for context in data["x"]["api_context"]}
    assert seen["draft"]["sample_width"] == 0
    assert spec.empty_set_model_x_system in seen["draft"]["prompt"]
    assert "matches none" in seen["draft"]["prompt"]
    assert seen["judge"]["expected_rest_api_list"] == []
    summary = counters.summary()
    assert summary[_phase2_metric("draft_total")] == 1
    assert summary[_phase2_metric("accepted_total")] == 1
    assert summary[_phase2_metric("empty_set_expected_total")] == 1
    assert summary[_phase2_metric("empty_set_match_rate")] == 1.0
    assert summary[_phase2_metric("sample_width", "k")] == 0


def test_builder_k0_uses_separate_budget_from_positive_api_limits(tmp_path: Path) -> None:
    """Repeated k=0 contexts consume only the separately bounded negative budget."""
    spec = load_phase2_labelled_requests_spec(_write_spec(tmp_path / "phase2.yaml"))
    records = tuple(_record(index) for index in range(8))
    budget = D1SamplingBudget(
        max_accepted_rows=10,
        max_candidates=10,
        max_accepted_per_combination=1,
        max_attempts_per_combination=1,
        max_accepted_per_api=1,
        max_empty_set_candidates=2,
    )
    calls = {"draft": 0, "judge": 0}

    def draft_provider(_request: dict) -> str:
        calls["draft"] += 1
        return "request an operation that matches none of the listed resources"

    def judge_provider(request: dict) -> str:
        calls["judge"] += 1
        return _judge_json(rest_api_list=request["expected_rest_api_list"])

    builder = Phase2LabelledRequestBuilder(
        spec,
        draft_provider=draft_provider,
        judge_provider=judge_provider,
        sampling_budget=budget,
    )

    first, _ = builder.build_one(records, k=0, rng=random.Random(17))
    second, _ = builder.build_one(records, k=0, rng=random.Random(17))
    exhausted, _ = builder.build_one(records, k=0, rng=random.Random(17))
    positive, _ = builder.build_one(records, k=1, rng=random.Random(17))

    assert first is not None
    assert second is not None
    assert exhausted is None
    assert positive is not None
    assert calls == {"draft": 3, "judge": 3}
    assert budget.empty_set_attempts_total == 2
    assert budget.empty_set_accepted_total == 2
    assert budget.accepted_by_combination.total() == 1
    assert budget.accepted_by_api.total() == 1


@pytest.mark.parametrize("width", [1, 2, 3])
def test_builder_positive_widths_keep_exact_selected_cardinality(
    tmp_path: Path,
    width: int,
) -> None:
    """k=1/2/3 still produce exact positive API sets plus configured distractors."""
    spec = load_phase2_labelled_requests_spec(_write_spec(tmp_path / "phase2.yaml"))
    records = tuple(_record(index) for index in range(10))
    seen: dict[str, dict] = {}

    def draft_provider(request: dict) -> str:
        seen["draft"] = request
        return f"show {width} selected Redfish resources"

    def judge_provider(request: dict) -> str:
        seen["judge"] = request
        return _judge_json(rest_api_list=request["expected_rest_api_list"])

    row, counters = Phase2LabelledRequestBuilder(
        spec,
        draft_provider=draft_provider,
        judge_provider=judge_provider,
    ).build_one(records, k=width, rng=random.Random(width))

    assert row is not None
    data = row.to_dict()
    assert len(data["y_true"]["rest_api_list"]) == width
    assert data["metadata"]["sample_width_k"] == width
    assert len(data["x"]["api_context"]) == width + spec.context_distractors
    context_apis = {context["rest_api"] for context in data["x"]["api_context"]}
    assert set(data["y_true"]["rest_api_list"]) <= context_apis
    assert len(context_apis - set(data["y_true"]["rest_api_list"])) >= 4
    assert seen["draft"]["sample_width"] == width
    assert spec.model_x_system in seen["draft"]["prompt"]
    assert set(seen["judge"]["expected_rest_api_list"]) == set(
        data["y_true"]["rest_api_list"],
    )
    assert len(seen["judge"]["expected_rest_api_list"]) == width
    assert counters.summary()[_phase2_metric("sample_width", "k")] == width


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

    records = tuple(_record(index) for index in range(1, 6))
    row, counters = builder.build_one(records, k=1, rng=random.Random(1))
    summary = counters.summary()

    assert row is None
    assert summary[_phase2_metric("draft_total")] == 1
    assert summary[_phase2_metric("accepted_total")] == 0
    assert summary[_phase2_metric("rejected_total")] == 1
    assert summary[_phase2_metric("pro_accept_rate")] == 0.0
    assert summary[_phase2_metric("rest_api_set_match_rate")] == 0.0


def test_builder_rejects_accepted_nonsense_even_when_set_matches(tmp_path: Path) -> None:
    """A contradictory judge response cannot enter accepted labelled requests."""
    spec = load_phase2_labelled_requests_spec(_write_spec(tmp_path / "phase2.yaml"))
    builder = Phase2LabelledRequestBuilder(
        spec,
        draft_provider=lambda request: "???",
        judge_provider=lambda request: _judge_json(
            accepted=True,
            rest_api_list=request["expected_rest_api_list"],
            nonsense=True,
        ),
    )

    records = tuple(_record(index) for index in range(1, 6))
    row, counters = builder.build_one(records, k=1, rng=random.Random(1))
    summary = counters.summary()

    assert row is None
    assert summary[_phase2_metric("draft_total")] == 1
    assert summary[_phase2_metric("accepted_total")] == 0
    assert summary[_phase2_metric("rejected_total")] == 1
    assert summary[_phase2_metric("nonsense_rate")] == 1.0
    assert summary[_phase2_metric("pro_accept_rate")] == 0.0
    assert summary[_phase2_metric("rest_api_set_match_rate")] == 1.0


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
    assert set(summary) == set(PHASE2_LABELLED_REQUESTS_WANDB_METRIC_KEYS)
    assert summary[_phase2_metric("draft_total")] == 3
    assert summary[_phase2_metric("accepted_total")] == 1
    assert summary[_phase2_metric("rejected_total")] == 2
    assert summary[_phase2_metric("nonsense_rate")] == pytest.approx(1 / 3)
    assert summary[_phase2_metric("invalid_json_rate")] == pytest.approx(1 / 3)
    assert summary[_phase2_metric("pro_accept_rate")] == pytest.approx(1 / 3)
    assert summary[_phase2_metric("rest_api_set_match_rate")] == pytest.approx(1 / 3)
    assert summary[_phase2_metric("natural_command_rate")] == pytest.approx(1 / 3)
    assert summary[_phase2_metric("ambiguous_rate")] == 0.0
    assert summary[_phase2_metric("duplicate_intent_rate")] == 0.0
    assert summary[_phase2_metric("extra_intent_rate")] == 0.0
    assert summary[_phase2_metric("method_semantics_valid_rate")] == pytest.approx(2 / 3)
    assert summary[_phase2_metric("empty_set_match_rate")] == 1.0
    assert summary[_phase2_metric("empty_set_expected_total")] == 1


def test_counter_summary_binds_semantic_metric_names() -> None:
    """Summary values are keyed by metric names, not tuple positions."""
    counters = Phase2LabelledRequestCounters(
        draft_total=7,
        accepted_total=5,
        rejected_total=2,
        pro_accept_total=6,
        nonsense_total=1,
        invalid_json_total=2,
        rest_api_set_match_total=4,
        natural_total=5,
        ambiguous_total=1,
        duplicate_intent_total=2,
        extra_intent_total=3,
        method_semantics_valid_total=4,
        empty_set_expected_total=3,
        empty_set_match_total=2,
        sample_width_k=3,
        vendor_source_corpus="vendor:corpus",
        prompt_spec_version="spec-v1",
        model_x_artifact_sha="sha256:abc",
        judge_model="pro",
        judge_profile="think-max",
    )

    summary = counters.summary()

    assert summary[_phase2_metric("draft_total")] == 7
    assert summary[_phase2_metric("accepted_total")] == 5
    assert summary[_phase2_metric("rejected_total")] == 2
    assert summary[_phase2_metric("nonsense_rate")] == pytest.approx(1 / 7)
    assert summary[_phase2_metric("invalid_json_rate")] == pytest.approx(2 / 7)
    assert summary[_phase2_metric("pro_accept_rate")] == pytest.approx(6 / 7)
    assert summary[_phase2_metric("rest_api_set_match_rate")] == pytest.approx(4 / 7)
    assert summary[_phase2_metric("natural_command_rate")] == pytest.approx(5 / 7)
    assert summary[_phase2_metric("ambiguous_rate")] == pytest.approx(1 / 7)
    assert summary[_phase2_metric("duplicate_intent_rate")] == pytest.approx(2 / 7)
    assert summary[_phase2_metric("extra_intent_rate")] == pytest.approx(3 / 7)
    assert summary[_phase2_metric("method_semantics_valid_rate")] == pytest.approx(4 / 7)
    assert summary[_phase2_metric("empty_set_match_rate")] == pytest.approx(2 / 3)
    assert summary[_phase2_metric("empty_set_expected_total")] == 3
    assert summary[_phase2_metric("sample_width", "k")] == 3
    assert summary[_phase2_metric("vendor", "source_corpus")] == "vendor:corpus"
    assert summary[_phase2_metric("prompt_spec_version")] == "spec-v1"
    assert summary[_phase2_metric("model_x", "artifact_sha")] == "sha256:abc"
    assert summary[_phase2_metric("judge", "model")] == "pro"
    assert summary[_phase2_metric("judge", "profile")] == "think-max"


def test_acceptance_thresholds_are_enforced_from_yaml(tmp_path: Path) -> None:
    """Configured acceptance thresholds decide whether summary metrics pass."""
    spec = load_phase2_labelled_requests_spec(_write_spec(tmp_path / "phase2.yaml"))
    passing = {
        _phase2_metric("pro_accept_rate"): 0.91,
        _phase2_metric("rest_api_set_match_rate"): 0.99,
        _phase2_metric("nonsense_rate"): 0.0,
        _phase2_metric("invalid_json_rate"): 0.0,
    }
    failing = {
        _phase2_metric("pro_accept_rate"): 0.89,
        _phase2_metric("rest_api_set_match_rate"): 0.99,
        _phase2_metric("nonsense_rate"): 0.0,
        _phase2_metric("invalid_json_rate"): 0.0,
    }

    assert phase2_acceptance_thresholds_pass(spec, passing)
    assert not phase2_acceptance_thresholds_pass(spec, failing)


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
        phase_metric(PHASE2_LABELLED_REQUESTS, "natural_command_rate"),
        phase_metric(PHASE2_LABELLED_REQUESTS, "ambiguous_rate"),
        phase_metric(PHASE2_LABELLED_REQUESTS, "duplicate_intent_rate"),
        phase_metric(PHASE2_LABELLED_REQUESTS, "extra_intent_rate"),
        phase_metric(PHASE2_LABELLED_REQUESTS, "method_semantics_valid_rate"),
        phase_metric(PHASE2_LABELLED_REQUESTS, "empty_set_match_rate"),
        phase_metric(PHASE2_LABELLED_REQUESTS, "empty_set_expected_total"),
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
    records = tuple(_record(index) for index in range(5))
    builder = Phase2LabelledRequestBuilder(
        spec,
        draft_provider=lambda request: "show system one",
        judge_provider=lambda request: _judge_json(
            rest_api_list=request["expected_rest_api_list"],
        ),
    )
    row, _ = builder.build_one(records, k=1, rng=random.Random(1))

    phase3_input = to_minimal_phase3_input(row)

    assert phase3_input["text"] == "show system one"
    assert row is not None
    assert phase3_input["rest_api_list"] == sorted(row.rest_api_list)
    assert "api_context" in phase3_input
    assert {
        context["rest_api"]
        for context in phase3_input["api_context"]
    } >= set(row.rest_api_list)
    assert "calls" not in phase3_input
    encoded = json.dumps(phase3_input)
    assert '"method":' not in encoded
    assert '"arguments":' not in encoded


def test_minimal_phase3_fixture_rejects_missing_phase2_row() -> None:
    """Phase 3 compatibility helpers fail before fabricating labels."""
    with pytest.raises(ValueError, match="phase2 row is required"):
        to_minimal_phase3_input(None)


def test_phase2_module_does_not_import_phase3_argument_runtime() -> None:
    """The labelled-request builder stays independent from Phase 3 call logic."""
    source = Path("igc/ds/phase2_labelled_requests.py").read_text(encoding="utf-8")
    module = ast.parse(source)
    imported_names: set[str] = set()
    for node in ast.walk(module):
        if isinstance(node, ast.ImportFrom):
            imported_names.update(alias.name for alias in node.names)

    assert "build_ordered_call_row" not in imported_names
    assert "parse_ordered_calls_y_pred" not in imported_names
    assert "build_call_row" not in imported_names
    assert "parse_calls_y_pred" not in imported_names


# Author: Mus mbayramo@stanford.edu
