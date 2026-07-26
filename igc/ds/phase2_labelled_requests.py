"""Offline Phase 2 labelled-request dataset plumbing.

Used by ``scripts/build_phase2_labelled_requests.py`` (the ``P2-LABELS`` dataset
CLI), which drives ``Phase2LabelledRequestBuilder`` with
``load_phase2_labelled_requests_spec`` and provider adapters to write accepted
``phase2_labelled_requests`` JSONL rows plus a metrics summary; focused tests run
through ``scripts/validate_phase2_labelled_requests.sh``. The module is
pure: it loads YAML specs, renders configured prompts, samples tiny records, and
parses injected judge responses without opening W&B, loading a model, or calling
the network.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from string import Formatter
from typing import Any, Callable, Mapping, Sequence

import yaml

from igc.modules.base.metric_keys import (
    PHASE2_LABELLED_REQUESTS,
    PHASE2_LABELLED_REQUESTS_WANDB_METRIC_KEYS,
    phase_metric,
)
from igc.ds.rest_goal_contract import (
    RedfishContext,
    build_d1_rest_api_list_row,
    d1_row_id,
)

D1_DATASET = "D1"
_DRAFT_TOTAL_KEY = phase_metric(PHASE2_LABELLED_REQUESTS, "draft_total")
_ACCEPTED_TOTAL_KEY = phase_metric(PHASE2_LABELLED_REQUESTS, "accepted_total")
_REJECTED_TOTAL_KEY = phase_metric(PHASE2_LABELLED_REQUESTS, "rejected_total")
_NONSENSE_RATE_KEY = phase_metric(PHASE2_LABELLED_REQUESTS, "nonsense_rate")
_INVALID_JSON_RATE_KEY = phase_metric(PHASE2_LABELLED_REQUESTS, "invalid_json_rate")
_PRO_ACCEPT_RATE_KEY = phase_metric(PHASE2_LABELLED_REQUESTS, "pro_accept_rate")
_REST_API_SET_MATCH_RATE_KEY = phase_metric(PHASE2_LABELLED_REQUESTS, "rest_api_set_match_rate")
_NATURAL_COMMAND_RATE_KEY = phase_metric(PHASE2_LABELLED_REQUESTS, "natural_command_rate")
_AMBIGUOUS_RATE_KEY = phase_metric(PHASE2_LABELLED_REQUESTS, "ambiguous_rate")
_DUPLICATE_INTENT_RATE_KEY = phase_metric(PHASE2_LABELLED_REQUESTS, "duplicate_intent_rate")
_EXTRA_INTENT_RATE_KEY = phase_metric(PHASE2_LABELLED_REQUESTS, "extra_intent_rate")
_METHOD_SEMANTICS_VALID_RATE_KEY = phase_metric(
    PHASE2_LABELLED_REQUESTS,
    "method_semantics_valid_rate",
)
_EMPTY_SET_MATCH_RATE_KEY = phase_metric(PHASE2_LABELLED_REQUESTS, "empty_set_match_rate")
_EMPTY_SET_EXPECTED_TOTAL_KEY = phase_metric(PHASE2_LABELLED_REQUESTS, "empty_set_expected_total")
_SAMPLE_WIDTH_KEY = phase_metric(PHASE2_LABELLED_REQUESTS, "sample_width", "k")
_VENDOR_SOURCE_CORPUS_KEY = phase_metric(PHASE2_LABELLED_REQUESTS, "vendor", "source_corpus")
_PROMPT_SPEC_VERSION_KEY = phase_metric(PHASE2_LABELLED_REQUESTS, "prompt_spec_version")
_MODEL_X_ARTIFACT_SHA_KEY = phase_metric(PHASE2_LABELLED_REQUESTS, "model_x", "artifact_sha")
_JUDGE_MODEL_KEY = phase_metric(PHASE2_LABELLED_REQUESTS, "judge", "model")
_JUDGE_PROFILE_KEY = phase_metric(PHASE2_LABELLED_REQUESTS, "judge", "profile")

_REQUIRED_ACCEPTANCE_KEYS = (
    "min_pro_accept_rate",
    "min_rest_api_set_match_rate",
    "max_nonsense_rate",
    "max_invalid_json_rate",
)
_REQUIRED_JUDGE_CALIBRATION_KEYS = (
    "min_precision",
    "min_recall",
    "max_false_accept_rate",
)
_PROVIDER_ADAPTERS = frozenset({"mock", "file", "openai-compatible"})


class Phase2LabelledRequestsSpecError(ValueError):
    """Raised when the labelled-request YAML spec is missing required fields."""


@dataclass(frozen=True)
class RestApiRecord:
    """One sampled Redfish REST API record.

    :param rest_api: concrete Redfish URI sampled from the corpus.
    :param allowed_methods: HTTP methods legal on this URI.
    :param json_body: Redfish response body for the URI.
    :param vendor: vendor/source family used for metric grouping.
    :param source_corpus: corpus artifact or fixture name that supplied the row.
    """

    rest_api: str  # concrete Redfish URI that becomes part of the known target set.
    allowed_methods: tuple[str, ...]  # HTTP methods legal on this URI.
    json_body: Mapping[str, Any]  # Redfish response body shown as evidence.
    operation_names: tuple[str, ...] = ()  # named actions/functions available on this URI.
    argument_schema: Mapping[str, Any] = field(default_factory=dict)
    vendor: str = ""  # vendor/source family for W&B grouping.
    source_corpus: str = ""  # corpus artifact or fixture name for provenance.

    def to_prompt_dict(self) -> dict[str, Any]:
        """Serialize the record shape shown to model and judge prompts."""
        return {
            "rest_api": self.rest_api,  # sampled API path the text must cover.
            "allowed_methods": list(self.allowed_methods),  # legal methods for this API.
            "operation_names": list(self.operation_names),
            "argument_schema": dict(self.argument_schema),
            "json": dict(self.json_body),  # JSON evidence for this API.
            "vendor": self.vendor,  # vendor/source family for provenance.
            "source_corpus": self.source_corpus,  # corpus artifact or fixture name.
        }


@dataclass(frozen=True)
class ModelXSpec:
    """Configured model_x identity for draft text generation."""

    model_id: str  # model identifier or runtime placeholder supplied by YAML.
    artifact_sha: str = ""  # Phase 1 artifact SHA or runtime placeholder.


@dataclass(frozen=True)
class JudgeSpec:
    """Configured private judge routing profile."""

    route: str  # judge route name, for example a private Pro route placeholder.
    model_id: str  # judge model identifier or runtime placeholder supplied by YAML.
    profile: str  # judge invocation profile or runtime placeholder.


@dataclass(frozen=True)
class ProviderAdapterSpec:
    """Configured provider adapter for draft or judge calls.

    :param adapter: adapter selector loaded from YAML, for example ``mock``.
    :param base_url_env: environment variable that supplies a live HTTP base URL.
    :param api_key_env: optional environment variable that supplies a bearer token.
    :param endpoint_path: HTTP path appended to the configured base URL.
    :param timeout_seconds: request timeout for live HTTP calls.
    :param response_text_path: dotted path to text in the provider JSON response.
    :param payload_request_fields: request fields copied into the live HTTP JSON body.
    """

    adapter: str  # provider selector; live modes are opt-in by config/CLI.
    base_url_env: str = ""  # env var containing the live provider base URL.
    api_key_env: str = ""  # optional env var containing a provider bearer token.
    endpoint_path: str = ""  # provider endpoint path such as a chat-completions route.
    timeout_seconds: float = 30.0  # live request timeout.
    response_text_path: str = ""  # dotted JSON path to generated text.
    payload_request_fields: tuple[str, ...] = ()  # extra request fields copied to payload.


@dataclass(frozen=True)
class Phase2LabelledRequestsSpec:
    """Loaded YAML contract for labelled-request generation.

    :param dataset_name: canonical emitted dataset name.
    :param prompt_spec_version: version string copied into rows and metrics.
    :param sample_widths: accepted sample widths, always one, two, and three.
    :param model_x: configured draft model identity.
    :param judge: configured private judge route and model profile.
    :param generation: generation knobs passed to the injected draft provider.
    :param model_x_system: system prompt text for the draft provider.
    :param model_x_template: prompt template for sampled records.
    :param judge_system: system prompt text for the judge provider.
    :param judge_template: prompt template for records plus draft text.
    :param draft_provider: configured adapter metadata for model_x drafts.
    :param judge_provider: configured adapter metadata for private judging.
    :param live_without_gate_max_candidates: max live candidates allowed without gate flag.
    :param wandb_namespace: W&B namespace for builder metrics.
    :param metric_keys: complete metric-key list from the spec.
    :param acceptance_thresholds: configured acceptance thresholds.
    """

    dataset_name: str  # canonical emitted dataset name.
    prompt_spec_version: str  # prompt/spec version copied into rows.
    sample_widths: tuple[int, ...]  # accepted sample widths.
    context_distractors: int  # distractor contexts added only after text is judged.
    empty_set_candidates: int  # separately bounded negative candidates for k=0.
    max_accepted_rows: int  # hard release-wide accepted-row ceiling.
    max_candidates: int  # hard release-wide provider-attempt ceiling.
    max_accepted_per_combination: int  # variants retained for one API set.
    max_attempts_per_combination: int  # provider attempts allowed for one API set.
    max_accepted_per_api: int  # accepted rows containing any one API.
    model_x: ModelXSpec  # draft model identity from YAML.
    judge: JudgeSpec  # judge route and model identity from YAML.
    generation: Mapping[str, Any]  # generation settings passed through unchanged.
    model_x_system: str  # YAML system prompt for model_x draft generation.
    model_x_template: str  # YAML prompt template for sampled records.
    empty_set_model_x_system: str  # system prompt for no-matching-API requests.
    empty_set_model_x_template: str  # template showing only distractor contexts.
    judge_system: str  # YAML system prompt for private judge review.
    judge_template: str  # YAML prompt template for judge input.
    empty_set_judge_system: str  # judge system prompt for no-match requests.
    empty_set_judge_template: str  # judge template for no-match requests.
    draft_provider: ProviderAdapterSpec  # draft adapter config from YAML.
    judge_provider: ProviderAdapterSpec  # judge adapter config from YAML.
    live_without_gate_max_candidates: int  # live candidate cap before gate flag is required.
    wandb_namespace: str  # W&B metric namespace.
    metric_keys: tuple[str, ...]  # metric keys declared by the spec.
    acceptance_thresholds: Mapping[str, float]  # acceptance threshold values.
    judge_calibration_thresholds: Mapping[str, float]  # labelled judge quality floors.


@dataclass(frozen=True)
class ProJudgeResult:
    """Parsed private judge decision for one draft text."""

    valid_json: bool  # true only when the complete strict verdict parsed.
    accepted: bool  # judge's own acceptance decision.
    natural: bool = False  # true when the text is a natural operator command.
    nonsense: bool = False  # true when the draft is junk or not an operator request.
    ambiguous: bool = False  # true when more than one API interpretation remains.
    duplicate_intent: bool = False  # true when an intent is repeated.
    extra_intents: bool = False  # true when unsupported work was added.
    method_semantics_valid: bool = False  # true when the request matches legal methods.
    covered_api_set: tuple[str, ...] = ()  # APIs actually covered by the text.
    reason: str = ""  # short non-secret judge reason.

    @property
    def invalid_json(self) -> bool:
        """Compatibility readout used by aggregate invalid-JSON metrics."""
        return not self.valid_json


@dataclass
class D1SamplingBudget:
    """Mutable bounded-build state shared across positive and negative generation."""

    max_accepted_rows: int
    max_candidates: int
    max_accepted_per_combination: int
    max_attempts_per_combination: int
    max_accepted_per_api: int
    attempts_total: int = 0
    accepted_total: int = 0
    attempts_by_combination: Counter[tuple[str, ...]] = field(default_factory=Counter)
    accepted_by_combination: Counter[tuple[str, ...]] = field(default_factory=Counter)
    accepted_by_api: Counter[str] = field(default_factory=Counter)

    @classmethod
    def from_spec(cls, spec: Phase2LabelledRequestsSpec) -> "D1SamplingBudget":
        """Create an empty budget from the YAML-owned finite controls."""
        return cls(
            max_accepted_rows=spec.max_accepted_rows,
            max_candidates=spec.max_candidates,
            max_accepted_per_combination=spec.max_accepted_per_combination,
            max_attempts_per_combination=spec.max_attempts_per_combination,
            max_accepted_per_api=spec.max_accepted_per_api,
        )

    def reserve_attempt(self, apis: Sequence[str]) -> bool:
        """Reserve one provider attempt only while every configured bound permits it."""
        combination = tuple(sorted(apis))
        if self.attempts_total >= self.max_candidates:
            return False
        if self.accepted_total >= self.max_accepted_rows:
            return False
        if self.attempts_by_combination[combination] >= self.max_attempts_per_combination:
            return False
        if self.accepted_by_combination[combination] >= self.max_accepted_per_combination:
            return False
        if any(self.accepted_by_api[api] >= self.max_accepted_per_api for api in combination):
            return False
        self.attempts_total += 1
        self.attempts_by_combination[combination] += 1
        return True

    def record_accept(self, apis: Sequence[str]) -> None:
        """Record one accepted row after a previously reserved provider attempt."""
        combination = tuple(sorted(apis))
        self.accepted_total += 1
        self.accepted_by_combination[combination] += 1
        self.accepted_by_api.update(combination)

    def summary(self) -> dict[str, Any]:
        """Return bounded, non-secret counters for manifests and metrics."""
        return {
            "limits": {
                "max_accepted_rows": self.max_accepted_rows,
                "max_candidates": self.max_candidates,
                "max_accepted_per_combination": self.max_accepted_per_combination,
                "max_attempts_per_combination": self.max_attempts_per_combination,
                "max_accepted_per_api": self.max_accepted_per_api,
            },
            "observed": {
                "attempts_total": self.attempts_total,
                "accepted_total": self.accepted_total,
                "unique_combinations_attempted": len(self.attempts_by_combination),
            },
        }


@dataclass(frozen=True)
class Phase2LabelledRequestRow:
    """Accepted Phase 2 labelled-request row."""

    text: str  # human request text accepted by the private judge.
    records: tuple[RestApiRecord, ...]  # sampled Redfish records used as context.
    rest_api_list: tuple[str, ...]  # canonical sampled REST API list.
    prompt_spec_version: str  # prompt/spec version used for this row.
    sample_width_k: int  # number of selected APIs represented by the text.
    validation: Mapping[str, Any] = field(default_factory=dict)  # judge validation flags.

    def to_dict(self) -> dict[str, Any]:
        """Serialize the accepted row as JSON-compatible data."""
        selected_apis = set(self.rest_api_list)
        target_records = tuple(
            record for record in self.records if record.rest_api in selected_apis
        )
        heldout_groups = sorted({
            record.vendor or record.source_corpus
            for record in target_records
            if record.vendor or record.source_corpus
        })
        if not heldout_groups and not selected_apis:
            heldout_groups = ["empty_set"]
        row = build_d1_rest_api_list_row(
            text=self.text,
            contexts=tuple(
                RedfishContext(
                    rest_api=record.rest_api,
                    allowed_methods=record.allowed_methods,
                    json=record.json_body,
                    operation_names=record.operation_names,
                    argument_schema=record.argument_schema,
                )
                for record in self.records
            ),
            rest_api_list=self.rest_api_list,
            validation=self.validation,
        )
        row["metadata"] = {
            "row_id": d1_row_id(row),
            "prompt_spec_version": self.prompt_spec_version,
            "sample_width_k": self.sample_width_k,
            "vendor": [record.vendor for record in self.records],
            "source_corpus": [record.source_corpus for record in self.records],
            "heldout_vendor_or_model": heldout_groups,
        }
        return row


DraftProvider = Callable[[dict[str, Any]], str]
JudgeProvider = Callable[[dict[str, Any]], str]


def load_phase2_labelled_requests_spec(path: str | Path) -> Phase2LabelledRequestsSpec:
    """Load and validate a labelled-request YAML spec.

    :param path: YAML spec path.
    :return: normalized spec object.
    :raises Phase2LabelledRequestsSpecError: if required fields are missing.
    """
    spec_path = Path(path)
    try:
        raw = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise Phase2LabelledRequestsSpecError(f"cannot read spec {spec_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise Phase2LabelledRequestsSpecError(f"cannot parse YAML in {spec_path}: {exc}") from exc

    if not isinstance(raw, Mapping):
        raise Phase2LabelledRequestsSpecError("phase2 labelled-request spec must be a mapping")

    dataset = _mapping(raw, "dataset", required=True)
    dataset_name = _required_string(dataset, "name", "dataset.name")
    if dataset_name != D1_DATASET:
        raise Phase2LabelledRequestsSpecError(
            f"dataset.name must be {D1_DATASET!r}",
        )

    sampling = _mapping(raw, "sampling", required=True)
    raw_sample_widths = _sequence(sampling, "sample_widths")
    if not all(
        isinstance(width, int) and not isinstance(width, bool)
        for width in raw_sample_widths
    ):
        raise Phase2LabelledRequestsSpecError(
            "sampling.sample_widths must be integer sequence [1, 2, 3]",
        )
    sample_widths = tuple(raw_sample_widths)
    if sample_widths != (1, 2, 3):
        raise Phase2LabelledRequestsSpecError("sampling.sample_widths must be [1, 2, 3]")
    context_distractors = _optional_non_negative_int(
        sampling,
        "context_distractors",
        default=4,
        label="sampling.context_distractors",
    )
    if context_distractors < 4:
        raise Phase2LabelledRequestsSpecError(
            "sampling.context_distractors must be at least 4",
        )
    empty_set_candidates = _optional_non_negative_int(
        sampling,
        "empty_set_candidates",
        default=0,
        label="sampling.empty_set_candidates",
    )
    if empty_set_candidates < 1:
        raise Phase2LabelledRequestsSpecError(
            "sampling.empty_set_candidates must be positive",
        )
    finite_controls: dict[str, int] = {}
    for key in (
        "max_accepted_rows",
        "max_candidates",
        "max_accepted_per_combination",
        "max_attempts_per_combination",
        "max_accepted_per_api",
    ):
        value = _optional_non_negative_int(
            sampling,
            key,
            default=0,
            label=f"sampling.{key}",
        )
        if value < 1:
            raise Phase2LabelledRequestsSpecError(f"sampling.{key} must be positive")
        finite_controls[key] = value
    if (
        finite_controls["max_accepted_per_combination"]
        > finite_controls["max_attempts_per_combination"]
    ):
        raise Phase2LabelledRequestsSpecError(
            "sampling.max_accepted_per_combination cannot exceed "
            "sampling.max_attempts_per_combination"
        )

    model_x_raw = _mapping(raw, "model_x", required=True)
    judge_raw = _mapping(raw, "judge", required=True)
    prompts = _mapping(raw, "prompts", required=True)
    model_prompt = _mapping(prompts, "model_x_draft", required=True)
    empty_set_prompt = _mapping(
        prompts,
        "model_x_empty_set_draft",
        required=True,
    )
    judge_prompt = _mapping(prompts, "pro_judge", required=True)
    empty_set_judge_prompt = _mapping(
        prompts,
        "pro_judge_empty_set",
        required=True,
    )
    model_x_system = _required_string(model_prompt, "system", "prompts.model_x_draft.system")
    model_x_template = _required_string(
        model_prompt,
        "template",
        "prompts.model_x_draft.template",
    )
    empty_set_model_x_system = _required_string(
        empty_set_prompt,
        "system",
        "prompts.model_x_empty_set_draft.system",
    )
    empty_set_model_x_template = _required_string(
        empty_set_prompt,
        "template",
        "prompts.model_x_empty_set_draft.template",
    )
    judge_system = _required_string(judge_prompt, "system", "prompts.pro_judge.system")
    judge_template = _required_string(judge_prompt, "template", "prompts.pro_judge.template")
    empty_set_judge_system = _required_string(
        empty_set_judge_prompt,
        "system",
        "prompts.pro_judge_empty_set.system",
    )
    empty_set_judge_template = _required_string(
        empty_set_judge_prompt,
        "template",
        "prompts.pro_judge_empty_set.template",
    )
    _validate_prompt_template(
        model_x_template,
        label="prompts.model_x_draft.template",
        required_fields=("records_json",),
        allowed_fields=("records_json",),
    )
    _validate_prompt_template(
        empty_set_model_x_template,
        label="prompts.model_x_empty_set_draft.template",
        required_fields=("records_json",),
        allowed_fields=("records_json",),
    )
    _validate_prompt_template(
        judge_template,
        label="prompts.pro_judge.template",
        required_fields=("records_json", "draft_text"),
        allowed_fields=("records_json", "draft_text"),
    )
    _validate_prompt_template(
        empty_set_judge_template,
        label="prompts.pro_judge_empty_set.template",
        required_fields=("records_json", "draft_text"),
        allowed_fields=("records_json", "draft_text"),
    )
    providers = _mapping(raw, "providers", required=True)
    draft_provider = _provider_adapter_spec(
        _mapping(providers, "draft"),
        label="providers.draft",
    )
    judge_provider = _provider_adapter_spec(
        _mapping(providers, "judge"),
        label="providers.judge",
    )
    safety = _mapping(raw, "safety")
    live_without_gate_max_candidates = _optional_non_negative_int(
        safety,
        "live_without_gate_max_candidates",
        default=3,
        label="safety.live_without_gate_max_candidates",
    )
    wandb = _mapping(raw, "wandb", required=True)

    metric_keys = tuple(str(key) for key in _sequence(wandb, "metric_keys"))
    if metric_keys != PHASE2_LABELLED_REQUESTS_WANDB_METRIC_KEYS:
        raise Phase2LabelledRequestsSpecError("wandb.metric_keys must match the registry")

    wandb_namespace = _required_string(wandb, "namespace", "wandb.namespace")
    if wandb_namespace != PHASE2_LABELLED_REQUESTS:
        raise Phase2LabelledRequestsSpecError(
            f"wandb.namespace must be {PHASE2_LABELLED_REQUESTS!r}",
        )

    acceptance = _mapping(raw, "acceptance", required=True)
    missing_acceptance = sorted(set(_REQUIRED_ACCEPTANCE_KEYS) - set(acceptance))
    if missing_acceptance:
        raise Phase2LabelledRequestsSpecError(
            f"acceptance missing required keys: {', '.join(missing_acceptance)}",
        )

    acceptance_thresholds: dict[str, float] = {}
    for key, value in acceptance.items():
        try:
            acceptance_thresholds[str(key)] = float(value)
        except (TypeError, ValueError) as exc:
            raise Phase2LabelledRequestsSpecError(
                f"acceptance.{key} must be numeric",
            ) from exc

    calibration = _mapping(raw, "judge_calibration", required=True)
    missing_calibration = sorted(
        set(_REQUIRED_JUDGE_CALIBRATION_KEYS) - set(calibration)
    )
    if missing_calibration:
        raise Phase2LabelledRequestsSpecError(
            "judge_calibration missing required keys: "
            + ", ".join(missing_calibration),
        )
    judge_calibration_thresholds: dict[str, float] = {}
    for key, value in calibration.items():
        try:
            judge_calibration_thresholds[str(key)] = float(value)
        except (TypeError, ValueError) as exc:
            raise Phase2LabelledRequestsSpecError(
                f"judge_calibration.{key} must be numeric",
            ) from exc

    return Phase2LabelledRequestsSpec(
        dataset_name=dataset_name,
        prompt_spec_version=_required_string(
            dataset,
            "prompt_spec_version",
            "dataset.prompt_spec_version",
        ),
        sample_widths=sample_widths,
        context_distractors=context_distractors,
        empty_set_candidates=empty_set_candidates,
        max_accepted_rows=finite_controls["max_accepted_rows"],
        max_candidates=finite_controls["max_candidates"],
        max_accepted_per_combination=finite_controls[
            "max_accepted_per_combination"
        ],
        max_attempts_per_combination=finite_controls[
            "max_attempts_per_combination"
        ],
        max_accepted_per_api=finite_controls["max_accepted_per_api"],
        model_x=ModelXSpec(
            model_id=_required_string(model_x_raw, "model_id", "model_x.model_id"),
            artifact_sha=str(model_x_raw.get("artifact_sha", "")),
        ),
        judge=JudgeSpec(
            route=_required_string(judge_raw, "route", "judge.route"),
            model_id=_required_string(judge_raw, "model_id", "judge.model_id"),
            profile=_required_string(judge_raw, "profile", "judge.profile"),
        ),
        generation=dict(_mapping(raw, "generation", required=True)),
        model_x_system=model_x_system,
        model_x_template=model_x_template,
        empty_set_model_x_system=empty_set_model_x_system,
        empty_set_model_x_template=empty_set_model_x_template,
        judge_system=judge_system,
        judge_template=judge_template,
        empty_set_judge_system=empty_set_judge_system,
        empty_set_judge_template=empty_set_judge_template,
        draft_provider=draft_provider,
        judge_provider=judge_provider,
        live_without_gate_max_candidates=live_without_gate_max_candidates,
        wandb_namespace=wandb_namespace,
        metric_keys=metric_keys,
        acceptance_thresholds=acceptance_thresholds,
        judge_calibration_thresholds=judge_calibration_thresholds,
    )


def render_model_x_prompt(
    spec: Phase2LabelledRequestsSpec,
    records: Sequence[RestApiRecord],
) -> str:
    """Render the model_x prompt from YAML-provided prompt fields."""
    return _render_prompt(
        system=spec.model_x_system,
        template=spec.model_x_template,
        records=records,
        draft_text="",
    )


def render_model_x_empty_set_prompt(
    spec: Phase2LabelledRequestsSpec,
    records: Sequence[RestApiRecord],
) -> str:
    """Render a request that must map to none of the shown API contexts."""
    return _render_prompt(
        system=spec.empty_set_model_x_system,
        template=spec.empty_set_model_x_template,
        records=records,
        draft_text="",
    )


def render_pro_judge_prompt(
    spec: Phase2LabelledRequestsSpec,
    records: Sequence[RestApiRecord],
    draft_text: str,
    *,
    empty_set: bool = False,
) -> str:
    """Render the private judge prompt from YAML-provided prompt fields."""
    return _render_prompt(
        system=(spec.empty_set_judge_system if empty_set else spec.judge_system),
        template=(
            spec.empty_set_judge_template if empty_set else spec.judge_template
        ),
        records=records,
        draft_text=draft_text,
    )


def sample_phase2_contexts(
    records: Sequence[RestApiRecord],
    *,
    k: int,
    rng: random.Random,
) -> tuple[RestApiRecord, ...]:
    """Sample one, two, or three REST API records without replacement."""
    if k not in (1, 2, 3):
        raise ValueError("sample width must be one of 1, 2, or 3")
    if len(records) < k:
        raise ValueError("not enough records for requested sample width")
    return tuple(rng.sample(list(records), k))


def parse_pro_judge_result(raw: str) -> ProJudgeResult:
    """Parse a private judge JSON result without raising on malformed output."""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        return ProJudgeResult(
            valid_json=False,
            accepted=False,
            reason=f"invalid_json: {exc.msg}",
        )
    if isinstance(parsed, Mapping) and isinstance(parsed.get("y_pred"), Mapping):
        parsed = parsed["y_pred"]
    if not isinstance(parsed, Mapping):
        return ProJudgeResult(
            valid_json=False,
            accepted=False,
            reason="judge result is not a mapping",
        )

    required_fields = {
        "accepted",
        "natural",
        "nonsense",
        "ambiguous",
        "duplicate_intent",
        "extra_intents",
        "method_semantics_valid",
        "covered_api_set",
        "reason",
    }
    if set(parsed) != required_fields:
        return ProJudgeResult(
            valid_json=False,
            accepted=False,
            reason=f"judge result must contain exactly {sorted(required_fields)}",
        )

    covered_api_value = parsed["covered_api_set"]
    if not isinstance(covered_api_value, list):
        return ProJudgeResult(
            valid_json=False,
            accepted=False,
            reason="covered_api_set is not a list",
        )
    if not all(isinstance(item, str) and item.strip() for item in covered_api_value):
        return ProJudgeResult(
            valid_json=False,
            accepted=False,
            reason="covered_api_set must contain only non-empty strings",
        )
    if len(covered_api_value) != len(set(covered_api_value)):
        return ProJudgeResult(
            valid_json=False,
            accepted=False,
            reason="covered_api_set must not contain duplicates",
        )

    bool_fields = (
        "accepted",
        "natural",
        "nonsense",
        "ambiguous",
        "duplicate_intent",
        "extra_intents",
        "method_semantics_valid",
    )
    invalid_bool_fields = [
        name for name in bool_fields if not isinstance(parsed.get(name), bool)
    ]
    if invalid_bool_fields:
        return ProJudgeResult(
            valid_json=False,
            accepted=False,
            reason="judge boolean fields are invalid: " + ", ".join(invalid_bool_fields),
        )
    if not isinstance(parsed["reason"], str):
        return ProJudgeResult(
            valid_json=False,
            accepted=False,
            reason="reason must be a string",
        )
    return ProJudgeResult(
        valid_json=True,
        accepted=parsed["accepted"],
        natural=parsed["natural"],
        nonsense=parsed["nonsense"],
        ambiguous=parsed["ambiguous"],
        duplicate_intent=parsed["duplicate_intent"],
        extra_intents=parsed["extra_intents"],
        method_semantics_valid=parsed["method_semantics_valid"],
        covered_api_set=tuple(covered_api_value),
        reason=parsed["reason"],
    )


def judge_result_is_accepted(
    verdict: ProJudgeResult,
    *,
    selected_api_set: Sequence[str],
) -> bool:
    """Apply the complete D1 judge acceptance predicate."""
    return (
        verdict.valid_json
        and verdict.accepted
        and verdict.natural
        and not verdict.nonsense
        and not verdict.ambiguous
        and not verdict.duplicate_intent
        and not verdict.extra_intents
        and verdict.method_semantics_valid
        and set(verdict.covered_api_set) == set(selected_api_set)
    )


def evaluate_judge_calibration(
    examples: Sequence[tuple[ProJudgeResult, Sequence[str], bool]],
) -> dict[str, float | int]:
    """Measure strict judge decisions against human calibration labels."""
    if not examples:
        raise ValueError("judge calibration requires at least one labelled example")
    true_positive = false_positive = true_negative = false_negative = 0
    positive_examples = sum(bool(human_accept) for _, _, human_accept in examples)
    negative_examples = len(examples) - positive_examples
    if positive_examples == 0 or negative_examples == 0:
        raise ValueError(
            "judge calibration requires both human-accepted and human-rejected examples"
        )
    for verdict, selected_api_set, human_accept in examples:
        predicted_accept = judge_result_is_accepted(
            verdict,
            selected_api_set=selected_api_set,
        )
        if predicted_accept and human_accept:
            true_positive += 1
        elif predicted_accept:
            false_positive += 1
        elif human_accept:
            false_negative += 1
        else:
            true_negative += 1
    precision = _rate(true_positive, true_positive + false_positive)
    recall = _rate(true_positive, true_positive + false_negative)
    false_accept_rate = (
        0.0
        if false_positive + true_negative == 0
        else false_positive / (false_positive + true_negative)
    )
    return {
        "precision": precision,
        "recall": recall,
        "false_accept_rate": false_accept_rate,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
        "positive_examples": positive_examples,
        "negative_examples": negative_examples,
        "examples": len(examples),
    }


def judge_calibration_passes(
    spec: Phase2LabelledRequestsSpec,
    metrics: Mapping[str, float | int],
) -> bool:
    """Apply the YAML-owned judge calibration floors."""
    thresholds = spec.judge_calibration_thresholds
    return (
        float(metrics.get("precision", 0.0)) >= thresholds["min_precision"]
        and float(metrics.get("recall", 0.0)) >= thresholds["min_recall"]
        and float(metrics.get("false_accept_rate", 1.0))
        <= thresholds["max_false_accept_rate"]
    )


def compare_rest_api_sets(expected: Sequence[str], predicted: Sequence[str]) -> bool:
    """Return true when two REST API lists name the same unordered set."""
    return set(expected) == set(predicted)


def empty_set_matches(expected: Sequence[str], predicted: Sequence[str]) -> bool:
    """Return true only when both expected and predicted API sets are empty."""
    return not expected and not predicted


@dataclass
class Phase2LabelledRequestCounters:
    """Counters and rates for offline labelled-request generation."""

    draft_total: int = 0  # number of draft text attempts.
    accepted_total: int = 0  # number of rows accepted into the dataset.
    rejected_total: int = 0  # number of rows rejected by parser/judge/set check.
    pro_accept_total: int = 0  # number of valid judge responses with accepted=true.
    nonsense_total: int = 0  # number of drafts flagged as nonsense.
    invalid_json_total: int = 0  # number of judge responses with invalid JSON.
    rest_api_set_match_total: int = 0  # number of rows with matching API sets.
    natural_total: int = 0  # number of strict verdicts marking the request natural.
    ambiguous_total: int = 0  # number of strict verdicts marking ambiguity.
    duplicate_intent_total: int = 0  # number of strict verdicts marking duplicate intent.
    extra_intent_total: int = 0  # number of strict verdicts marking extra intent.
    method_semantics_valid_total: int = 0  # strict verdicts with legal method semantics.
    empty_set_expected_total: int = 0  # number of valid judged no-action rows.
    empty_set_match_total: int = 0  # number of valid judged no-action matches.
    sample_width_k: int = 0  # sampled REST API count for this candidate.
    vendor_source_corpus: str = ""  # compact vendor/corpus provenance label.
    prompt_spec_version: str = ""  # prompt spec version copied from YAML.
    model_x_artifact_sha: str = ""  # Phase 1 model_x artifact SHA or placeholder.
    judge_model: str = ""  # private judge model identifier or placeholder.
    judge_profile: str = ""  # private judge profile identifier or placeholder.

    def observe_draft(self, text: str) -> None:
        """Count one draft attempt.

        :param text: generated text; stored only as quality signal, never logged.
        """
        _ = text
        self.draft_total += 1

    def observe_judge(
        self,
        result: ProJudgeResult,
        *,
        expected_rest_api_list: Sequence[str],
    ) -> None:
        """Count one parsed judge decision against the known sampled API set."""
        set_match = result.valid_json and compare_rest_api_sets(
            expected_rest_api_list,
            result.covered_api_set,
        )
        accepted = judge_result_is_accepted(
            result,
            selected_api_set=expected_rest_api_list,
        )

        if result.nonsense:
            self.nonsense_total += 1
        if not result.valid_json:
            self.invalid_json_total += 1
        if result.natural and result.valid_json:
            self.natural_total += 1
        if result.ambiguous and result.valid_json:
            self.ambiguous_total += 1
        if result.duplicate_intent and result.valid_json:
            self.duplicate_intent_total += 1
        if result.extra_intents and result.valid_json:
            self.extra_intent_total += 1
        if result.method_semantics_valid and result.valid_json:
            self.method_semantics_valid_total += 1
        if accepted:
            self.pro_accept_total += 1
        if set_match:
            self.rest_api_set_match_total += 1
        if result.valid_json and not expected_rest_api_list:
            self.empty_set_expected_total += 1
            if empty_set_matches(expected_rest_api_list, result.covered_api_set):
                self.empty_set_match_total += 1
        if accepted:
            self.accepted_total += 1
        else:
            self.rejected_total += 1

    def summary(self) -> dict[str, float | int | str]:
        """Return all registered builder metrics for W&B/TensorBoard logging."""
        return {
            _DRAFT_TOTAL_KEY: self.draft_total,
            _ACCEPTED_TOTAL_KEY: self.accepted_total,
            _REJECTED_TOTAL_KEY: self.rejected_total,
            _NONSENSE_RATE_KEY: _rate(self.nonsense_total, self.draft_total),
            _INVALID_JSON_RATE_KEY: _rate(self.invalid_json_total, self.draft_total),
            _PRO_ACCEPT_RATE_KEY: _rate(self.pro_accept_total, self.draft_total),
            _REST_API_SET_MATCH_RATE_KEY: _rate(self.rest_api_set_match_total, self.draft_total),
            _NATURAL_COMMAND_RATE_KEY: _rate(self.natural_total, self.draft_total),
            _AMBIGUOUS_RATE_KEY: _rate(self.ambiguous_total, self.draft_total),
            _DUPLICATE_INTENT_RATE_KEY: _rate(
                self.duplicate_intent_total,
                self.draft_total,
            ),
            _EXTRA_INTENT_RATE_KEY: _rate(self.extra_intent_total, self.draft_total),
            _METHOD_SEMANTICS_VALID_RATE_KEY: _rate(
                self.method_semantics_valid_total,
                self.draft_total,
            ),
            _EMPTY_SET_MATCH_RATE_KEY: _rate(
                self.empty_set_match_total,
                self.empty_set_expected_total,
            ),
            _EMPTY_SET_EXPECTED_TOTAL_KEY: self.empty_set_expected_total,
            _SAMPLE_WIDTH_KEY: self.sample_width_k,
            _VENDOR_SOURCE_CORPUS_KEY: self.vendor_source_corpus,
            _PROMPT_SPEC_VERSION_KEY: self.prompt_spec_version,
            _MODEL_X_ARTIFACT_SHA_KEY: self.model_x_artifact_sha,
            _JUDGE_MODEL_KEY: self.judge_model,
            _JUDGE_PROFILE_KEY: self.judge_profile,
        }


class Phase2LabelledRequestBuilder:
    """Offline builder that uses injected model_x and judge providers."""

    def __init__(
        self,
        spec: Phase2LabelledRequestsSpec,
        *,
        draft_provider: DraftProvider,
        judge_provider: JudgeProvider,
        sampling_budget: D1SamplingBudget | None = None,
    ) -> None:
        """Create a builder with pure injected provider callables."""
        self._spec = spec
        self._draft_provider = draft_provider
        self._judge_provider = judge_provider
        self._sampling_budget = sampling_budget or D1SamplingBudget.from_spec(spec)

    def build_one(
        self,
        records: Sequence[RestApiRecord],
        *,
        k: int,
        rng: random.Random,
    ) -> tuple[Phase2LabelledRequestRow | None, Phase2LabelledRequestCounters]:
        """Build and judge one accepted row candidate.

        :param records: candidate REST API records.
        :param k: target width, zero through three.
        :param rng: deterministic RNG supplied by the caller.
        :return: accepted row plus counters, or ``None`` plus rejection counters.
        """
        rest_apis = [record.rest_api for record in records]
        if len(rest_apis) != len(set(rest_apis)):
            raise ValueError("Phase 2 source pool must contain unique rest_api values")
        required_records = (
            self._spec.context_distractors
            if k == 0
            else k + self._spec.context_distractors
        )
        if len(records) < required_records:
            raise ValueError(
                "Phase 2 source pool cannot satisfy targets plus distractors: "
                f"need {required_records}, got {len(records)}"
            )

        if k == 0:
            sampled = tuple(
                rng.sample(list(records), self._spec.context_distractors)
            )
            expected_rest_api_list: tuple[str, ...] = ()
            budget_identity = tuple(
                f"empty_context:{record.rest_api}" for record in sampled
            )
        else:
            sampled = sample_phase2_contexts(records, k=k, rng=rng)
            expected_rest_api_list = tuple(record.rest_api for record in sampled)
            budget_identity = expected_rest_api_list
        counters = Phase2LabelledRequestCounters(
            sample_width_k=k,
            vendor_source_corpus=_source_corpus_label(sampled),
            prompt_spec_version=self._spec.prompt_spec_version,
            model_x_artifact_sha=self._spec.model_x.artifact_sha,
            judge_model=self._spec.judge.model_id,
            judge_profile=self._spec.judge.profile,
        )
        if not self._sampling_budget.reserve_attempt(budget_identity):
            return None, counters

        draft_request = {
            "prompt": (
                render_model_x_empty_set_prompt(self._spec, sampled)
                if k == 0
                else render_model_x_prompt(self._spec, sampled)
            ),
            "model_id": self._spec.model_x.model_id,  # draft model from YAML.
            "generation": dict(self._spec.generation),  # generation knobs from YAML.
            "sample_width": k,  # current sample width metric value.
        }
        draft_text = self._draft_provider(draft_request)
        counters.observe_draft(draft_text)

        judge_request = {
            "prompt": render_pro_judge_prompt(
                self._spec,
                sampled,
                draft_text,
                empty_set=k == 0,
            ),
            "model_id": self._spec.judge.model_id,  # judge model from YAML.
            "profile": self._spec.judge.profile,  # judge profile from YAML.
            "route": self._spec.judge.route,  # judge route from YAML.
            "expected_rest_api_list": list(expected_rest_api_list),  # known sampled set.
        }
        judge_result = parse_pro_judge_result(self._judge_provider(judge_request))
        counters.observe_judge(judge_result, expected_rest_api_list=expected_rest_api_list)

        set_match = compare_rest_api_sets(
            expected_rest_api_list,
            judge_result.covered_api_set,
        )
        accepted = judge_result_is_accepted(
            judge_result,
            selected_api_set=expected_rest_api_list,
        )
        if not accepted:
            return None, counters
        self._sampling_budget.record_accept(budget_identity)

        selected_apis = set(expected_rest_api_list)
        context_records = list(sampled)
        if k > 0:
            distractor_pool = [
                record for record in records if record.rest_api not in selected_apis
            ]
            context_records.extend(
                rng.sample(distractor_pool, self._spec.context_distractors)
            )
        rng.shuffle(context_records)

        row = Phase2LabelledRequestRow(
            text=draft_text,
            records=tuple(context_records),
            rest_api_list=expected_rest_api_list,
            prompt_spec_version=self._spec.prompt_spec_version,
            sample_width_k=k,
            validation={
                "text_source": "model_x_then_private_judge",  # draft then judge path.
                "review_judged": True,  # private judge parsed successfully.
                "valid_json": judge_result.valid_json,
                "accepted": judge_result.accepted,
                "natural": judge_result.natural,
                "nonsense": judge_result.nonsense,
                "ambiguous": judge_result.ambiguous,
                "duplicate_intent": judge_result.duplicate_intent,
                "extra_intents": judge_result.extra_intents,
                "method_semantics_valid": judge_result.method_semantics_valid,
                "covered_api_set": sorted(judge_result.covered_api_set),
                "set_coverage_preserved": set_match,  # unordered API set contract.
            },
        )
        return row, counters


def to_minimal_phase3_input(row: Phase2LabelledRequestRow | None) -> dict[str, Any]:
    """Convert an accepted Phase 2 row into a Phase 3 input fixture."""
    if row is None:
        raise ValueError("phase2 row is required")
    return {
        "text": row.text,  # accepted human request text.
        "rest_api_list": sorted(row.rest_api_list),  # canonical form of the unordered set.
        "api_context": [
            RedfishContext(
                rest_api=record.rest_api,
                allowed_methods=record.allowed_methods,
                json=record.json_body,
            ).to_dict()
            for record in row.records
        ],
    }


def phase2_acceptance_thresholds_pass(
    spec: Phase2LabelledRequestsSpec,
    summary: Mapping[str, float | int | str],
) -> bool:
    """Return true when observed builder metrics satisfy YAML thresholds.

    :param spec: loaded Phase 2 labelled-request spec.
    :param summary: summary returned by :meth:`Phase2LabelledRequestCounters.summary`.
    :return: true when all configured min/max thresholds pass.
    """
    thresholds = spec.acceptance_thresholds
    missing_acceptance = sorted(set(_REQUIRED_ACCEPTANCE_KEYS) - set(thresholds))
    if missing_acceptance:
        raise Phase2LabelledRequestsSpecError(
            f"acceptance missing required keys: {', '.join(missing_acceptance)}",
        )
    return (
        float(summary.get(_PRO_ACCEPT_RATE_KEY, 0.0))
        >= thresholds["min_pro_accept_rate"]
        and float(summary.get(_REST_API_SET_MATCH_RATE_KEY, 0.0))
        >= thresholds["min_rest_api_set_match_rate"]
        and float(summary.get(_NONSENSE_RATE_KEY, 1.0))
        <= thresholds["max_nonsense_rate"]
        and float(summary.get(_INVALID_JSON_RATE_KEY, 1.0))
        <= thresholds["max_invalid_json_rate"]
    )


def _render_prompt(
    *,
    system: str,
    template: str,
    records: Sequence[RestApiRecord],
    draft_text: str,
) -> str:
    """Render a configured prompt template with JSON-safe record payloads."""
    records_json = json.dumps(
        [record.to_prompt_dict() for record in records],
        indent=2,
        sort_keys=True,
    )
    body = template.format(records_json=records_json, draft_text=draft_text)
    return f"{system.rstrip()}\n\n{body.strip()}"


def _mapping(source: Mapping[str, Any], key: str, *, required: bool = False) -> Mapping[str, Any]:
    """Read a YAML child mapping."""
    value = source.get(key)
    if value is None and not required:
        return {}
    if not isinstance(value, Mapping):
        raise Phase2LabelledRequestsSpecError(f"{key} must be a mapping")
    return value


def _sequence(source: Mapping[str, Any], key: str) -> Sequence[Any]:
    """Read a YAML sequence field."""
    value = source.get(key)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise Phase2LabelledRequestsSpecError(f"{key} must be a sequence")
    return value


def _optional_sequence(source: Mapping[str, Any], key: str) -> Sequence[Any]:
    """Read an optional YAML sequence field."""
    value = source.get(key, ())
    if value == ():
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise Phase2LabelledRequestsSpecError(f"{key} must be a sequence")
    return value


def _required_string(source: Mapping[str, Any], key: str, label: str) -> str:
    """Read a required non-empty YAML string field."""
    value = source.get(key)
    if not isinstance(value, str) or not value.strip():
        raise Phase2LabelledRequestsSpecError(f"{label} must be a non-empty string")
    return value


def _validate_prompt_template(
    template: str,
    *,
    label: str,
    required_fields: Sequence[str],
    allowed_fields: Sequence[str],
) -> None:
    """Validate configured prompt placeholders before a build starts."""
    try:
        parsed_fields: set[str] = set()
        for _, field_name, _, _ in Formatter().parse(template):
            if field_name is None:
                continue
            if not field_name:
                raise Phase2LabelledRequestsSpecError(
                    f"{label} has unnamed format fields",
                )
            parsed_fields.add(field_name)
    except Phase2LabelledRequestsSpecError:
        raise
    except ValueError as exc:
        raise Phase2LabelledRequestsSpecError(f"{label} has malformed format fields") from exc

    unknown_fields = sorted(parsed_fields - set(allowed_fields))
    if unknown_fields:
        raise Phase2LabelledRequestsSpecError(
            f"{label} has unknown fields: {', '.join(unknown_fields)}",
        )

    missing_fields = sorted(set(required_fields) - parsed_fields)
    if missing_fields:
        raise Phase2LabelledRequestsSpecError(
            f"{label} missing required fields: {', '.join(missing_fields)}",
        )


def _optional_string(source: Mapping[str, Any], key: str) -> str:
    """Read an optional YAML string field."""
    value = source.get(key, "")
    if value is None:
        return ""
    if not isinstance(value, str):
        raise Phase2LabelledRequestsSpecError(f"{key} must be a string")
    return value.strip()


def _optional_non_negative_int(
    source: Mapping[str, Any],
    key: str,
    *,
    default: int,
    label: str,
) -> int:
    """Read an optional non-negative integer YAML field."""
    value = source.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise Phase2LabelledRequestsSpecError(f"{label} must be a non-negative integer")
    return value


def _optional_positive_float(source: Mapping[str, Any], key: str, *, default: float) -> float:
    """Read an optional positive numeric YAML field."""
    value = source.get(key, default)
    if isinstance(value, bool):
        raise Phase2LabelledRequestsSpecError(f"{key} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise Phase2LabelledRequestsSpecError(f"{key} must be numeric") from exc
    if result <= 0:
        raise Phase2LabelledRequestsSpecError(f"{key} must be positive")
    return result


def _provider_adapter_spec(raw: Mapping[str, Any], *, label: str) -> ProviderAdapterSpec:
    """Load one provider adapter block from YAML."""
    adapter = _optional_string(raw, "adapter") or "mock"
    if adapter not in _PROVIDER_ADAPTERS:
        raise Phase2LabelledRequestsSpecError(
            f"{label}.adapter must be one of {', '.join(sorted(_PROVIDER_ADAPTERS))}",
        )
    payload_values = _optional_sequence(raw, "payload_request_fields")
    if not all(isinstance(item, str) and item.strip() for item in payload_values):
        raise Phase2LabelledRequestsSpecError(
            f"{label}.payload_request_fields must contain strings",
        )
    payload_fields = tuple(item.strip() for item in payload_values)

    endpoint_path = _optional_string(raw, "endpoint_path")
    response_text_path = _optional_string(raw, "response_text_path")
    base_url_env = _optional_string(raw, "base_url_env")
    if adapter == "openai-compatible":
        if not base_url_env:
            raise Phase2LabelledRequestsSpecError(
                f"{label}.base_url_env is required for live providers",
            )
        if not endpoint_path:
            raise Phase2LabelledRequestsSpecError(
                f"{label}.endpoint_path is required for live providers",
            )
        if not response_text_path:
            raise Phase2LabelledRequestsSpecError(
                f"{label}.response_text_path is required for live providers",
            )

    return ProviderAdapterSpec(
        adapter=adapter,
        base_url_env=base_url_env,
        api_key_env=_optional_string(raw, "api_key_env"),
        endpoint_path=endpoint_path,
        timeout_seconds=_optional_positive_float(raw, "timeout_seconds", default=30.0),
        response_text_path=response_text_path,
        payload_request_fields=payload_fields,
    )


def _optional_bool(source: Mapping[str, Any], key: str) -> bool | None:
    """Read an optional judge boolean; return ``None`` for malformed values."""
    if key not in source:
        return None
    value = source[key]
    if not isinstance(value, bool):
        return None
    return value


def _rate(numerator: int, denominator: int) -> float:
    """Compute a stable zero-safe rate."""
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _source_corpus_label(records: Sequence[RestApiRecord]) -> str:
    """Return a compact stable vendor/source-corpus metric value."""
    labels = {
        f"{record.vendor or 'unknown'}:{record.source_corpus or 'unknown'}"
        for record in records
    }
    return ",".join(sorted(labels))


# Author: Mus mbayramo@stanford.edu
