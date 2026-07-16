"""Offline Phase 2 labelled-request dataset plumbing.

Used by tests and future dataset-build scripts when constructing
``phase2_labelled_requests`` rows from Redfish REST API evidence. The module is
pure: it loads YAML specs, renders configured prompts, samples tiny records, and
parses injected judge responses without opening W&B, loading a model, or calling
the network.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import random
from typing import Any, Callable, Mapping, Sequence

import yaml

from igc.modules.base.metric_keys import (
    PHASE2_LABELLED_REQUESTS,
    PHASE2_LABELLED_REQUESTS_WANDB_METRIC_KEYS,
    phase_metric,
)

_DRAFT_TOTAL_KEY = phase_metric(PHASE2_LABELLED_REQUESTS, "draft_total")
_ACCEPTED_TOTAL_KEY = phase_metric(PHASE2_LABELLED_REQUESTS, "accepted_total")
_REJECTED_TOTAL_KEY = phase_metric(PHASE2_LABELLED_REQUESTS, "rejected_total")
_NONSENSE_RATE_KEY = phase_metric(PHASE2_LABELLED_REQUESTS, "nonsense_rate")
_INVALID_JSON_RATE_KEY = phase_metric(PHASE2_LABELLED_REQUESTS, "invalid_json_rate")
_PRO_ACCEPT_RATE_KEY = phase_metric(PHASE2_LABELLED_REQUESTS, "pro_accept_rate")
_REST_API_SET_MATCH_RATE_KEY = phase_metric(PHASE2_LABELLED_REQUESTS, "rest_api_set_match_rate")
_EMPTY_SET_MATCH_RATE_KEY = phase_metric(PHASE2_LABELLED_REQUESTS, "empty_set_match_rate")
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
    vendor: str = ""  # vendor/source family for W&B grouping.
    source_corpus: str = ""  # corpus artifact or fixture name for provenance.

    def to_prompt_dict(self) -> dict[str, Any]:
        """Serialize the record shape shown to model and judge prompts."""
        return {
            "rest_api": self.rest_api,  # sampled API path the text must cover.
            "allowed_methods": list(self.allowed_methods),  # legal methods for this API.
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
    :param wandb_namespace: W&B namespace for builder metrics.
    :param metric_keys: complete metric-key list from the spec.
    :param acceptance_thresholds: configured acceptance thresholds.
    """

    dataset_name: str  # canonical emitted dataset name.
    prompt_spec_version: str  # prompt/spec version copied into rows.
    sample_widths: tuple[int, ...]  # accepted sample widths.
    model_x: ModelXSpec  # draft model identity from YAML.
    judge: JudgeSpec  # judge route and model identity from YAML.
    generation: Mapping[str, Any]  # generation settings passed through unchanged.
    model_x_system: str  # YAML system prompt for model_x draft generation.
    model_x_template: str  # YAML prompt template for sampled records.
    judge_system: str  # YAML system prompt for private judge review.
    judge_template: str  # YAML prompt template for judge input.
    wandb_namespace: str  # W&B metric namespace.
    metric_keys: tuple[str, ...]  # metric keys declared by the spec.
    acceptance_thresholds: Mapping[str, float]  # acceptance threshold values.


@dataclass(frozen=True)
class ProJudgeResult:
    """Parsed private judge decision for one draft text."""

    accepted: bool  # whether the judge says this text may enter the dataset.
    rest_api_list: tuple[str, ...] = ()  # API set the judge extracted from text.
    nonsense: bool = False  # true when the draft is junk or not an operator request.
    invalid_json: bool = False  # true when the judge response could not be parsed.
    reason: str = ""  # short non-secret judge reason.
    order_evidence: str = "none"  # explicit order signal, or ``none``.


@dataclass(frozen=True)
class Phase2LabelledRequestRow:
    """Accepted Phase 2 labelled-request row."""

    text: str  # human request text accepted by the private judge.
    records: tuple[RestApiRecord, ...]  # sampled Redfish records used as context.
    rest_api_list: tuple[str, ...]  # canonical sampled REST API list.
    order_evidence: str  # explicit order signal, or ``none``.
    prompt_spec_version: str  # prompt/spec version used for this row.
    validation: Mapping[str, Any] = field(default_factory=dict)  # judge validation flags.

    def to_dict(self) -> dict[str, Any]:
        """Serialize the accepted row as JSON-compatible data."""
        allowed_methods = {
            record.rest_api: list(record.allowed_methods)
            for record in self.records
        }
        return {
            "phase": 2,  # Phase number for labelled-request rows.
            "dataset": PHASE2_LABELLED_REQUESTS,  # canonical dataset name.
            "task": "text_to_rest_api_list",  # Phase 2 list field, evaluated as an API set.
            "x": {
                "text": self.text,  # accepted human request text.
                "json": [dict(record.json_body) for record in self.records],  # JSON context.
                "allowed_methods": allowed_methods,  # legal methods per REST API.
                "rest_api_list": list(self.rest_api_list),  # context API list.
            },
            "y_true": {
                "rest_api_list": list(self.rest_api_list),  # unordered target API set.
                "order_evidence": self.order_evidence,  # explicit order signal if any.
            },
            "validation": dict(self.validation),  # judge verdict details.
            "metadata": {
                "prompt_spec_version": self.prompt_spec_version,  # prompt/config version.
                "vendor": [record.vendor for record in self.records],  # vendor provenance.
                "source_corpus": [record.source_corpus for record in self.records],  # corpus provenance.
            },
        }


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
    if dataset_name != PHASE2_LABELLED_REQUESTS:
        raise Phase2LabelledRequestsSpecError(
            f"dataset.name must be {PHASE2_LABELLED_REQUESTS!r}",
        )

    sampling = _mapping(raw, "sampling", required=True)
    raw_sample_widths = _sequence(sampling, "sample_widths")
    if not all(isinstance(width, int) and not isinstance(width, bool) for width in raw_sample_widths):
        raise Phase2LabelledRequestsSpecError(
            "sampling.sample_widths must be integer sequence [1, 2, 3]",
        )
    sample_widths = tuple(raw_sample_widths)
    if sample_widths != (1, 2, 3):
        raise Phase2LabelledRequestsSpecError("sampling.sample_widths must be [1, 2, 3]")

    model_x_raw = _mapping(raw, "model_x", required=True)
    judge_raw = _mapping(raw, "judge", required=True)
    prompts = _mapping(raw, "prompts", required=True)
    model_prompt = _mapping(prompts, "model_x_draft", required=True)
    judge_prompt = _mapping(prompts, "pro_judge", required=True)
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

    return Phase2LabelledRequestsSpec(
        dataset_name=dataset_name,
        prompt_spec_version=_required_string(
            dataset,
            "prompt_spec_version",
            "dataset.prompt_spec_version",
        ),
        sample_widths=sample_widths,
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
        model_x_system=_required_string(model_prompt, "system", "prompts.model_x_draft.system"),
        model_x_template=_required_string(
            model_prompt,
            "template",
            "prompts.model_x_draft.template",
        ),
        judge_system=_required_string(judge_prompt, "system", "prompts.pro_judge.system"),
        judge_template=_required_string(judge_prompt, "template", "prompts.pro_judge.template"),
        wandb_namespace=wandb_namespace,
        metric_keys=metric_keys,
        acceptance_thresholds=acceptance_thresholds,
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


def render_pro_judge_prompt(
    spec: Phase2LabelledRequestsSpec,
    records: Sequence[RestApiRecord],
    draft_text: str,
) -> str:
    """Render the private judge prompt from YAML-provided prompt fields."""
    return _render_prompt(
        system=spec.judge_system,
        template=spec.judge_template,
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
            accepted=False,
            invalid_json=True,
            reason=f"invalid_json: {exc.msg}",
        )
    if isinstance(parsed, Mapping) and isinstance(parsed.get("y_pred"), Mapping):
        parsed = parsed["y_pred"]
    if not isinstance(parsed, Mapping):
        return ProJudgeResult(accepted=False, invalid_json=True, reason="judge result is not a mapping")

    if "rest_api_list" in parsed:
        rest_api_value = parsed["rest_api_list"]
        rest_api_field = "rest_api_list"
    elif "rest_api_set" in parsed:
        rest_api_value = parsed["rest_api_set"]
        rest_api_field = "rest_api_set"
    else:
        return ProJudgeResult(
            accepted=False,
            invalid_json=True,
            reason="rest_api_list or rest_api_set is required",
        )

    if not isinstance(rest_api_value, list):
        return ProJudgeResult(
            accepted=False,
            invalid_json=True,
            reason=f"{rest_api_field} is not a list",
        )
    if not all(isinstance(item, str) for item in rest_api_value):
        return ProJudgeResult(
            accepted=False,
            invalid_json=True,
            reason=f"{rest_api_field} must contain only strings",
        )

    if "accepted" in parsed:
        accepted_key = "accepted"
    elif "accept" in parsed:
        accepted_key = "accept"
    else:
        return ProJudgeResult(
            accepted=False,
            invalid_json=True,
            reason="accepted or accept is required",
        )
    accepted = _optional_bool(parsed, accepted_key)
    if accepted is None:
        return ProJudgeResult(
            accepted=False,
            invalid_json=True,
            reason=f"{accepted_key} must be a boolean when present",
        )
    nonsense = _optional_bool(parsed, "nonsense")
    if nonsense is None:
        return ProJudgeResult(
            accepted=False,
            invalid_json=True,
            reason="nonsense must be a boolean when present",
        )

    return ProJudgeResult(
        accepted=accepted,
        rest_api_list=tuple(rest_api_value),
        nonsense=nonsense,
        invalid_json=False,
        reason=str(parsed.get("reason", "")),
        order_evidence=str(parsed.get("order_evidence", "none")),
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
        set_match = (
            not result.invalid_json
            and compare_rest_api_sets(expected_rest_api_list, result.rest_api_list)
        )
        accepted = result.accepted and set_match and not result.nonsense and not result.invalid_json

        if result.nonsense:
            self.nonsense_total += 1
        if result.invalid_json:
            self.invalid_json_total += 1
        if result.accepted and not result.invalid_json:
            self.pro_accept_total += 1
        if set_match:
            self.rest_api_set_match_total += 1
        if not result.invalid_json and not expected_rest_api_list:
            self.empty_set_expected_total += 1
            if empty_set_matches(expected_rest_api_list, result.rest_api_list):
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
            _EMPTY_SET_MATCH_RATE_KEY: _rate(
                self.empty_set_match_total,
                self.empty_set_expected_total,
            ),
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
    ) -> None:
        """Create a builder with pure injected provider callables."""
        self._spec = spec
        self._draft_provider = draft_provider
        self._judge_provider = judge_provider

    def build_one(
        self,
        records: Sequence[RestApiRecord],
        *,
        k: int,
        rng: random.Random,
    ) -> tuple[Phase2LabelledRequestRow | None, Phase2LabelledRequestCounters]:
        """Build and judge one accepted row candidate.

        :param records: candidate REST API records.
        :param k: sample width, one through three.
        :param rng: deterministic RNG supplied by the caller.
        :return: accepted row plus counters, or ``None`` plus rejection counters.
        """
        sampled = sample_phase2_contexts(records, k=k, rng=rng)
        expected_rest_api_list = tuple(record.rest_api for record in sampled)
        counters = Phase2LabelledRequestCounters(
            sample_width_k=k,
            vendor_source_corpus=_source_corpus_label(sampled),
            prompt_spec_version=self._spec.prompt_spec_version,
            model_x_artifact_sha=self._spec.model_x.artifact_sha,
            judge_model=self._spec.judge.model_id,
            judge_profile=self._spec.judge.profile,
        )

        draft_request = {
            "prompt": render_model_x_prompt(self._spec, sampled),  # full configured prompt.
            "model_id": self._spec.model_x.model_id,  # draft model from YAML.
            "generation": dict(self._spec.generation),  # generation knobs from YAML.
            "sample_width": k,  # current sample width metric value.
        }
        draft_text = self._draft_provider(draft_request)
        counters.observe_draft(draft_text)

        judge_request = {
            "prompt": render_pro_judge_prompt(self._spec, sampled, draft_text),  # judge prompt.
            "model_id": self._spec.judge.model_id,  # judge model from YAML.
            "profile": self._spec.judge.profile,  # judge profile from YAML.
            "route": self._spec.judge.route,  # judge route from YAML.
            "expected_rest_api_list": list(expected_rest_api_list),  # known sampled set.
        }
        judge_result = parse_pro_judge_result(self._judge_provider(judge_request))
        counters.observe_judge(judge_result, expected_rest_api_list=expected_rest_api_list)

        set_match = compare_rest_api_sets(expected_rest_api_list, judge_result.rest_api_list)
        accepted = (
            judge_result.accepted
            and set_match
            and not judge_result.nonsense
            and not judge_result.invalid_json
        )
        if not accepted:
            return None, counters

        row = Phase2LabelledRequestRow(
            text=draft_text,
            records=sampled,
            rest_api_list=expected_rest_api_list,
            order_evidence=judge_result.order_evidence,
            prompt_spec_version=self._spec.prompt_spec_version,
            validation={
                "text_source": "model_x_then_private_judge",  # draft then judge path.
                "review_judged": True,  # private judge parsed successfully.
                "all_rest_api_present": set(expected_rest_api_list) <= set(judge_result.rest_api_list),
                "extra_rest_api_present": not set(judge_result.rest_api_list) <= set(expected_rest_api_list),
                "set_coverage_preserved": set_match,  # unordered API set contract.
                "nonsense": judge_result.nonsense,  # judge nonsense flag.
            },
        )
        return row, counters


def to_minimal_phase3_input(row: Phase2LabelledRequestRow | None) -> dict[str, Any]:
    """Convert an accepted Phase 2 row into a Phase 3 input fixture."""
    if row is None:
        raise ValueError("phase2 row is required")
    allowed_methods = {
        record.rest_api: list(record.allowed_methods)
        for record in row.records
    }
    return {
        "text": row.text,  # accepted human request text.
        "rest_api_list": list(row.rest_api_list),  # Phase 2 API-set target.
        "json": [dict(record.json_body) for record in row.records],  # same-row JSON context.
        "allowed_methods": allowed_methods,  # same-row allowed methods.
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


def _required_string(source: Mapping[str, Any], key: str, label: str) -> str:
    """Read a required non-empty YAML string field."""
    value = source.get(key)
    if not isinstance(value, str) or not value.strip():
        raise Phase2LabelledRequestsSpecError(f"{label} must be a non-empty string")
    return value


def _optional_bool(source: Mapping[str, Any], key: str) -> bool | None:
    """Read an optional judge boolean; return ``None`` for malformed values."""
    if key not in source:
        return False
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
