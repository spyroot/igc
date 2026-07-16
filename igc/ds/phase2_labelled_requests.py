"""Offline plumbing for the Phase 2 labelled request dataset.

The builder samples Redfish REST API records, renders configured prompts, and
parses configured judge responses. This module is deliberately pure: it does not
call models, W&B, GPUs, Redfish hosts, or private runtime endpoints.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import yaml

from igc.modules.base.metric_keys import phase_metric


class Phase2LabelledRequestsSpecError(ValueError):
    """Raised when the Phase 2 labelled request YAML spec is invalid."""


_TOP_LEVEL_KEYS = {
    "version",
    "dataset_name",
    "sample_widths",
    "wandb",
    "model_x",
    "judge",
    "generation",
    "acceptance_thresholds",
    "prompts",
}
_PHASE2_SAMPLE_WIDTHS = (1, 2, 3)


@dataclass(frozen=True)
class PromptSpec:
    """One prompt template loaded from ``configs/phase2_labelled_requests.yaml``."""

    name: str
    version: str
    template: str


@dataclass(frozen=True)
class Phase2LabelledRequestsSpec:
    """Normalized runtime spec loaded from ``configs/phase2_labelled_requests.yaml``."""

    path: Path
    version: str
    dataset_name: str
    sample_widths: tuple[int, ...]
    wandb: dict[str, Any]
    model_x: dict[str, Any]
    judge: dict[str, Any]
    generation: dict[str, Any]
    acceptance_thresholds: dict[str, float]
    prompts: dict[str, PromptSpec]


@dataclass(frozen=True)
class Phase2RestApiRecord:
    """One Redfish REST API record available to the labelled-request builder."""

    rest_api: str
    allowed_methods: tuple[str, ...]
    json_body: Mapping[str, Any]
    vendor: str = ""
    source_corpus: str = ""

    def __post_init__(self) -> None:
        """Normalize method storage while preserving public record data."""
        if not self.rest_api:
            raise ValueError("rest_api must be non-empty")
        object.__setattr__(self, "allowed_methods", tuple(self.allowed_methods))

    def to_prompt_dict(self) -> dict[str, Any]:
        """Return the public-safe record shape used by configured prompts."""
        return {
            "rest_api": self.rest_api,
            "allowed_methods": list(self.allowed_methods),
            "json": dict(self.json_body),
            "vendor": self.vendor,
            "source_corpus": self.source_corpus,
        }


@dataclass(frozen=True)
class Phase2RequestSample:
    """A sampled set of one to three REST API records."""

    records: tuple[Phase2RestApiRecord, ...]

    @property
    def sample_width(self) -> int:
        """Return the number of sampled REST API records."""
        return len(self.records)

    @property
    def rest_api_list(self) -> tuple[str, ...]:
        """Return sampled REST APIs in deterministic stored order."""
        return tuple(record.rest_api for record in self.records)

    @property
    def allowed_methods_by_rest_api(self) -> dict[str, list[str]]:
        """Return allowed HTTP methods keyed by REST API."""
        return {
            record.rest_api: list(record.allowed_methods)
            for record in self.records
        }

    def to_prompt_payload(self) -> list[dict[str, Any]]:
        """Return the sampled record payload for prompt rendering."""
        return [record.to_prompt_dict() for record in self.records]


@dataclass(frozen=True)
class ProJudgeResult:
    """Parsed private-judge decision for one drafted request label."""

    valid_json: bool
    accepted: bool
    pro_accept: bool
    rest_api_set_match: bool
    empty_set_match: bool
    nonsense: bool
    invalid_json: bool
    rest_api_list: tuple[str, ...] = ()
    reason: str = ""
    raw: str = ""


@dataclass
class Phase2LabelledRequestCounters:
    """Offline counters for Phase 2 labelled request generation metrics."""

    draft_total: int = 0
    accepted_total: int = 0
    rejected_total: int = 0
    nonsense_total: int = 0
    invalid_json_total: int = 0
    pro_accept_total: int = 0
    rest_api_set_match_total: int = 0
    empty_set_match_total: int = 0
    last_sample_width: int = 0
    last_vendor: str = ""
    last_source_corpus: str = ""

    def record_outcome(
        self,
        result: ProJudgeResult,
        *,
        sample_width: int,
        vendor: str,
        source_corpus: str,
        spec: Phase2LabelledRequestsSpec,
    ) -> None:
        """Record one draft/judge outcome without opening a live metrics run."""
        if sample_width not in spec.sample_widths:
            raise ValueError(f"sample width must be one of {spec.sample_widths}")

        self.draft_total += 1
        self.last_sample_width = sample_width
        self.last_vendor = vendor
        self.last_source_corpus = source_corpus

        if result.accepted:
            self.accepted_total += 1
        else:
            self.rejected_total += 1
        if result.nonsense:
            self.nonsense_total += 1
        if result.invalid_json:
            self.invalid_json_total += 1
        if result.pro_accept:
            self.pro_accept_total += 1
        if result.rest_api_set_match:
            self.rest_api_set_match_total += 1
        if result.empty_set_match:
            self.empty_set_match_total += 1

    def to_wandb_metrics(self, spec: Phase2LabelledRequestsSpec) -> dict[str, Any]:
        """Return metric-key/value pairs suitable for an offline W&B logger seam."""
        namespace = _required_string(spec.wandb, "namespace", context="wandb")
        source_value = (
            f"{self.last_vendor}/{self.last_source_corpus}"
            if self.last_vendor or self.last_source_corpus else ""
        )
        return {
            phase_metric(namespace, "build", "draft_total"): self.draft_total,
            phase_metric(namespace, "build", "accepted_total"): self.accepted_total,
            phase_metric(namespace, "build", "rejected_total"): self.rejected_total,
            phase_metric(namespace, "eval", "nonsense_rate"): self._rate(self.nonsense_total),
            phase_metric(namespace, "eval", "invalid_json_rate"): self._rate(
                self.invalid_json_total
            ),
            phase_metric(namespace, "eval", "pro_accept_rate"): self._rate(
                self.pro_accept_total
            ),
            phase_metric(namespace, "eval", "rest_api_set_match_rate"): self._rate(
                self.rest_api_set_match_total
            ),
            phase_metric(namespace, "eval", "empty_set_match_rate"): self._rate(
                self.empty_set_match_total
            ),
            phase_metric(namespace, "sample_width", "k"): self.last_sample_width,
            phase_metric(namespace, "vendor", "source_corpus"): source_value,
            phase_metric(namespace, "spec", "prompt_spec_version"): spec.version,
            phase_metric(namespace, "model", "model_x_artifact_sha"): _required_string(
                spec.model_x, "artifact_sha", context="model_x"
            ),
            phase_metric(namespace, "judge", "model"): _required_string(
                spec.judge, "model", context="judge"
            ),
            phase_metric(namespace, "judge", "profile"): _required_string(
                spec.judge, "profile", context="judge"
            ),
        }

    def _rate(self, numerator: int) -> float:
        """Return a stable zero-safe rate over drafted examples."""
        if self.draft_total == 0:
            return 0.0
        return numerator / self.draft_total


def build_phase2_labelled_request_row(
    spec: Phase2LabelledRequestsSpec,
    sample: Phase2RequestSample,
    *,
    draft_text: str,
    judge_result: ProJudgeResult,
) -> dict[str, Any]:
    """Build one accepted Phase 2 labelled-request dataset row.

    :param spec: YAML-backed Phase 2 labelled-request spec.
    :param sample: sampled Redfish records that define the expected REST API set.
    :param draft_text: natural operator request drafted by the configured provider.
    :param judge_result: parsed private-judge result for ``draft_text``.
    :return: JSON-compatible accepted row for ``phase2_labelled_requests``.
    :raises ValueError: when the judge result is not accepted.
    """
    if not judge_result.accepted:
        raise ValueError("cannot build accepted row from rejected judge result")
    return {
        "phase": 2,
        "dataset": spec.dataset_name,
        "task": "text_to_rest_api_set",
        "x": {
            "text": draft_text,
            "records": sample.to_prompt_payload(),
        },
        "y_true": {
            "rest_api_set": list(sample.rest_api_list),
            "order_evidence": "none",
        },
        "validation": {
            "text_source": "model_x_then_private_pro_judge",
            "pro_judged": True,
            "rest_api_set_match": judge_result.rest_api_set_match,
            "empty_set_match": judge_result.empty_set_match,
            "nonsense": judge_result.nonsense,
            "invalid_json": judge_result.invalid_json,
            "judge_reason": judge_result.reason,
        },
    }


def load_phase2_labelled_requests_spec(
    path: str | Path,
) -> Phase2LabelledRequestsSpec:
    """Load and validate the Phase 2 labelled request YAML spec."""
    spec_path = Path(path)
    try:
        raw = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise Phase2LabelledRequestsSpecError(
            f"cannot read Phase 2 labelled request spec {spec_path}: {exc}"
        ) from exc
    except yaml.YAMLError as exc:
        raise Phase2LabelledRequestsSpecError(
            f"cannot parse YAML in Phase 2 labelled request spec {spec_path}: {exc}"
        ) from exc

    if not isinstance(raw, dict):
        raise Phase2LabelledRequestsSpecError("Phase 2 spec must be a YAML mapping")

    unknown = sorted(set(raw) - _TOP_LEVEL_KEYS)
    if unknown:
        raise Phase2LabelledRequestsSpecError(
            f"unknown top-level keys: {', '.join(unknown)}"
        )

    version = _required_string(raw, "version")
    dataset_name = _required_string(raw, "dataset_name")
    sample_widths = _sample_widths(raw.get("sample_widths"))
    prompts = _prompt_specs(_mapping(raw, "prompts", required=True))

    return Phase2LabelledRequestsSpec(
        path=spec_path,
        version=version,
        dataset_name=dataset_name,
        sample_widths=sample_widths,
        wandb=_mapping(raw, "wandb", required=True),
        model_x=_mapping(raw, "model_x", required=True),
        judge=_mapping(raw, "judge", required=True),
        generation=_mapping(raw, "generation", required=True),
        acceptance_thresholds=_float_mapping(
            _mapping(raw, "acceptance_thresholds", required=True),
            "acceptance_thresholds",
        ),
        prompts=prompts,
    )


def sample_rest_api_records(
    records: Sequence[Phase2RestApiRecord],
    *,
    k: int,
    rng: random.Random,
    allowed_widths: Sequence[int],
) -> Phase2RequestSample:
    """Sample ``k`` REST API records without replacement."""
    allowed = tuple(allowed_widths)
    if k not in allowed:
        raise ValueError(f"sample width must be one of {allowed}")
    if len(records) < k:
        raise ValueError("not enough records to sample requested width")
    return Phase2RequestSample(tuple(rng.sample(list(records), k)))


def render_phase2_prompt(
    spec: Phase2LabelledRequestsSpec,
    prompt_name: str,
    sample: Phase2RequestSample,
    *,
    draft_text: str = "",
) -> str:
    """Render a configured Phase 2 prompt for one sampled record set."""
    try:
        prompt = spec.prompts[prompt_name]
    except KeyError as exc:
        raise Phase2LabelledRequestsSpecError(
            f"unknown prompt in Phase 2 spec: {prompt_name}"
        ) from exc

    variables = {
        "dataset_name": spec.dataset_name,
        "prompt_spec_version": prompt.version,
        "sample_width": sample.sample_width,
        "records_json": _json(sample.to_prompt_payload()),
        "rest_api_list_json": _json(list(sample.rest_api_list)),
        "allowed_methods_json": _json(sample.allowed_methods_by_rest_api),
        "draft_text": draft_text,
        "judge_model": _required_string(spec.judge, "model", context="judge"),
        "judge_profile": _required_string(spec.judge, "profile", context="judge"),
        "model_x_id": _required_string(spec.model_x, "id", context="model_x"),
    }
    try:
        return prompt.template.format(**variables)
    except KeyError as exc:
        missing = exc.args[0]
        raise Phase2LabelledRequestsSpecError(
            f"prompt {prompt_name} references unknown variable {missing}"
        ) from exc


def rest_api_sets_equal(
    expected_rest_apis: Sequence[str],
    actual_rest_apis: Sequence[str],
) -> bool:
    """Return unordered REST API set equality; empty set equals empty set."""
    return frozenset(expected_rest_apis) == frozenset(actual_rest_apis)


def parse_pro_judge_result(
    raw: str,
    *,
    expected_rest_apis: Sequence[str],
) -> ProJudgeResult:
    """Parse a private Pro judge JSON response for one drafted label."""
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as exc:
        return ProJudgeResult(
            valid_json=False,
            accepted=False,
            pro_accept=False,
            rest_api_set_match=False,
            empty_set_match=False,
            nonsense=False,
            invalid_json=True,
            reason=str(exc),
            raw=raw,
        )

    if not isinstance(decoded, Mapping):
        return ProJudgeResult(
            valid_json=False,
            accepted=False,
            pro_accept=False,
            rest_api_set_match=False,
            empty_set_match=False,
            nonsense=False,
            invalid_json=True,
            reason="judge response must be a JSON object",
            raw=raw,
        )

    if "rest_api_list" in decoded:
        rest_api_value = decoded["rest_api_list"]
        rest_api_field = "rest_api_list"
    elif "rest_api_set" in decoded:
        rest_api_value = decoded["rest_api_set"]
        rest_api_field = "rest_api_set"
    else:
        # A judge response with NEITHER field is malformed output, not an empty
        # set: silently substituting [] would let a bare {"accepted": true}
        # accept a hard-negative row with no REST API evidence at all.
        return ProJudgeResult(
            valid_json=False,
            accepted=False,
            pro_accept=False,
            rest_api_set_match=False,
            empty_set_match=False,
            nonsense=False,
            invalid_json=True,
            reason="missing rest_api_list or rest_api_set field",
            raw=raw,
        )

    try:
        rest_api_list = _string_tuple(rest_api_value, rest_api_field)
    except Phase2LabelledRequestsSpecError as exc:
        return ProJudgeResult(
            valid_json=False,
            accepted=False,
            pro_accept=False,
            rest_api_set_match=False,
            empty_set_match=False,
            nonsense=False,
            invalid_json=True,
            reason=str(exc),
            raw=raw,
        )
    try:
        nonsense = _optional_bool(decoded, "nonsense", default=False)
        if "accepted" in decoded:
            requested_accept = _optional_bool(decoded, "accepted", default=False)
        else:
            requested_accept = _optional_bool(decoded, "accept", default=False)
    except Phase2LabelledRequestsSpecError as exc:
        return ProJudgeResult(
            valid_json=False,
            accepted=False,
            pro_accept=False,
            rest_api_set_match=False,
            empty_set_match=False,
            nonsense=False,
            invalid_json=True,
            reason=str(exc),
            raw=raw,
        )
    computed_set_match = rest_api_sets_equal(expected_rest_apis, rest_api_list)
    judge_set_match = decoded.get("rest_api_set_match")
    rest_api_set_match = (
        computed_set_match if not isinstance(judge_set_match, bool)
        else judge_set_match and computed_set_match
    )
    empty_set_match = not expected_rest_apis and not rest_api_list
    pro_accept = requested_accept and rest_api_set_match and not nonsense

    return ProJudgeResult(
        valid_json=True,
        accepted=pro_accept,
        pro_accept=pro_accept,
        rest_api_set_match=rest_api_set_match,
        empty_set_match=empty_set_match,
        nonsense=nonsense,
        invalid_json=False,
        rest_api_list=rest_api_list,
        reason=str(decoded.get("reason", "")),
        raw=raw,
    )


def _mapping(raw: Mapping[str, Any], key: str, *, required: bool = False) -> dict[str, Any]:
    """Return a YAML mapping field as a dict."""
    value = raw.get(key)
    if value is None:
        if required:
            raise Phase2LabelledRequestsSpecError(f"missing required field: {key}")
        return {}
    if not isinstance(value, Mapping):
        raise Phase2LabelledRequestsSpecError(f"{key} must be a mapping")
    return dict(value)


def _required_string(
    raw: Mapping[str, Any],
    key: str,
    *,
    context: str | None = None,
) -> str:
    """Return a non-empty string field from a mapping."""
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        prefix = f"{context}." if context else ""
        raise Phase2LabelledRequestsSpecError(f"{prefix}{key} must be a non-empty string")
    return value


def _sample_widths(value: Any) -> tuple[int, ...]:
    """Validate configured sample widths."""
    if not isinstance(value, list) or not value:
        raise Phase2LabelledRequestsSpecError("sample_widths must be a non-empty list")
    widths = tuple(value)
    if widths != _PHASE2_SAMPLE_WIDTHS:
        raise Phase2LabelledRequestsSpecError(
            f"sample_widths must be {list(_PHASE2_SAMPLE_WIDTHS)}"
        )
    return widths


def _prompt_specs(raw: Mapping[str, Any]) -> dict[str, PromptSpec]:
    """Normalize prompt specs from the YAML mapping."""
    prompts: dict[str, PromptSpec] = {}
    for name, value in raw.items():
        if not isinstance(value, Mapping):
            raise Phase2LabelledRequestsSpecError(f"prompt {name} must be a mapping")
        prompt = dict(value)
        prompts[name] = PromptSpec(
            name=name,
            version=_required_string(prompt, "version", context=f"prompts.{name}"),
            template=_required_string(prompt, "template", context=f"prompts.{name}"),
        )
    return prompts


def _float_mapping(raw: Mapping[str, Any], context: str) -> dict[str, float]:
    """Return a mapping whose values are numeric thresholds."""
    out: dict[str, float] = {}
    for key, value in raw.items():
        if not isinstance(value, (int, float)):
            raise Phase2LabelledRequestsSpecError(f"{context}.{key} must be numeric")
        out[str(key)] = float(value)
    return out


def _string_tuple(value: Any, context: str) -> tuple[str, ...]:
    """Return a tuple of strings from a JSON list field."""
    if not isinstance(value, list):
        raise Phase2LabelledRequestsSpecError(f"{context} must be a list")
    if not all(isinstance(item, str) for item in value):
        raise Phase2LabelledRequestsSpecError(f"{context} entries must be strings")
    return tuple(value)


def _optional_bool(
    raw: Mapping[str, Any],
    key: str,
    *,
    default: bool,
) -> bool:
    """Return an optional judge boolean without Python truthiness coercion."""
    if key not in raw:
        return default
    value = raw[key]
    if not isinstance(value, bool):
        raise Phase2LabelledRequestsSpecError(f"{key} must be a boolean")
    return value


def _json(value: Any) -> str:
    """Return stable pretty JSON for prompt variables."""
    return json.dumps(value, indent=2, sort_keys=True)


# Author: Mus mbayramo@stanford.edu
