"""Offline Phase 1 held-out prediction metrics.

This module consumes already-produced Phase 1 prediction JSONL artifacts. It
does not load a model, touch a GPU, call W&B, or read corpora. Rows are matched
by an explicit row id when present, otherwise by JSONL position.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping

from igc.modules.base.metric_keys import PHASE1_FINETUNE, phase_metric

JsonValue = Any


class Phase1GoldenError(ValueError):
    """Raised when a prediction artifact cannot be evaluated safely."""


@dataclass(frozen=True)
class PredictionRecord:
    """Normalized view of one Phase 1 prediction row."""

    key: str
    line_number: int
    rest_api: str
    target_json: JsonValue | None
    prediction_json: JsonValue | None
    parse_ok: bool
    parse_error: str = ""
    exact_match: bool | None = None
    odata_id_match: bool | None = None
    top_k_match: bool | None = None
    sequence_length: int | None = None
    generated_tokens: int | None = None
    padding_tokens: int | None = None
    latency_sec: float | None = None
    memory_peak_mb: float | None = None
    confidence: float | None = None
    log_prob: float | None = None
    log_prob_per_token: float | None = None


@dataclass(frozen=True)
class ArtifactSummary:
    """Public-safe metadata for an input JSONL artifact."""

    role: str
    name: str
    sha256: str
    bytes: int
    rows: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable summary."""
        return dict(self.__dict__)


@dataclass
class ArtifactMetrics:
    """Metrics computed for one prediction artifact."""

    role: str
    artifact: ArtifactSummary
    row_count: int
    parsed_rows: int
    missing_target_rows: int
    missing_prediction_rows: int
    exact_match_count: int
    exact_match_rows: int
    odata_id_match_count: int
    odata_id_rows: int
    top_k_match_count: int
    top_k_rows: int
    total_generated_tokens: int
    total_latency_sec: float
    sequence_lengths: list[int] = field(default_factory=list)
    padding_ratios: list[float] = field(default_factory=list)
    latencies_sec: list[float] = field(default_factory=list)
    memory_peaks_mb: list[float] = field(default_factory=list)
    row_log_prob_per_token: list[tuple[float, int]] = field(default_factory=list)
    calibration_pairs: list[tuple[float, bool]] = field(default_factory=list)

    def _rate(self, numerator: int, denominator: int) -> float | None:
        return None if denominator == 0 else numerator / denominator

    def to_metric_dict(self, *, ece_bins: int = 10) -> dict[str, float | int | None]:
        """Return standard Phase 1 metric keys for this artifact."""
        return {
            "row_count": self.row_count,
            "missing_target_rows": self.missing_target_rows,
            "missing_prediction_rows": self.missing_prediction_rows,
            phase_metric(PHASE1_FINETUNE, "eval", "json_parse_rate"): (
                self._rate(self.parsed_rows, self.row_count)
            ),
            phase_metric(PHASE1_FINETUNE, "eval", "json_exact_match_rate"): (
                self._rate(self.exact_match_count, self.exact_match_rows)
            ),
            phase_metric(PHASE1_FINETUNE, "eval", "odata_id_match_rate"): (
                self._rate(self.odata_id_match_count, self.odata_id_rows)
            ),
            phase_metric(PHASE1_FINETUNE, "eval", "top_k_accuracy"): (
                self._rate(self.top_k_match_count, self.top_k_rows)
            ),
            phase_metric(PHASE1_FINETUNE, "throughput", "eval_tokens_per_sec"): (
                None
                if self.total_latency_sec <= 0
                else self.total_generated_tokens / self.total_latency_sec
            ),
            phase_metric(PHASE1_FINETUNE, "throughput", "eval_samples_per_sec"): (
                None if self.total_latency_sec <= 0 else self.row_count / self.total_latency_sec
            ),
            phase_metric(PHASE1_FINETUNE, "data", "padding_ratio"): (
                None if not self.padding_ratios else mean(self.padding_ratios)
            ),
            phase_metric(PHASE1_FINETUNE, "data", "mean_sequence_length"): (
                None if not self.sequence_lengths else mean(self.sequence_lengths)
            ),
            phase_metric(PHASE1_FINETUNE, "data", "max_sequence_length"): (
                None if not self.sequence_lengths else max(self.sequence_lengths)
            ),
            phase_metric(PHASE1_FINETUNE, "calibration", "log_prob_per_token"): (
                _weighted_mean(self.row_log_prob_per_token)
            ),
            phase_metric(PHASE1_FINETUNE, "calibration", "ece"): (
                expected_calibration_error(self.calibration_pairs, bins=ece_bins)
            ),
            phase_metric(PHASE1_FINETUNE, "test", "latency_sec_p50"): percentile(
                self.latencies_sec,
                50,
            ),
            phase_metric(PHASE1_FINETUNE, "test", "latency_sec_p95"): percentile(
                self.latencies_sec,
                95,
            ),
            phase_metric(PHASE1_FINETUNE, "test", "memory_peak_mb"): (
                None if not self.memory_peaks_mb else max(self.memory_peaks_mb)
            ),
        }

    def to_evidence_dict(self, *, ece_bins: int = 10) -> dict[str, Any]:
        """Return compact evidence for artifact-level metrics."""
        return {
            "artifact": self.artifact.to_dict(),
            "counts": {
                "rows": self.row_count,
                "parsed_rows": self.parsed_rows,
                "missing_target_rows": self.missing_target_rows,
                "missing_prediction_rows": self.missing_prediction_rows,
                "exact_match_rows": self.exact_match_rows,
                "exact_match_count": self.exact_match_count,
                "odata_id_rows": self.odata_id_rows,
                "odata_id_match_count": self.odata_id_match_count,
                "top_k_rows": self.top_k_rows,
                "top_k_match_count": self.top_k_match_count,
            },
            "metrics": self.to_metric_dict(ece_bins=ece_bins),
        }


def _weighted_mean(values: Iterable[tuple[float, int]]) -> float | None:
    total_weight = 0
    total = 0.0
    for value, weight in values:
        if weight <= 0:
            continue
        total += value * weight
        total_weight += weight
    return None if total_weight == 0 else total / total_weight


def percentile(values: Iterable[float], pct: float) -> float | None:
    """Return a linear-interpolated percentile for non-empty values."""
    ordered = sorted(values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def expected_calibration_error(
    pairs: Iterable[tuple[float, bool]],
    *,
    bins: int = 10,
) -> float | None:
    """Compute equal-width ECE from ``(confidence, correct)`` pairs."""
    clean = [(conf, correct) for conf, correct in pairs if 0.0 <= conf <= 1.0]
    if not clean:
        return None
    if bins <= 0:
        raise Phase1GoldenError("ece bins must be positive")
    total = len(clean)
    ece = 0.0
    for index in range(bins):
        low = index / bins
        high = (index + 1) / bins
        if index == bins - 1:
            bucket = [(c, ok) for c, ok in clean if low <= c <= high]
        else:
            bucket = [(c, ok) for c, ok in clean if low <= c < high]
        if not bucket:
            continue
        confidence = mean(c for c, _ in bucket)
        accuracy = mean(1.0 if ok else 0.0 for _, ok in bucket)
        ece += (len(bucket) / total) * abs(accuracy - confidence)
    return ece


def artifact_summary(path: str | Path, role: str, rows: int) -> ArtifactSummary:
    """Hash a JSONL artifact without exposing its parent path."""
    p = Path(path)
    digest = hashlib.sha256()
    size = 0
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return ArtifactSummary(
        role=role,
        name=p.name,
        sha256=digest.hexdigest(),
        bytes=size,
        rows=rows,
    )


def load_prediction_rows(path: str | Path) -> list[dict[str, Any]]:
    """Load a non-empty JSONL prediction artifact."""
    rows: list[dict[str, Any]] = []
    p = Path(path)
    with p.open(encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as err:
                raise Phase1GoldenError(
                    f"{p.name}:{line_number}: invalid JSONL row"
                ) from err
            if not isinstance(row, dict):
                raise Phase1GoldenError(
                    f"{p.name}:{line_number}: prediction row must be an object"
                )
            rows.append(row)
    if not rows:
        raise Phase1GoldenError(f"{p.name}: no prediction rows")
    return rows


def evaluate_prediction_jsonl(
    path: str | Path,
    *,
    role: str,
    ece_bins: int = 10,
) -> ArtifactMetrics:
    """Evaluate one Phase 1 prediction JSONL artifact."""
    raw_rows = load_prediction_rows(path)
    records = [normalize_prediction_row(row, index) for index, row in enumerate(raw_rows)]
    metrics = _metrics_from_records(
        role=role,
        records=records,
        artifact=artifact_summary(path, role, len(records)),
    )
    if ece_bins <= 0:
        raise Phase1GoldenError("ece bins must be positive")
    return metrics


def _metrics_from_records(
    *,
    role: str,
    records: list[PredictionRecord],
    artifact: ArtifactSummary,
) -> ArtifactMetrics:
    metrics = ArtifactMetrics(
        role=role,
        artifact=artifact,
        row_count=len(records),
        parsed_rows=sum(1 for record in records if record.parse_ok),
        missing_target_rows=sum(1 for record in records if record.target_json is None),
        missing_prediction_rows=sum(1 for record in records if not record.parse_ok),
        exact_match_count=sum(1 for record in records if record.exact_match is True),
        exact_match_rows=sum(1 for record in records if record.exact_match is not None),
        odata_id_match_count=sum(1 for record in records if record.odata_id_match is True),
        odata_id_rows=sum(1 for record in records if record.odata_id_match is not None),
        top_k_match_count=sum(1 for record in records if record.top_k_match is True),
        top_k_rows=sum(1 for record in records if record.top_k_match is not None),
        total_generated_tokens=sum(record.generated_tokens or 0 for record in records),
        total_latency_sec=sum(record.latency_sec or 0.0 for record in records),
    )
    metrics.latencies_sec.extend(
        record.latency_sec for record in records if record.latency_sec is not None
    )
    metrics.sequence_lengths.extend(
        record.sequence_length for record in records if record.sequence_length is not None
    )
    metrics.padding_ratios.extend(_padding_ratios(records))
    metrics.memory_peaks_mb.extend(
        record.memory_peak_mb for record in records if record.memory_peak_mb is not None
    )
    metrics.row_log_prob_per_token.extend(_log_prob_values(records))
    metrics.calibration_pairs.extend(
        (record.confidence, record.exact_match)
        for record in records
        if record.confidence is not None and record.exact_match is not None
    )
    return metrics


def normalize_prediction_row(row: Mapping[str, Any], index: int) -> PredictionRecord:
    """Normalize one raw prediction row into evaluator fields."""
    target = _target_json(row)
    rest_api = _rest_api(row, target)
    prediction_raw = _prediction_value(row)
    prediction_json, parse_ok, parse_error = parse_json_value(prediction_raw)
    exact_match = None
    if target is not None:
        exact_match = parse_ok and prediction_json == target
    odata_match = _odata_id_match(prediction_json, rest_api) if parse_ok else False
    if not rest_api:
        odata_match = None
    generated_tokens = _int_from_paths(row, _GENERATED_TOKEN_PATHS)
    sequence_length = _int_from_paths(row, _SEQUENCE_LENGTH_PATHS)
    padding_tokens = _int_from_paths(row, _PADDING_TOKEN_PATHS)
    return PredictionRecord(
        key=_row_key(row, index),
        line_number=index + 1,
        rest_api=rest_api,
        target_json=target,
        prediction_json=prediction_json,
        parse_ok=parse_ok,
        parse_error=parse_error,
        exact_match=exact_match,
        odata_id_match=odata_match,
        top_k_match=_top_k_match(row, target),
        sequence_length=sequence_length,
        generated_tokens=generated_tokens,
        padding_tokens=padding_tokens,
        latency_sec=_float_from_paths(row, _LATENCY_PATHS),
        memory_peak_mb=_float_from_paths(row, _MEMORY_PATHS),
        confidence=_confidence(row),
        log_prob=_float_from_paths(row, _LOG_PROB_PATHS),
        log_prob_per_token=_float_from_paths(row, _LOG_PROB_PER_TOKEN_PATHS),
    )


def compare_prediction_artifacts(
    baseline: ArtifactMetrics,
    model: ArtifactMetrics,
    *,
    ece_bins: int = 10,
) -> dict[str, Any]:
    """Compare baseline and ``model_x`` metrics."""
    baseline_metrics = baseline.to_metric_dict(ece_bins=ece_bins)
    model_metrics = model.to_metric_dict(ece_bins=ece_bins)
    delta: dict[str, float | None] = {}
    for key, model_value in model_metrics.items():
        baseline_value = baseline_metrics.get(key)
        if isinstance(model_value, (int, float)) and isinstance(baseline_value, (int, float)):
            delta[key] = model_value - baseline_value
        else:
            delta[key] = None
    return {
        "baseline_role": baseline.role,
        "model_role": model.role,
        "row_count_delta": model.row_count - baseline.row_count,
        "delta": delta,
    }


def build_phase1_golden_payload(
    *,
    baseline_jsonl: str | Path,
    model_jsonl: str | Path,
    ece_bins: int = 10,
) -> dict[str, Any]:
    """Compute baseline-vs-model_x metrics and compact evidence."""
    baseline = evaluate_prediction_jsonl(
        baseline_jsonl,
        role="baseline",
        ece_bins=ece_bins,
    )
    model = evaluate_prediction_jsonl(
        model_jsonl,
        role="model_x",
        ece_bins=ece_bins,
    )
    return {
        "schema_version": "phase1_golden_metrics.v1",
        "phase": 1,
        "task": "redfish_json_reconstruction",
        "metric_namespace": PHASE1_FINETUNE,
        "metrics": {
            "baseline": baseline.to_metric_dict(ece_bins=ece_bins),
            "model_x": model.to_metric_dict(ece_bins=ece_bins),
        },
        "comparison": compare_prediction_artifacts(
            baseline,
            model,
            ece_bins=ece_bins,
        ),
        "evidence": {
            "baseline": baseline.to_evidence_dict(ece_bins=ece_bins),
            "model_x": model.to_evidence_dict(ece_bins=ece_bins),
        },
    }


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Write a compact JSON payload."""
    with Path(path).open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")


def parse_json_value(value: Any) -> tuple[JsonValue | None, bool, str]:
    """Parse a prediction value as JSON, allowing a single fenced/object fragment."""
    if value is None:
        return None, False, "missing prediction"
    if isinstance(value, (dict, list)):
        return value, True, ""
    if not isinstance(value, str):
        return None, False, f"unsupported prediction type {type(value).__name__}"
    text = _strip_fence(value.strip())
    try:
        return json.loads(text), True, ""
    except json.JSONDecodeError:
        fragment = _first_json_fragment(text)
        if fragment is None:
            return None, False, "prediction is not JSON"
        try:
            return json.loads(fragment), True, ""
        except json.JSONDecodeError as err:
            return None, False, f"prediction JSON parse failed: {err.msg}"


def _strip_fence(text: str) -> str:
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if len(lines) >= 3 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return text


def _first_json_fragment(text: str) -> str | None:
    start = _first_json_start(text)
    if start is None:
        return None
    stack: list[str] = []
    in_string = False
    escape = False
    for index, char in enumerate(text[start:], start=start):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char in "{[":
            stack.append("}" if char == "{" else "]")
            continue
        if char in "}]":
            if not stack or char != stack[-1]:
                return None
            stack.pop()
            if not stack:
                return text[start : index + 1]
    return None


def _first_json_start(text: str) -> int | None:
    """Return the first JSON opener that is not inside surrounding quotes."""
    in_string = False
    escape = False
    for index, char in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char in "{[":
            return index
    return None


def _target_json(row: Mapping[str, Any]) -> JsonValue | None:
    value = _first_path(
        row,
        (
            ("y_true", "json"),
            ("target", "json"),
            ("expected", "json"),
            ("reference", "json"),
            ("y_true",),
            ("target",),
            ("expected",),
            ("reference",),
        ),
    )
    parsed, ok, _ = parse_json_value(value)
    return parsed if ok else None


def _prediction_value(row: Mapping[str, Any]) -> Any:
    return _first_path(
        row,
        (
            ("y_pred", "json"),
            ("y_pred", "redfish_json"),
            ("y_pred", "output_json"),
            ("y_pred", "prediction"),
            ("y_pred", "text"),
            ("y_pred", "raw"),
            ("prediction", "json"),
            ("prediction", "text"),
            ("prediction", "raw"),
            ("prediction",),
            ("predicted_json",),
            ("output_json",),
            ("generated_json",),
            ("generated_text",),
            ("completion",),
            ("output",),
            ("y_pred",),
        ),
    )


def _rest_api(row: Mapping[str, Any], target: JsonValue | None) -> str:
    value = _first_path(row, (("x", "rest_api"), ("rest_api",), ("uri",), ("url",)))
    if isinstance(value, str) and value:
        return value
    if isinstance(target, dict) and isinstance(target.get("@odata.id"), str):
        return target["@odata.id"]
    return ""


def _row_key(row: Mapping[str, Any], index: int) -> str:
    value = _first_path(
        row,
        (("id",), ("row_id",), ("sample_id",), ("example_id",), ("uid",)),
    )
    return str(value) if value is not None else str(index)


def _odata_id_match(prediction: JsonValue | None, rest_api: str) -> bool | None:
    if not rest_api:
        return None
    if not isinstance(prediction, dict):
        return False
    return prediction.get("@odata.id") == rest_api


def _top_k_match(row: Mapping[str, Any], target: JsonValue | None) -> bool | None:
    if target is None:
        return None
    candidates = _first_path(
        row,
        (
            ("y_pred_top_k",),
            ("top_k_predictions",),
            ("top_k",),
            ("candidates",),
            ("prediction", "top_k"),
            ("y_pred", "top_k"),
        ),
    )
    if candidates is None:
        return None
    if not isinstance(candidates, list):
        return False
    for candidate in candidates:
        candidate_json, ok, _ = parse_json_value(_unwrap_prediction_candidate(candidate))
        if ok and candidate_json == target:
            return True
    return False


def _unwrap_prediction_candidate(candidate: Any) -> Any:
    if not isinstance(candidate, dict):
        return candidate
    return _first_path(
        candidate,
        (
            ("json",),
            ("redfish_json",),
            ("prediction",),
            ("text",),
            ("raw",),
        ),
        default=candidate,
    )


def _padding_ratios(records: Iterable[PredictionRecord]) -> list[float]:
    ratios = []
    for record in records:
        if record.sequence_length and record.padding_tokens is not None:
            ratios.append(record.padding_tokens / record.sequence_length)
    return ratios


def _log_prob_values(records: Iterable[PredictionRecord]) -> list[tuple[float, int]]:
    values = []
    for record in records:
        weight = record.generated_tokens or record.sequence_length or 1
        if record.log_prob_per_token is not None:
            values.append((record.log_prob_per_token, weight))
        elif record.log_prob is not None and weight > 0:
            values.append((record.log_prob / weight, weight))
    return values


def _confidence(row: Mapping[str, Any]) -> float | None:
    value = _float_from_paths(
        row,
        (
            ("confidence",),
            ("probability",),
            ("mean_token_probability",),
            ("calibration", "confidence"),
            ("metrics", "confidence"),
            ("y_pred", "confidence"),
            ("prediction", "confidence"),
        ),
    )
    if value is None:
        return None
    return max(0.0, min(1.0, value))


def _first_path(
    row: Mapping[str, Any],
    paths: Iterable[tuple[str, ...]],
    *,
    default: Any = None,
) -> Any:
    for path in paths:
        current: Any = row
        found = True
        for key in path:
            if not isinstance(current, Mapping) or key not in current:
                found = False
                break
            current = current[key]
        if found:
            return current
    return default


def _float_from_paths(
    row: Mapping[str, Any],
    paths: Iterable[tuple[str, ...]],
) -> float | None:
    value = _first_path(row, paths)
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_from_paths(
    row: Mapping[str, Any],
    paths: Iterable[tuple[str, ...]],
) -> int | None:
    value = _float_from_paths(row, paths)
    if value is None:
        return None
    return int(value)


_GENERATED_TOKEN_PATHS = (
    ("generated_tokens",),
    ("completion_tokens",),
    ("output_tokens",),
    ("tokens", "generated"),
    ("usage", "completion_tokens"),
    ("metrics", "generated_tokens"),
)

_SEQUENCE_LENGTH_PATHS = (
    ("sequence_length",),
    ("seq_len",),
    ("tokens", "sequence_length"),
    ("usage", "total_tokens"),
    ("metrics", "sequence_length"),
)

_PADDING_TOKEN_PATHS = (
    ("padding_tokens",),
    ("tokens", "padding"),
    ("metrics", "padding_tokens"),
)

_LATENCY_PATHS = (
    ("latency_sec",),
    ("elapsed_sec",),
    ("duration_sec",),
    ("test_time_sec",),
    ("metrics", "latency_sec"),
    ("timing", "latency_sec"),
)

_MEMORY_PATHS = (
    ("memory_peak_mb",),
    ("peak_memory_mb",),
    ("metrics", "memory_peak_mb"),
)

_LOG_PROB_PATHS = (
    ("log_prob",),
    ("logprob",),
    ("sequence_log_prob",),
    ("completion_log_prob",),
    ("metrics", "log_prob"),
)

_LOG_PROB_PER_TOKEN_PATHS = (
    ("log_prob_per_token",),
    ("mean_log_prob",),
    ("metrics", "log_prob_per_token"),
)


# Author: Mus mbayramo@stanford.edu
