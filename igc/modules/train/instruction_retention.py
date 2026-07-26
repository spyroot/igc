"""Paired Phase 1 instruction-retention metrics for future D1 generation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


class InstructionRetentionError(ValueError):
    """Raised when paired foundation/model_x judge artifacts are not comparable."""


@dataclass(frozen=True)
class RetentionRow:
    """One fixed API-combination prompt and its external judge decision."""

    row_id: str
    k: int
    expected: frozenset[str]
    judged: frozenset[str]
    natural_command: bool
    accepted: bool
    nonsense: bool


def load_retention_jsonl(path: str | Path) -> list[RetentionRow]:
    """Load strict judged output rows for one model role."""
    rows: list[RetentionRow] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise InstructionRetentionError(
                    f"line {line_number}: invalid JSON"
                ) from exc
            rows.append(_parse_row(raw, line_number))
    if not rows:
        raise InstructionRetentionError("instruction-retention artifact has no rows")
    ids = [row.row_id for row in rows]
    if len(ids) != len(set(ids)):
        raise InstructionRetentionError("instruction-retention row_id values must be unique")
    if {row.k for row in rows} != {1, 2, 3}:
        raise InstructionRetentionError("instruction-retention artifact must cover k=1,2,3")
    return rows


def retention_metrics(rows: Sequence[RetentionRow]) -> dict[str, Any]:
    """Compute aggregate and per-k instruction-retention rates."""
    return {
        "rows": len(rows),
        **_rates(rows),
        "by_k": {
            str(k): {"rows": sum(row.k == k for row in rows), **_rates([
                row for row in rows if row.k == k
            ])}
            for k in (1, 2, 3)
        },
    }


def compare_retention_artifacts(
    foundation: Sequence[RetentionRow],
    model_x: Sequence[RetentionRow],
) -> dict[str, Any]:
    """Compare model_x with foundation on exactly the same fixed combinations."""
    foundation_by_id = {row.row_id: row for row in foundation}
    model_by_id = {row.row_id: row for row in model_x}
    if set(foundation_by_id) != set(model_by_id):
        raise InstructionRetentionError(
            "foundation and model_x retention artifacts must cover identical row IDs"
        )
    for row_id, foundation_row in foundation_by_id.items():
        model_row = model_by_id[row_id]
        if (
            foundation_row.k != model_row.k
            or foundation_row.expected != model_row.expected
        ):
            raise InstructionRetentionError(
                f"row {row_id!r} changed its fixed API combination"
            )
    foundation_metrics = retention_metrics(foundation)
    model_metrics = retention_metrics(model_x)
    metric_names = (
        "natural_command_rate",
        "judge_acceptance_rate",
        "missing_intent_rate",
        "extra_intent_rate",
        "nonsense_rate",
    )
    return {
        "foundation": foundation_metrics,
        "model_x": model_metrics,
        "delta": {
            name: model_metrics[name] - foundation_metrics[name]
            for name in metric_names
        },
    }


def _parse_row(raw: Any, line_number: int) -> RetentionRow:
    if not isinstance(raw, Mapping) or set(raw) != {
        "row_id",
        "k",
        "expected_rest_api_list",
        "text",
        "judge",
    }:
        raise InstructionRetentionError(f"line {line_number}: invalid row fields")
    if not isinstance(raw["text"], str) or not raw["text"].strip():
        raise InstructionRetentionError(f"line {line_number}: text must be non-empty")
    expected = _api_set(raw["expected_rest_api_list"], line_number, "expected")
    k = raw["k"]
    if not isinstance(k, int) or k not in (1, 2, 3) or len(expected) != k:
        raise InstructionRetentionError(f"line {line_number}: k must match 1-3 intents")
    judge = raw["judge"]
    if not isinstance(judge, Mapping) or set(judge) != {
        "natural_command",
        "accepted",
        "rest_api_list",
        "nonsense",
    }:
        raise InstructionRetentionError(f"line {line_number}: invalid judge fields")
    for key in ("natural_command", "accepted", "nonsense"):
        if not isinstance(judge[key], bool):
            raise InstructionRetentionError(f"line {line_number}: judge.{key} must be bool")
    row_id = raw["row_id"]
    if not isinstance(row_id, str) or not row_id:
        raise InstructionRetentionError(f"line {line_number}: row_id must be non-empty")
    return RetentionRow(
        row_id=row_id,
        k=k,
        expected=expected,
        judged=_api_set(judge["rest_api_list"], line_number, "judge"),
        natural_command=judge["natural_command"],
        accepted=judge["accepted"],
        nonsense=judge["nonsense"],
    )


def _api_set(value: Any, line_number: int, label: str) -> frozenset[str]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise InstructionRetentionError(f"line {line_number}: {label} APIs must be list[str]")
    if len(value) != len(set(value)):
        raise InstructionRetentionError(f"line {line_number}: {label} APIs must be unique")
    return frozenset(value)


def _rates(rows: Sequence[RetentionRow]) -> dict[str, float]:
    if not rows:
        raise InstructionRetentionError("cannot score an empty retention group")
    expected_total = sum(len(row.expected) for row in rows)
    judged_total = sum(len(row.judged) for row in rows)
    missing = sum(len(row.expected - row.judged) for row in rows)
    extra = sum(len(row.judged - row.expected) for row in rows)
    return {
        "natural_command_rate": sum(row.natural_command for row in rows) / len(rows),
        "judge_acceptance_rate": sum(row.accepted for row in rows) / len(rows),
        "missing_intent_rate": missing / expected_total if expected_total else 0.0,
        "extra_intent_rate": extra / judged_total if judged_total else (1.0 if extra else 0.0),
        "nonsense_rate": sum(row.nonsense for row in rows) / len(rows),
    }
