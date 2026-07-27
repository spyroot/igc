"""Focused tests for paired Phase 1 instruction-retention metrics."""

from __future__ import annotations

import json

import pytest

from igc.modules.train.instruction_retention import (
    InstructionRetentionError,
    compare_retention_artifacts,
    load_retention_jsonl,
)


def _row(row_id: str, k: int, *, accepted: bool = True) -> dict:
    apis = [f"/redfish/v1/fixture/{index}" for index in range(k)]
    return {
        "row_id": row_id,
        "k": k,
        "expected_rest_api_list": apis,
        "text": "perform the requested operations",
        "judge": {
            "natural_command": True,
            "accepted": accepted,
            "rest_api_list": apis,
            "nonsense": False,
        },
    }


def _write(path, rows) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_retention_compares_identical_k_1_2_3_rows(tmp_path) -> None:
    rows = [_row("one", 1), _row("two", 2), _row("three", 3)]
    foundation_path = tmp_path / "foundation.jsonl"
    model_path = tmp_path / "model.jsonl"
    _write(foundation_path, rows)
    _write(model_path, rows)
    comparison = compare_retention_artifacts(
        load_retention_jsonl(foundation_path),
        load_retention_jsonl(model_path),
    )
    assert comparison["model_x"]["judge_acceptance_rate"] == 1.0
    assert comparison["delta"]["judge_acceptance_rate"] == 0.0
    assert set(comparison["model_x"]["by_k"]) == {"1", "2", "3"}


def test_retention_rejects_changed_fixed_combination(tmp_path) -> None:
    foundation_path = tmp_path / "foundation.jsonl"
    model_path = tmp_path / "model.jsonl"
    _write(foundation_path, [_row("one", 1), _row("two", 2), _row("three", 3)])
    changed = [_row("one", 1), _row("two", 2), _row("three", 3)]
    changed[0]["expected_rest_api_list"] = ["/redfish/v1/other"]
    changed[0]["judge"]["rest_api_list"] = ["/redfish/v1/other"]
    _write(model_path, changed)
    with pytest.raises(InstructionRetentionError, match="changed"):
        compare_retention_artifacts(
            load_retention_jsonl(foundation_path),
            load_retention_jsonl(model_path),
        )
