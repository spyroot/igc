"""Deterministic contract tests for lossless Phase 1 JSON chunking."""

from __future__ import annotations

import hashlib

import torch

from igc.ds.phase1_chunking import (
    Phase1ChunkingPolicy,
    chunk_phase1_row,
    phase1_row_token_count,
    phase1_token_distribution,
    reassemble_phase1_rows,
)
from igc.ds.phase1_render import build_phase1_row, validate_phase1_row


class _CharacterTokenizer:
    """Minimal tokenizer whose token count equals Unicode code-point count."""

    pad_token = "<pad>"
    eos_token = "<eos>"
    pad_token_id = 0

    def __init__(self) -> None:
        self.characters_seen = 0

    def __call__(
        self,
        text,
        *,
        padding=False,
        truncation=False,
        return_tensors="pt",
        add_special_tokens=False,
    ):
        del padding, truncation, return_tensors, add_special_tokens
        self.characters_seen += len(text)
        ids = torch.arange(1, len(text) + 1, dtype=torch.long).unsqueeze(0)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def _sha(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _row(body: dict, *, rest_api: str = "/redfish/v1/Systems/1") -> dict:
    return build_phase1_row(
        rest_api=rest_api,
        allowed_methods=["GET", "PATCH"],
        input_json=body,
        target_json=body,
        metadata={
            "row_id": _sha(f"source\0{rest_api}"),
            "source_corpus": "real_vendor:test:systems",
            "trust_level": "REAL",
            "vendor": "test",
        },
    )


def _policy(max_tokens: int = 320) -> Phase1ChunkingPolicy:
    return Phase1ChunkingPolicy(
        transform="phase1.lossless-json-chunk.v1",
        tokenizer_sha=_sha("tokenizer"),
        max_tokens=max_tokens,
        telemetry_rest_api_markers=("/TelemetryService", "/Sensor"),
    )


def test_chunking_preserves_nested_objects_arrays_and_long_strings() -> None:
    """Every original value survives object, array, and string partitioning."""

    body = {
        "@odata.id": "/redfish/v1/Systems/1",
        "Name": "system-" + "x" * 280,
        "Nested": {
            "Enabled": True,
            "Items": [
                {"Id": str(index), "Payload": chr(65 + index) * 120}
                for index in range(4)
            ],
        },
    }
    tokenizer = _CharacterTokenizer()

    chunks = chunk_phase1_row(_row(body), tokenizer=tokenizer, policy=_policy())

    assert len(chunks) > 3
    assert reassemble_phase1_rows(list(reversed(chunks))) == body
    assert all(
        phase1_row_token_count(chunk, tokenizer=tokenizer) <= _policy().max_tokens
        for chunk in chunks
    )
    assert len({chunk["metadata"]["row_id"] for chunk in chunks}) == len(chunks)
    for index, chunk in enumerate(chunks):
        validate_phase1_row(chunk)
        metadata = chunk["metadata"]["chunk"]
        assert metadata["index"] == index
        assert metadata["count"] == len(chunks)
        assert metadata["original_row_id"] == _row(body)["metadata"]["row_id"]


def test_fitting_resource_still_gets_transform_and_reassembly_lineage() -> None:
    """One-row resources carry the same immutable transform identity as split rows."""

    body = {"@odata.id": "/redfish/v1/Systems/1", "PowerState": "On"}
    tokenizer = _CharacterTokenizer()

    chunks = chunk_phase1_row(
        _row(body),
        tokenizer=tokenizer,
        policy=_policy(max_tokens=2048),
    )

    assert len(chunks) == 1
    assert chunks[0]["metadata"]["chunk"]["kind"] == "whole_document"
    assert reassemble_phase1_rows(chunks) == body


def test_array_reassembly_combines_oversized_element_and_later_slice() -> None:
    """A recursively split item and a compact tail share one original array length."""

    body = {
        "Members": [
            {"Payload": "x" * 400},
            {"Id": "small-1"},
            {"Id": "small-2"},
        ]
    }
    tokenizer = _CharacterTokenizer()

    chunks = chunk_phase1_row(_row(body), tokenizer=tokenizer, policy=_policy())

    assert any(
        chunk["metadata"]["chunk"]["kind"] == "array_slice" for chunk in chunks
    )
    assert reassemble_phase1_rows(chunks) == body


def test_token_distribution_reports_telemetry_without_filtering() -> None:
    """Telemetry classification changes only the report, never row membership."""

    tokenizer = _CharacterTokenizer()
    rows = [
        _row({"Value": "x" * 10}),
        _row(
            {"Value": "y" * 200},
            rest_api="/redfish/v1/TelemetryService/MetricReports/1",
        ),
    ]

    report = phase1_token_distribution(
        rows,
        tokenizer=tokenizer,
        telemetry_rest_api_markers=_policy().telemetry_rest_api_markers,
    )

    assert report["all_resources"]["count"] == 2
    assert report["non_telemetry_resources"]["count"] == 1
    assert report["telemetry_resources"]["count"] == 1
    assert report["all_resources"]["mean"] is not None


def test_long_string_partitioning_does_not_retokenize_the_whole_remainder() -> None:
    """Chunk search work stays bounded by local windows for giant scalar values."""

    tokenizer = _CharacterTokenizer()
    body = {"Payload": "x" * 3000}

    chunks = chunk_phase1_row(_row(body), tokenizer=tokenizer, policy=_policy())

    assert len(chunks) > 20
    assert reassemble_phase1_rows(chunks) == body
    assert tokenizer.characters_seen < 500_000
