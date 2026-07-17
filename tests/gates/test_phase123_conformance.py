"""Tests for the deterministic Phase 1 -> D1 -> Phase 2 -> Phase 3 gate."""
from __future__ import annotations

from pathlib import Path

from scripts.gates.phase123_conformance import assert_report, build_report


def test_phase123_conformance_report_has_expected_keys_and_shapes(tmp_path: Path) -> None:
    """The conformance gate walks the full deterministic contract ladder."""

    report = build_report(tmp_path)
    assert_report(report)

    assert report["phase1"]["tensor_keys"] == ["attention_mask", "input_ids", "labels"]
    assert report["d1"]["judge"]["rest_api_set_match"] is True
    assert report["phase2"]["parsed_rest_api_list"] == ["/redfish/v1/Systems/1"]
    assert report["phase3"]["parsed_call_count"] == 1
