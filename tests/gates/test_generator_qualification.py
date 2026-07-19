"""Offline tests for the generator-qualification gate.

Prove the committed baseline qualifies, the rates compute correctly, and a junk or
duplicate-heavy generator is caught by the thresholds.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

from pathlib import Path

from scripts.gates.generator_qualification import load_config, run

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = REPO_ROOT / "configs/gates/generator_qualification.yaml"


def test_baseline_config_qualifies() -> None:
    """The committed baseline sample stays inside every threshold."""
    config = load_config(CONFIG)
    report = run(config["records"], config["thresholds"])
    assert report["qualified"], report["violations"]
    assert report["rates"]["valid_text_rate"] == 1.0
    assert 0.5 <= report["rates"]["judge_acceptance_rate"] <= 1.0


def test_junk_generator_fails_valid_text() -> None:
    """A generator emitting only nonsense drops valid_text_rate below the floor."""
    junk = [
        {"text": "zzz", "allowed_methods": {}, "judge_response": '{"accepted": false, "rest_api_list": [], "nonsense": true}'}
        for _ in range(4)
    ]
    report = run(junk, load_config(CONFIG)["thresholds"])
    assert report["rates"]["valid_text_rate"] == 0.0
    assert not report["qualified"]


def test_duplicate_heavy_generator_trips_threshold() -> None:
    """Too many duplicate-intent drafts exceed max_duplicate_intent_rate."""
    dup = [
        {
            "text": "read system 1 twice",
            "allowed_methods": {"/redfish/v1/Systems/1": ["GET"]},
            "judge_response": (
                '{"accepted": false, "natural": true, "nonsense": false, '
                '"ambiguous": false, "duplicate_intent": true, '
                '"method_semantics_valid": true, "coverage": '
                '[{"rest_api": "/redfish/v1/Systems/1", "text_span": "read system 1", "supported": true}, '
                '{"rest_api": "/redfish/v1/Systems/1", "text_span": "twice", "supported": true}], '
                '"extra_intents": [], "reason": "duplicate"}'
            ),
        }
        for _ in range(4)
    ]
    report = run(dup, load_config(CONFIG)["thresholds"])
    assert report["rates"]["duplicate_intent_rate"] == 1.0
    assert not report["qualified"]


def test_unsupported_intent_detected() -> None:
    """An extracted API outside the allowed context counts as unsupported."""
    rec = [{
        "text": "reset manager 1",
        "allowed_methods": {"/redfish/v1/Systems/1": ["GET"]},
        "judge_response": (
            '{"accepted": false, "natural": true, "nonsense": false, '
            '"ambiguous": false, "duplicate_intent": false, '
            '"method_semantics_valid": false, "coverage": '
            '[{"rest_api": "/redfish/v1/Managers/1", "text_span": "reset manager 1", "supported": true}], '
            '"extra_intents": [], "reason": "outside allowed context"}'
        ),
    }]
    report = run(rec, load_config(CONFIG)["thresholds"])
    assert report["rates"]["unsupported_intent_rate"] == 1.0


# Author: Mus mbayramo@stanford.edu
