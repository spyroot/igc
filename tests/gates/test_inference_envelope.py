"""Gate tests: the Phase 2/3 inference envelope stays contract-shaped.

Two layers, matching requirement 4:

* An OFFLINE, deterministic layer. A fixed fixture "model response" (the exact
  canonical target a perfect model would emit) is fed through the ONE canonical
  parse/eval path in ``igc/ds/rest_goal_contract.py`` and must EXACT-MATCH the
  expected ``rest_api_list`` (Phase 2) and unordered ``calls`` (Phase 3). A wrong
  fixture response is also checked so a green result cannot be a vacuous pass.
  This layer imports no torch/transformers, needs no GPU, and is collected by the
  default offline gate (``pytest -q`` over ``tests/``).

* A real-model/vLLM layer, marked ``@pytest.mark.gpu`` and therefore excluded from
  the offline gate by ``pytest.ini`` ``addopts``. It is additionally guarded so it
  SKIPS unless a GPU and an approved local checkpoint directory are present. It
  checks only the ENVELOPE — model loadability, that the request/response keys are
  present, that the completion JSON-parses, and that the parsed output has the
  right SHAPE — never stochastic answer quality.

  BLOCKED (2026-07): the GB300/NV72 surface is powered off, so the ``gpu`` layer
  cannot run anywhere right now. It is authored as a real, clearly-marked skipped
  stage, not a fake pass. When GB300 is powered on, run it inside an approved
  remote container with ``IGC_ENVELOPE_MODEL_DIR`` pointing at a Phase 2/3
  checkpoint on BeeGFS ``/models``; never on the operator laptop.

Used by ``scripts/gates/contract_authority.py`` (the contract-authority gate runner
imports/collects this file) and by ``.github/workflows/ci.yml`` via the default
offline pytest step.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

import json
import os

import pytest

from igc.ds.rest_goal_contract import (
    RedfishContext,
    build_d1_rest_api_list_row,
    build_call_row,
    evaluate_calls_y_pred,
    parse_calls_y_pred,
    parse_rest_api_list_y_pred,
    render_call_example,
    render_rest_api_list_example,
)

# Fixed, deterministic fixture shared by the offline envelope checks. Values are
# arbitrary but stable so the expected parse/eval result is fully determined.
_FIXTURE_REST_API = "/redfish/v1/Systems/1"  # One concrete legal Redfish target.
_FIXTURE_TEXT = "power on system 1"          # Operator sentence tied to the target.


def _phase2_row() -> dict:
    """Build the canonical Phase 2 row from the fixed fixture."""
    contexts = [
        RedfishContext(
            rest_api=_FIXTURE_REST_API,
            allowed_methods=["GET", "PATCH"],
            json={"@odata.id": _FIXTURE_REST_API, "PowerState": "Off"},
        )
    ]
    return build_d1_rest_api_list_row(
        text=_FIXTURE_TEXT,
        contexts=contexts,
        rest_api_list=[_FIXTURE_REST_API],
    )


def _phase3_row() -> dict:
    """Build the canonical Phase 3 row from the fixed fixture."""
    contexts = [
        RedfishContext(
            rest_api=_FIXTURE_REST_API,
            allowed_methods=["GET", "PATCH"],
            json={"@odata.id": _FIXTURE_REST_API, "PowerState": "Off"},
        )
    ]
    return build_call_row(
        text=_FIXTURE_TEXT,
        contexts=contexts,
        rest_api_list=[_FIXTURE_REST_API],
        method_by_api={_FIXTURE_REST_API: "PATCH"},
        arguments_by_api={_FIXTURE_REST_API: {"PowerState": "On"}},
    )


# --------------------------------------------------------------------------- #
# Offline deterministic layer (default gate; no torch, no GPU).
# --------------------------------------------------------------------------- #
def test_phase2_perfect_response_exact_matches_rest_api_list() -> None:
    """The canonical target parses back to the expected ordered rest_api_list."""
    row = _phase2_row()
    rendered = render_rest_api_list_example(row)
    # A perfect model emits exactly the canonical target JSON completion.
    parsed = parse_rest_api_list_y_pred(rendered.target_json)
    assert parsed == list(row["y_true"]["rest_api_list"])


def test_phase2_wrapped_y_pred_envelope_parses_identically() -> None:
    """The ``{"y_pred": {...}}`` transport envelope parses to the same list."""
    row = _phase2_row()
    expected = list(row["y_true"]["rest_api_list"])
    wrapped = json.dumps({"y_pred": {"rest_api_list": expected}})
    assert parse_rest_api_list_y_pred(wrapped) == expected


def test_phase2_wrong_response_does_not_match() -> None:
    """A wrong Phase 2 response must NOT exact-match — the check is not vacuous."""
    row = _phase2_row()
    wrong = json.dumps({"rest_api_list": ["/redfish/v1/Chassis/9"]})
    assert parse_rest_api_list_y_pred(wrong) != list(row["y_true"]["rest_api_list"])


def test_phase2_malformed_response_raises() -> None:
    """A non-JSON / mis-shaped Phase 2 response is rejected, never coerced."""
    with pytest.raises((ValueError, json.JSONDecodeError)):
        parse_rest_api_list_y_pred("not json at all")
    with pytest.raises(ValueError):
        parse_rest_api_list_y_pred(json.dumps({"rest_api_list": "not-a-list"}))


def test_phase3_perfect_response_exact_matches_calls() -> None:
    """The canonical Phase 3 target evaluates to an exact call-set match."""
    row = _phase3_row()
    rendered = render_call_example(row)
    result = evaluate_calls_y_pred(row, rendered.target_json)
    assert result["parsed"] is True
    assert result["parse_error"] == ""
    assert result["call_set_exact_match"] is True
    assert result["call_set_exact_match_rate"] == 1.0
    assert result["method_exact_match_rate"] == 1.0
    assert result["arguments_exact_match_rate"] == 1.0
    # The parsed calls round-trip to the row's y_true calls exactly.
    assert parse_calls_y_pred(rendered.target_json) == list(row["y_true"]["calls"])


def test_phase3_wrong_method_is_not_exact_match() -> None:
    """A Phase 3 response with the wrong method must not report an exact match."""
    row = _phase3_row()
    # Legal method (GET is in allowed_methods) but not the labelled PATCH call.
    wrong = json.dumps(
        {
            "calls": [
                {
                    "rest_api": _FIXTURE_REST_API,
                    "http_method": "GET",
                    "operation_name": None,
                    "arguments": {},
                }
            ]
        }
    )
    result = evaluate_calls_y_pred(row, wrong)
    assert result["parsed"] is True
    assert result["call_set_exact_match"] is False
    assert result["method_exact_match_rate"] == 0.0


def test_phase3_invalid_json_reports_parse_failure() -> None:
    """An unparseable Phase 3 response is a recorded failure, not an exception."""
    result = evaluate_calls_y_pred(_phase3_row(), "{ broken json")
    assert result["parsed"] is False
    assert result["call_set_exact_match"] is False
    assert result["parse_error"]


# --------------------------------------------------------------------------- #
# Real-model / vLLM envelope layer (opt-in; GPU; BLOCKED while GB300 is off).
# --------------------------------------------------------------------------- #
def _extract_json_object(text: str) -> dict:
    """Extract the last top-level JSON object from a model completion.

    Envelope helper only: locates a ``{...}`` span and parses it. Used by the
    GPU stage to prove the completion is JSON-shaped without judging its content.

    :param text: raw decoded model completion.
    :return: the parsed JSON object.
    :raises ValueError: if no JSON object can be located/parsed.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no JSON object found in completion")
    return json.loads(text[start : end + 1])


@pytest.mark.gpu
def test_phase2_model_response_envelope_only() -> None:
    """Load a real Phase 2 checkpoint and check ONLY the response envelope.

    BLOCKED while GB300/NV72 is powered off — this test skips unless a GPU AND an
    approved checkpoint directory (``IGC_ENVELOPE_MODEL_DIR``) are both present, so
    it never fabricates a pass. It asserts the model loads, the completion
    JSON-parses, the ``rest_api_list`` key is present, and the parsed value has the
    right SHAPE (a list of strings). It never asserts which APIs were produced.

    :raises AssertionError: only on an envelope breach (unparseable output or a
        missing/mis-typed ``rest_api_list`` key), never on answer quality.
    """
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires a local CUDA-capable GPU (GB300 powered off)")
    model_dir = os.environ.get("IGC_ENVELOPE_MODEL_DIR")
    if not model_dir or not os.path.isdir(model_dir):
        pytest.skip(
            "set IGC_ENVELOPE_MODEL_DIR to an approved Phase 2 checkpoint on /models "
            "(BLOCKED while GB300 is off)"
        )
    transformers = pytest.importorskip("transformers")

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_dir)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    model.eval()

    row = _phase2_row()
    prompt = render_rest_api_list_example(row).prompt
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,  # deterministic decode; envelope, not sampling quality
        )
    completion = tokenizer.decode(
        generated[0][inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
    )

    # Envelope: the completion must be JSON-shaped and carry the contract key.
    obj = _extract_json_object(completion)
    assert "rest_api_list" in obj, obj
    parsed = parse_rest_api_list_y_pred(obj)
    assert isinstance(parsed, list)
    assert all(isinstance(item, str) for item in parsed)


# Author: Mus mbayramo@stanford.edu
