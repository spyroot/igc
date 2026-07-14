"""Offline tests for the ordered REST-goal dataset contracts.

These tests pin the Phase 2/3 mock-plumbing rows without training a model,
running W&B, touching captured corpora, or inventing a regex extractor.

Author:
Mus mbayramo@stanford.edu
"""

import json
from pathlib import Path

import pytest

from igc.ds.rest_goal_contract import (
    D0,
    D1,
    MODEL_X,
    PHASE2_GOAL_EXTRACT_METRIC_KEYS,
    PHASE3_ARGUMENT_EXTRACT_METRIC_KEYS,
    RedfishContext,
    build_d1_rest_api_list_row,
    build_ordered_call_row,
    inference_ordered_goals_json,
    parse_ordered_calls_y_pred,
    parse_rest_api_list_y_pred,
    render_ordered_call_example,
    render_rest_api_list_example,
)


def _context(rest_api: str, allowed_methods: tuple[str, ...], body: dict) -> RedfishContext:
    """Build one tiny Redfish context row for contract tests."""
    return RedfishContext(rest_api=rest_api, allowed_methods=allowed_methods, json=body)


def test_d1_row_preserves_operator_order_independent_of_context_order() -> None:
    """The label order follows the operator-stated order, not JSON context order."""
    systems = _context(
        "/redfish/v1/Systems",
        ("GET", "HEAD"),
        {"@odata.id": "/redfish/v1/Systems", "Name": "Systems"},
    )
    tasks = _context(
        "/redfish/v1/TaskService/Tasks",
        ("GET", "HEAD"),
        {"@odata.id": "/redfish/v1/TaskService/Tasks", "Name": "Tasks"},
    )

    row = build_d1_rest_api_list_row(
        text="check the task queue, then list the systems",
        contexts=(systems, tasks),
        rest_api_list=("/redfish/v1/TaskService/Tasks", "/redfish/v1/Systems"),
    )

    assert row["phase"] == 2
    assert row["dataset"] == D1
    assert row["source_dataset"] == D0
    assert row["model_x"] == MODEL_X
    assert row["x"]["json"] == [systems.json, tasks.json]
    assert row["y_true"]["rest_api_list"] == [
        "/redfish/v1/TaskService/Tasks",
        "/redfish/v1/Systems",
    ]
    assert row["x"]["allowed_methods"] == {
        "/redfish/v1/Systems": ["GET", "HEAD"],
        "/redfish/v1/TaskService/Tasks": ["GET", "HEAD"],
    }
    assert row["validation"] == {
        "text_source": "mock_fixture",
        "review_judged": False,
        "all_rest_api_present": True,
        "extra_rest_api_present": False,
        "order_preserved": True,
    }


def test_phase23_rows_pin_locked_field_names() -> None:
    """Phase 2/3 rows expose only the locked contract field names."""
    context = _context(
        "/redfish/v1/Systems",
        ("GET", "HEAD"),
        {"@odata.id": "/redfish/v1/Systems", "Name": "Systems"},
    )

    assert set(context.to_dict()) == {"rest_api", "allowed_methods", "json"}

    phase2 = build_d1_rest_api_list_row(
        text="list systems",
        contexts=(context,),
        rest_api_list=("/redfish/v1/Systems",),
    )
    phase3 = build_ordered_call_row(
        text="list systems",
        contexts=(context,),
        rest_api_list=("/redfish/v1/Systems",),
    )

    assert set(phase2) == {
        "phase",
        "dataset",
        "source_dataset",
        "model_x",
        "task",
        "x",
        "y_true",
        "validation",
    }
    assert set(phase2["x"]) == {"text", "json", "allowed_methods"}
    assert set(phase2["y_true"]) == {"rest_api_list", "order_evidence"}
    assert set(phase3) == {"phase", "source_dataset", "model_x", "task", "x", "y_true"}
    assert set(phase3["x"]) == {"text", "rest_api_list", "json", "allowed_methods"}
    assert set(phase3["y_true"]) == {"calls"}
    assert set(phase3["y_true"]["calls"][0]) == {
        "rest_api",
        "allowed_methods",
        "method",
        "arguments",
    }


def test_phase3_get_calls_preserve_order_and_keep_arguments_empty() -> None:
    """Read-only GET calls keep ordered rest_api rows and never copy scalar JSON values."""
    row = build_ordered_call_row(
        text="check task state, then inspect system power",
        contexts=(
            _context(
                "/redfish/v1/Systems/1",
                ("GET", "HEAD"),
                {
                    "@odata.id": "/redfish/v1/Systems/1",
                    "PowerState": "On",
                },
            ),
            _context(
                "/redfish/v1/TaskService/Tasks",
                ("GET", "HEAD"),
                {
                    "@odata.id": "/redfish/v1/TaskService/Tasks",
                    "Members@odata.count": 0,
                },
            ),
        ),
        rest_api_list=("/redfish/v1/TaskService/Tasks", "/redfish/v1/Systems/1"),
    )

    assert row["phase"] == 3
    assert row["x"]["rest_api_list"] == [
        "/redfish/v1/TaskService/Tasks",
        "/redfish/v1/Systems/1",
    ]
    assert row["y_true"]["calls"] == [
        {
            "rest_api": "/redfish/v1/TaskService/Tasks",
            "allowed_methods": ["GET", "HEAD"],
            "method": "GET",
            "arguments": {},
        },
        {
            "rest_api": "/redfish/v1/Systems/1",
            "allowed_methods": ["GET", "HEAD"],
            "method": "GET",
            "arguments": {},
        },
    ]


def test_phase3_mutation_arguments_must_be_supplied_explicitly() -> None:
    """PATCH rows do not infer arguments from arbitrary scalar values in GET JSON."""
    settings = _context(
        "/redfish/v1/Systems/1/Bios/Settings",
        ("GET", "PATCH"),
        {
            "@odata.id": "/redfish/v1/Systems/1/Bios/Settings",
            "Attributes": {"BootMode": "Uefi"},
        },
    )

    without_arguments = build_ordered_call_row(
        text="set bios boot mode",
        contexts=(settings,),
        rest_api_list=("/redfish/v1/Systems/1/Bios/Settings",),
        method_by_api={"/redfish/v1/Systems/1/Bios/Settings": "PATCH"},
    )
    with_arguments = build_ordered_call_row(
        text="set bios boot mode to Uefi",
        contexts=(settings,),
        rest_api_list=("/redfish/v1/Systems/1/Bios/Settings",),
        method_by_api={"/redfish/v1/Systems/1/Bios/Settings": "PATCH"},
        arguments_by_api={
            "/redfish/v1/Systems/1/Bios/Settings": {
                "Attributes": {"BootMode": "Uefi"},
            },
        },
    )

    assert without_arguments["y_true"]["calls"][0]["arguments"] == {}
    assert with_arguments["y_true"]["calls"][0]["arguments"] == {
        "Attributes": {"BootMode": "Uefi"},
    }


def test_rows_reject_missing_and_duplicate_contexts() -> None:
    """Rows fail fast when labels cannot be resolved to one current context."""
    context = _context(
        "/redfish/v1/Systems",
        ("GET", "HEAD"),
        {"@odata.id": "/redfish/v1/Systems"},
    )

    with pytest.raises(ValueError, match="not present"):
        build_d1_rest_api_list_row(
            text="list chassis",
            contexts=(context,),
            rest_api_list=("/redfish/v1/Chassis",),
        )
    with pytest.raises(ValueError, match="duplicate rest_api"):
        build_ordered_call_row(
            text="list systems twice",
            contexts=(context, context),
            rest_api_list=("/redfish/v1/Systems",),
        )


def test_phase3_rejects_methods_outside_allowed_methods() -> None:
    """The selected method must be present in allowed_methods, including empty sets."""
    read_only = _context(
        "/redfish/v1/Systems",
        ("GET", "HEAD"),
        {"@odata.id": "/redfish/v1/Systems"},
    )
    no_methods = _context(
        "/redfish/v1/Managers",
        (),
        {"@odata.id": "/redfish/v1/Managers"},
    )

    with pytest.raises(ValueError, match="not in allowed_methods"):
        build_ordered_call_row(
            text="delete systems",
            contexts=(read_only,),
            rest_api_list=("/redfish/v1/Systems",),
            method_by_api={"/redfish/v1/Systems": "DELETE"},
        )
    with pytest.raises(ValueError, match="not in allowed_methods"):
        build_ordered_call_row(
            text="list managers",
            contexts=(no_methods,),
            rest_api_list=("/redfish/v1/Managers",),
        )


def test_rendered_examples_have_prompt_target_boundary_and_canonical_json() -> None:
    """Rendered rows separate x prompt from the y_true JSON completion."""
    context = _context(
        "/redfish/v1/Systems",
        ("GET", "HEAD"),
        {"@odata.id": "/redfish/v1/Systems"},
    )
    phase2 = build_d1_rest_api_list_row(
        text="list systems",
        contexts=(context,),
        rest_api_list=("/redfish/v1/Systems",),
    )
    phase3 = build_ordered_call_row(
        text="list systems",
        contexts=(context,),
        rest_api_list=("/redfish/v1/Systems",),
    )

    rendered2 = render_rest_api_list_example(phase2)
    rendered3 = render_ordered_call_example(phase3)

    assert rendered2.target_char_start == len(rendered2.prompt)
    assert json.loads(rendered2.target_json) == {"rest_api_list": ["/redfish/v1/Systems"]}
    assert "### Ordered REST API List" in rendered2.prompt
    assert rendered2.full_text == rendered2.prompt + rendered2.target_json
    assert json.loads(rendered3.target_json) == {
        "calls": phase3["y_true"]["calls"],
    }
    assert "### Ordered REST Calls" in rendered3.prompt


def test_inference_json_uses_ordered_goals_shape() -> None:
    """The combined inference handoff uses ordered_goals with Phase 3 call fields."""
    context = _context(
        "/redfish/v1/TaskService/Tasks",
        ("GET", "HEAD"),
        {"@odata.id": "/redfish/v1/TaskService/Tasks"},
    )
    row = build_ordered_call_row(
        text="check task queue",
        contexts=(context,),
        rest_api_list=("/redfish/v1/TaskService/Tasks",),
    )

    assert inference_ordered_goals_json(row) == {
        "text": "check task queue",
        "ordered_goals": row["y_true"]["calls"],
    }


def test_y_pred_parsers_preserve_order_and_report_bad_contracts() -> None:
    """Parsed y_pred JSON preserves order and rejects malformed call objects clearly."""
    assert parse_rest_api_list_y_pred({
        "y_pred": {"rest_api_list": ["/redfish/v1/B", "/redfish/v1/A"]},
    }) == ["/redfish/v1/B", "/redfish/v1/A"]
    calls = [{
        "rest_api": "/redfish/v1/Systems",
        "allowed_methods": ["get", "head"],
        "method": "get",
        "arguments": {},
    }]

    assert parse_ordered_calls_y_pred(json.dumps({"y_pred": {"calls": calls}})) == [{
        "rest_api": "/redfish/v1/Systems",
        "allowed_methods": ["GET", "HEAD"],
        "method": "GET",
        "arguments": {},
    }]
    with pytest.raises(ValueError, match="rest_api"):
        parse_ordered_calls_y_pred({
            "y_pred": {
                "calls": [{
                    "allowed_methods": ["GET"],
                    "method": "GET",
                    "arguments": {},
                }],
            },
        })


def test_wandb_metric_keys_are_stage_scoped_and_not_m3_names() -> None:
    """Phase 2/3 metric constants use dedicated W&B namespaces."""
    assert "phase2_goal_extract/eval/ordered_exact_match_rate" in PHASE2_GOAL_EXTRACT_METRIC_KEYS
    assert "phase2_goal_extract/eval/set_match_rate" in PHASE2_GOAL_EXTRACT_METRIC_KEYS
    assert "phase3_argument_extract/eval/readonly_empty_arguments_rate" in PHASE3_ARGUMENT_EXTRACT_METRIC_KEYS
    assert "phase3_argument_extract/eval/arguments_exact_match_rate" in PHASE3_ARGUMENT_EXTRACT_METRIC_KEYS
    assert all(k.startswith("phase2_goal_extract/") for k in PHASE2_GOAL_EXTRACT_METRIC_KEYS)
    assert all(k.startswith("phase3_argument_extract/") for k in PHASE3_ARGUMENT_EXTRACT_METRIC_KEYS)
    assert not any(k.startswith("m3_") for k in PHASE2_GOAL_EXTRACT_METRIC_KEYS)
    assert not any(k.startswith("m3_") for k in PHASE3_ARGUMENT_EXTRACT_METRIC_KEYS)


def test_training_docs_pin_phase23_metric_constants() -> None:
    """Training docs name the Phase 2/3 constants and representative key groups."""
    docs_path = Path(__file__).resolve().parents[2] / "docs" / "TRAINING.md"
    training_doc = docs_path.read_text(encoding="utf-8")

    assert "PHASE2_GOAL_EXTRACT_METRIC_KEYS" in training_doc
    assert "PHASE3_ARGUMENT_EXTRACT_METRIC_KEYS" in training_doc
    assert "phase2_goal_extract/eval/{loss,perplexity,token_accuracy" in training_doc
    assert "phase2_goal_extract/test/{latency_sec_p50,latency_sec_p95" in training_doc
    assert "phase3_argument_extract/eval/{allowed_methods_exact_match_rate" in training_doc
    assert "phase3_argument_extract/data/{avg_num_calls,avg_arguments_length" in training_doc


# Author: Mus mbayramo@stanford.edu
