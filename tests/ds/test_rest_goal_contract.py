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
    inference_target_calls_json,
    parse_ordered_calls_y_pred,
    parse_rest_api_list_y_pred,
    render_ordered_call_example,
    render_rest_api_list_example,
)
from igc.modules.base.metric_keys import (
    PHASE2_WANDB_METRIC_KEYS,
    PHASE3_WANDB_METRIC_KEYS,
    PHASE23_WANDB_METRIC_KEYS,
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
    assert row["x"]["text"] == "check the task queue, then list the systems"
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


def test_d1_row_does_not_leak_extra_context_into_target_list() -> None:
    """Extra current context stays in x and never becomes an unrequested target."""
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
    chassis = _context(
        "/redfish/v1/Chassis",
        ("GET", "HEAD"),
        {"@odata.id": "/redfish/v1/Chassis", "Name": "Chassis"},
    )

    row = build_d1_rest_api_list_row(
        text="check the task queue, then list systems",
        contexts=(systems, tasks, chassis),
        rest_api_list=("/redfish/v1/TaskService/Tasks", "/redfish/v1/Systems"),
    )

    assert row["x"]["json"] == [systems.json, tasks.json, chassis.json]
    assert row["y_true"]["rest_api_list"] == [
        "/redfish/v1/TaskService/Tasks",
        "/redfish/v1/Systems",
    ]
    assert "/redfish/v1/Chassis" not in row["y_true"]["rest_api_list"]


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
    assert phase2["task"] == "text_to_rest_api_list"
    assert set(phase2["x"]) == {"text", "json", "allowed_methods"}
    assert set(phase2["y_true"]) == {"rest_api_list", "order_evidence"}
    assert set(phase3) == {"phase", "source_dataset", "model_x", "task", "x", "y_true"}
    assert phase3["task"] == "text_and_rest_api_list_to_calls"
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
    assert row["x"]["text"] == "check task state, then inspect system power"
    assert row["x"]["json"] == [
        {
            "@odata.id": "/redfish/v1/Systems/1",
            "PowerState": "On",
        },
        {
            "@odata.id": "/redfish/v1/TaskService/Tasks",
            "Members@odata.count": 0,
        },
    ]
    assert row["x"]["allowed_methods"] == {
        "/redfish/v1/Systems/1": ["GET", "HEAD"],
        "/redfish/v1/TaskService/Tasks": ["GET", "HEAD"],
    }
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


def test_phase3_get_calls_discard_supplied_arguments() -> None:
    """Read-only GET labels keep arguments empty even if caller supplies body-like data."""
    row = build_ordered_call_row(
        text="inspect system power",
        contexts=(
            _context(
                "/redfish/v1/Systems/1",
                ("GET", "PATCH"),
                {
                    "@odata.id": "/redfish/v1/Systems/1",
                    "PowerState": "On",
                },
            ),
        ),
        rest_api_list=("/redfish/v1/Systems/1",),
        method_by_api={"/redfish/v1/Systems/1": "GET"},
        arguments_by_api={"/redfish/v1/Systems/1": {"PowerState": "On"}},
    )

    assert row["y_true"]["calls"][0] == {
        "rest_api": "/redfish/v1/Systems/1",
        "allowed_methods": ["GET", "PATCH"],
        "method": "GET",
        "arguments": {},
    }


def test_phase3_head_calls_discard_supplied_arguments() -> None:
    """Read-only HEAD labels keep arguments empty like GET labels."""
    row = build_ordered_call_row(
        text="check system headers",
        contexts=(
            _context(
                "/redfish/v1/Systems/1",
                ("GET", "HEAD"),
                {
                    "@odata.id": "/redfish/v1/Systems/1",
                    "PowerState": "On",
                },
            ),
        ),
        rest_api_list=("/redfish/v1/Systems/1",),
        method_by_api={"/redfish/v1/Systems/1": "HEAD"},
        arguments_by_api={"/redfish/v1/Systems/1": {"PowerState": "On"}},
    )

    assert row["y_true"]["calls"][0] == {
        "rest_api": "/redfish/v1/Systems/1",
        "allowed_methods": ["GET", "HEAD"],
        "method": "HEAD",
        "arguments": {},
    }


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


def test_phase3_patch_does_not_infer_top_level_get_scalars() -> None:
    """PATCH rows do not turn arbitrary top-level GET values into arguments."""
    system = _context(
        "/redfish/v1/Systems/1",
        ("GET", "PATCH"),
        {
            "@odata.id": "/redfish/v1/Systems/1",
            "PowerState": "On",
            "Name": "System",
        },
    )

    row = build_ordered_call_row(
        text="set the system power state",
        contexts=(system,),
        rest_api_list=("/redfish/v1/Systems/1",),
        method_by_api={"/redfish/v1/Systems/1": "PATCH"},
    )

    assert row["y_true"]["calls"][0] == {
        "rest_api": "/redfish/v1/Systems/1",
        "allowed_methods": ["GET", "PATCH"],
        "method": "PATCH",
        "arguments": {},
    }


def test_phase3_post_does_not_infer_action_arguments_from_get_scalars() -> None:
    """POST rows do not turn observed action metadata into arguments."""
    action_target = _context(
        "/redfish/v1/Systems/1/Actions/ComputerSystem.Reset",
        ("POST",),
        {
            "@odata.id": "/redfish/v1/Systems/1",
            "ResetType": "GracefulRestart",
        },
    )

    row = build_ordered_call_row(
        text="reset the system",
        contexts=(action_target,),
        rest_api_list=("/redfish/v1/Systems/1/Actions/ComputerSystem.Reset",),
        method_by_api={
            "/redfish/v1/Systems/1/Actions/ComputerSystem.Reset": "POST",
        },
    )

    assert row["y_true"]["calls"][0] == {
        "rest_api": "/redfish/v1/Systems/1/Actions/ComputerSystem.Reset",
        "allowed_methods": ["POST"],
        "method": "POST",
        "arguments": {},
    }


def test_phase3_ignores_unselected_method_and_argument_labels() -> None:
    """Phase 3 emits exactly one call per ordered rest_api_list entry."""
    system = _context(
        "/redfish/v1/Systems/1",
        ("GET", "HEAD"),
        {"@odata.id": "/redfish/v1/Systems/1"},
    )
    reset = _context(
        "/redfish/v1/Systems/1/Actions/ComputerSystem.Reset",
        ("POST",),
        {"@odata.id": "/redfish/v1/Systems/1"},
    )

    row = build_ordered_call_row(
        text="inspect the system",
        contexts=(system, reset),
        rest_api_list=("/redfish/v1/Systems/1",),
        method_by_api={
            "/redfish/v1/Systems/1/Actions/ComputerSystem.Reset": "POST",
        },
        arguments_by_api={
            "/redfish/v1/Systems/1/Actions/ComputerSystem.Reset": {
                "ResetType": "GracefulRestart",
            },
        },
    )

    assert row["x"]["rest_api_list"] == ["/redfish/v1/Systems/1"]
    assert row["y_true"]["calls"] == [{
        "rest_api": "/redfish/v1/Systems/1",
        "allowed_methods": ["GET", "HEAD"],
        "method": "GET",
        "arguments": {},
    }]


def test_phase3_default_method_prefers_get_over_mutating_methods() -> None:
    """Default Phase 3 labels prefer read-only GET even if PATCH appears first."""
    system = _context(
        "/redfish/v1/Systems/1",
        ("PATCH", "GET"),
        {
            "@odata.id": "/redfish/v1/Systems/1",
            "PowerState": "On",
        },
    )

    row = build_ordered_call_row(
        text="inspect system power",
        contexts=(system,),
        rest_api_list=("/redfish/v1/Systems/1",),
        arguments_by_api={"/redfish/v1/Systems/1": {"PowerState": "On"}},
    )

    assert row["y_true"]["calls"][0] == {
        "rest_api": "/redfish/v1/Systems/1",
        "allowed_methods": ["PATCH", "GET"],
        "method": "GET",
        "arguments": {},
    }


def test_phase3_mixed_calls_preserve_order_case_and_per_api_arguments() -> None:
    """Mixed read/write calls keep order, normalize methods, and isolate arguments."""
    virtual_media = _context(
        "/redfish/v1/Managers/1/VirtualMedia/CD",
        ("get", "head"),
        {
            "@odata.id": "/redfish/v1/Managers/1/VirtualMedia/CD",
            "Image": "old.iso",
        },
    )
    bios_settings = _context(
        "/redfish/v1/Systems/1/Bios/Settings",
        ("get", "patch"),
        {
            "@odata.id": "/redfish/v1/Systems/1/Bios/Settings",
            "Attributes": {"BootMode": "LegacyBios"},
        },
    )

    row = build_ordered_call_row(
        text="inspect virtual media, then set bios boot mode to Uefi",
        contexts=(bios_settings, virtual_media),
        rest_api_list=(
            "/redfish/v1/Managers/1/VirtualMedia/CD",
            "/redfish/v1/Systems/1/Bios/Settings",
        ),
        method_by_api={"/redfish/v1/Systems/1/Bios/Settings": "patch"},
        arguments_by_api={
            "/redfish/v1/Systems/1/Bios/Settings": {
                "Attributes": {"BootMode": "Uefi"},
            },
            "/redfish/v1/Managers/1/VirtualMedia/CD": {"Image": "old.iso"},
        },
    )

    assert row["x"]["text"] == "inspect virtual media, then set bios boot mode to Uefi"
    assert row["x"]["json"] == [bios_settings.json, virtual_media.json]
    assert row["x"]["allowed_methods"] == {
        "/redfish/v1/Systems/1/Bios/Settings": ["GET", "PATCH"],
        "/redfish/v1/Managers/1/VirtualMedia/CD": ["GET", "HEAD"],
    }
    assert row["y_true"]["calls"] == [
        {
            "rest_api": "/redfish/v1/Managers/1/VirtualMedia/CD",
            "allowed_methods": ["GET", "HEAD"],
            "method": "GET",
            "arguments": {},
        },
        {
            "rest_api": "/redfish/v1/Systems/1/Bios/Settings",
            "allowed_methods": ["GET", "PATCH"],
            "method": "PATCH",
            "arguments": {"Attributes": {"BootMode": "Uefi"}},
        },
    ]


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


def test_empty_ordered_rows_are_supported_for_noop_context() -> None:
    """Empty mock rows encode no selected REST goals without inventing context."""
    phase2 = build_d1_rest_api_list_row(
        text="nothing to do",
        contexts=(),
        rest_api_list=(),
        order_evidence="empty_request",
    )
    phase3 = build_ordered_call_row(
        text="nothing to do",
        contexts=(),
        rest_api_list=(),
    )

    assert phase2["x"]["json"] == []
    assert phase2["x"]["allowed_methods"] == {}
    assert phase2["y_true"]["rest_api_list"] == []
    assert phase2["y_true"]["order_evidence"] == "empty_request"
    assert phase3["x"]["rest_api_list"] == []
    assert phase3["x"]["json"] == []
    assert phase3["x"]["allowed_methods"] == {}
    assert phase3["y_true"]["calls"] == []


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
    assert rendered2.target_json == (
        "{\n"
        '  "rest_api_list": [\n'
        '    "/redfish/v1/Systems"\n'
        "  ]\n"
        "}"
    )
    assert json.loads(rendered2.target_json) == {"rest_api_list": ["/redfish/v1/Systems"]}
    assert "### Ordered REST API List" in rendered2.prompt
    assert rendered2.full_text == rendered2.prompt + rendered2.target_json
    assert rendered3.target_json == (
        "{\n"
        '  "calls": [\n'
        "    {\n"
        '      "allowed_methods": [\n'
        '        "GET",\n'
        '        "HEAD"\n'
        "      ],\n"
        '      "arguments": {},\n'
        '      "method": "GET",\n'
        '      "rest_api": "/redfish/v1/Systems"\n'
        "    }\n"
        "  ]\n"
        "}"
    )
    assert json.loads(rendered3.target_json) == {
        "calls": phase3["y_true"]["calls"],
    }
    assert "### Ordered REST Calls" in rendered3.prompt


def test_rendered_phase3_patch_example_keeps_explicit_arguments() -> None:
    """Rendered Phase 3 labels preserve explicit non-GET arguments."""
    context = _context(
        "/redfish/v1/Systems/1/Bios/Settings",
        ("GET", "PATCH"),
        {
            "@odata.id": "/redfish/v1/Systems/1/Bios/Settings",
            "Attributes": {"BootMode": "Uefi"},
        },
    )
    row = build_ordered_call_row(
        text="set bios boot mode to Uefi",
        contexts=(context,),
        rest_api_list=("/redfish/v1/Systems/1/Bios/Settings",),
        method_by_api={"/redfish/v1/Systems/1/Bios/Settings": "PATCH"},
        arguments_by_api={
            "/redfish/v1/Systems/1/Bios/Settings": {
                "Attributes": {"BootMode": "Uefi"},
            },
        },
    )

    rendered = render_ordered_call_example(row)

    assert rendered.target_char_start == len(rendered.prompt)
    assert json.loads(rendered.target_json) == {
        "calls": [{
            "rest_api": "/redfish/v1/Systems/1/Bios/Settings",
            "allowed_methods": ["GET", "PATCH"],
            "method": "PATCH",
            "arguments": {"Attributes": {"BootMode": "Uefi"}},
        }],
    }
    assert '"PATCH"' in rendered.target_json
    assert '"Attributes": {' in rendered.target_json
    assert "### Ordered REST Calls" in rendered.prompt
    assert "/redfish/v1/Systems/1/Bios/Settings" in rendered.prompt


def test_inference_json_uses_target_calls_shape() -> None:
    """The combined inference handoff uses target_calls with Phase 3 call fields."""
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

    assert inference_target_calls_json(row) == {
        "text": "check task queue",
        "target_calls": row["y_true"]["calls"],
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
    with pytest.raises(ValueError, match="method"):
        parse_ordered_calls_y_pred({
            "calls": [{
                "rest_api": "/redfish/v1/Systems",
                "allowed_methods": ["GET"],
                "arguments": {},
            }],
        })
    with pytest.raises(ValueError, match="arguments"):
        parse_ordered_calls_y_pred({
            "calls": [{
                "rest_api": "/redfish/v1/Systems",
                "allowed_methods": ["GET"],
                "method": "GET",
            }],
        })


def test_rest_api_list_parser_rejects_non_string_items() -> None:
    """Phase 2 y_pred parsing rejects non-string REST API labels."""
    with pytest.raises(ValueError, match="rest_api_list item"):
        parse_rest_api_list_y_pred({
            "y_pred": {"rest_api_list": ["/redfish/v1/Systems", 42]},
        })


def test_rest_api_list_parser_rejects_non_object_top_level_json() -> None:
    """Phase 2 y_pred parsing rejects JSON that is not an object."""
    for y_pred in ('["/redfish/v1/Systems"]', '"not an object"'):
        with pytest.raises(ValueError, match="y_pred must be an object"):
            parse_rest_api_list_y_pred(y_pred)


def test_rest_api_list_parser_rejects_non_object_y_pred_envelope() -> None:
    """Phase 2 y_pred parsing rejects a malformed y_pred envelope value."""
    for y_pred in ({"y_pred": ["/redfish/v1/Systems"]}, {"y_pred": "not an object"}):
        with pytest.raises(ValueError, match="y_pred.y_pred must be an object"):
            parse_rest_api_list_y_pred(y_pred)


def test_ordered_calls_parser_preserves_multiple_call_order() -> None:
    """Phase 3 y_pred parsing keeps the model-emitted call sequence intact."""
    calls = [
        {
            "rest_api": "/redfish/v1/TaskService/Tasks",
            "allowed_methods": ["GET", "HEAD"],
            "method": "GET",
            "arguments": {},
        },
        {
            "rest_api": "/redfish/v1/Systems",
            "allowed_methods": ["GET", "HEAD"],
            "method": "GET",
            "arguments": {},
        },
    ]

    assert parse_ordered_calls_y_pred({"calls": calls}) == calls


def test_ordered_calls_parser_rejects_non_string_contract_fields() -> None:
    """Phase 3 y_pred parsing rejects non-string REST API, method, and allowed methods."""
    with pytest.raises(ValueError, match="rest_api"):
        parse_ordered_calls_y_pred({
            "calls": [{
                "rest_api": 42,
                "allowed_methods": ["GET"],
                "method": "GET",
                "arguments": {},
            }],
        })
    with pytest.raises(ValueError, match="allowed_methods"):
        parse_ordered_calls_y_pred({
            "calls": [{
                "rest_api": "/redfish/v1/Systems",
                "allowed_methods": ["GET", 42],
                "method": "GET",
                "arguments": {},
            }],
        })
    with pytest.raises(ValueError, match="method"):
        parse_ordered_calls_y_pred({
            "calls": [{
                "rest_api": "/redfish/v1/Systems",
                "allowed_methods": ["GET"],
                "method": 42,
                "arguments": {},
            }],
        })


def test_ordered_calls_parser_rejects_non_object_top_level_json() -> None:
    """Phase 3 y_pred parsing rejects JSON that is not an object."""
    for y_pred in ("[]", "42"):
        with pytest.raises(ValueError, match="y_pred must be an object"):
            parse_ordered_calls_y_pred(y_pred)


def test_ordered_calls_parser_rejects_non_object_y_pred_envelope() -> None:
    """Phase 3 y_pred parsing rejects a malformed y_pred envelope value."""
    for y_pred in ({"y_pred": []}, {"y_pred": 42}):
        with pytest.raises(ValueError, match="y_pred.y_pred must be an object"):
            parse_ordered_calls_y_pred(y_pred)


def test_ordered_calls_parser_rejects_invalid_method_and_arguments_shape() -> None:
    """Phase 3 y_pred parsing rejects invalid method and argument contracts."""
    with pytest.raises(ValueError, match="not in allowed_methods"):
        parse_ordered_calls_y_pred({
            "calls": [{
                "rest_api": "/redfish/v1/Systems",
                "allowed_methods": ["GET", "HEAD"],
                "method": "PATCH",
                "arguments": {},
            }],
        })
    with pytest.raises(ValueError, match="arguments"):
        parse_ordered_calls_y_pred({
            "calls": [{
                "rest_api": "/redfish/v1/Systems",
                "allowed_methods": ["GET", "HEAD"],
                "method": "GET",
                "arguments": ["PowerState", "On"],
            }],
        })


def test_ordered_calls_parser_rejects_readonly_arguments() -> None:
    """Phase 3 y_pred parsing rejects non-empty GET/HEAD argument objects."""
    for method in ("GET", "HEAD"):
        with pytest.raises(ValueError, match="read-only"):
            parse_ordered_calls_y_pred({
                "calls": [{
                    "rest_api": "/redfish/v1/Systems",
                    "allowed_methods": ["GET", "HEAD", "PATCH"],
                    "method": method,
                    "arguments": {"PowerState": "On"},
                }],
            })


def test_wandb_metric_keys_are_stage_scoped_and_not_m3_names() -> None:
    """Phase 2/3 contract constants reuse the shared W&B metric registry."""
    assert PHASE2_GOAL_EXTRACT_METRIC_KEYS == PHASE2_WANDB_METRIC_KEYS
    assert PHASE3_ARGUMENT_EXTRACT_METRIC_KEYS == PHASE3_WANDB_METRIC_KEYS
    assert (
        PHASE2_GOAL_EXTRACT_METRIC_KEYS + PHASE3_ARGUMENT_EXTRACT_METRIC_KEYS
        == PHASE23_WANDB_METRIC_KEYS
    )
    assert (
        "phase2_goal_extract/eval/ordered_exact_match_rate"
        in PHASE2_GOAL_EXTRACT_METRIC_KEYS
    )
    assert "phase2_goal_extract/eval/set_match_rate" in PHASE2_GOAL_EXTRACT_METRIC_KEYS
    assert (
        "phase3_argument_extract/eval/readonly_empty_arguments_rate"
        in PHASE3_ARGUMENT_EXTRACT_METRIC_KEYS
    )
    assert (
        "phase3_argument_extract/eval/arguments_exact_match_rate"
        in PHASE3_ARGUMENT_EXTRACT_METRIC_KEYS
    )
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
    assert "PHASE2_WANDB_METRIC_KEYS" in training_doc
    assert "PHASE3_WANDB_METRIC_KEYS" in training_doc
    assert "phase2_goal_extract/eval/{ordered_exact_match_rate,set_match_rate" in training_doc
    assert "phase2_goal_extract/order/{kendall_tau,edit_distance}" in training_doc
    assert "phase3_argument_extract/eval/{call_ordered_exact_match_rate" in training_doc
    assert "phase3_argument_extract/order/{kendall_tau,edit_distance}" in training_doc


# Author: Mus mbayramo@stanford.edu
