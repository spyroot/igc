"""Offline tests for the REST-goal dataset contracts.

These tests pin the Phase 2/3 mock-plumbing rows without training a model,
running W&B, touching captured corpora, or inventing a regex extractor.
Phase 2/3 are UNORDERED tasks; execution order is separate RL-oracle evidence.

Author:
Mus mbayramo@stanford.edu
"""

import json
from pathlib import Path

import pytest

import igc.ds.rest_goal_contract as rest_goal_contract
from igc.ds.rest_goal_contract import (
    D0,
    D1,
    MODEL_X,
    PHASE2_GOAL_EXTRACT_METRIC_KEYS,
    PHASE3_ARGUMENT_EXTRACT_METRIC_KEYS,
    RedfishContext,
    build_d1_rest_api_list_row,
    build_phase2_labelled_request_row,
    build_call_row,
    evaluate_rest_api_list_y_pred,
    evaluate_calls_y_pred,
    parse_calls_y_pred,
    parse_rest_api_list_y_pred,
    render_call_example,
    render_rest_api_list_example,
)
from igc.modules.base.metric_keys import (
    PHASE2_LABELLED_REQUESTS,
    PHASE2_WANDB_METRIC_KEYS,
    PHASE3_WANDB_METRIC_KEYS,
    PHASE23_WANDB_METRIC_KEYS,
)


def _context(rest_api: str, allowed_methods: tuple[str, ...], body: dict) -> RedfishContext:
    """Build one tiny Redfish context row for contract tests."""
    return RedfishContext(rest_api=rest_api, allowed_methods=allowed_methods, json=body)


def test_phase23_locked_name_constants_use_literal_contract_values() -> None:
    """Locked Phase 2/3 names stay literal, not only internally self-consistent."""
    assert MODEL_X == "model_x"
    assert D0 == "D0"
    assert D1 == "D1"
    assert PHASE2_LABELLED_REQUESTS == "phase2_labelled_requests"


def test_legacy_phase2_builder_name_emits_locked_d1_shape() -> None:
    """The old builder name is only an alias, not a second row contract."""
    context = _context(
        "/redfish/v1/Systems",
        ("GET", "HEAD"),
        {"@odata.id": "/redfish/v1/Systems"},
    )

    assert build_phase2_labelled_request_row(
        text="list systems",
        contexts=(context,),
        rest_api_list=("/redfish/v1/Systems",),
    ) == build_d1_rest_api_list_row(
        text="list systems",
        contexts=(context,),
        rest_api_list=("/redfish/v1/Systems",),
    )


def test_phase2_row_preserves_operator_order_independent_of_context_order() -> None:
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
    assert row["target_semantics"] == "unordered_unique_set"
    assert row["x"]["text"] == "check the task queue, then list the systems"
    assert set(row["x"]) == {"text"}
    assert row["y_true"]["rest_api_list"] == [
        "/redfish/v1/TaskService/Tasks",
        "/redfish/v1/Systems",
    ]
    assert row["validation"] == {
        "text_source": "mock_fixture",
        "review_judged": False,
        "natural": True,
        "exact_api_coverage": True,
        "extra_intent": False,
        "duplicate_intent": False,
        "ambiguous": False,
        "nonsense": False,
        "method_semantics_valid": True,
    }


def test_phase2_row_does_not_leak_extra_context_into_target_list() -> None:
    """Extra generator context never leaks into D1 x or the target set."""
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

    assert set(row["x"]) == {"text"}
    assert row["y_true"]["rest_api_list"] == [
        "/redfish/v1/TaskService/Tasks",
        "/redfish/v1/Systems",
    ]
    assert "/redfish/v1/Chassis" not in row["y_true"]["rest_api_list"]


def test_phase23_rows_pin_locked_field_names() -> None:
    """Phase 2/3 rows expose only the locked contract field names."""
    body = {"@odata.id": "/redfish/v1/Systems", "Name": "Systems"}
    context = _context(
        "/redfish/v1/Systems",
        ("GET", "HEAD"),
        body,
    )

    serialized = context.to_dict()

    assert set(serialized) == {"rest_api", "allowed_methods", "json"}
    assert serialized == {
        "rest_api": "/redfish/v1/Systems",
        "allowed_methods": ["GET", "HEAD"],
        "json": {"@odata.id": "/redfish/v1/Systems", "Name": "Systems"},
    }
    assert serialized["allowed_methods"] is not context.allowed_methods
    assert serialized["json"] is not body

    phase2 = build_d1_rest_api_list_row(
        text="list systems",
        contexts=(context,),
        rest_api_list=("/redfish/v1/Systems",),
    )
    phase3 = build_call_row(
        text="list systems",
        contexts=(context,),
        rest_api_list=("/redfish/v1/Systems",),
        method_by_api={"/redfish/v1/Systems": "GET"},
    )

    assert set(phase2) == {
        "phase",
        "dataset",
        "source_dataset",
        "model_x",
        "task",
        "target_semantics",
        "x",
        "y_true",
        "validation",
    }
    assert phase2["task"] == "text_to_rest_api_list"
    assert phase2["dataset"] == D1
    assert phase2["target_semantics"] == "unordered_unique_set"
    assert set(phase2["x"]) == {"text"}
    assert set(phase2["y_true"]) == {"rest_api_list"}
    assert set(phase3) == {
        "phase",
        "source_dataset",
        "model_x",
        "task",
        "target_semantics",
        "x",
        "y_true",
    }
    assert phase3["task"] == "text_and_rest_api_list_to_calls"
    assert phase3["source_dataset"] == D1
    assert phase3["target_semantics"] == "unordered_call_set"
    assert set(phase3["x"]) == {"text", "rest_api_list", "json", "allowed_methods"}
    assert set(phase3["y_true"]) == {"calls"}
    # A Call is exactly rest_api/method/arguments; allowed_methods stays in x.
    assert set(phase3["y_true"]["calls"][0]) == {
        "rest_api",
        "method",
        "arguments",
    }


def test_phase3_get_calls_form_canonical_set_and_keep_arguments_empty() -> None:
    """Read-only GET calls form a canonical (sorted) set and never copy scalar JSON values."""
    row = build_call_row(
        text="check task state and inspect system power",
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
        method_by_api={
            "/redfish/v1/TaskService/Tasks": "GET",
            "/redfish/v1/Systems/1": "GET",
        },
    )

    assert row["phase"] == 3
    assert row["x"]["text"] == "check task state and inspect system power"
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
    # Canonical (sorted) set identity — caller mention order is NOT preserved,
    # because Phase 3 output is a set, not an execution plan.
    assert row["x"]["rest_api_list"] == [
        "/redfish/v1/Systems/1",
        "/redfish/v1/TaskService/Tasks",
    ]
    assert row["y_true"]["calls"] == [
        {
            "rest_api": "/redfish/v1/Systems/1",
            "method": "GET",
            "arguments": {},
        },
        {
            "rest_api": "/redfish/v1/TaskService/Tasks",
            "method": "GET",
            "arguments": {},
        },
    ]


def test_phase3_readonly_calls_reject_supplied_arguments() -> None:
    """GET/HEAD labels raise on body-like arguments instead of silently discarding them."""
    for method in ("GET", "HEAD"):
        with pytest.raises(ValueError, match="read-only"):
            build_call_row(
                text="inspect system power",
                contexts=(
                    _context(
                        "/redfish/v1/Systems/1",
                        ("GET", "HEAD", "PATCH"),
                        {
                            "@odata.id": "/redfish/v1/Systems/1",
                            "PowerState": "On",
                        },
                    ),
                ),
                rest_api_list=("/redfish/v1/Systems/1",),
                method_by_api={"/redfish/v1/Systems/1": method},
                arguments_by_api={"/redfish/v1/Systems/1": {"PowerState": "On"}},
            )


def test_phase3_mutation_arguments_must_be_supplied_explicitly() -> None:
    """Mutation rows require an explicit binding; missing args never become {} silently."""
    settings = _context(
        "/redfish/v1/Systems/1/Bios/Settings",
        ("GET", "PATCH"),
        {
            "@odata.id": "/redfish/v1/Systems/1/Bios/Settings",
            "Attributes": {"BootMode": "Uefi"},
        },
    )

    # A PATCH with NO explicit binding raises — the builder never infers {} or
    # scrapes arguments from the GET JSON scalars.
    with pytest.raises(ValueError, match="explicit arguments"):
        build_call_row(
            text="set bios boot mode",
            contexts=(settings,),
            rest_api_list=("/redfish/v1/Systems/1/Bios/Settings",),
            method_by_api={"/redfish/v1/Systems/1/Bios/Settings": "PATCH"},
        )
    with_arguments = build_call_row(
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

    assert with_arguments["y_true"]["calls"][0]["arguments"] == {
        "Attributes": {"BootMode": "Uefi"},
    }


def test_phase3_no_argument_action_binds_empty_object_explicitly() -> None:
    """A no-argument function/action explicitly binds {} — still an object, never inferred."""
    action_target = _context(
        "/redfish/v1/Systems/1/Actions/ComputerSystem.Reset",
        ("POST",),
        {
            "@odata.id": "/redfish/v1/Systems/1",
            "ResetType": "GracefulRestart",
        },
    )

    row = build_call_row(
        text="reset the system",
        contexts=(action_target,),
        rest_api_list=("/redfish/v1/Systems/1/Actions/ComputerSystem.Reset",),
        method_by_api={
            "/redfish/v1/Systems/1/Actions/ComputerSystem.Reset": "POST",
        },
        arguments_by_api={
            # Explicit empty binding: the caller says "no arguments", the
            # builder does not scrape ResetType from the observed GET JSON.
            "/redfish/v1/Systems/1/Actions/ComputerSystem.Reset": {},
        },
    )

    assert row["y_true"]["calls"][0] == {
        "rest_api": "/redfish/v1/Systems/1/Actions/ComputerSystem.Reset",
        "method": "POST",
        "arguments": {},
    }


def test_phase3_ignores_unselected_method_and_argument_labels() -> None:
    """Phase 3 emits exactly one call per selected API; unselected labels are ignored."""
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

    row = build_call_row(
        text="inspect the system",
        contexts=(system, reset),
        rest_api_list=("/redfish/v1/Systems/1",),
        method_by_api={
            "/redfish/v1/Systems/1": "GET",
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
        "method": "GET",
        "arguments": {},
    }]


def test_phase3_missing_method_raises_instead_of_defaulting() -> None:
    """A selected API without an explicit method raises — GET is never a silent default."""
    system = _context(
        "/redfish/v1/Systems/1",
        ("PATCH", "GET"),
        {
            "@odata.id": "/redfish/v1/Systems/1",
            "PowerState": "On",
        },
    )

    with pytest.raises(ValueError, match="explicit method"):
        build_call_row(
            text="inspect system power",
            contexts=(system,),
            rest_api_list=("/redfish/v1/Systems/1",),
            method_by_api={},
        )


def test_phase3_mixed_calls_normalize_case_and_isolate_per_api_arguments() -> None:
    """Mixed read/write calls normalize method case and isolate per-API arguments."""
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

    row = build_call_row(
        text="inspect virtual media and set bios boot mode to Uefi",
        contexts=(bios_settings, virtual_media),
        rest_api_list=(
            "/redfish/v1/Systems/1/Bios/Settings",
            "/redfish/v1/Managers/1/VirtualMedia/CD",
        ),
        method_by_api={
            "/redfish/v1/Managers/1/VirtualMedia/CD": "get",
            "/redfish/v1/Systems/1/Bios/Settings": "patch",
        },
        arguments_by_api={
            "/redfish/v1/Systems/1/Bios/Settings": {
                "Attributes": {"BootMode": "Uefi"},
            },
        },
    )

    assert row["x"]["text"] == "inspect virtual media and set bios boot mode to Uefi"
    assert row["x"]["json"] == [bios_settings.json, virtual_media.json]
    assert row["x"]["allowed_methods"] == {
        "/redfish/v1/Systems/1/Bios/Settings": ["GET", "PATCH"],
        "/redfish/v1/Managers/1/VirtualMedia/CD": ["GET", "HEAD"],
    }
    # Canonical sorted set: Managers/... sorts before Systems/... regardless of
    # the caller-supplied mention order.
    assert row["y_true"]["calls"] == [
        {
            "rest_api": "/redfish/v1/Managers/1/VirtualMedia/CD",
            "method": "GET",
            "arguments": {},
        },
        {
            "rest_api": "/redfish/v1/Systems/1/Bios/Settings",
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
        build_d1_rest_api_list_row(
            text="list systems",
            contexts=(context, context),
            rest_api_list=("/redfish/v1/Systems",),
        )
    with pytest.raises(ValueError, match="unique set"):
        build_d1_rest_api_list_row(
            text="list systems twice",
            contexts=(context,),
            rest_api_list=("/redfish/v1/Systems", "/redfish/v1/Systems"),
        )
    with pytest.raises(ValueError, match="not present"):
        build_call_row(
            text="list chassis",
            contexts=(context,),
            rest_api_list=("/redfish/v1/Chassis",),
            method_by_api={},
        )
    with pytest.raises(ValueError, match="duplicate rest_api"):
        build_call_row(
            text="list systems twice",
            contexts=(context, context),
            rest_api_list=("/redfish/v1/Systems",),
            method_by_api={"/redfish/v1/Systems": "GET"},
        )
    with pytest.raises(ValueError, match="unique set"):
        build_call_row(
            text="list systems twice",
            contexts=(context,),
            rest_api_list=("/redfish/v1/Systems", "/redfish/v1/Systems"),
            method_by_api={"/redfish/v1/Systems": "GET"},
        )


def test_empty_rows_are_supported_for_noop_context() -> None:
    """Empty mock rows encode no selected REST goals without inventing context."""
    phase2 = build_d1_rest_api_list_row(
        text="nothing to do",
        contexts=(),
        rest_api_list=(),
    )
    phase3 = build_call_row(
        text="nothing to do",
        contexts=(),
        rest_api_list=(),
        method_by_api={},
    )

    assert phase2["x"] == {"text": "nothing to do"}
    assert phase2["y_true"]["rest_api_list"] == []
    assert set(phase2["y_true"]) == {"rest_api_list"}
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
        build_call_row(
            text="delete systems",
            contexts=(read_only,),
            rest_api_list=("/redfish/v1/Systems",),
            method_by_api={"/redfish/v1/Systems": "DELETE"},
        )
    with pytest.raises(ValueError, match="not in allowed_methods"):
        build_call_row(
            text="list managers",
            contexts=(no_methods,),
            rest_api_list=("/redfish/v1/Managers",),
            method_by_api={"/redfish/v1/Managers": "GET"},
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
    phase3 = build_call_row(
        text="list systems",
        contexts=(context,),
        rest_api_list=("/redfish/v1/Systems",),
        method_by_api={"/redfish/v1/Systems": "GET"},
    )

    rendered2 = render_rest_api_list_example(phase2)
    rendered3 = render_call_example(phase3)

    assert rendered2.target_char_start == len(rendered2.prompt)
    assert rendered2.target_json == (
        "{\n"
        '  "rest_api_list": [\n'
        '    "/redfish/v1/Systems"\n'
        "  ]\n"
        "}"
    )
    assert json.loads(rendered2.target_json) == {"rest_api_list": ["/redfish/v1/Systems"]}
    assert "### REST API Set" in rendered2.prompt
    assert rendered2.full_text == rendered2.prompt + rendered2.target_json
    assert rendered3.target_json == (
        "{\n"
        '  "calls": [\n'
        "    {\n"
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
    assert "### REST Calls" in rendered3.prompt
    assert "Ordered" not in rendered3.prompt


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
    row = build_call_row(
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

    rendered = render_call_example(row)

    assert rendered.target_char_start == len(rendered.prompt)
    assert json.loads(rendered.target_json) == {
        "calls": [{
            "rest_api": "/redfish/v1/Systems/1/Bios/Settings",
            "method": "PATCH",
            "arguments": {"Attributes": {"BootMode": "Uefi"}},
        }],
    }
    assert '"PATCH"' in rendered.target_json
    assert '"Attributes": {' in rendered.target_json
    assert "### REST Calls" in rendered.prompt
    assert "/redfish/v1/Systems/1/Bios/Settings" in rendered.prompt


def test_rendered_phase3_targets_use_canonical_set_order() -> None:
    """Phase 2 keeps its stored list; Phase 3 targets render the canonical sorted set."""
    systems = _context(
        "/redfish/v1/Systems",
        ("GET", "HEAD"),
        {"@odata.id": "/redfish/v1/Systems"},
    )
    tasks = _context(
        "/redfish/v1/TaskService/Tasks",
        ("GET", "HEAD"),
        {"@odata.id": "/redfish/v1/TaskService/Tasks"},
    )

    phase2 = build_d1_rest_api_list_row(
        text="check tasks and list systems",
        contexts=(systems, tasks),
        rest_api_list=("/redfish/v1/TaskService/Tasks", "/redfish/v1/Systems"),
    )
    phase3 = build_call_row(
        text="check tasks and list systems",
        contexts=(systems, tasks),
        rest_api_list=("/redfish/v1/TaskService/Tasks", "/redfish/v1/Systems"),
        method_by_api={
            "/redfish/v1/TaskService/Tasks": "GET",
            "/redfish/v1/Systems": "GET",
        },
    )

    # Phase 2 stores the list as supplied (serialization only, evaluated as a set).
    assert json.loads(render_rest_api_list_example(phase2).target_json) == {
        "rest_api_list": [
            "/redfish/v1/TaskService/Tasks",
            "/redfish/v1/Systems",
        ],
    }
    # Phase 3 canonicalizes to the sorted unique set — list order is identity,
    # never an execution plan.
    assert json.loads(render_call_example(phase3).target_json) == {
        "calls": [
            {
                "rest_api": "/redfish/v1/Systems",
                "method": "GET",
                "arguments": {},
            },
            {
                "rest_api": "/redfish/v1/TaskService/Tasks",
                "method": "GET",
                "arguments": {},
            },
        ],
    }


def test_contract_module_exports_no_ordered_handoff() -> None:
    """The ordered inference handoff is gone: order belongs to the RL oracle only."""
    assert not hasattr(rest_goal_contract, "inference_ordered_goals_json")
    assert not hasattr(rest_goal_contract, "build_ordered_call_row")
    assert not hasattr(rest_goal_contract, "_default_method")


def test_y_pred_parsers_report_bad_contracts() -> None:
    """Parsed y_pred JSON keeps predicted items and rejects malformed calls clearly."""
    assert parse_rest_api_list_y_pred({
        "y_pred": {"rest_api_list": ["/redfish/v1/B", "/redfish/v1/A"]},
    }) == ["/redfish/v1/B", "/redfish/v1/A"]
    assert parse_rest_api_list_y_pred({
        "rest_api_list": ["/redfish/v1/Systems", "/redfish/v1/Managers"],
    }) == ["/redfish/v1/Systems", "/redfish/v1/Managers"]
    calls = [{
        "rest_api": "/redfish/v1/Systems",
        "method": "get",
        "arguments": {},
    }]

    assert parse_calls_y_pred(json.dumps({"y_pred": {"calls": calls}})) == [{
        "rest_api": "/redfish/v1/Systems",
        "method": "GET",
        "arguments": {},
    }]
    with pytest.raises(ValueError, match="rest_api"):
        parse_calls_y_pred({
            "y_pred": {
                "calls": [{
                    "method": "GET",
                    "arguments": {},
                }],
            },
        })
    with pytest.raises(ValueError, match="method"):
        parse_calls_y_pred({
            "calls": [{
                "rest_api": "/redfish/v1/Systems",
                "arguments": {},
            }],
        })
    with pytest.raises(ValueError, match="arguments"):
        parse_calls_y_pred({
            "calls": [{
                "rest_api": "/redfish/v1/Systems",
                "method": "GET",
            }],
        })


def test_rest_api_list_evaluator_ignores_order_and_rejects_duplicates() -> None:
    """Phase 2 target scoring is set equality plus no duplicate predictions."""
    row = build_d1_rest_api_list_row(
        text="check managers and systems",
        contexts=(
            _context("/redfish/v1/Systems", ("GET",), {"@odata.id": "/redfish/v1/Systems"}),
            _context("/redfish/v1/Managers", ("GET",), {"@odata.id": "/redfish/v1/Managers"}),
        ),
        rest_api_list=("/redfish/v1/Systems", "/redfish/v1/Managers"),
    )

    reordered = evaluate_rest_api_list_y_pred(row, {
        "rest_api_list": ["/redfish/v1/Managers", "/redfish/v1/Systems"],
    })
    duplicated = evaluate_rest_api_list_y_pred(row, {
        "rest_api_list": [
            "/redfish/v1/Managers",
            "/redfish/v1/Systems",
            "/redfish/v1/Systems",
        ],
    })
    extra = evaluate_rest_api_list_y_pred(row, {
        "rest_api_list": ["/redfish/v1/Systems", "/redfish/v1/Chassis"],
    })

    assert reordered["parse_ok"] is True
    assert reordered["set_match"] is True
    assert duplicated["set_match"] is False
    assert duplicated["duplicate_prediction"] is True
    assert extra["set_match"] is False
    assert extra["missing_rest_api"] == ["/redfish/v1/Managers"]
    assert extra["extra_rest_api"] == ["/redfish/v1/Chassis"]


def test_calls_parser_preserves_mutation_arguments_and_normalizes_method() -> None:
    """Phase 3 y_pred parsing keeps PATCH arguments and normalizes method case."""
    calls = [{
        "rest_api": "/redfish/v1/Systems/1/Bios/Settings",
        "method": "patch",
        "arguments": {"Attributes": {"BootMode": "Uefi"}},
    }]

    assert parse_calls_y_pred({"y_pred": {"calls": calls}}) == [{
        "rest_api": "/redfish/v1/Systems/1/Bios/Settings",
        "method": "PATCH",
        "arguments": {"Attributes": {"BootMode": "Uefi"}},
    }]


def test_calls_parser_strips_context_fields_from_predicted_calls() -> None:
    """allowed_methods is row evidence — a prediction carrying it is not echoed back."""
    calls = [{
        "rest_api": "/redfish/v1/Systems",
        "allowed_methods": ["GET", "HEAD"],
        "method": "GET",
        "arguments": {},
    }]

    assert parse_calls_y_pred({"calls": calls}) == [{
        "rest_api": "/redfish/v1/Systems",
        "method": "GET",
        "arguments": {},
    }]


def test_rest_api_list_parser_rejects_non_string_items() -> None:
    """Phase 2 y_pred parsing rejects non-string REST API labels."""
    with pytest.raises(ValueError, match="rest_api_list item"):
        parse_rest_api_list_y_pred({
            "y_pred": {"rest_api_list": ["/redfish/v1/Systems", 42]},
        })


def test_rest_api_list_parser_rejects_non_list_target() -> None:
    """Phase 2 y_pred parsing rejects scalar REST API labels."""
    with pytest.raises(ValueError, match="rest_api_list must be a list"):
        parse_rest_api_list_y_pred({
            "y_pred": {"rest_api_list": "/redfish/v1/Systems"},
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


def test_calls_parser_keeps_predicted_sequence_for_set_evaluation() -> None:
    """Parsing keeps predicted items as emitted; set semantics apply at evaluation."""
    calls = [
        {
            "rest_api": "/redfish/v1/TaskService/Tasks",
            "method": "GET",
            "arguments": {},
        },
        {
            "rest_api": "/redfish/v1/Systems",
            "method": "GET",
            "arguments": {},
        },
    ]

    assert parse_calls_y_pred({"calls": calls}) == calls


def test_calls_parser_rejects_non_string_contract_fields() -> None:
    """Phase 3 y_pred parsing rejects non-string REST API and method values."""
    with pytest.raises(ValueError, match="rest_api"):
        parse_calls_y_pred({
            "calls": [{
                "rest_api": 42,
                "method": "GET",
                "arguments": {},
            }],
        })
    with pytest.raises(ValueError, match="method"):
        parse_calls_y_pred({
            "calls": [{
                "rest_api": "/redfish/v1/Systems",
                "method": 42,
                "arguments": {},
            }],
        })


def test_calls_parser_rejects_non_list_calls_and_items() -> None:
    """Phase 3 y_pred parsing rejects malformed calls containers and items."""
    with pytest.raises(ValueError, match="calls must be a list"):
        parse_calls_y_pred({"calls": "not a list"})
    with pytest.raises(ValueError, match="calls item must be an object"):
        parse_calls_y_pred({"calls": ["not an object"]})


def test_calls_parser_rejects_non_object_top_level_json() -> None:
    """Phase 3 y_pred parsing rejects JSON that is not an object."""
    for y_pred in ("[]", "42"):
        with pytest.raises(ValueError, match="y_pred must be an object"):
            parse_calls_y_pred(y_pred)


def test_calls_parser_rejects_non_object_y_pred_envelope() -> None:
    """Phase 3 y_pred parsing rejects a malformed y_pred envelope value."""
    for y_pred in ({"y_pred": []}, {"y_pred": 42}):
        with pytest.raises(ValueError, match="y_pred.y_pred must be an object"):
            parse_calls_y_pred(y_pred)


def test_calls_parser_rejects_non_object_arguments() -> None:
    """Phase 3 y_pred parsing rejects arguments that are not an object."""
    with pytest.raises(ValueError, match="arguments"):
        parse_calls_y_pred({
            "calls": [{
                "rest_api": "/redfish/v1/Systems",
                "method": "GET",
                "arguments": ["PowerState", "On"],
            }],
        })


def test_calls_parser_rejects_readonly_arguments() -> None:
    """Phase 3 y_pred parsing rejects non-empty GET/HEAD argument objects."""
    for method in ("GET", "HEAD"):
        with pytest.raises(ValueError, match="read-only"):
            parse_calls_y_pred({
                "calls": [{
                    "rest_api": "/redfish/v1/Systems",
                    "method": method,
                    "arguments": {"PowerState": "On"},
                }],
            })


def _two_call_row() -> dict:
    """A two-GET Phase 3 fixture row shared by the evaluator tests."""
    systems = _context(
        "/redfish/v1/Systems",
        ("GET", "HEAD"),
        {"@odata.id": "/redfish/v1/Systems"},
    )
    tasks = _context(
        "/redfish/v1/TaskService/Tasks",
        ("GET", "HEAD"),
        {"@odata.id": "/redfish/v1/TaskService/Tasks"},
    )
    return build_call_row(
        text="check tasks and list systems",
        contexts=(systems, tasks),
        rest_api_list=("/redfish/v1/TaskService/Tasks", "/redfish/v1/Systems"),
        method_by_api={
            "/redfish/v1/TaskService/Tasks": "GET",
            "/redfish/v1/Systems": "GET",
        },
    )


def test_call_evaluation_ignores_order() -> None:
    """A reordered but otherwise identical prediction is a full set match."""
    row = _two_call_row()
    reordered = list(reversed(row["y_true"]["calls"]))

    evaluation = evaluate_calls_y_pred(row, {"calls": reordered})

    assert evaluation["parsed"] is True
    assert evaluation["duplicate_prediction"] is False
    assert evaluation["call_set_exact_match"] is True
    assert evaluation["call_set_exact_match_rate"] == 1.0
    assert evaluation["rest_api_set_match_rate"] == 1.0
    assert evaluation["method_exact_match_rate"] == 1.0
    assert evaluation["invalid_method_rate"] == 0.0


def test_call_evaluation_rejects_duplicate_predictions() -> None:
    """A duplicated predicted call fails the set match — no dedup forgiveness."""
    row = _two_call_row()
    predicted_calls = list(row["y_true"]["calls"]) + [row["y_true"]["calls"][0]]

    evaluation = evaluate_calls_y_pred(row, {"calls": predicted_calls})

    assert evaluation["parsed"] is True
    assert evaluation["expected_call_count"] == 2
    assert evaluation["predicted_call_count"] == 3
    assert evaluation["call_count_match"] is False
    assert evaluation["duplicate_prediction"] is True
    assert evaluation["call_set_exact_match"] is False
    assert evaluation["call_set_exact_match_rate"] == 0.0
    assert evaluation["rest_api_set_match_rate"] == 0.0


def test_call_evaluation_counts_distinct_extra_predictions_as_failures() -> None:
    """A distinct extra predicted call fails the set match without hiding shared hits."""
    row = _two_call_row()
    predicted_calls = list(row["y_true"]["calls"]) + [{
        "rest_api": "/redfish/v1/Chassis",
        "method": "GET",
        "arguments": {},
    }]

    evaluation = evaluate_calls_y_pred(row, {"calls": predicted_calls})

    assert evaluation["parsed"] is True
    assert evaluation["duplicate_prediction"] is False
    assert evaluation["call_set_exact_match"] is False
    assert evaluation["rest_api_set_match_rate"] == 0.0
    # Shared-API agreement is still visible: both expected calls matched.
    assert evaluation["method_exact_match_rate"] == pytest.approx(2 / 3)
    assert evaluation["arguments_exact_match_rate"] == pytest.approx(2 / 3)


def test_call_evaluation_reports_invalid_method_via_row_evidence() -> None:
    """A method illegal for the row's allowed_methods is an invalid-method failure."""
    context = _context(
        "/redfish/v1/Systems",
        ("GET", "HEAD"),
        {"@odata.id": "/redfish/v1/Systems"},
    )
    row = build_call_row(
        text="list systems",
        contexts=(context,),
        rest_api_list=("/redfish/v1/Systems",),
        method_by_api={"/redfish/v1/Systems": "GET"},
    )

    evaluation = evaluate_calls_y_pred(
        row,
        {
            "calls": [{
                "rest_api": "/redfish/v1/Systems",
                "method": "PATCH",
                "arguments": {"PowerState": "On"},
            }],
        },
    )

    # The prediction parses (structurally valid JSON), but the method is not
    # legal for this API in the row's evidence, and the arguments are unsafe.
    assert evaluation["parsed"] is True
    assert evaluation["arguments_json_validity_rate"] == 1.0
    assert evaluation["invalid_method_rate"] == 1.0
    assert evaluation["call_set_exact_match"] is False
    assert evaluation["method_exact_match_rate"] == 0.0
    assert evaluation["unsafe_argument_rejection_rate"] == 0.0


def test_call_evaluation_scores_required_and_no_argument_calls() -> None:
    """Required-arg coverage, no-arg accuracy, and unsafe-arg rejection score per call."""
    bios = _context(
        "/redfish/v1/Systems/1/Bios/Settings",
        ("GET", "PATCH"),
        {"@odata.id": "/redfish/v1/Systems/1/Bios/Settings"},
    )
    tasks = _context(
        "/redfish/v1/TaskService/Tasks",
        ("GET", "HEAD"),
        {"@odata.id": "/redfish/v1/TaskService/Tasks"},
    )
    row = build_call_row(
        text="set bios boot mode to Uefi and check tasks",
        contexts=(bios, tasks),
        rest_api_list=(
            "/redfish/v1/Systems/1/Bios/Settings",
            "/redfish/v1/TaskService/Tasks",
        ),
        method_by_api={
            "/redfish/v1/Systems/1/Bios/Settings": "PATCH",
            "/redfish/v1/TaskService/Tasks": "GET",
        },
        arguments_by_api={
            "/redfish/v1/Systems/1/Bios/Settings": {
                "Attributes": {"BootMode": "Uefi"},
            },
        },
    )

    exact = evaluate_calls_y_pred(row, {"calls": row["y_true"]["calls"]})
    assert exact["call_set_exact_match"] is True
    assert exact["required_argument_coverage_rate"] == 1.0
    assert exact["no_argument_accuracy_rate"] == 1.0
    assert exact["unsafe_argument_rejection_rate"] == 1.0

    # Missing the required PATCH argument key: coverage drops, no unsafe injection.
    missing_required = evaluate_calls_y_pred(row, {"calls": [
        {
            "rest_api": "/redfish/v1/Systems/1/Bios/Settings",
            "method": "PATCH",
            "arguments": {},
        },
        {
            "rest_api": "/redfish/v1/TaskService/Tasks",
            "method": "GET",
            "arguments": {},
        },
    ]})
    assert missing_required["required_argument_coverage_rate"] == 0.0
    assert missing_required["no_argument_accuracy_rate"] == 1.0
    assert missing_required["unsafe_argument_rejection_rate"] == 1.0
    assert missing_required["call_set_exact_match"] is False

    # Injecting an unsupported argument key: unsafe rejection drops.
    unsafe = evaluate_calls_y_pred(row, {"calls": [
        {
            "rest_api": "/redfish/v1/Systems/1/Bios/Settings",
            "method": "PATCH",
            "arguments": {
                "Attributes": {"BootMode": "Uefi"},
                "UnsupportedKnob": True,
            },
        },
        {
            "rest_api": "/redfish/v1/TaskService/Tasks",
            "method": "GET",
            "arguments": {},
        },
    ]})
    assert unsafe["required_argument_coverage_rate"] == 1.0
    assert unsafe["unsafe_argument_rejection_rate"] == pytest.approx(1 / 2)
    assert unsafe["call_set_exact_match"] is False


def test_call_evaluation_empty_set_matches_empty_prediction() -> None:
    """A hard-negative row ([] expected) matches only an empty prediction."""
    row = build_call_row(
        text="nothing to do",
        contexts=(),
        rest_api_list=(),
        method_by_api={},
    )

    empty = evaluate_calls_y_pred(row, {"calls": []})
    assert empty["call_set_exact_match"] is True
    assert empty["rest_api_set_match_rate"] == 1.0

    nonempty = evaluate_calls_y_pred(row, {"calls": [{
        "rest_api": "/redfish/v1/Systems",
        "method": "GET",
        "arguments": {},
    }]})
    assert nonempty["call_set_exact_match"] is False
    assert nonempty["rest_api_set_match_rate"] == 0.0


def test_wandb_metric_keys_are_stage_scoped_and_not_m3_names() -> None:
    """Phase 2/3 contract constants reuse the shared W&B metric registry."""
    expected_phase2 = (
        "phase2_goal_extraction/train/loss",
        "phase2_goal_extraction/train/perplexity",
        "phase2_goal_extraction/train/optimizer_step",
        "phase2_goal_extraction/eval/set_match_rate",
        "phase2_goal_extraction/eval/precision",
        "phase2_goal_extraction/eval/recall",
        "phase2_goal_extraction/eval/f1",
        "phase2_goal_extraction/eval/invalid_rest_rate",
        "phase2_goal_extraction/eval/hard_negative_accuracy",
    )
    expected_phase3 = (
        "phase3_argument_extraction/train/loss",
        "phase3_argument_extraction/train/perplexity",
        "phase3_argument_extraction/train/optimizer_step",
        "phase3_argument_extraction/eval/call_set_exact_match_rate",
        "phase3_argument_extraction/eval/method_exact_match_rate",
        "phase3_argument_extraction/eval/arguments_json_validity_rate",
        "phase3_argument_extraction/eval/arguments_exact_match_rate",
        "phase3_argument_extraction/eval/required_argument_coverage_rate",
        "phase3_argument_extraction/eval/no_argument_accuracy_rate",
        "phase3_argument_extraction/eval/unsafe_argument_rejection_rate",
    )

    assert PHASE2_GOAL_EXTRACT_METRIC_KEYS == PHASE2_WANDB_METRIC_KEYS
    assert PHASE3_ARGUMENT_EXTRACT_METRIC_KEYS == PHASE3_WANDB_METRIC_KEYS
    assert PHASE2_GOAL_EXTRACT_METRIC_KEYS == expected_phase2
    assert PHASE3_ARGUMENT_EXTRACT_METRIC_KEYS == expected_phase3
    assert (
        PHASE2_GOAL_EXTRACT_METRIC_KEYS + PHASE3_ARGUMENT_EXTRACT_METRIC_KEYS
        == PHASE23_WANDB_METRIC_KEYS
    )
    assert len(PHASE2_GOAL_EXTRACT_METRIC_KEYS) == len(set(PHASE2_GOAL_EXTRACT_METRIC_KEYS))
    assert len(PHASE3_ARGUMENT_EXTRACT_METRIC_KEYS) == len(set(PHASE3_ARGUMENT_EXTRACT_METRIC_KEYS))
    assert "phase2_goal_extraction/eval/set_match_rate" in PHASE2_GOAL_EXTRACT_METRIC_KEYS
    assert (
        "phase3_argument_extraction/eval/call_set_exact_match_rate"
        in PHASE3_ARGUMENT_EXTRACT_METRIC_KEYS
    )
    assert (
        "phase3_argument_extraction/eval/arguments_exact_match_rate"
        in PHASE3_ARGUMENT_EXTRACT_METRIC_KEYS
    )
    # Phase 2/3 are unordered tasks: no order/* namespace, no ordered-* metric.
    assert not any("/order/" in k for k in PHASE23_WANDB_METRIC_KEYS)
    assert not any("ordered" in k for k in PHASE23_WANDB_METRIC_KEYS)
    assert all(k.startswith("phase2_goal_extraction/") for k in PHASE2_GOAL_EXTRACT_METRIC_KEYS)
    assert all(k.startswith("phase3_argument_extraction/") for k in PHASE3_ARGUMENT_EXTRACT_METRIC_KEYS)
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
    assert "phase2_goal_extraction/eval/{set_match_rate,precision" in training_doc
    assert "phase3_argument_extraction/eval/{call_set_exact_match_rate" in training_doc
    # The unordered contract leaves no order/* keys in the training docs.
    assert "order/{kendall_tau,edit_distance}" not in training_doc
    for metric_key in PHASE2_GOAL_EXTRACT_METRIC_KEYS:
        assert metric_key in training_doc
    for metric_key in PHASE3_ARGUMENT_EXTRACT_METRIC_KEYS:
        assert metric_key in training_doc


# Author: Mus mbayramo@stanford.edu
