"""Canonical Phase 2 REST-set and Phase 3 call-set contract tests."""

from __future__ import annotations

import json

import pytest

from igc.ds.rest_goal_contract import (
    CALL_FIELDS,
    RedfishContext,
    build_call_row,
    build_d1_master_record,
    build_d1_rest_api_list_row,
    d1_row_id,
    evaluate_calls_y_pred,
    evaluate_rest_api_set,
    inference_calls_json,
    parse_calls_y_pred,
    parse_rest_api_list_y_pred,
    render_call_example,
    render_d1_master_views,
    render_rest_api_list_example,
)


SYSTEM = "/redfish/v1/Systems/1"
BIOS = f"{SYSTEM}/Bios/Settings"


def _context(
    rest_api: str,
    methods: tuple[str, ...] = ("GET",),
    *,
    operations: tuple[str, ...] = ("get_resource",),
    argument_schema: dict | None = None,
) -> RedfishContext:
    return RedfishContext(
        rest_api=rest_api,
        allowed_methods=methods,
        operation_names=operations,
        argument_schema={} if argument_schema is None else argument_schema,
        json={"@odata.id": rest_api, "Name": "fixture"},
    )


def _phase2_catalog(*targets: RedfishContext) -> tuple[RedfishContext, ...]:
    distractors = tuple(
        _context(f"/redfish/v1/Managers/{index}")
        for index in range(4)
    )
    return (*targets, *distractors)


def _phase3_row() -> dict:
    return build_call_row(
        text="read the system and set boot mode to Uefi",
        contexts=(
            _context(SYSTEM),
            _context(
                BIOS,
                ("GET", "PATCH"),
                operations=("set_bios_attributes",),
                argument_schema={"Attributes": {"BootMode": "string"}},
            ),
        ),
        rest_api_list=(BIOS, SYSTEM),
        method_by_api={SYSTEM: "GET", BIOS: "PATCH"},
        operation_name_by_api={
            SYSTEM: "get_resource",
            BIOS: "set_bios_attributes",
        },
        arguments_by_api={
            SYSTEM: {},
            BIOS: {"Attributes": {"BootMode": "Uefi"}},
        },
    )


def test_phase2_row_is_unordered_and_target_free_in_x() -> None:
    row = build_d1_rest_api_list_row(
        text="read bios and system",
        contexts=_phase2_catalog(_context(BIOS), _context(SYSTEM)),
        rest_api_list=(SYSTEM, BIOS),
    )
    assert row["phase"] == 2
    assert row["dataset"] == "D1"
    assert row["source_dataset"] == "D0"
    assert row["target_semantics"] == "unordered_unique_rest_api_set"
    assert row["x"].keys() == {"text", "api_context"}
    assert "rest_api_list" not in row["x"]
    assert row["y_true"] == {"rest_api_list": sorted((SYSTEM, BIOS))}
    context_apis = {context["rest_api"] for context in row["x"]["api_context"]}
    assert {SYSTEM, BIOS} <= context_apis
    assert len(context_apis - {SYSTEM, BIOS}) >= 4
    assert all("selected" not in context for context in row["x"]["api_context"])


def test_phase2_context_policy_requires_all_targets_and_four_hidden_distractors() -> None:
    """Training rows must include every target plus at least four unmarked distractors."""
    target = _context(SYSTEM)
    with pytest.raises(ValueError, match="present in current context"):
        build_d1_rest_api_list_row(
            text="read the system",
            contexts=_phase2_catalog(),
            rest_api_list=(SYSTEM,),
        )

    with pytest.raises(ValueError, match="at least 4 distractors"):
        build_d1_rest_api_list_row(
            text="read the system",
            contexts=(target, _context("/redfish/v1/Managers/0")),
            rest_api_list=(SYSTEM,),
        )

    row = build_d1_rest_api_list_row(
        text="read the system",
        contexts=_phase2_catalog(target),
        rest_api_list=(SYSTEM,),
    )
    row["x"]["api_context"][0]["selected"] = True
    with pytest.raises(ValueError, match="Redfish context"):
        render_rest_api_list_example(row)

    row = build_d1_rest_api_list_row(
        text="read the system",
        contexts=_phase2_catalog(target),
        rest_api_list=(SYSTEM,),
    )
    row["x"]["target_indices"] = [0]
    with pytest.raises(ValueError, match="Phase 2 x"):
        render_rest_api_list_example(row)


def test_phase2_renderer_rejects_target_leakage() -> None:
    row = build_d1_rest_api_list_row(
        text="read the system",
        contexts=_phase2_catalog(_context(SYSTEM)),
        rest_api_list=(SYSTEM,),
    )
    row["x"]["rest_api_list"] = [SYSTEM]
    with pytest.raises(ValueError, match="Phase 2 x"):
        render_rest_api_list_example(row)


@pytest.mark.parametrize(
    "prediction",
    [
        {"rest_api": SYSTEM},
        {"rest_api_list": SYSTEM},
        {"rest_api_list": [SYSTEM, SYSTEM]},
        {"ordered_goals": [SYSTEM]},
        {"rest_api_list": [SYSTEM], "extra": True},
    ],
)
def test_phase2_parser_rejects_scalar_alias_duplicate_and_extra_shapes(prediction) -> None:
    with pytest.raises(ValueError):
        parse_rest_api_list_y_pred(prediction)


def test_phase2_empty_set_equals_empty_set() -> None:
    assert evaluate_rest_api_set([], []) == {
        "set_exact_match": True,
        "set_match_rate": 1.0,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "expected_count": 0,
        "predicted_count": 0,
    }


@pytest.mark.parametrize(
    "expected",
    [
        ["/redfish/v1/Systems/1"],
        ["/redfish/v1/Systems/1", "/redfish/v1/Managers/1"],
        ["/redfish/v1/Systems/1", "/redfish/v1/Managers/1", "/redfish/v1/Chassis/1"],
    ],
)
def test_phase2_set_metrics_ignore_permutation_and_penalize_distractors(
    expected: list[str],
) -> None:
    """Promotion metrics are object-shaped and set-based for k=1, k=2, and k=3."""
    exact = evaluate_rest_api_set(expected, list(reversed(expected)))
    with_distractor = evaluate_rest_api_set(
        expected,
        [*reversed(expected), "/redfish/v1/Unexpected"],
    )
    missing_target = evaluate_rest_api_set(expected, expected[:-1])

    assert parse_rest_api_list_y_pred({"rest_api_list": list(reversed(expected))}) == sorted(
        expected,
    )
    assert exact["set_exact_match"] is True
    assert exact["set_match_rate"] == 1.0
    assert exact["precision"] == 1.0
    assert exact["recall"] == 1.0
    assert with_distractor["set_exact_match"] is False
    assert with_distractor["precision"] < 1.0
    assert with_distractor["recall"] == 1.0
    assert missing_target["set_exact_match"] is False
    expected_precision = 1.0 if expected[:-1] else 0.0
    assert missing_target["precision"] == expected_precision
    assert missing_target["recall"] < 1.0


def test_phase2_render_target_is_only_the_api_list() -> None:
    row = build_d1_rest_api_list_row(
        text="read the system",
        contexts=_phase2_catalog(_context(SYSTEM)),
        rest_api_list=(SYSTEM,),
    )
    rendered = render_rest_api_list_example(row)
    assert json.loads(rendered.target_json) == {"rest_api_list": [SYSTEM]}
    assert rendered.full_text == rendered.prompt + rendered.target_json
    assert d1_row_id(row).startswith("sha256:")


def test_phase3_requires_explicit_method_operation_and_argument_maps() -> None:
    kwargs = {
        "text": "read the system",
        "contexts": (_context(SYSTEM),),
        "rest_api_list": (SYSTEM,),
        "method_by_api": {SYSTEM: "GET"},
        "operation_name_by_api": {SYSTEM: "get_resource"},
        "arguments_by_api": {SYSTEM: {}},
    }
    for missing in ("method_by_api", "operation_name_by_api", "arguments_by_api"):
        candidate = dict(kwargs)
        candidate[missing] = {}
        with pytest.raises(ValueError, match="keys must exactly match"):
            build_call_row(**candidate)


def test_phase3_rejects_implicit_mutation_arguments_and_read_arguments() -> None:
    context = _context(
        BIOS,
        ("GET", "PATCH"),
        operations=("set_bios_attributes",),
        argument_schema={"Attributes": {"BootMode": "string"}},
    )
    with pytest.raises(ValueError, match="keys must exactly match"):
        build_call_row(
            text="set boot mode",
            contexts=(context,),
            rest_api_list=(BIOS,),
            method_by_api={BIOS: "PATCH"},
            operation_name_by_api={BIOS: "set_bios_attributes"},
            arguments_by_api={},
        )
    with pytest.raises(ValueError, match="GET call arguments must be empty"):
        build_call_row(
            text="read bios",
            contexts=(context,),
            rest_api_list=(BIOS,),
            method_by_api={BIOS: "GET"},
            operation_name_by_api={BIOS: None},
            arguments_by_api={BIOS: {"unexpected": True}},
        )


def test_phase3_build_call_row_rejects_operation_absent_from_context() -> None:
    """Phase 3 labels cannot name operations outside the public API context."""
    context = _context(
        BIOS,
        ("PATCH",),
        operations=("set_bios_attributes",),
        argument_schema={"Attributes": {"BootMode": "string"}},
    )

    with pytest.raises(ValueError, match="is not declared"):
        build_call_row(
            text="set boot mode",
            contexts=(context,),
            rest_api_list=(BIOS,),
            method_by_api={BIOS: "PATCH"},
            operation_name_by_api={BIOS: "delete_bios"},
            arguments_by_api={BIOS: {"Attributes": {"BootMode": "Uefi"}}},
        )


def test_phase3_build_call_row_rejects_arguments_that_violate_context_schema() -> None:
    """Phase 3 mutation arguments must satisfy the context argument schema."""
    context = _context(
        BIOS,
        ("PATCH",),
        operations=("set_bios_attributes",),
        argument_schema={"Attributes": {"BootMode": "string"}},
    )

    with pytest.raises(ValueError, match="arguments do not match schema"):
        build_call_row(
            text="set boot mode",
            contexts=(context,),
            rest_api_list=(BIOS,),
            method_by_api={BIOS: "PATCH"},
            operation_name_by_api={BIOS: "set_bios_attributes"},
            arguments_by_api={BIOS: {"Attributes": {"BootMode": 7}}},
        )


def test_phase3_call_set_has_exact_fields_and_preserves_values() -> None:
    row = _phase3_row()
    assert row["source_dataset"] == "D1"
    assert row["target_semantics"] == "unordered_unique_call_set"
    calls = row["y_true"]["calls"]
    assert all(set(call) == CALL_FIELDS for call in calls)
    bios = next(call for call in calls if call["rest_api"] == BIOS)
    assert bios["http_method"] == "PATCH"
    assert bios["operation_name"] == "set_bios_attributes"
    assert bios["arguments"] == {"Attributes": {"BootMode": "Uefi"}}


def test_phase3_reordered_prediction_is_exact() -> None:
    row = _phase3_row()
    prediction = {"calls": list(reversed(row["y_true"]["calls"]))}
    evaluation = evaluate_calls_y_pred(row, prediction)
    assert evaluation["parsed"] is True
    assert evaluation["accepted"] is True
    assert evaluation["call_set_exact_match_rate"] == 1.0
    assert evaluation["rest_api_set_match_rate"] == 1.0


@pytest.mark.parametrize(
    "prediction",
    [
        {"call": {}},
        {"calls": {}},
        {"ordered_goals": []},
        {"calls": [{"rest_api": SYSTEM}]},
        {"calls": [{
            "rest_api": SYSTEM,
            "http_method": "GET",
            "operation_name": "get_resource",
            "arguments": {},
            "allowed_methods": ["GET"],
        }]},
    ],
)
def test_phase3_parser_rejects_scalar_alias_and_extra_shapes(prediction) -> None:
    with pytest.raises(ValueError):
        parse_calls_y_pred(prediction)


def test_phase3_parser_rejects_method_not_allowed_by_context() -> None:
    prediction = {"calls": [{
        "rest_api": SYSTEM,
        "http_method": "PATCH",
        "operation_name": "set_asset_tag",
        "arguments": {"AssetTag": "rack-7"},
    }]}
    with pytest.raises(ValueError, match="not allowed"):
        parse_calls_y_pred(prediction, contexts=(_context(SYSTEM),))


def test_phase3_parser_rejects_operation_absent_from_context() -> None:
    """Context-aware prediction parsing rejects undeclared operation names."""
    prediction = {"calls": [{
        "rest_api": BIOS,
        "http_method": "PATCH",
        "operation_name": "delete_bios",
        "arguments": {"Attributes": {"BootMode": "Uefi"}},
    }]}
    context = _context(
        BIOS,
        ("PATCH",),
        operations=("set_bios_attributes",),
        argument_schema={"Attributes": {"BootMode": "string"}},
    )

    with pytest.raises(ValueError, match="is not declared"):
        parse_calls_y_pred(prediction, contexts=(context,))


def test_phase3_parser_rejects_arguments_that_violate_context_schema() -> None:
    """Context-aware prediction parsing rejects malformed mutation arguments."""
    prediction = {"calls": [{
        "rest_api": BIOS,
        "http_method": "PATCH",
        "operation_name": "set_bios_attributes",
        "arguments": {"Attributes": {"BootMode": 7}},
    }]}
    context = _context(
        BIOS,
        ("PATCH",),
        operations=("set_bios_attributes",),
        argument_schema={"Attributes": {"BootMode": "string"}},
    )

    with pytest.raises(ValueError, match="arguments do not match schema"):
        parse_calls_y_pred(prediction, contexts=(context,))


def test_phase3_render_and_inference_use_calls_only() -> None:
    row = _phase3_row()
    rendered = render_call_example(row)
    assert json.loads(rendered.target_json) == {"calls": row["y_true"]["calls"]}
    assert inference_calls_json(row) == {"calls": row["y_true"]["calls"]}


def _validation_for(targets: tuple[str, ...]) -> dict[str, object]:
    """Strict accepted Phase 2 judge evidence for a target API set."""
    return {
        "valid_json": True,
        "accepted": True,
        "natural": True,
        "nonsense": False,
        "ambiguous": False,
        "duplicate_intent": False,
        "extra_intents": False,
        "method_semantics_valid": True,
        "covered_api_set": sorted(targets),
    }


def _master_contexts() -> tuple[RedfishContext, ...]:
    """Phase 2 context catalog with selected APIs plus four hidden distractors."""
    return _phase2_catalog(
        _context(SYSTEM, ("GET",), operations=("get_resource",)),
        _context(
            BIOS,
            ("GET", "PATCH"),
            operations=("set_bios_attributes",),
            argument_schema={
                "properties": {
                    "Attributes": {
                        "type": "object",
                        "properties": {"BootMode": {"type": "string"}},
                        "required": ["BootMode"],
                    },
                },
                "required": ["Attributes"],
            },
        ),
    )


def _master_metadata(*, text: str, contexts: tuple[RedfishContext, ...]) -> dict[str, object]:
    """Metadata with row_id derived from the matching Phase 2 D1 view."""
    phase2 = build_d1_rest_api_list_row(
        text=text,
        contexts=contexts,
        rest_api_list=(SYSTEM, BIOS),
        validation=_validation_for((SYSTEM, BIOS)),
    )
    return {
        "row_id": d1_row_id(phase2),
        "sample_width_k": 2,
        "vendor": ["unit"],
        "source_corpus": ["unit-fixture"],
    }


def _master_label_kwargs(
    *,
    grounding: dict[str, dict[str, object]] | None = None,
) -> dict[str, object]:
    """Return one fully grounded D1 master-label fixture."""
    text = "read the system and set BIOS boot mode to Uefi"
    contexts = _master_contexts()
    return {
        "text": text,
        "contexts": contexts,
        "method_by_api": {SYSTEM: "GET", BIOS: "PATCH"},
        "operation_name_by_api": {
            SYSTEM: "get_resource",
            BIOS: "set_bios_attributes",
        },
        "arguments_by_api": {
            SYSTEM: {},
            BIOS: {"Attributes": {"BootMode": "Uefi"}},
        },
        "argument_value_grounding_by_api": grounding
        or {
            SYSTEM: {"grounded": False, "sources": []},
            BIOS: {"grounded": True, "sources": ["argument_schema", "operator_text"]},
        },
        "validation": _validation_for((SYSTEM, BIOS)),
        "metadata": _master_metadata(text=text, contexts=contexts),
    }


def test_d1_master_record_renders_exact_phase2_phase3_views() -> None:
    """One master record produces strict Phase 2/3 views over the same API set."""
    master = build_d1_master_record(**_master_label_kwargs())

    phase2, phase3 = render_d1_master_views(master)

    assert master["schema_version"] == "d1_master.v1"
    assert phase2["target_semantics"] == "unordered_unique_rest_api_set"
    assert phase3["target_semantics"] == "unordered_unique_call_set"
    assert phase2["metadata"]["row_id"] == d1_row_id(phase2)
    assert phase3["metadata"]["row_id"] == phase2["metadata"]["row_id"]
    phase2_apis = set(phase2["y_true"]["rest_api_list"])
    phase3_apis = {call["rest_api"] for call in phase3["y_true"]["calls"]}
    assert phase2_apis == {SYSTEM, BIOS}
    assert phase3_apis == phase2_apis
    bios_call = next(
        call for call in phase3["y_true"]["calls"]
        if call["rest_api"] == BIOS
    )
    assert bios_call["http_method"] == "PATCH"
    assert bios_call["operation_name"] == "set_bios_attributes"
    assert bios_call["arguments"] == {"Attributes": {"BootMode": "Uefi"}}
    assert render_rest_api_list_example(phase2).target_json
    assert render_call_example(phase3).target_json


def test_d1_master_record_accepts_operation_definition_grounding_alternative() -> None:
    """Mutation arguments may use operator text plus operation-definition evidence."""
    master = build_d1_master_record(
        **_master_label_kwargs(
            grounding={
                SYSTEM: {"grounded": False, "sources": []},
                BIOS: {
                    "grounded": True,
                    "sources": ["operator_text", "operation_definition"],
                },
            },
        ),
    )

    assert master["label_evidence"]["argument_value_grounding_by_api"][BIOS] == {
        "grounded": True,
        "sources": ["operation_definition", "operator_text"],
    }


def test_d1_master_views_reject_api_set_drift() -> None:
    """Phase 2/3 API sets cannot diverge under a reused D1 row identity."""
    master = build_d1_master_record(**_master_label_kwargs())
    master["calls"] = [
        call for call in master["calls"]
        if call["rest_api"] != BIOS
    ]

    with pytest.raises(ValueError, match="metadata.row_id"):
        render_d1_master_views(master)


@pytest.mark.parametrize(
    ("grounding", "message"),
    [
        (
            {
                SYSTEM: {"grounded": False, "sources": []},
                BIOS: {"grounded": True, "sources": ["current_json"]},
            },
            "unsupported sources",
        ),
        (
            {
                SYSTEM: {"grounded": False, "sources": []},
                BIOS: {"grounded": True, "sources": ["operator_text", "fixture"]},
            },
            "unsupported sources",
        ),
        (
            {
                SYSTEM: {"grounded": False, "sources": []},
                BIOS: {"grounded": True, "sources": ["operator_text"]},
            },
            "operator_text plus argument_schema",
        ),
        (
            {
                SYSTEM: {"grounded": False, "sources": []},
                BIOS: {"grounded": False, "sources": []},
            },
            "positive grounding",
        ),
        (
            {
                SYSTEM: {"grounded": False, "sources": []},
            },
            "grounding keys",
        ),
    ],
)
def test_d1_master_record_rejects_current_json_and_missing_grounding(
    grounding: dict[str, dict[str, object]],
    message: str,
) -> None:
    """Mutation values require text plus schema/action evidence from known sources."""
    with pytest.raises(ValueError, match=message):
        build_d1_master_record(**_master_label_kwargs(grounding=grounding))
