#!/usr/bin/env python3
"""Fail when Phase 2/3 regress to superseded scalar, ordered, or defaulted shapes."""

from __future__ import annotations

import copy
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from igc.ds.rest_goal_contract import (  # noqa: E402
    RedfishContext,
    build_call_row,
    build_d1_rest_api_list_row,
    evaluate_calls,
    parse_calls_y_pred,
    parse_rest_api_list_y_pred,
    render_call_example,
    render_rest_api_list_example,
)
from igc.modules.base.metric_keys import (  # noqa: E402
    PHASE2_WANDB_METRIC_KEYS,
    PHASE3_WANDB_METRIC_KEYS,
)


def _must_reject(label: str, operation: Callable[[], Any], failures: list[str]) -> None:
    try:
        operation()
    except (TypeError, ValueError):
        return
    failures.append(f"{label}: invalid shape was accepted")


def run_gate() -> list[str]:
    """Return contract violations; an empty list is a pass."""
    failures: list[str] = []
    root = Path(__file__).resolve().parents[2]
    rest_goal_spec = _load_yaml(root / "configs/contracts/rest_goal.yaml")
    d1_spec = _load_yaml(root / "configs/contracts/d1_contract.yaml")
    _check_machine_contracts(rest_goal_spec, d1_spec, failures)
    context = RedfishContext(
        rest_api="/redfish/v1/Systems/1",
        allowed_methods=("GET", "PATCH"),
        operation_names=("get_system", "set_asset_tag"),
        argument_schema={"AssetTag": "string"},
        json={"@odata.id": "/redfish/v1/Systems/1", "AssetTag": "old"},
    )
    distractors = tuple(
        RedfishContext(
            rest_api=f"/redfish/v1/Managers/{index}",
            allowed_methods=("GET",),
            operation_names=("get_manager",),
            json={"@odata.id": f"/redfish/v1/Managers/{index}"},
        )
        for index in range(4)
    )
    phase2 = build_d1_rest_api_list_row(
        text="read the system",
        contexts=(context, *distractors),
        rest_api_list=(context.rest_api,),
    )

    _must_reject(
        "scalar rest_api",
        lambda: parse_rest_api_list_y_pred({"rest_api": context.rest_api}),
        failures,
    )
    _must_reject(
        "scalar rest_api_list",
        lambda: parse_rest_api_list_y_pred({"rest_api_list": context.rest_api}),
        failures,
    )
    _must_reject(
        "scalar call",
        lambda: parse_calls_y_pred({"call": {}}),
        failures,
    )
    _must_reject(
        "scalar calls",
        lambda: parse_calls_y_pred({"calls": {}}),
        failures,
    )
    _must_reject(
        "ordered_goals alias",
        lambda: parse_calls_y_pred({"ordered_goals": []}),
        failures,
    )

    leaked = copy.deepcopy(phase2)
    leaked["x"]["rest_api_list"] = [context.rest_api]
    _must_reject(
        "Phase 2 target copied into x",
        lambda: render_rest_api_list_example(leaked),
        failures,
    )
    only_target = copy.deepcopy(phase2)
    only_target["x"]["api_context"] = [only_target["x"]["api_context"][0]]
    _must_reject(
        "Phase 2 context contains only target APIs",
        lambda: render_rest_api_list_example(only_target),
        failures,
    )
    selected = copy.deepcopy(phase2)
    selected["x"]["api_context"][0]["selected"] = True
    _must_reject(
        "Phase 2 selected membership metadata",
        lambda: render_rest_api_list_example(selected),
        failures,
    )
    target_indices = copy.deepcopy(phase2)
    target_indices["x"]["target_indices"] = [0]
    _must_reject(
        "Phase 2 target indices",
        lambda: render_rest_api_list_example(target_indices),
        failures,
    )
    prompt = render_rest_api_list_example(phase2).prompt
    if '"rest_api_list"' in prompt:
        failures.append("Phase 2 target list field was rendered in the prompt")
    _must_reject(
        "implicit GET default",
        lambda: build_call_row(
            text="read the system",
            contexts=(context,),
            rest_api_list=(context.rest_api,),
            method_by_api={},
            operation_name_by_api={context.rest_api: "get_system"},
            arguments_by_api={context.rest_api: {}},
        ),
        failures,
    )
    _must_reject(
        "implicit empty mutation arguments",
        lambda: build_call_row(
            text="set the asset tag to rack-7",
            contexts=(context,),
            rest_api_list=(context.rest_api,),
            method_by_api={context.rest_api: "PATCH"},
            operation_name_by_api={context.rest_api: "set_asset_tag"},
            arguments_by_api={},
        ),
        failures,
    )

    phase3 = build_call_row(
        text="set the asset tag and read the manager",
        contexts=(context, distractors[0]),
        rest_api_list=(context.rest_api, distractors[0].rest_api),
        method_by_api={
            context.rest_api: "PATCH",
            distractors[0].rest_api: "GET",
        },
        operation_name_by_api={
            context.rest_api: "set_asset_tag",
            distractors[0].rest_api: None,
        },
        arguments_by_api={
            context.rest_api: {"AssetTag": "rack-7"},
            distractors[0].rest_api: {},
        },
    )
    rendered = render_call_example(phase3)
    parsed = parse_calls_y_pred(phase3["y_true"])
    reversed_calls = list(reversed(parsed))
    if not evaluate_calls(parsed, reversed_calls)["call_set_exact_match"]:
        failures.append("Phase 3 call order changed unordered set equality")
    if "allowed_methods" in json.loads(rendered.target_json)["calls"][0]:
        failures.append("Phase 3 output leaked input-only allowed_methods")

    missing_operation = copy.deepcopy(phase3["y_true"])
    del missing_operation["calls"][0]["operation_name"]
    _must_reject(
        "implicit operation_name",
        lambda: parse_calls_y_pred(missing_operation),
        failures,
    )
    missing_call = copy.deepcopy(phase3)
    missing_call["y_true"]["calls"] = missing_call["y_true"]["calls"][:1]
    _must_reject(
        "missing Phase 3 call",
        lambda: render_call_example(missing_call),
        failures,
    )

    metric_keys = PHASE2_WANDB_METRIC_KEYS + PHASE3_WANDB_METRIC_KEYS
    forbidden_metric_terms = ("ordered", "kendall", "edit_distance")
    for metric in metric_keys:
        if any(term in metric for term in forbidden_metric_terms):
            failures.append(f"ordered Phase 2/3 metric remains: {metric}")

    return failures


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML object")
    return value


def _expect(
    failures: list[str],
    label: str,
    observed: Any,
    expected: Any,
) -> None:
    if observed != expected:
        failures.append(f"{label}: expected {expected!r}, got {observed!r}")


def _check_machine_contracts(
    rest_goal: dict[str, Any],
    d1: dict[str, Any],
    failures: list[str],
) -> None:
    authority = rest_goal.get("authority", {})
    _expect(failures, "Phase 2 authority", authority.get("phase2_output"), "rest_api_list")
    _expect(failures, "Phase 3 authority", authority.get("phase3_output"), "calls")
    _expect(failures, "order authority", authority.get("order_semantics"), "unordered")

    phase2 = rest_goal.get("phase2", {})
    _expect(failures, "Phase 2 input fields", phase2.get("input_fields"), ["text", "api_context"])
    _expect(failures, "Phase 2 output fields", phase2.get("output_fields"), ["rest_api_list"])
    phase3 = rest_goal.get("phase3", {})
    _expect(
        failures,
        "Phase 3 input fields",
        phase3.get("input_fields"),
        ["text", "rest_api_list", "api_context"],
    )
    _expect(failures, "Phase 3 output fields", phase3.get("output_fields"), ["calls"])
    _expect(
        failures,
        "Phase 3 call fields",
        phase3.get("call_fields"),
        ["rest_api", "http_method", "operation_name", "arguments"],
    )

    d1_row = d1.get("row", {})
    _expect(failures, "D1 input fields", d1_row.get("input_fields"), ["text", "api_context"])
    context = d1.get("context", {})
    _expect(failures, "D1 distractor minimum", context.get("min_distractors_per_row"), 4)
    _expect(failures, "D1 hidden membership", context.get("selected_membership_hidden"), True)
    promotion = rest_goal.get("promotion_evidence", {})
    _expect(
        failures,
        "Phase 2 promotion parent",
        promotion.get("phase2", {}).get("expected_parent_role"),
        "model_x",
    )
    _expect(
        failures,
        "Phase 3 promotion parent",
        promotion.get("phase3", {}).get("expected_parent_role"),
        "goal_extractor",
    )


def main() -> int:
    """CLI entry point for the project contract gate."""
    failures = run_gate()
    if failures:
        print(json.dumps({"status": "fail", "failures": failures}, indent=2))
        return 1
    print(json.dumps({"status": "pass", "gate": "phase23.contract"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
