#!/usr/bin/env python3
"""Run the offline Phase 3 method/argument extraction smoke.

The runner is spec-driven and provider-injected. It uses tiny built-in
fixtures, renders them through the ordered-goal contract, and parses fake model
outputs from either deterministic mock mode or a local JSONL file. It never
loads model weights, opens W&B, downloads corpora, calls Redfish, or uses a GPU.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import yaml

# Running as ``python scripts/build_phase3_argument_smoke.py`` puts scripts/ on
# sys.path; add the repo root so ``import igc`` works without editable install.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from igc.ds.rest_goal_contract import (
    RedfishContext,
    build_ordered_call_row,
    evaluate_ordered_calls_y_pred,
    inference_target_calls_json,
    render_ordered_call_example,
)
from igc.modules.base.metric_keys import PHASE3_ARGUMENT_EXTRACT, phase_metric

PHASE3_ARGUMENT_SMOKE = "phase3_argument_extractor_smoke"

_CALL_ORDERED_EXACT_KEY = phase_metric(
    PHASE3_ARGUMENT_EXTRACT,
    "eval",
    "call_ordered_exact_match_rate",
)
_CALL_ORDER_CORRECT_KEY = phase_metric(
    PHASE3_ARGUMENT_EXTRACT,
    "eval",
    "call_order_correct_rate",
)
_REST_API_EXACT_KEY = phase_metric(PHASE3_ARGUMENT_EXTRACT, "eval", "rest_api_exact_match_rate")
_ALLOWED_METHODS_EXACT_KEY = phase_metric(
    PHASE3_ARGUMENT_EXTRACT,
    "eval",
    "allowed_methods_exact_match_rate",
)
_METHOD_EXACT_KEY = phase_metric(PHASE3_ARGUMENT_EXTRACT, "eval", "method_exact_match_rate")
_ARGS_PARSE_KEY = phase_metric(PHASE3_ARGUMENT_EXTRACT, "eval", "arguments_json_parse_rate")
_ARGS_EXACT_KEY = phase_metric(PHASE3_ARGUMENT_EXTRACT, "eval", "arguments_exact_match_rate")
_INVALID_METHOD_KEY = phase_metric(PHASE3_ARGUMENT_EXTRACT, "eval", "invalid_method_rate")
_READONLY_EMPTY_KEY = phase_metric(
    PHASE3_ARGUMENT_EXTRACT,
    "eval",
    "readonly_empty_arguments_rate",
)
_ROWS_TOTAL_KEY = phase_metric(PHASE3_ARGUMENT_EXTRACT, "smoke", "rows_total")
_PARSED_TOTAL_KEY = phase_metric(PHASE3_ARGUMENT_EXTRACT, "smoke", "parsed_total")
_ACCEPTED_TOTAL_KEY = phase_metric(PHASE3_ARGUMENT_EXTRACT, "smoke", "accepted_total")
_PROMPT_SPEC_VERSION_KEY = phase_metric(
    PHASE3_ARGUMENT_EXTRACT,
    "smoke",
    "prompt_spec_version",
)
_WEIGHTS_ROLE_KEY = phase_metric(PHASE3_ARGUMENT_EXTRACT, "smoke", "weights_role")

_RATE_KEYS = (
    _CALL_ORDERED_EXACT_KEY,
    _CALL_ORDER_CORRECT_KEY,
    _REST_API_EXACT_KEY,
    _ALLOWED_METHODS_EXACT_KEY,
    _METHOD_EXACT_KEY,
    _ARGS_PARSE_KEY,
    _ARGS_EXACT_KEY,
    _INVALID_METHOD_KEY,
    _READONLY_EMPTY_KEY,
)

PredictionProvider = Callable[[dict[str, Any]], str]


class Phase3ArgumentSmokeSpecError(ValueError):
    """Raised when the Phase 3 smoke YAML spec is invalid."""


@dataclass(frozen=True)
class Phase3ArgumentSmokeSpec:
    """Loaded YAML contract for the Phase 3 argument-extractor smoke."""

    profile_name: str
    prompt_spec_version: str
    task: str
    base_weights_role: str
    weights_role: str
    renderer: str
    fixtures: tuple[str, ...]
    provider_modes: tuple[str, ...]
    wandb_namespace: str
    metric_keys: tuple[str, ...]
    acceptance_thresholds: Mapping[str, float]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spec",
        default="configs/inference/phase3_argument_extractor_smoke.yaml",
        help="YAML smoke profile with fixtures, metric keys, and thresholds.",
    )
    parser.add_argument(
        "--output-jsonl",
        required=True,
        help="Destination JSONL for rendered fixture rows and prediction results.",
    )
    parser.add_argument(
        "--metrics-out",
        required=True,
        help="Destination JSON file for aggregate smoke metrics.",
    )
    parser.add_argument(
        "--provider-mode",
        choices=("mock", "file"),
        default="mock",
        help="Use deterministic exact-match fake outputs or local prediction fixtures.",
    )
    parser.add_argument(
        "--predictions-jsonl",
        default="",
        help="Provider-mode file: one raw argument_extractor JSON prediction per fixture.",
    )
    parser.add_argument(
        "--allow-threshold-failure",
        action="store_true",
        help="Write artifacts and exit 0 even when YAML thresholds fail.",
    )
    return parser.parse_args(argv)


def load_phase3_argument_smoke_spec(path: str | Path) -> Phase3ArgumentSmokeSpec:
    """Load and validate the Phase 3 argument smoke YAML spec."""
    spec_path = Path(path)
    try:
        raw = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise Phase3ArgumentSmokeSpecError(f"cannot read spec {spec_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise Phase3ArgumentSmokeSpecError(f"cannot parse YAML in {spec_path}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise Phase3ArgumentSmokeSpecError("Phase 3 argument smoke spec must be a mapping")

    profile = _mapping(raw, "profile", required=True)
    smoke = _mapping(raw, "smoke", required=True)
    wandb = _mapping(raw, "wandb", required=True)
    acceptance = _mapping(raw, "acceptance", required=True)

    profile_name = _required_string(profile, "name", "profile.name")
    if profile_name != PHASE3_ARGUMENT_SMOKE:
        raise Phase3ArgumentSmokeSpecError(
            f"profile.name must be {PHASE3_ARGUMENT_SMOKE!r}",
        )
    task = _required_string(profile, "task", "profile.task")
    if task != "text_and_rest_api_list_to_calls":
        raise Phase3ArgumentSmokeSpecError(
            "profile.task must be 'text_and_rest_api_list_to_calls'",
        )
    weights_role = _required_string(profile, "weights_role", "profile.weights_role")
    if weights_role != "argument_extractor":
        raise Phase3ArgumentSmokeSpecError("profile.weights_role must be 'argument_extractor'")
    renderer = _required_string(profile, "renderer", "profile.renderer")
    if renderer != "render_ordered_call_example":
        raise Phase3ArgumentSmokeSpecError(
            "profile.renderer must be 'render_ordered_call_example'",
        )

    fixtures = tuple(str(item) for item in _sequence(smoke, "fixtures"))
    if fixtures != ("read_only_ordered_get", "patch_with_arguments"):
        raise Phase3ArgumentSmokeSpecError(
            "smoke.fixtures must be read_only_ordered_get and patch_with_arguments",
        )
    provider_modes = tuple(str(item) for item in _sequence(smoke, "provider_modes"))
    if provider_modes != ("mock", "file"):
        raise Phase3ArgumentSmokeSpecError("smoke.provider_modes must be mock and file")

    wandb_namespace = _required_string(wandb, "namespace", "wandb.namespace")
    if wandb_namespace != PHASE3_ARGUMENT_EXTRACT:
        raise Phase3ArgumentSmokeSpecError(
            f"wandb.namespace must be {PHASE3_ARGUMENT_EXTRACT!r}",
        )
    metric_keys = tuple(str(item) for item in _sequence(wandb, "metric_keys"))
    missing_metrics = sorted(set(_expected_metric_keys()) - set(metric_keys))
    if missing_metrics:
        raise Phase3ArgumentSmokeSpecError(
            f"wandb.metric_keys missing required keys: {', '.join(missing_metrics)}",
        )

    thresholds = _acceptance_thresholds(acceptance)
    return Phase3ArgumentSmokeSpec(
        profile_name=profile_name,
        prompt_spec_version=_required_string(
            profile,
            "prompt_spec_version",
            "profile.prompt_spec_version",
        ),
        task=task,
        base_weights_role=_required_string(
            profile,
            "base_weights_role",
            "profile.base_weights_role",
        ),
        weights_role=weights_role,
        renderer=renderer,
        fixtures=fixtures,
        provider_modes=provider_modes,
        wandb_namespace=wandb_namespace,
        metric_keys=metric_keys,
        acceptance_thresholds=thresholds,
    )


def default_phase3_smoke_rows() -> tuple[dict[str, Any], ...]:
    """Return the tiny offline fixtures used by the smoke runner."""
    systems = RedfishContext(
        rest_api="/redfish/v1/Systems",
        allowed_methods=("GET", "HEAD"),
        json={
            "@odata.id": "/redfish/v1/Systems",
            "@odata.type": "#ComputerSystemCollection.ComputerSystemCollection",
            "Members": [{"@odata.id": "/redfish/v1/Systems/System.Embedded.1"}],
            "Members@odata.count": 1,
            "Name": "Computer System Collection",
        },
    )
    tasks = RedfishContext(
        rest_api="/redfish/v1/TaskService/Tasks",
        allowed_methods=("GET", "HEAD"),
        json={
            "@odata.id": "/redfish/v1/TaskService/Tasks",
            "@odata.type": "#TaskCollection.TaskCollection",
            "Members": [],
            "Members@odata.count": 0,
            "Name": "Task Collection",
        },
    )
    bios_settings = RedfishContext(
        rest_api="/redfish/v1/Systems/System.Embedded.1/Bios/Settings",
        allowed_methods=("GET", "PATCH"),
        json={
            "@odata.id": "/redfish/v1/Systems/System.Embedded.1/Bios/Settings",
            "Attributes": {"BootMode": "LegacyBios"},
        },
    )

    return (
        build_ordered_call_row(
            text="check the task queue, then list the available computer systems",
            contexts=(systems, tasks),
            rest_api_list=("/redfish/v1/TaskService/Tasks", "/redfish/v1/Systems"),
        ),
        build_ordered_call_row(
            text="set the embedded system bios boot mode to Uefi",
            contexts=(bios_settings,),
            rest_api_list=("/redfish/v1/Systems/System.Embedded.1/Bios/Settings",),
            method_by_api={
                "/redfish/v1/Systems/System.Embedded.1/Bios/Settings": "PATCH",
            },
            arguments_by_api={
                "/redfish/v1/Systems/System.Embedded.1/Bios/Settings": {
                    "Attributes": {"BootMode": "Uefi"},
                },
            },
        ),
    )


def build_phase3_argument_smoke(
    *,
    spec: Phase3ArgumentSmokeSpec,
    rows: Sequence[Mapping[str, Any]],
    prediction_provider: PredictionProvider,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Render rows, parse fake predictions, and aggregate smoke metrics."""
    artifacts: list[dict[str, Any]] = []
    evaluations: list[dict[str, Any]] = []

    for row_index, row in enumerate(rows):
        rendered = render_ordered_call_example(row)
        request = {
            "row_index": row_index,
            "profile": spec.profile_name,
            "prompt_spec_version": spec.prompt_spec_version,
            "weights_role": spec.weights_role,
            "prompt": rendered.prompt,
            "target_json": rendered.target_json,
            "expected_calls": list(row["y_true"]["calls"]),
        }
        raw_prediction = prediction_provider(request)
        evaluation = evaluate_ordered_calls_y_pred(row, raw_prediction)
        evaluations.append(evaluation)
        artifacts.append({
            "dataset": spec.profile_name,
            "prompt_spec_version": spec.prompt_spec_version,
            "task": spec.task,
            "row_index": row_index,
            "x": dict(row["x"]),
            "y_true": dict(row["y_true"]),
            "rendered": {
                "renderer": spec.renderer,
                "target_char_start": rendered.target_char_start,
                "prompt_char_count": len(rendered.prompt),
                "target_json": rendered.target_json,
            },
            "y_pred_raw": raw_prediction,
            "evaluation": evaluation,
            "inference": inference_target_calls_json(row),
        })

    summary = aggregate_phase3_argument_smoke_metrics(spec, evaluations)
    return artifacts, summary


def aggregate_phase3_argument_smoke_metrics(
    spec: Phase3ArgumentSmokeSpec,
    evaluations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate row-level evaluation records into spec metric keys."""
    rows_total = len(evaluations)
    summary: dict[str, Any] = {
        "dataset": spec.profile_name,
        "task": spec.task,
        _ROWS_TOTAL_KEY: rows_total,
        _PARSED_TOTAL_KEY: sum(1 for item in evaluations if item.get("parsed")),
        _ACCEPTED_TOTAL_KEY: sum(
            1
            for item in evaluations
            if item.get("call_ordered_exact_match")
        ),
        _PROMPT_SPEC_VERSION_KEY: spec.prompt_spec_version,
        _WEIGHTS_ROLE_KEY: spec.weights_role,
    }
    for key in _RATE_KEYS:
        field = key.rsplit("/", 1)[-1]
        summary[key] = _average(float(item.get(field, 0.0)) for item in evaluations)
    summary["thresholds_pass"] = phase3_argument_smoke_thresholds_pass(spec, summary)
    return summary


def phase3_argument_smoke_thresholds_pass(
    spec: Phase3ArgumentSmokeSpec,
    summary: Mapping[str, Any],
) -> bool:
    """Return true when aggregate smoke metrics satisfy YAML thresholds."""
    thresholds = spec.acceptance_thresholds
    return (
        float(summary.get(_CALL_ORDERED_EXACT_KEY, 0.0))
        >= thresholds["min_call_ordered_exact_match_rate"]
        and float(summary.get(_CALL_ORDER_CORRECT_KEY, 0.0))
        >= thresholds["min_call_order_correct_rate"]
        and float(summary.get(_METHOD_EXACT_KEY, 0.0))
        >= thresholds["min_method_exact_match_rate"]
        and float(summary.get(_ARGS_PARSE_KEY, 0.0))
        >= thresholds["min_arguments_json_parse_rate"]
        and float(summary.get(_ARGS_EXACT_KEY, 0.0))
        >= thresholds["min_arguments_exact_match_rate"]
        and float(summary.get(_READONLY_EMPTY_KEY, 0.0))
        >= thresholds["min_readonly_empty_arguments_rate"]
        and float(summary.get(_INVALID_METHOD_KEY, 1.0))
        <= thresholds["max_invalid_method_rate"]
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    """Write JSONL rows and return the number written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True))
            handle.write("\n")
            count += 1
    return count


def write_metrics(path: Path, summary: Mapping[str, Any]) -> None:
    """Write aggregate smoke metrics JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint.

    :return: process-style exit code.
    """
    args = parse_args(argv)
    spec = load_phase3_argument_smoke_spec(args.spec)
    prediction_provider = _prediction_provider(args)
    artifacts, metrics = build_phase3_argument_smoke(
        spec=spec,
        rows=default_phase3_smoke_rows(),
        prediction_provider=prediction_provider,
    )
    rows_written = write_jsonl(Path(args.output_jsonl), artifacts)
    write_metrics(Path(args.metrics_out), metrics)
    print(
        f"wrote dataset={metrics['dataset']} "
        f"rows={rows_written} "
        f"accepted={metrics[_ACCEPTED_TOTAL_KEY]} "
        f"thresholds_pass={metrics['thresholds_pass']} "
        f"output_jsonl={args.output_jsonl} "
        f"metrics_out={args.metrics_out}"
    )
    if not metrics["thresholds_pass"] and not args.allow_threshold_failure:
        return 2
    return 0


def _prediction_provider(args: argparse.Namespace) -> PredictionProvider:
    """Return the fake prediction provider selected by CLI flags."""
    if args.provider_mode == "mock":
        return _mock_prediction_provider
    if not args.predictions_jsonl:
        raise SystemExit("--predictions-jsonl is required for provider-mode file")
    return _JsonLineProvider(Path(args.predictions_jsonl), label="prediction")


def _mock_prediction_provider(request: dict[str, Any]) -> str:
    """Return an exact-match fake argument_extractor output."""
    return json.dumps({"calls": request["expected_calls"]}, sort_keys=True)


class _JsonLineProvider:
    """Sequential fake provider backed by non-blank local JSONL text lines."""

    def __init__(self, path: Path, *, label: str) -> None:
        """Load fake provider lines."""
        self._path = path
        self._label = label
        self._lines = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self._index = 0
        if not self._lines:
            raise SystemExit(f"{path}: no {label} provider lines found")

    def __call__(self, request: dict[str, Any]) -> str:
        """Return the next provider line."""
        _ = request
        if self._index >= len(self._lines):
            raise SystemExit(f"{self._path}: not enough {self._label} provider lines")
        line = self._lines[self._index]
        self._index += 1
        return line


def _expected_metric_keys() -> tuple[str, ...]:
    """Return the metric keys required from the YAML profile."""
    return _RATE_KEYS + (
        _ROWS_TOTAL_KEY,
        _PARSED_TOTAL_KEY,
        _ACCEPTED_TOTAL_KEY,
        _PROMPT_SPEC_VERSION_KEY,
        _WEIGHTS_ROLE_KEY,
    )


def _acceptance_thresholds(raw: Mapping[str, Any]) -> dict[str, float]:
    """Validate and normalize acceptance thresholds."""
    required = {
        "min_call_ordered_exact_match_rate",
        "min_call_order_correct_rate",
        "min_method_exact_match_rate",
        "min_arguments_json_parse_rate",
        "min_arguments_exact_match_rate",
        "min_readonly_empty_arguments_rate",
        "max_invalid_method_rate",
    }
    missing = sorted(required - set(raw))
    if missing:
        raise Phase3ArgumentSmokeSpecError(
            f"acceptance missing required keys: {', '.join(missing)}",
        )
    thresholds: dict[str, float] = {}
    for key, value in raw.items():
        try:
            thresholds[str(key)] = float(value)
        except (TypeError, ValueError) as exc:
            raise Phase3ArgumentSmokeSpecError(f"acceptance.{key} must be numeric") from exc
    return thresholds


def _mapping(source: Mapping[str, Any], key: str, *, required: bool = False) -> Mapping[str, Any]:
    """Read a nested mapping from a YAML dictionary."""
    value = source.get(key)
    if value is None and not required:
        return {}
    if not isinstance(value, Mapping):
        raise Phase3ArgumentSmokeSpecError(f"{key} must be a mapping")
    return value


def _sequence(source: Mapping[str, Any], key: str) -> Sequence[Any]:
    """Read a sequence from a YAML dictionary."""
    value = source.get(key)
    if not isinstance(value, (list, tuple)):
        raise Phase3ArgumentSmokeSpecError(f"{key} must be a sequence")
    return value


def _required_string(source: Mapping[str, Any], key: str, label: str) -> str:
    """Read a required non-empty string from a YAML dictionary."""
    value = source.get(key)
    if not isinstance(value, str) or not value.strip():
        raise Phase3ArgumentSmokeSpecError(f"{label} must be a non-empty string")
    return value


def _average(values: Iterable[float]) -> float:
    """Return the average of an iterable, or zero when it is empty."""
    values_list = list(values)
    if not values_list:
        return 0.0
    return sum(values_list) / len(values_list)


if __name__ == "__main__":
    raise SystemExit(main())


# Author: Mus mbayramo@stanford.edu
