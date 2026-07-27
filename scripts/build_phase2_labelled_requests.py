#!/usr/bin/env python3
"""Build ``phase2_labelled_requests`` JSONL rows from Redfish records.

The provider path is selected explicitly: deterministic mocks and local files
support contract checks, while the OpenAI-compatible adapter serves real model_x
and private judge runs. The script never calls a Redfish host.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
import random
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

# Running as ``python scripts/build_phase2_labelled_requests.py`` puts
# scripts/ on sys.path; add the repo root so ``import igc`` works without an
# editable install.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from igc.ds.phase2_labelled_requests import (
    D1SamplingBudget,
    PHASE2_LABELLED_REQUESTS,
    Phase2LabelledRequestBuilder,
    Phase2LabelledRequestCounters,
    Phase2LabelledRequestsSpec,
    ProviderAdapterSpec,
    RestApiRecord,
    load_phase2_labelled_requests_spec,
    phase2_acceptance_thresholds_pass,
)
from igc.ds.d1_release import release_d1_jsonl
from igc.modules.train.promotion_evidence import is_sha256

DraftProvider = Callable[[dict[str, Any]], str]
JudgeProvider = Callable[[dict[str, Any]], str]
JsonTransport = Callable[[str, Mapping[str, Any], Mapping[str, str], float], Mapping[str, Any]]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spec",
        default="configs/phase2_labelled_requests.yaml",
        help="YAML builder spec that owns prompts, model IDs, metrics, and thresholds.",
    )
    parser.add_argument(
        "--records-jsonl",
        required=True,
        help="Input JSONL with REST context, methods, operation names, argument schema, "
             "vendor, and source corpus.",
    )
    parser.add_argument(
        "--output-release-dir",
        required=True,
        help=(
            f"Immutable release directory for accepted {PHASE2_LABELLED_REQUESTS} "
            "rows; contains data.jsonl and manifest.json."
        ),
    )
    parser.add_argument(
        "--metrics-out",
        required=True,
        help="Destination JSON file for aggregate offline builder metrics.",
    )
    parser.add_argument(
        "--metric-report",
        choices=("none", "wandb"),
        default="none",
        help="Optional remote metric sink. Use wandb only in the approved lab runtime.",
    )
    width_group = parser.add_mutually_exclusive_group(required=True)
    width_group.add_argument(
        "--sample-width",
        type=int,
        help="Target API count per candidate; 0 is the judged empty-set case.",
    )
    width_group.add_argument(
        "--all-sample-widths",
        action="store_true",
        help=(
            "Build one canonical D1 release across k=1/2/3 plus the separately "
            "bounded judged k=0 cases."
        ),
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Number of labelled-request candidates to attempt.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Deterministic sampler seed.",
    )
    parser.add_argument(
        "--provider-mode",
        choices=("config", "mock", "file", "openai-compatible"),
        default="config",
        help=(
            "Select providers from YAML, local mocks, local files, "
            "or live OpenAI-compatible HTTP."
        ),
    )
    parser.add_argument(
        "--draft-provider-adapter",
        choices=("mock", "file", "openai-compatible"),
        default="",
        help="Override only the model_x draft provider adapter from YAML.",
    )
    parser.add_argument(
        "--judge-provider-adapter",
        choices=("mock", "file", "openai-compatible"),
        default="",
        help="Override only the private judge provider adapter from YAML.",
    )
    parser.add_argument(
        "--drafts-jsonl",
        default="",
        help="File adapter: one draft text line per candidate.",
    )
    parser.add_argument(
        "--judges-jsonl",
        default="",
        help="File adapter: one raw judge JSON line per candidate.",
    )
    parser.add_argument(
        "--live-provider-gate-passed",
        action="store_true",
        help="Allow live provider runs above the YAML safety.live_without_gate_max_candidates cap.",
    )
    parser.add_argument(
        "--allow-threshold-failure",
        action="store_true",
        help="Write artifacts and exit 0 even when YAML acceptance thresholds fail.",
    )
    return parser.parse_args(argv)


def load_rest_api_records(path: Path) -> tuple[RestApiRecord, ...]:
    """Load fixture REST API records from JSONL with line-numbered errors."""
    records: list[RestApiRecord] = []
    seen_rest_apis: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
        if not isinstance(row, Mapping):
            raise SystemExit(f"{path}:{line_number}: row must be a JSON object")

        record = _record_from_mapping(row, path=path, line_number=line_number)
        if record.rest_api in seen_rest_apis:
            raise SystemExit(f"{path}:{line_number}: duplicate rest_api {record.rest_api}")
        seen_rest_apis.add(record.rest_api)
        records.append(record)

    if not records:
        raise SystemExit(f"{path}: no REST API records found")
    return tuple(records)


def build_phase2_labelled_requests(
    *,
    spec: Phase2LabelledRequestsSpec,
    records: tuple[RestApiRecord, ...],
    sample_width: int,
    count: int,
    seed: int,
    draft_provider: DraftProvider,
    judge_provider: JudgeProvider,
    sampling_budget: D1SamplingBudget | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build accepted rows plus aggregate metrics using injected providers."""
    if sample_width not in spec.sample_widths and sample_width != 0:
        raise SystemExit("sample-width must be present in the YAML sampling.sample_widths")
    if count < 1:
        raise SystemExit("count must be positive")
    required_records = (
        spec.context_distractors
        if sample_width == 0
        else sample_width + spec.context_distractors
    )
    if len(records) < required_records:
        raise ValueError(
            "not enough REST API records for targets plus distractors: "
            f"targets={sample_width}, distractors={spec.context_distractors}"
        )

    budget = sampling_budget or D1SamplingBudget.from_spec(spec)
    builder = Phase2LabelledRequestBuilder(
        spec,
        draft_provider=draft_provider,
        judge_provider=judge_provider,
        sampling_budget=budget,
    )
    rng = random.Random(seed)
    accepted_rows: list[dict[str, Any]] = []
    counters = Phase2LabelledRequestCounters(
        sample_width_k=sample_width,
        prompt_spec_version=spec.prompt_spec_version,
        model_x_artifact_sha=spec.model_x.artifact_sha,
        judge_model=spec.judge.model_id,
        judge_profile=spec.judge.profile,
    )
    source_labels: set[str] = set()

    for _ in range(count):
        row, candidate = builder.build_one(records, k=sample_width, rng=rng)
        _merge_counters(counters, candidate)
        if candidate.vendor_source_corpus:
            source_labels.update(candidate.vendor_source_corpus.split(","))
        if row is not None:
            accepted_rows.append(row.to_dict())

    counters.vendor_source_corpus = ",".join(sorted(source_labels))
    summary = counters.summary()
    summary.update({
        "dataset": spec.dataset_name,
        "records_in": len(records),
        "requested_candidates": count,
        "accepted_rows": len(accepted_rows),
        "sampling_budget": budget.summary(),
        "thresholds_pass": phase2_acceptance_thresholds_pass(spec, summary),
    })
    return accepted_rows, summary


def write_metrics(path: Path, summary: Mapping[str, Any]) -> None:
    """Write aggregate metrics JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint.

    :return: process-style exit code.
    """
    args = parse_args(argv)
    spec = load_phase2_labelled_requests_spec(args.spec)
    records = load_rest_api_records(Path(args.records_jsonl))
    if args.all_sample_widths:
        build_plan = [(width, args.count) for width in spec.sample_widths]
        build_plan.append((0, spec.empty_set_candidates))
    else:
        build_plan = [(args.sample_width, args.count)]
    widths = tuple(width for width, _ in build_plan)
    requested_candidates = sum(count for _, count in build_plan)
    if requested_candidates > spec.max_candidates:
        raise SystemExit(
            "requested candidates exceed sampling.max_candidates: "
            f"requested={requested_candidates} limit={spec.max_candidates}"
        )
    draft_adapter, judge_adapter = _selected_adapters(args, spec)
    spec = _resolve_live_identities(
        spec,
        draft_adapter=draft_adapter,
        judge_adapter=judge_adapter,
        env=os.environ,
    )
    _enforce_live_provider_gate(
        args,
        spec=spec,
        draft_adapter=draft_adapter,
        judge_adapter=judge_adapter,
        candidate_count=requested_candidates,
    )
    draft_provider, judge_provider = _providers(
        args,
        spec=spec,
        draft_adapter=draft_adapter,
        judge_adapter=judge_adapter,
    )
    rows_by_width: dict[int, list[dict[str, Any]]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    sampling_budget = D1SamplingBudget.from_spec(spec)
    for width, candidate_count in build_plan:
        width_rows, width_metrics = build_phase2_labelled_requests(
            spec=spec,
            records=records,
            sample_width=width,
            count=candidate_count,
            seed=args.seed if len(widths) == 1 else args.seed + width,
            draft_provider=draft_provider,
            judge_provider=judge_provider,
            sampling_budget=sampling_budget,
        )
        rows_by_width[width] = width_rows
        summaries[str(width)] = width_metrics
    rows = (
        _balanced_release_rows(rows_by_width, spec.sample_widths)
        if args.all_sample_widths
        else list(rows_by_width[widths[0]])
    )
    thresholds_pass = all(
        bool(summary["thresholds_pass"])
        for summary in summaries.values()
    )
    if len(widths) == 1:
        metrics = next(iter(summaries.values()))
    else:
        metrics = {
            "dataset": spec.dataset_name,
            "sample_widths": list(widths),
            "requested_candidates": requested_candidates,
            "accepted_rows": len(rows),
            "accepted_rows_before_balance": sum(
                len(width_rows) for width_rows in rows_by_width.values()
            ),
            "thresholds_pass": thresholds_pass,
            "by_sample_width": summaries,
            "sampling_budget": sampling_budget.summary(),
        }
    write_metrics(Path(args.metrics_out), metrics)
    if args.metric_report == "wandb":
        _log_wandb_metrics(
            spec=spec,
            summaries=summaries,
            aggregate=metrics,
            seed=args.seed,
        )
    rows_written = 0
    if thresholds_pass:
        release_d1_jsonl(
            output_dir=args.output_release_dir,
            rows=rows,
            expected_widths=widths,
            release_metadata={
                "prompt_spec_version": spec.prompt_spec_version,
                "model_x_artifact_sha": spec.model_x.artifact_sha,
                "judge_route": spec.judge.route,
                "judge_model": spec.judge.model_id,
                "judge_profile": spec.judge.profile,
                "draft_provider_adapter": draft_adapter,
                "judge_provider_adapter": judge_adapter,
                "sampling_budget": sampling_budget.summary(),
            },
        )
        rows_written = len(rows)
    print(
        f"wrote dataset={metrics['dataset']} "
        f"attempted={metrics['requested_candidates']} "
        f"accepted={rows_written} "
        f"thresholds_pass={thresholds_pass} "
        f"released={thresholds_pass} "
        f"output_release_dir={args.output_release_dir} "
        f"metrics_out={args.metrics_out}"
    )
    if not thresholds_pass and not args.allow_threshold_failure:
        return 2
    return 0


def _balanced_release_rows(
    rows_by_width: Mapping[int, list[dict[str, Any]]],
    positive_widths: tuple[int, ...],
) -> list[dict[str, Any]]:
    """Return deterministic balanced k=1/2/3 rows plus all accepted k=0 rows.

    Provider acceptance rates differ by width, so equal candidate counts do not
    imply equal accepted counts.  The release contract balances accepted positive
    rows; deterministic prefix selection preserves the seeded provider order and
    keeps empty-set negatives on their independent bound.
    """
    missing = [width for width in positive_widths if not rows_by_width.get(width)]
    if missing:
        raise SystemExit(
            "cannot balance D1 release because accepted positive widths are missing: "
            f"{missing}"
        )
    accepted_per_width = min(
        len(rows_by_width[width]) for width in positive_widths
    )
    balanced: list[dict[str, Any]] = []
    for width in positive_widths:
        balanced.extend(rows_by_width[width][:accepted_per_width])
    balanced.extend(rows_by_width.get(0, []))
    return balanced


def _log_wandb_metrics(
    *,
    spec: Phase2LabelledRequestsSpec,
    summaries: Mapping[str, Mapping[str, Any]],
    aggregate: Mapping[str, Any],
    seed: int,
) -> None:
    """Log D1 quality metrics without placing secret provider values in config.

    :param spec: resolved D1 builder specification with immutable identities.
    :param summaries: one aggregate metric mapping per sample width.
    :param aggregate: complete build summary written to the metrics artifact.
    :param seed: deterministic sampler seed.
    :raises RuntimeError: when W&B cannot initialize or record the required metrics.
    """
    run = None
    try:
        import wandb

        run = wandb.init(
            project=os.environ.get("WANDB_PROJECT"),
            entity=os.environ.get("WANDB_ENTITY"),
            name=f"d1-build-{spec.model_x.artifact_sha.removeprefix('sha256:')[:12]}",
            group=spec.wandb_namespace,
            job_type="dataset-build",
            config={
                "dataset": spec.dataset_name,
                "prompt_spec_version": spec.prompt_spec_version,
                "model_x_model": spec.model_x.model_id,
                "model_x_artifact_sha": spec.model_x.artifact_sha,
                "judge_model": spec.judge.model_id,
                "judge_profile": spec.judge.profile,
                "sample_widths": sorted(int(width) for width in summaries),
                "seed": seed,
                "sampling_limits": {
                    "max_accepted_rows": spec.max_accepted_rows,
                    "max_candidates": spec.max_candidates,
                    "max_accepted_per_combination": (
                        spec.max_accepted_per_combination
                    ),
                    "max_attempts_per_combination": (
                        spec.max_attempts_per_combination
                    ),
                    "max_accepted_per_api": spec.max_accepted_per_api,
                    "max_empty_set_candidates": spec.empty_set_candidates,
                },
            },
        )
        if run is None:
            raise RuntimeError("wandb.init returned no run")
        for width, summary in sorted(summaries.items(), key=lambda item: int(item[0])):
            payload = {
                key: float(value)
                for key, value in summary.items()
                if key in spec.metric_keys
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
            }
            payload[f"{spec.wandb_namespace}/sample_width/k"] = int(width)
            run.log(payload, step=int(width))
        run.summary["requested_candidates"] = int(aggregate["requested_candidates"])
        run.summary["accepted_rows"] = int(aggregate["accepted_rows"])
        run.summary["thresholds_pass"] = bool(aggregate["thresholds_pass"])
        run.finish()
    except Exception as exc:
        cleanup_error = None
        if run is not None:
            try:
                run.finish(exit_code=1)
            except Exception as finish_exc:
                cleanup_error = finish_exc
        if cleanup_error is not None:
            raise RuntimeError("D1 W&B logging and cleanup both failed") from cleanup_error
        raise RuntimeError("required D1 W&B metric logging failed") from exc


def _record_from_mapping(row: Mapping[str, Any], *, path: Path, line_number: int) -> RestApiRecord:
    """Normalize one JSONL row into a :class:`RestApiRecord`."""
    source = row.get("x") if isinstance(row.get("x"), Mapping) else row
    rest_api = source.get("rest_api")
    if not isinstance(rest_api, str) or not rest_api.strip():
        raise SystemExit(f"{path}:{line_number}: rest_api must be a non-empty string")

    raw_methods = source.get("allowed_methods")
    if not isinstance(raw_methods, list) or not all(isinstance(item, str) for item in raw_methods):
        raise SystemExit(f"{path}:{line_number}: allowed_methods must be a list of strings")

    json_body = source.get("json")
    if not isinstance(json_body, Mapping):
        raise SystemExit(f"{path}:{line_number}: json must be an object")

    raw_operations = source.get("operation_names", [])
    if not isinstance(raw_operations, list) or not all(
        isinstance(item, str) and item.strip() for item in raw_operations
    ):
        raise SystemExit(f"{path}:{line_number}: operation_names must be a list of strings")
    argument_schema = source.get("argument_schema", {})
    if not isinstance(argument_schema, Mapping):
        raise SystemExit(f"{path}:{line_number}: argument_schema must be an object")

    return RestApiRecord(
        rest_api=rest_api,
        allowed_methods=tuple(method.upper() for method in raw_methods),
        json_body=dict(json_body),
        operation_names=tuple(raw_operations),
        argument_schema=dict(argument_schema),
        vendor=str(row.get("vendor") or source.get("vendor") or ""),
        source_corpus=str(row.get("source_corpus") or source.get("source_corpus") or ""),
    )


def _selected_adapters(
    args: argparse.Namespace,
    spec: Phase2LabelledRequestsSpec,
) -> tuple[str, str]:
    """Resolve draft and judge adapter names from YAML plus CLI overrides."""
    if args.provider_mode == "config":
        draft_adapter = spec.draft_provider.adapter
        judge_adapter = spec.judge_provider.adapter
    else:
        draft_adapter = args.provider_mode
        judge_adapter = args.provider_mode
    if args.draft_provider_adapter:
        draft_adapter = args.draft_provider_adapter
    if args.judge_provider_adapter:
        judge_adapter = args.judge_provider_adapter
    return draft_adapter, judge_adapter


def _resolve_live_identities(
    spec: Phase2LabelledRequestsSpec,
    *,
    draft_adapter: str,
    judge_adapter: str,
    env: Mapping[str, str],
) -> Phase2LabelledRequestsSpec:
    """Resolve identities before a live run so metrics and release evidence agree.

    Mock and file providers retain literal placeholders for deterministic contract
    checks. A live provider must resolve every identity it owns before the first
    request; otherwise its HTTP payload and release manifest could name different
    artifacts.
    """
    model_x = spec.model_x
    judge = spec.judge
    if draft_adapter == "openai-compatible":
        model_x = replace(
            model_x,
            model_id=_resolve_env_value(model_x.model_id, env, "model_x.model_id"),
            artifact_sha=_resolve_env_value(
                model_x.artifact_sha,
                env,
                "model_x.artifact_sha",
            ),
        )
        if not is_sha256(model_x.artifact_sha):
            raise SystemExit("model_x.artifact_sha must resolve to sha256:<64 hex>")
    if judge_adapter == "openai-compatible":
        judge = replace(
            judge,
            route=_resolve_env_value(judge.route, env, "judge.route"),
            model_id=_resolve_env_value(judge.model_id, env, "judge.model_id"),
            profile=_resolve_env_value(judge.profile, env, "judge.profile"),
        )
    return replace(spec, model_x=model_x, judge=judge)


def _enforce_live_provider_gate(
    args: argparse.Namespace,
    *,
    spec: Phase2LabelledRequestsSpec,
    draft_adapter: str,
    judge_adapter: str,
    candidate_count: int,
) -> None:
    """Block dataset-scale live provider runs until an explicit gate flag is passed."""
    uses_live_adapter = "openai-compatible" in {draft_adapter, judge_adapter}
    if not uses_live_adapter or args.live_provider_gate_passed:
        return
    if candidate_count > spec.live_without_gate_max_candidates:
        raise SystemExit(
            "live provider runs above safety.live_without_gate_max_candidates "
            "require --live-provider-gate-passed",
        )


def _providers(
    args: argparse.Namespace,
    *,
    spec: Phase2LabelledRequestsSpec,
    draft_adapter: str,
    judge_adapter: str,
) -> tuple[DraftProvider, JudgeProvider]:
    """Return draft and judge providers for the resolved adapters."""
    return (
        _provider_for_adapter(
            draft_adapter,
            config=spec.draft_provider,
            text_path=args.drafts_jsonl,
            label="draft",
        ),
        _provider_for_adapter(
            judge_adapter,
            config=spec.judge_provider,
            text_path=args.judges_jsonl,
            label="judge",
        ),
    )


def _provider_for_adapter(
    adapter: str,
    *,
    config: ProviderAdapterSpec,
    text_path: str,
    label: str,
) -> DraftProvider:
    """Build one provider callable for the selected adapter."""
    if adapter == "mock":
        if label == "draft":
            return _mock_draft_provider
        return _mock_judge_provider
    if adapter == "file":
        if not text_path:
            raise SystemExit(f"--{label}s-jsonl is required for {label} file provider")
        return _TextLineProvider(Path(text_path), label=label)
    if adapter == "openai-compatible":
        _require_live_provider_config(config, label=label)
        return _OpenAICompatibleChatProvider(config, label=label)
    raise SystemExit(f"unknown {label} provider adapter {adapter!r}")


def _mock_draft_provider(request: dict[str, Any]) -> str:
    """Return deterministic fixture text for offline smoke tests."""
    return f"fixture request covering {request['sample_width']} Redfish API record(s)"


def _mock_judge_provider(request: dict[str, Any]) -> str:
    """Return a deterministic accepting judge response for offline smoke tests."""
    return json.dumps({
        "accepted": True,
        "natural": True,
        "nonsense": False,
        "ambiguous": False,
        "duplicate_intent": False,
        "extra_intents": False,
        "method_semantics_valid": True,
        "covered_api_set": request["expected_rest_api_list"],
        "reason": "mock exact-set match",
    })


def _require_live_provider_config(config: ProviderAdapterSpec, *, label: str) -> None:
    """Fail early when a CLI override selects live HTTP without YAML routing fields."""
    missing = [
        field_name
        for field_name in ("base_url_env", "endpoint_path", "response_text_path")
        if not getattr(config, field_name)
    ]
    if missing:
        joined = ", ".join(f"providers.{label}.{field_name}" for field_name in missing)
        raise SystemExit(
            f"{label} openai-compatible provider requires {joined}",
        )


class _OpenAICompatibleChatProvider:
    """Small OpenAI-compatible chat-completions provider with injectable HTTP."""

    def __init__(
        self,
        config: ProviderAdapterSpec,
        *,
        label: str,
        env: Mapping[str, str] | None = None,
        transport: JsonTransport | None = None,
    ) -> None:
        """Create a live provider from YAML config and environment variables."""
        self._config = config
        self._label = label
        self._env = os.environ if env is None else env
        self._transport = _urlopen_json_transport if transport is None else transport

    def __call__(self, request: dict[str, Any]) -> str:
        """Send one prompt to an OpenAI-compatible endpoint and return text."""
        base_url = _required_env(
            self._env,
            self._config.base_url_env,
            f"{self._label}.base_url_env",
        )
        model_id = _resolve_env_value(
            str(request["model_id"]),
            self._env,
            f"{self._label}.model_id",
        )
        payload: dict[str, Any] = {
            "model": model_id,
            "messages": [{"role": "user", "content": request["prompt"]}],
        }
        generation = request.get("generation")
        if isinstance(generation, Mapping):
            payload.update(
                _resolve_env_values(
                    dict(generation),
                    self._env,
                    f"{self._label}.generation",
                ),
            )
        for field_name in self._config.payload_request_fields:
            if field_name in request:
                payload[field_name] = _resolve_env_values(
                    request[field_name],
                    self._env,
                    f"{self._label}.{field_name}",
                )

        response = self._transport(
            _join_url(base_url, self._config.endpoint_path),
            payload,
            _headers(self._config, self._env),
            self._config.timeout_seconds,
        )
        return _extract_response_text(
            response,
            self._config.response_text_path,
            label=self._label,
        )


class _TextLineProvider:
    """Sequential provider backed by non-blank local text lines."""

    def __init__(self, path: Path, *, label: str) -> None:
        """Load provider fixture lines."""
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


def _urlopen_json_transport(
    url: str,
    payload: Mapping[str, Any],
    headers: Mapping[str, str],
    timeout: float,
) -> Mapping[str, Any]:
    """POST JSON to a live provider and parse the JSON response."""
    data = json.dumps(dict(payload)).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers=dict(headers),
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.URLError as exc:
        raise SystemExit(f"live provider request failed: {exc.reason}") from exc
    except OSError as exc:
        raise SystemExit(f"live provider request failed: {exc}") from exc
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"live provider returned invalid JSON: {exc.msg}") from exc
    if not isinstance(parsed, Mapping):
        raise SystemExit("live provider response must be a JSON object")
    return parsed


def _headers(config: ProviderAdapterSpec, env: Mapping[str, str]) -> dict[str, str]:
    """Build HTTP headers without exposing bearer values."""
    headers = {"Content-Type": "application/json"}
    if config.api_key_env:
        api_key = env.get(config.api_key_env, "")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _join_url(base_url: str, endpoint_path: str) -> str:
    """Join a configured base URL and endpoint path."""
    return f"{base_url.rstrip('/')}/{endpoint_path.lstrip('/')}"


def _required_env(env: Mapping[str, str], key: str, label: str) -> str:
    """Read a required environment variable named by YAML."""
    if not key:
        raise SystemExit(f"{label} must name an environment variable")
    value = env.get(key, "")
    if not value:
        raise SystemExit(f"missing environment variable {key} for {label}")
    return value


def _resolve_env_values(value: Any, env: Mapping[str, str], label: str) -> Any:
    """Resolve exact ``${VAR}`` placeholders in nested provider values."""
    if isinstance(value, str):
        return _resolve_env_value(value, env, label)
    if isinstance(value, Mapping):
        return {
            str(child_key): _resolve_env_values(child_value, env, f"{label}.{child_key}")
            for child_key, child_value in value.items()
        }
    if isinstance(value, list):
        return [
            _resolve_env_values(child_value, env, f"{label}[]")
            for child_value in value
        ]
    return value


def _resolve_env_value(value: str, env: Mapping[str, str], label: str) -> str:
    """Resolve an exact ``${VAR}`` placeholder or return a literal string."""
    if not value.startswith("${") or not value.endswith("}"):
        return value
    key = value[2:-1]
    if not key:
        raise SystemExit(f"{label} contains an empty environment placeholder")
    resolved = env.get(key, "")
    if not resolved:
        raise SystemExit(f"missing environment variable {key} for {label}")
    return resolved


def _extract_response_text(response: Mapping[str, Any], path: str, *, label: str) -> str:
    """Extract generated text from a JSON response by dotted path."""
    current: Any = response
    for part in path.split("."):
        if isinstance(current, Mapping):
            if part not in current:
                raise SystemExit(f"{label} provider response missing {path}")
            current = current[part]
            continue
        if isinstance(current, list):
            try:
                index = int(part)
            except ValueError as exc:
                raise SystemExit(f"{label} provider response path {path} is not valid") from exc
            try:
                current = current[index]
            except IndexError as exc:
                raise SystemExit(f"{label} provider response path {path} is out of range") from exc
            continue
        raise SystemExit(f"{label} provider response path {path} cannot be traversed")
    if not isinstance(current, str) or not current.strip():
        raise SystemExit(f"{label} provider response path {path} did not contain text")
    return current.strip()


def _merge_counters(
    aggregate: Phase2LabelledRequestCounters,
    candidate: Phase2LabelledRequestCounters,
) -> None:
    """Merge one candidate counter object into the aggregate counter object."""
    aggregate.draft_total += candidate.draft_total
    aggregate.accepted_total += candidate.accepted_total
    aggregate.rejected_total += candidate.rejected_total
    aggregate.pro_accept_total += candidate.pro_accept_total
    aggregate.nonsense_total += candidate.nonsense_total
    aggregate.invalid_json_total += candidate.invalid_json_total
    aggregate.rest_api_set_match_total += candidate.rest_api_set_match_total
    aggregate.natural_total += candidate.natural_total
    aggregate.ambiguous_total += candidate.ambiguous_total
    aggregate.duplicate_intent_total += candidate.duplicate_intent_total
    aggregate.extra_intent_total += candidate.extra_intent_total
    aggregate.method_semantics_valid_total += candidate.method_semantics_valid_total
    aggregate.empty_set_expected_total += candidate.empty_set_expected_total
    aggregate.empty_set_match_total += candidate.empty_set_match_total


if __name__ == "__main__":
    raise SystemExit(main())


# Author: Mus mbayramo@stanford.edu
