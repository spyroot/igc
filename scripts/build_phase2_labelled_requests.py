#!/usr/bin/env python3
"""Build offline ``phase2_labelled_requests`` JSONL rows from fixture records.

The script is deliberately provider-injected: it loads the YAML spec, reads a
tiny JSONL record fixture, and uses either deterministic mock providers or
local draft/judge fixture files. It never opens W&B, downloads a model, calls a
Redfish host, or reaches the network.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

import argparse
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
    PHASE2_LABELLED_REQUESTS,
    Phase2LabelledRequestBuilder,
    Phase2LabelledRequestCounters,
    Phase2LabelledRequestsSpec,
    ProviderAdapterSpec,
    RestApiRecord,
    load_phase2_labelled_requests_spec,
    phase2_acceptance_thresholds_pass,
)

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
        help="Input JSONL with rest_api, allowed_methods, json, vendor, and source_corpus.",
    )
    parser.add_argument(
        "--output-jsonl",
        required=True,
        help=f"Destination JSONL for accepted {PHASE2_LABELLED_REQUESTS} rows.",
    )
    parser.add_argument(
        "--metrics-out",
        required=True,
        help="Destination JSON file for aggregate offline builder metrics.",
    )
    parser.add_argument(
        "--sample-width",
        type=int,
        required=True,
        help="Number of REST API records sampled per candidate; must be 1, 2, or 3.",
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
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build accepted rows plus aggregate metrics using injected providers."""
    if sample_width not in spec.sample_widths:
        raise SystemExit("sample-width must be present in the YAML sampling.sample_widths")
    if count < 1:
        raise SystemExit("count must be positive")
    if len(records) < sample_width:
        raise SystemExit("not enough records for requested sample-width")

    builder = Phase2LabelledRequestBuilder(
        spec,
        draft_provider=draft_provider,
        judge_provider=judge_provider,
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
        "thresholds_pass": phase2_acceptance_thresholds_pass(spec, summary),
    })
    return accepted_rows, summary


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    """Write JSONL rows and return the number written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")
            count += 1
    return count


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
    draft_adapter, judge_adapter = _selected_adapters(args, spec)
    _enforce_live_provider_gate(
        args,
        spec=spec,
        draft_adapter=draft_adapter,
        judge_adapter=judge_adapter,
    )
    draft_provider, judge_provider = _providers(
        args,
        spec=spec,
        draft_adapter=draft_adapter,
        judge_adapter=judge_adapter,
    )
    rows, metrics = build_phase2_labelled_requests(
        spec=spec,
        records=records,
        sample_width=args.sample_width,
        count=args.count,
        seed=args.seed,
        draft_provider=draft_provider,
        judge_provider=judge_provider,
    )
    rows_written = write_jsonl(Path(args.output_jsonl), rows)
    write_metrics(Path(args.metrics_out), metrics)
    print(
        f"wrote dataset={metrics['dataset']} "
        f"attempted={metrics['requested_candidates']} "
        f"accepted={rows_written} "
        f"thresholds_pass={metrics['thresholds_pass']} "
        f"output_jsonl={args.output_jsonl} "
        f"metrics_out={args.metrics_out}"
    )
    if not metrics["thresholds_pass"] and not args.allow_threshold_failure:
        return 2
    return 0


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

    return RestApiRecord(
        rest_api=rest_api,
        allowed_methods=tuple(method.upper() for method in raw_methods),
        json_body=dict(json_body),
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


def _enforce_live_provider_gate(
    args: argparse.Namespace,
    *,
    spec: Phase2LabelledRequestsSpec,
    draft_adapter: str,
    judge_adapter: str,
) -> None:
    """Block dataset-scale live provider runs until an explicit gate flag is passed."""
    uses_live_adapter = "openai-compatible" in {draft_adapter, judge_adapter}
    if not uses_live_adapter or args.live_provider_gate_passed:
        return
    if args.count > spec.live_without_gate_max_candidates:
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
        "rest_api_list": request["expected_rest_api_list"],
        "nonsense": False,
        "reason": "fixture",
        "order_evidence": "none",
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
    aggregate.empty_set_expected_total += candidate.empty_set_expected_total
    aggregate.empty_set_match_total += candidate.empty_set_match_total


if __name__ == "__main__":
    raise SystemExit(main())


# Author: Mus mbayramo@stanford.edu
