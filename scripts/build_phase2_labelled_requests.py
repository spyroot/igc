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
    RestApiRecord,
    load_phase2_labelled_requests_spec,
    phase2_acceptance_thresholds_pass,
)

DraftProvider = Callable[[dict[str, Any]], str]
JudgeProvider = Callable[[dict[str, Any]], str]


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
        choices=("mock", "file"),
        default="mock",
        help="Use deterministic local mock providers or local fixture files.",
    )
    parser.add_argument(
        "--drafts-jsonl",
        default="",
        help="Provider-mode file: one draft text line per candidate.",
    )
    parser.add_argument(
        "--judges-jsonl",
        default="",
        help="Provider-mode file: one raw judge JSON line per candidate.",
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
    draft_provider, judge_provider = _providers(args)
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


def _providers(args: argparse.Namespace) -> tuple[DraftProvider, JudgeProvider]:
    """Return local draft and judge providers for the requested mode."""
    if args.provider_mode == "mock":
        return _mock_draft_provider, _mock_judge_provider
    if not args.drafts_jsonl or not args.judges_jsonl:
        raise SystemExit("--drafts-jsonl and --judges-jsonl are required for provider-mode file")
    return (
        _TextLineProvider(Path(args.drafts_jsonl), label="draft"),
        _TextLineProvider(Path(args.judges_jsonl), label="judge"),
    )


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


class _TextLineProvider:
    """Sequential provider backed by non-blank local text lines."""

    def __init__(self, path: Path, *, label: str) -> None:
        """Load provider fixture lines."""
        self._path = path
        self._label = label
        self._lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
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
