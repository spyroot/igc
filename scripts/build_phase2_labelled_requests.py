#!/usr/bin/env python3
"""Build Phase 2 labelled-request rows from offline provider outputs.

The command loads ``configs/phase2_labelled_requests.yaml``, samples already
materialized Redfish records, consumes explicit JSONL outputs from the model_x
draft and private-judge provider lanes, and writes accepted JSONL rows plus a
metrics summary. It never calls W&B, GPUs, Redfish hosts, or model endpoints.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

# Running as ``python scripts/build_phase2_labelled_requests.py`` puts scripts/
# on sys.path; add the repo root so ``import igc`` works without installation.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from igc.ds.phase2_labelled_requests import (
    Phase2LabelledRequestCounters,
    Phase2RestApiRecord,
    build_phase2_labelled_request_row,
    load_phase2_labelled_requests_spec,
    parse_pro_judge_result,
    sample_rest_api_records,
)
from igc.ds.sources import RedfishFixtureSource, TrustLevel
from igc.ds.sources.base import SourceRecord


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spec",
        default="configs/phase2_labelled_requests.yaml",
        help="Phase 2 labelled-request YAML spec.",
    )
    parser.add_argument(
        "--capture-root",
        action="append",
        default=[],
        help="Materialized Redfish capture root; may be repeated.",
    )
    parser.add_argument(
        "--corpus-manifest",
        default="",
        help="redfish_ctl manifest.v1.json path for materialized corpora.",
    )
    parser.add_argument(
        "--corpus-root",
        default="",
        help="Materialized redfish_ctl corpus root used with --corpus-manifest.",
    )
    parser.add_argument(
        "--corpus-kind",
        default="dataset",
        help="redfish_ctl corpus kind selector, normally dataset.",
    )
    parser.add_argument(
        "--corpus-id",
        action="append",
        default=[],
        help="Optional redfish_ctl corpus id allow-list; may be repeated.",
    )
    parser.add_argument(
        "--drafts-jsonl",
        required=True,
        help="JSONL provider output with one object containing text per sample.",
    )
    parser.add_argument(
        "--judges-jsonl",
        required=True,
        help="JSONL private-judge output with one JSON object per sample.",
    )
    parser.add_argument(
        "--output-jsonl",
        required=True,
        help="Destination JSONL for accepted phase2_labelled_requests rows.",
    )
    parser.add_argument(
        "--metrics-out",
        required=True,
        help="Destination JSON summary for counters and W&B-compatible keys.",
    )
    parser.add_argument(
        "--sample-width",
        action="append",
        type=int,
        default=[],
        help="Sample width k to cycle through; defaults to spec sample_widths.",
    )
    parser.add_argument("--max-samples", type=int, default=0, help="Limit provider rows.")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic sampling seed.")
    parser.add_argument("--vendor", default="", help="Vendor override for --capture-root rows.")
    parser.add_argument("--source", default="phase2_capture", help="Source label for captures.")
    parser.add_argument("--glob-pattern", default="**/*.json", help="Capture filename glob.")
    return parser.parse_args(argv)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file into object rows."""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise SystemExit(f"{path}:{line_number}: JSONL row must be an object")
            rows.append(row)
    return rows


def _draft_text(row: Mapping[str, Any], *, index: int) -> str:
    """Return draft text from one provider row."""
    text = row.get("text", row.get("draft_text"))
    if not isinstance(text, str) or not text.strip():
        raise SystemExit(f"draft row {index} must contain non-empty text")
    return text


def _phase2_record(record: SourceRecord) -> Phase2RestApiRecord:
    """Convert a generic source record into the Phase 2 builder record shape."""
    methods = record.allowed_methods or ["GET", "HEAD"]
    return Phase2RestApiRecord(
        rest_api=record.url,
        allowed_methods=tuple(str(method).upper() for method in methods),
        json_body=record.response,
        vendor=record.vendor or "",
        source_corpus=str(record.provenance.get("corpus_id") or record.source),
    )


def _iter_sources(args: argparse.Namespace) -> Iterable[RedfishFixtureSource]:
    """Yield configured Redfish source adapters."""
    if args.corpus_manifest or args.corpus_root:
        if not args.corpus_manifest or not args.corpus_root:
            raise SystemExit("--corpus-manifest and --corpus-root must be set together")
        yield from RedfishFixtureSource.from_redfish_ctl_manifest(
            args.corpus_manifest,
            args.corpus_root,
            trust_level=TrustLevel.REAL,
            kind=args.corpus_kind,
            corpus_ids=args.corpus_id,
        )
    for index, root in enumerate(args.capture_root, start=1):
        source = args.source if len(args.capture_root) == 1 else f"{args.source}_{index}"
        yield RedfishFixtureSource(
            root,
            source=source,
            trust_level=TrustLevel.REAL,
            vendor=args.vendor or None,
            glob_pattern=args.glob_pattern,
        )


def _load_phase2_records(args: argparse.Namespace) -> list[Phase2RestApiRecord]:
    """Load Phase 2 REST API records from all configured source adapters."""
    sources = list(_iter_sources(args))
    if not sources:
        raise SystemExit("at least one --capture-root or --corpus-manifest is required")
    records: list[Phase2RestApiRecord] = []
    for source in sources:
        records.extend(_phase2_record(record) for record in source.iter_records())
    if not records:
        raise SystemExit("no Redfish records found for Phase 2 labelled-request build")
    return records


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    """Write JSONL rows and return the count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            count += 1
    return count


def _write_metrics(path: Path, summary: Mapping[str, Any]) -> None:
    """Write a metrics summary JSON document."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Run the offline Phase 2 labelled-request build."""
    args = parse_args(argv)
    spec = load_phase2_labelled_requests_spec(args.spec)
    records = _load_phase2_records(args)
    drafts = _read_jsonl(Path(args.drafts_jsonl))
    judges = _read_jsonl(Path(args.judges_jsonl))
    limit = args.max_samples or min(len(drafts), len(judges))
    if limit <= 0:
        raise SystemExit("no provider rows available")
    if len(drafts) < limit or len(judges) < limit:
        raise SystemExit("draft and judge JSONL files must cover --max-samples")

    widths = tuple(args.sample_width or spec.sample_widths)
    rng = random.Random(args.seed)
    counters = Phase2LabelledRequestCounters()
    accepted_rows: list[dict[str, Any]] = []

    for index in range(limit):
        width = widths[index % len(widths)]
        sample = sample_rest_api_records(
            records,
            k=width,
            rng=rng,
            allowed_widths=spec.sample_widths,
        )
        draft_text = _draft_text(drafts[index], index=index + 1)
        judge_raw = json.dumps(judges[index], sort_keys=True)
        result = parse_pro_judge_result(judge_raw, expected_rest_apis=sample.rest_api_list)
        vendor = sample.records[0].vendor if sample.records else ""
        source_corpus = sample.records[0].source_corpus if sample.records else ""
        counters.record_outcome(
            result,
            sample_width=sample.sample_width,
            vendor=vendor,
            source_corpus=source_corpus,
            spec=spec,
        )
        if result.accepted:
            accepted_rows.append(
                build_phase2_labelled_request_row(
                    spec,
                    sample,
                    draft_text=draft_text,
                    judge_result=result,
                )
            )

    accepted_count = _write_jsonl(Path(args.output_jsonl), accepted_rows)
    metrics = counters.to_wandb_metrics(spec)
    _write_metrics(
        Path(args.metrics_out),
        {
            "dataset": spec.dataset_name,
            "spec_version": spec.version,
            "drafted_rows": counters.draft_total,
            "accepted_rows": accepted_count,
            "rejected_rows": counters.rejected_total,
            "metrics": metrics,
        },
    )
    print(
        f"wrote {accepted_count} accepted rows from {counters.draft_total} drafts "
        f"to {args.output_jsonl}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
