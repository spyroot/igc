#!/usr/bin/env python3
"""Build GoalExtractor / GoalEncoder datasets from captured Redfish JSON.

The script is offline by default. It always writes deterministic goal surfaces
(``true_y``). It can optionally call a configurable paraphrase provider to draft
operator text (``x``). The model provider only writes natural language; labels
come from captured JSON and are never copied from model output.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Running as ``python scripts/build_goal_dataset.py`` puts scripts/ on sys.path;
# add the repo root so ``import igc`` works without installation.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from igc.ds.goal_dataset import write_goal_surfaces
from igc.ds.goal_dataset_builder import build_goal_surfaces
from igc.ds.goal_paraphrases import (
    OpenAICompatibleParaphraseProvider,
    StaticParaphraseProvider,
    generate_goal_text_drafts,
)
from igc.ds.sources import RedfishFixtureSource, TrustLevel

_VENDOR_ROOT_MARKERS = (
    ("idrac_fixtures", "dell"),
    ("dell", "dell"),
    ("hpe_fixtures", "hpe"),
    ("ilo", "hpe"),
    ("supermicro_gb300_corpus", "supermicro"),
    ("supermicro_fixtures", "supermicro"),
    ("supermicro", "supermicro"),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--capture-root",
        action="append",
        required=True,
        help="Directory of captured Redfish JSON files; may be repeated.",
    )
    parser.add_argument("--vendor", default="", help="Vendor label for emitted SourceRecords.")
    parser.add_argument("--source", default="real", help="Source label for emitted SourceRecords.")
    parser.add_argument(
        "--surfaces-out",
        required=True,
        help="Output JSONL path for deterministic GoalSurface rows.",
    )
    parser.add_argument(
        "--manifest-out",
        default="",
        help="Optional JSON path for dataset counts and input summary.",
    )
    parser.add_argument(
        "--text-out",
        default="",
        help="Optional output JSONL path for LLM-drafted GoalTextExample rows.",
    )
    parser.add_argument(
        "--paraphrase-mode",
        choices=("none", "static", "openai"),
        default="none",
        help="How to produce candidate operator text.",
    )
    parser.add_argument(
        "--goal-id",
        action="append",
        default=[],
        help="Goal IDs to include in one generated text target.",
    )
    parser.add_argument(
        "--generate-all-goals",
        action="store_true",
        help="Generate text rows for every unique atomic GoalRef discovered.",
    )
    parser.add_argument(
        "--static-text",
        action="append",
        default=[],
        help="Candidate text for --paraphrase-mode static; may be repeated.",
    )
    parser.add_argument(
        "--base-url-env",
        default="GOAL_PARAPHRASE_BASE_URL",
        help="Env var holding OpenAI-compatible base URL for --paraphrase-mode openai.",
    )
    parser.add_argument(
        "--model-env",
        default="GOAL_PARAPHRASE_MODEL",
        help="Env var holding model name for --paraphrase-mode openai.",
    )
    parser.add_argument(
        "--api-key-env",
        default="GOAL_PARAPHRASE_API_KEY",
        help="Env var holding optional API key for --paraphrase-mode openai.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=8,
        help="Requested paraphrase count.",
    )
    parser.add_argument(
        "--glob-pattern",
        default="*.json",
        help="Capture filename glob passed to RedfishFixtureSource; use **/*.json for trees.",
    )
    return parser.parse_args(argv)


def _infer_vendor_from_root(root: str) -> str:
    """Infer vendor provenance from known redfish_ctl corpus root names."""
    normalized = [part.lower() for part in Path(root).parts]
    for marker, vendor in _VENDOR_ROOT_MARKERS:
        if marker in normalized:
            return vendor
    return ""


def _load_records(args: argparse.Namespace):
    """Load all capture roots into SourceRecord rows."""
    records = []
    for root in args.capture_root:
        vendor = args.vendor or _infer_vendor_from_root(root) or None
        source = RedfishFixtureSource(
            root,
            source=args.source,
            trust_level=TrustLevel.REAL,
            vendor=vendor,
            glob_pattern=args.glob_pattern,
        )
        records.extend(source.iter_records())
    return records


def _provider(args: argparse.Namespace):
    """Build the requested paraphrase provider."""
    if args.paraphrase_mode == "static":
        return StaticParaphraseProvider(tuple(args.static_text))
    if args.paraphrase_mode == "openai":
        base_url = os.environ.get(args.base_url_env, "")
        model = os.environ.get(args.model_env, "")
        if not base_url or not model:
            raise SystemExit(
                f"missing {args.base_url_env} or {args.model_env} for --paraphrase-mode openai"
            )
        return OpenAICompatibleParaphraseProvider(
            base_url=base_url,
            model=model,
            api_key=os.environ.get(args.api_key_env, ""),
        )
    return None


def _write_text_examples(args: argparse.Namespace, surfaces) -> int:
    """Generate and write draft text examples when requested."""
    if not args.text_out or args.paraphrase_mode == "none":
        return 0
    by_id = {surface.goal_ref.goal_id: surface.goal_ref for surface in surfaces}
    missing = [goal_id for goal_id in args.goal_id if goal_id not in by_id]
    if missing:
        raise SystemExit(f"unknown goal id(s): {', '.join(missing)}")
    if args.goal_id and args.generate_all_goals:
        raise SystemExit("--goal-id and --generate-all-goals are mutually exclusive")
    if not args.goal_id and not args.generate_all_goals:
        raise SystemExit(
            "--goal-id or --generate-all-goals is required when writing text examples"
        )

    provider = _provider(args)
    targets = (
        [(tuple(by_id[goal_id] for goal_id in args.goal_id))]
        if args.goal_id
        else [(by_id[goal_id],) for goal_id in sorted(by_id)]
    )
    examples = []
    for refs in targets:
        examples.extend(generate_goal_text_drafts(
            provider,
            goal_refs=refs,
            count=args.count,
        ))
    from igc.ds.goal_dataset import GoalTextExample

    GoalTextExample.write_jsonl(Path(args.text_out), examples)
    return len(examples)


def _manifest(args: argparse.Namespace, records, surfaces, num_text: int) -> dict:
    """Build a public-safe dataset count manifest."""
    goal_ids = {surface.goal_ref.goal_id for surface in surfaces}
    vendors = sorted({record.vendor for record in records if record.vendor})
    sources = sorted({record.source for record in records if record.source})
    by_resource_type: dict[str, int] = {}
    by_family: dict[str, int] = {}
    for surface in surfaces:
        by_resource_type[surface.goal_ref.resource_type] = (
            by_resource_type.get(surface.goal_ref.resource_type, 0) + 1
        )
        by_family[surface.goal_ref.family] = by_family.get(surface.goal_ref.family, 0) + 1
    return {
        "capture_roots": list(args.capture_root),
        "glob_pattern": args.glob_pattern,
        "capture_records": len(records),
        "goal_surfaces": len(surfaces),
        "unique_goal_ids": len(goal_ids),
        "text_examples": num_text,
        "sources": sources,
        "vendors": vendors,
        "surfaces_out": args.surfaces_out,
        "text_out": args.text_out,
        "by_goal_family": dict(sorted(by_family.items())),
        "by_resource_type": dict(sorted(by_resource_type.items())),
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint.

    :return: process-style exit code.
    """
    args = parse_args(argv)
    records = _load_records(args)
    surfaces = build_goal_surfaces(records)
    write_goal_surfaces(Path(args.surfaces_out), surfaces)
    num_text = _write_text_examples(args, surfaces)
    manifest = _manifest(args, records, surfaces, num_text)
    if args.manifest_out:
        Path(args.manifest_out).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(
        f"wrote capture_records={manifest['capture_records']} "
        f"surfaces={manifest['goal_surfaces']} "
        f"unique_goal_ids={manifest['unique_goal_ids']} "
        f"text_examples={manifest['text_examples']} "
        f"surfaces_out={args.surfaces_out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# Author: Mus mbayramo@stanford.edu
