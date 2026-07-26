#!/usr/bin/env python3
"""Build the Phase 1 train/held-out corpus from the YAML source registry."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from igc.ds.source_registry import materialize_phase1_registry_corpus  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse source-registry and deterministic split settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-registry",
        default="configs/data/redfish_sources.yaml",
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--corpus-kind", default="dataset")
    parser.add_argument("--eval-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--summary-json", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Materialize all required sources and write one public-safe summary."""
    args = parse_args(argv)
    try:
        summary = materialize_phase1_registry_corpus(
            registry_path=args.source_registry,
            output_root=args.output_root,
            corpus_kind=args.corpus_kind,
            eval_fraction=args.eval_fraction,
            seed=args.seed,
        )
        destination = Path(args.summary_json)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({
            "status": "pass",
            "train_rows": summary["train_rows"],
            "heldout_rows": summary["heldout_rows"],
            "summary": str(destination),
        }, sort_keys=True))
        return 0
    except (KeyError, OSError, TypeError, ValueError) as exc:
        print(json.dumps({
            "status": "blocked",
            "error": str(exc),
            "safe_next_step": "repair the source registry or its named remote environment",
        }), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
