#!/usr/bin/env python3
"""Build the Phase 1 train/held-out corpus from the YAML source registry."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from igc.ds.source_registry import materialize_phase1_registry_corpus  # noqa: E402
from igc.ds.phase1_chunking import load_phase1_chunking_policy  # noqa: E402
from igc.modules.train.profiles import resolve_profile  # noqa: E402


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
    parser.add_argument(
        "--training-profile",
        required=True,
        help="Profile whose model, tokenizer SHA, and seq_len define chunking",
    )
    parser.add_argument("--summary-json", required=True)
    return parser.parse_args(argv)


def _load_materialization_runtime(profile_name: str):
    """Load the profile-pinned tokenizer only on the approved build surface."""

    from transformers import AutoTokenizer

    profile = resolve_profile(profile_name)
    if profile.task != "redfish_json_reconstruction":
        raise ValueError("Phase 1 materialization requires a reconstruction profile")
    if not profile.tokenizer_sha:
        raise ValueError("Phase 1 training profile must pin tokenizer_sha")
    tokenizer = AutoTokenizer.from_pretrained(profile.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    policy = load_phase1_chunking_policy(
        tokenizer_sha=profile.tokenizer_sha,
        max_tokens=profile.seq_len,
    )
    return profile, tokenizer, policy


def main(argv: list[str] | None = None) -> int:
    """Materialize all required sources and write one public-safe summary."""
    args = parse_args(argv)
    try:
        profile, tokenizer, chunking_policy = _load_materialization_runtime(
            args.training_profile
        )
        summary = materialize_phase1_registry_corpus(
            registry_path=args.source_registry,
            output_root=args.output_root,
            corpus_kind=args.corpus_kind,
            eval_fraction=args.eval_fraction,
            seed=args.seed,
            tokenizer=tokenizer,
            chunking_policy=chunking_policy,
        )
        summary["training_profile"] = profile.name
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
            "training_profile": profile.name,
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
