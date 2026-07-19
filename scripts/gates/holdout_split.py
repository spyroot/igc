"""Gate: holdout.split-verification — P1-GOLDEN scored the split it trained on.

Phase 1 acceptance compares a golden-producer eval score against the model that was
trained; that comparison is only honest if BOTH ran over the *same* deterministic
train/eval split, with no train/eval overlap. Each run stamps two fair-comparison
keys into its run manifest via ``DataManifest.to_run_manifest_fields``
(``igc/ds/sources/mixer.py``):

* ``data_manifest`` — ``DataManifest.content_hash()``, the exact record mix.
* ``eval_split``    — ``DataManifest.eval_split_id()``, the split policy
  (trust floor / eval fraction / seed) that drew the held-out set.

This gate fails when the P1-GOLDEN producer and the training run disagree on either
key, or when either key is missing/empty on either side — a mismatch means the golden
score was computed on a different corpus or a different held-out split than the model
saw, so it cannot certify no train/eval overlap.

Offline (CI): pure dict comparison of two already-serialized manifest-field mappings
(the producer's and the training run's). No cluster read, no model load, no network.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

# The two fair-comparison keys DataManifest.to_run_manifest_fields() emits; both must
# match across the golden producer and the training run for the split to be verified.
REQUIRED_FIELDS: tuple[str, ...] = ("data_manifest", "eval_split")


def _is_missing(value: Any) -> bool:
    """True when a manifest field is absent or blank.

    A field counts as missing when it is ``None`` or (for a string) empty/whitespace;
    an empty hash or split id can never certify a split.

    :param value: the field value pulled from a manifest-field mapping.
    :return: True if the value cannot stand in as a real key.
    """
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def verify_split(producer_fields: Mapping[str, Any],
                 training_fields: Mapping[str, Any]) -> list[str]:
    """Check that a P1-GOLDEN producer scored the split the model trained on.

    Compares the two fair-comparison keys (:data:`REQUIRED_FIELDS`) between the golden
    producer's manifest fields and the training run's manifest fields. Fails when a key
    is missing/empty on either side, or when the two sides disagree on a key — either
    condition means the held-out eval set was not provably the one the model trained
    against, so train/eval non-overlap is unverified.

    :param producer_fields: the P1-GOLDEN producer's ``to_run_manifest_fields`` mapping.
    :param training_fields: the training run's ``to_run_manifest_fields`` mapping.
    :return: a list of human-readable violation strings; empty means the split is
        verified (same data mix, same split policy on both sides).
    """
    violations: list[str] = []
    for key in REQUIRED_FIELDS:
        producer_value = producer_fields.get(key)
        training_value = training_fields.get(key)
        producer_missing = _is_missing(producer_value)
        training_missing = _is_missing(training_value)
        if producer_missing:
            violations.append(f"producer manifest missing '{key}'")
        if training_missing:
            violations.append(f"training manifest missing '{key}'")
        # Only a real mismatch is worth reporting when both sides supplied a value.
        if not producer_missing and not training_missing and producer_value != training_value:
            violations.append(
                f"{key} mismatch: producer={producer_value!r} != training={training_value!r}"
            )
    return violations


def run(producer_fields: Mapping[str, Any],
        training_fields: Mapping[str, Any]) -> dict[str, Any]:
    """Run the split verification and build the gate report.

    :param producer_fields: the P1-GOLDEN producer's manifest fields.
    :param training_fields: the training run's manifest fields.
    :return: a report dict with the checked keys, violations, and a ``verified`` flag.
    """
    violations = verify_split(producer_fields, training_fields)
    return {
        "gate": "holdout.split-verification",
        "required_fields": list(REQUIRED_FIELDS),
        "producer_fields": {k: producer_fields.get(k) for k in REQUIRED_FIELDS},
        "training_fields": {k: training_fields.get(k) for k in REQUIRED_FIELDS},
        "violations": violations,
        "verified": not violations,
    }


def _load_fields(path: Path) -> dict[str, Any]:
    """Load one run/producer manifest JSON and return its fair-comparison keys.

    Accepts either a bare ``{data_manifest, eval_split}`` mapping or a larger run
    manifest that carries those keys among others.

    :param path: path to a JSON file holding the manifest fields.
    :return: a mapping restricted to :data:`REQUIRED_FIELDS` present in the file.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"manifest fields malformed (expected an object): {path}")
    return {k: data[k] for k in REQUIRED_FIELDS if k in data}


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the holdout split-verification gate (offline dict compare)."""
    parser = argparse.ArgumentParser(
        description="Verify the P1-GOLDEN producer scored the split the model trained on.")
    parser.add_argument("--producer", required=True,
                        help="JSON with the producer's data_manifest / eval_split fields.")
    parser.add_argument("--training", required=True,
                        help="JSON with the training run's data_manifest / eval_split fields.")
    parser.add_argument("--out", default="reports/gate-report-holdout-split.json")
    args = parser.parse_args(argv)

    report = run(_load_fields(Path(args.producer)), _load_fields(Path(args.training)))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps({"verified": report["verified"], "violations": report["violations"]}, indent=2))
    if not report["verified"]:
        print("BLOCKER: golden producer did not score the split the model trained on.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# Author: Mus mbayramo@stanford.edu
