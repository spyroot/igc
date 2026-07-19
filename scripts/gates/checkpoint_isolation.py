"""Gate: checkpoint.isolation — Phase 2/3 outputs must never clobber model_x.

Phase 1 Redfish JSON pretraining produces the ``model_x`` checkpoint on BeeGFS
``/models`` (the accepted Phase 1 artifact). Phase 2 REST-goal extraction and
Phase 3 method/argument extraction each write their OWN adapter/checkpoint
outputs, and those output directories must be **disjoint** from the ``model_x``
checkpoint directory. A Phase 2/3 run that writes at, under, or above the
Phase 1 checkpoint path would overwrite or truncate the accepted ``model_x``
artifact — silently poisoning every downstream phase that loads it.

This gate is pure path logic: given the resolved ``model_x`` directory and the
Phase 2 / Phase 3 output directories, it reports a violation when an output path
is equal to, nested under, or a parent of ``model_x_dir`` (checked with
``Path.is_relative_to`` in both directions). It performs no filesystem writes and
loads no model, so it runs in the offline CI gate.

Used by: the mount/output launch gate evidence step — call ``check_isolation``
with the rendered Phase 1 checkpoint dir and the Phase 2/3 output dirs before a
Phase 2/3 GPU launch; a non-empty return marks the launch ``BLOCKED``.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _overlaps(a: Path, b: Path) -> bool:
    """Return True if ``a`` and ``b`` are the same path or one contains the other.

    Overlap means writes to one directory can touch the other: identical paths,
    ``a`` nested under ``b``, or ``b`` nested under ``a``. Both directions are
    checked so a parent-directory overwrite is caught, not just a child one.

    :param a: first resolved directory path.
    :param b: second resolved directory path.
    :return: True if the two directories overlap (unsafe), else False.
    """
    return a == b or a.is_relative_to(b) or b.is_relative_to(a)


def check_isolation(model_x_dir: str, phase2_out: str, phase3_out: str) -> list[str]:
    """Assert Phase 2/3 outputs never overwrite the Phase 1 ``model_x`` checkpoint.

    Fails if ``phase2_out`` or ``phase3_out`` is equal to, under, or a parent of
    ``model_x_dir``. Paths are resolved before comparison so ``.``/``..`` and
    symlink-free relative forms normalize to the same absolute path; identical
    paths are treated as a violation.

    :param model_x_dir: the Phase 1 ``model_x`` checkpoint directory (protected).
    :param phase2_out: the Phase 2 REST-goal extraction output directory.
    :param phase3_out: the Phase 3 method/argument extraction output directory.
    :return: list of human-readable violation strings; empty means isolated/safe.
    """
    model_x = Path(model_x_dir).resolve()
    violations: list[str] = []

    for label, raw in (("phase2_out", phase2_out), ("phase3_out", phase3_out)):
        candidate = Path(raw).resolve()
        if not _overlaps(candidate, model_x):
            continue
        if candidate == model_x:
            relation = "is identical to"
        elif candidate.is_relative_to(model_x):
            relation = "is nested under"
        else:  # model_x is nested under candidate -> parent-directory overwrite
            relation = "is a parent of"
        violations.append(
            f"{label} ({candidate}) {relation} model_x_dir ({model_x}); "
            "Phase 2/3 output would overwrite the Phase 1 checkpoint"
        )

    return violations


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the checkpoint-isolation gate.

    :param argv: optional argument vector (defaults to ``sys.argv``).
    :return: process exit code (0 = isolated, 1 = violation found).
    """
    parser = argparse.ArgumentParser(
        description="Assert Phase 2/3 outputs never overwrite the Phase 1 model_x checkpoint."
    )
    parser.add_argument("--model-x-dir", required=True, help="Phase 1 model_x checkpoint dir")
    parser.add_argument("--phase2-out", required=True, help="Phase 2 output dir")
    parser.add_argument("--phase3-out", required=True, help="Phase 3 output dir")
    args = parser.parse_args(argv)

    violations = check_isolation(args.model_x_dir, args.phase2_out, args.phase3_out)
    if violations:
        for v in violations:
            print(f"BLOCKER: {v}", file=sys.stderr)
        return 1
    print("checkpoint.isolation: OK — Phase 2/3 outputs are disjoint from model_x.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# Author: Mus mbayramo@stanford.edu
