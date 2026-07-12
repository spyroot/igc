#!/usr/bin/env python
"""Safe dispatcher for spec-driven igc launch plans.

This first implementation slice supports dry-run rendering only. Live Docker
and Slurm mutation will be added behind explicit backend adapters after the
spec contract and tests are stable.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from igc.shared.run_spec import RunSpecError, load_run_spec, render_plan


def main(argv: list[str] | None = None) -> int:
    """Render or refuse a run spec."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, help="YAML run spec to launch")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="render the launch plan without executing it",
    )
    args = parser.parse_args(argv)

    if not args.dry_run:
        print(
            "RUN SPEC ERROR: refusing live launch without --dry-run in this "
            "implementation slice",
            file=sys.stderr,
        )
        return 2

    try:
        spec = load_run_spec(args.spec)
        sys.stdout.write(render_plan(spec))
    except RunSpecError as exc:
        print(f"RUN SPEC ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
