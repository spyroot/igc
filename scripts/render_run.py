#!/usr/bin/env python
"""Render an igc run spec without executing Docker or Slurm."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from igc.shared.run_spec import RunSpecError, load_run_spec, render_plan


def main(argv: list[str] | None = None) -> int:
    """Render a run spec and return a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, help="YAML run spec to render")
    args = parser.parse_args(argv)

    try:
        spec = load_run_spec(args.spec)
        sys.stdout.write(render_plan(spec))
    except RunSpecError as exc:
        print(f"RUN SPEC ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
