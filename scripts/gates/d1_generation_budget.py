"""Gate: d1.generation_budget — Phase 2 D1 generation must run under finite caps.

Phase 2 builds the ``D1`` dataset by sampling API sets (widths 1/2/3) from the
Phase 1 ``D0`` corpus, asking ``model_x`` to draft the operator ``text`` for each
sampled set, and having the Pro judge accept or reject the draft. That loop is
open-ended by nature: a weak generator, a lenient judge, or a pathological corpus
combination can spin candidates forever. This gate refuses to let a D1 build start
unless the budget config declares **every** stopping limit as a finite, positive
integer — there is no "run until it looks done" mode.

The budget it enforces (all mandatory, all finite):

* ``sample_widths``               — the API-set widths to sample (e.g. ``[1, 2, 3]``).
* ``max_accepted_rows``           — hard cap on accepted D1 rows for the whole build.
* ``max_candidates``              — hard cap on draft/judge attempts for the whole build.
* ``max_accepted_per_combination``— accepted rows kept per sampled API combination.
* ``max_attempts_per_combination``— draft attempts spent per sampled API combination.
* ``max_accepted_per_api``        — accepted rows any single API may appear in.
* ``require_single_api_coverage`` — whether every selected API must be span-covered.

``validate_budget`` is the gate check: a non-empty list of messages means the config
is unsafe to run. ``should_stop`` / ``combination_full`` / ``api_full`` are the
predicates a generator loop calls to honour the global, per-combination, and per-API
caps. This is pure logic — no model, network, or torch — so it runs in the offline
CI gate; the real ``model_x`` + Pro-judge generation loop consumes the same config on
the cluster.

Used by:
* ``scripts/gates/d1_generation_budget.py:main`` validates the committed
  ``configs/gates/d1_budget.yaml`` in the offline gate.
* A Phase 2 D1 build loop calls ``should_stop`` / ``combination_full`` / ``api_full``
  each iteration so the caps in the same YAML actually bound the run.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

import yaml

# The finite integer caps every budget config must declare. Each must be a real,
# positive ``int`` (bool/float/None/<=0 are all unbounded-or-invalid and rejected).
REQUIRED_INT_LIMITS: tuple[str, ...] = (
    "max_accepted_rows",             # total accepted D1 rows for the build
    "max_candidates",                # total draft/judge attempts for the build
    "max_accepted_per_combination",  # accepted rows kept per sampled API combination
    "max_attempts_per_combination",  # draft attempts spent per sampled API combination
    "max_accepted_per_api",          # accepted rows any single API may appear in
)


def load_config(path: Path) -> dict[str, Any]:
    """Load the D1 budget YAML into a plain dict.

    :param path: path to the budget config (e.g. ``configs/gates/d1_budget.yaml``).
    :return: the parsed config mapping.
    :raises ValueError: if the file does not parse to a mapping.
    """
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"d1 budget config malformed (not a mapping): {path}")
    return data


def _is_positive_int(value: Any) -> bool:
    """True only for a real, positive ``int`` — never a bool, float, or None.

    ``bool`` is a subclass of ``int`` (``True == 1``); a boolean is never a valid
    numeric limit, so it is rejected here rather than silently read as ``1``.

    :param value: candidate limit value.
    :return: True if ``value`` is an ``int`` strictly greater than zero and not a bool.
    """
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _check_positive_int_limit(errors: list[str], config: Mapping[str, Any], key: str) -> None:
    """Append a message to ``errors`` if ``config[key]`` is not a finite positive int.

    Rejects the unbounded/invalid forms the gate exists to catch: missing key,
    ``None``/null, ``0`` or negative, ``bool``, and non-finite/non-integer floats
    (``.inf``, ``.nan``, ``2.5``).

    :param errors: accumulator list mutated in place.
    :param config: the budget config.
    :param key: the limit name to validate.
    """
    if key not in config:
        errors.append(f"{key}: missing (required finite limit)")
        return
    value = config[key]
    if isinstance(value, bool):
        errors.append(f"{key}: must be a positive int, got bool {value!r}")
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            errors.append(f"{key}: non-finite ({value!r}); the limit must be bounded")
        else:
            errors.append(f"{key}: must be an int, got float {value!r}")
        return
    if not _is_positive_int(value):
        errors.append(f"{key}: must be a positive int, got {value!r}")


def validate_budget(config: Mapping[str, Any]) -> list[str]:
    """Validate that every mandatory D1 generation limit is finite and positive.

    The gate fails (returns a non-empty list) when any required limit is missing,
    ``None``, non-finite, a non-positive int, a bool, or a float; when
    ``sample_widths`` is absent/empty or holds a non-positive-int width; or when
    ``require_single_api_coverage`` is missing or not a bool. An empty list means the
    config is safe to drive a bounded build.

    :param config: the parsed budget config.
    :return: list of human-readable failure messages (empty == valid).
    """
    errors: list[str] = []

    # Every global/per-combination/per-api cap must be a finite positive int.
    for key in REQUIRED_INT_LIMITS:
        _check_positive_int_limit(errors, config, key)

    # sample_widths: a non-empty list of positive-int widths (e.g. [1, 2, 3]).
    if "sample_widths" not in config:
        errors.append("sample_widths: missing (required)")
    else:
        widths = config["sample_widths"]
        if not isinstance(widths, (list, tuple)) or len(widths) == 0:
            errors.append(f"sample_widths: must be a non-empty list, got {widths!r}")
        else:
            for width in widths:
                if not _is_positive_int(width):
                    errors.append(f"sample_widths: each width must be a positive int, got {width!r}")

    # require_single_api_coverage: an explicit boolean policy flag, never implicit.
    if "require_single_api_coverage" not in config:
        errors.append("require_single_api_coverage: missing (required bool)")
    elif not isinstance(config["require_single_api_coverage"], bool):
        errors.append(
            f"require_single_api_coverage: must be a bool, got {config['require_single_api_coverage']!r}"
        )

    return errors


def should_stop(accepted_rows: int, attempted_candidates: int, config: Mapping[str, Any]) -> bool:
    """Global stop predicate: has the build hit its accepted-row or candidate cap?

    :param accepted_rows: accepted D1 rows so far in the build.
    :param attempted_candidates: draft/judge attempts so far in the build.
    :param config: the (validated) budget config.
    :return: True once ``accepted_rows >= max_accepted_rows`` OR
             ``attempted_candidates >= max_candidates``.
    """
    return (
        accepted_rows >= int(config["max_accepted_rows"])
        or attempted_candidates >= int(config["max_candidates"])
    )


def combination_full(accepted_in_combo: int, attempts_in_combo: int, config: Mapping[str, Any]) -> bool:
    """Per-combination stop predicate for one sampled API set.

    A sampled API combination is exhausted once it has yielded its allowed number of
    accepted rows or has burned its per-combination attempt budget (which protects a
    build from a combination the generator/judge can never satisfy).

    :param accepted_in_combo: accepted rows kept for this API combination.
    :param attempts_in_combo: draft attempts spent on this API combination.
    :param config: the (validated) budget config.
    :return: True once this combination reaches its accepted or attempt cap.
    """
    return (
        accepted_in_combo >= int(config["max_accepted_per_combination"])
        or attempts_in_combo >= int(config["max_attempts_per_combination"])
    )


def api_full(accepted_for_api: int, config: Mapping[str, Any]) -> bool:
    """Per-API stop predicate: is a single API already at its accepted-row cap?

    Keeps one popular API (e.g. a Systems collection) from dominating D1 while rare
    APIs stay under-represented.

    :param accepted_for_api: accepted rows this API already appears in.
    :param config: the (validated) budget config.
    :return: True once ``accepted_for_api >= max_accepted_per_api``.
    """
    return accepted_for_api >= int(config["max_accepted_per_api"])


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: validate the committed D1 budget config (offline gate).

    :param argv: optional argument vector (defaults to ``sys.argv``).
    :return: process exit code — 0 when the budget is valid, 1 otherwise.
    """
    parser = argparse.ArgumentParser(description="Validate the Phase 2 D1 generation budget.")
    parser.add_argument("--config", default="configs/gates/d1_budget.yaml")
    parser.add_argument("--out", default="reports/gate-report-d1-generation-budget.json")
    args = parser.parse_args(argv)

    config = load_config(Path(args.config))
    errors = validate_budget(config)
    report = {
        "gate": "d1.generation_budget",
        "config": args.config,
        "errors": errors,
        "valid": not errors,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps({"valid": report["valid"], "errors": errors}, indent=2))
    if errors:
        print("BLOCKER: D1 generation budget is missing a finite limit.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# Author: Mus mbayramo@stanford.edu
