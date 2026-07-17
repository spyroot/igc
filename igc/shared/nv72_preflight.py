"""Fleet-health preflight evaluation for GB300 training jobs.

TEAM_GUIDE makes checking the NV72 fleet dashboard a hard blocker before any
GB300 work: RoCE degraded, the shared ``/models`` mount missing, or a required
model endpoint absent means "stop and report a BLOCKER", never "submit anyway".
This module holds the pure decision logic — it never talks to the network.
``scripts/preflight_nv72.sh`` (which owns the curl against
``$NV72_FLEET_DASHBOARD_URL``, defined in the caller's environment) pipes the
``/api/v1/state`` JSON into :func:`main`; tests feed fixture payloads straight
into :func:`evaluate_state` per TEAM_GUIDE's offline-testing mandate.

Author:
Mus mbayramo@stanford.edu
"""

import argparse
import json
import sys
from typing import List, Optional, Tuple


def _required_int_count(
        section: dict,
        key: str,
        *,
        context: str,
        reasons: List[str],
        min_value: int = 0,
) -> Optional[int]:
    """Read a dashboard count field, rejecting missing or nonsensical values."""
    value = section.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < min_value:
        reasons.append(f"{context} has invalid {key}: {value!r}")
        return None
    return value


def evaluate_state(
        state: dict,
        require_endpoint: Optional[str] = None,
) -> Tuple[bool, List[str]]:
    """Decide whether the fleet is healthy enough to submit a GB300 job.

    :param state: the parsed ``/api/v1/state`` payload (or a test fixture).
    :param require_endpoint: optionally require a serving model endpoint group
        (e.g. ``"flash"`` or ``"pro"``) to be non-empty.
    :return: ``(ok, reasons)`` — ``reasons`` lists every blocker found.
    """
    reasons: List[str] = []
    summaries = state.get("summaries") or {}

    roce = summaries.get("roce") or {}
    if not roce:
        reasons.append("state has no summaries.roce section (dashboard degraded?)")
    else:
        nodes = _required_int_count(
            roce, "nodes", context="summaries.roce", reasons=reasons, min_value=1)
        rdma_active = _required_int_count(
            roce, "rdma_active", context="summaries.roce", reasons=reasons)
        if nodes is not None and rdma_active is not None and rdma_active != nodes:
            reasons.append(f"RoCE degraded: rdma_active {rdma_active}/{nodes}")

    mount = summaries.get("models_mount") or {}
    if not mount:
        reasons.append("state has no summaries.models_mount section")
    else:
        nodes = _required_int_count(
            mount, "nodes", context="summaries.models_mount", reasons=reasons, min_value=1)
        mounted = _required_int_count(
            mount, "mounted", context="summaries.models_mount", reasons=reasons)
        if nodes is not None and mounted is not None and mounted != nodes:
            reasons.append(f"/models mount degraded: mounted {mounted}/{nodes}")

    if require_endpoint is not None:
        endpoints = (state.get("model_endpoints") or {}).get(require_endpoint) or []
        if not endpoints:
            reasons.append(
                f"required model endpoint group {require_endpoint!r} has no serving instance")

    return (not reasons, reasons)


def main(argv: Optional[List[str]] = None) -> int:
    """Read the state JSON from stdin and print the verdict.

    :param argv: CLI args (``--require-endpoint flash|pro``).
    :return: process exit code — 0 healthy, 1 blocked, 2 unreadable input.
    """
    parser = argparse.ArgumentParser(description="NV72 fleet preflight verdict.")
    parser.add_argument(
        "--require-endpoint", default=None,
        help="also require this model endpoint group (e.g. flash, pro) to be serving.")
    args = parser.parse_args(argv)

    try:
        state = json.load(sys.stdin)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        print(f"BLOCKER: fleet state unreadable ({e}) — is the dashboard up?")
        return 2

    ok, reasons = evaluate_state(state, require_endpoint=args.require_endpoint)
    if ok:
        print("fleet preflight OK")
        return 0
    for reason in reasons:
        print(f"BLOCKER: {reason}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())


# Author: Mus mbayramo@stanford.edu
