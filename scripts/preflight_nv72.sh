#!/usr/bin/env bash
# preflight_nv72.sh — mandatory fleet-health gate before any GB300 job.
#
# TEAM_GUIDE's blocker rule: check the NV72 fleet dashboard before GB300 work;
# an unavailable API or unhealthy state (RoCE down, /models unmounted, required
# model endpoint absent) is a hard stop. The dashboard address comes ONLY from
# the environment — never hardcoded (internal endpoint policy):
#
#   export NV72_FLEET_DASHBOARD_URL=...   # from your ops notes / TEAM_GUIDE
#   ./scripts/preflight_nv72.sh                       # infra health only
#   ./scripts/preflight_nv72.sh --require-endpoint pro  # also require Pro serving
#
# Decision logic lives in igc/shared/nv72_preflight.py (offline-unit-tested);
# this wrapper only owns the curl.
set -euo pipefail

: "${NV72_FLEET_DASHBOARD_URL:?set NV72_FLEET_DASHBOARD_URL (internal dashboard base URL) before GB300 work}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! STATE_JSON=$(curl -fsS --max-time "${NV72_PREFLIGHT_TIMEOUT:-10}" \
        "${NV72_FLEET_DASHBOARD_URL}/api/v1/state"); then
    echo "BLOCKER: fleet dashboard unreachable at \$NV72_FLEET_DASHBOARD_URL/api/v1/state" >&2
    exit 1
fi

printf '%s' "${STATE_JSON}" | PYTHONPATH="${HERE}" "${PYTHON_BIN}" -m igc.shared.nv72_preflight "$@"
