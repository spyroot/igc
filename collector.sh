#!/usr/bin/env bash
# collector.sh — capture a Redfish resource tree for the igc dataset.
#
# Runs redfish_ctl's `discovery` crawl against ONE approved, non-production BMC and
# writes ~/.json_responses/<ip>/<path>.json per resource plus rest_api_map.npy — the
# numpy {"url_file_mapping", "allowed_methods_mapping"} dict the igc pipeline loads
# (igc/ds/ds_rest_trajectories.py). Output contract is unchanged from idrac_ctl.
#
# SAFETY: a naive recursive crawl has knocked a live BMC offline. Only run against an
# approved, NON-PRODUCTION host, with pacing. Never hardcode a host or credentials —
# they come from the environment below, never from this file or the repo.
#
# Config (environment only, no secrets in git):
#   REDFISH_IP / REDFISH_USERNAME / REDFISH_PASSWORD   target BMC + credentials (required;
#                                                      legacy IDRAC_* are still honored)
#   REDFISH_DISCOVERY_PACE_MS   delay between requests, ms   (default 200)
#   REDFISH_DISCOVERY_RETRIES   retries per request          (default 3)
#   REDFISH_DISCOVERY_BACKOFF   retry backoff factor         (default 2)
#   REDFISH_CTL_BIN             redfish_ctl binary to use    (default: redfish_ctl on PATH)
#                               The crawl tool currently lives in an isolated venv while the
#                               upstream requests-pin conflict with igc's datasets is resolved,
#                               e.g. REDFISH_CTL_BIN=$HOME/.venvs/redfish_ctl/bin/redfish_ctl
#
# Usage:
#   export REDFISH_IP=<approved-nonprod-bmc> REDFISH_USERNAME=... REDFISH_PASSWORD=...
#   ./collector.sh
set -euo pipefail

REDFISH_CTL_BIN="${REDFISH_CTL_BIN:-redfish_ctl}"

: "${REDFISH_IP:?set REDFISH_IP to an approved, non-production BMC — never a production host}"
: "${REDFISH_USERNAME:?set REDFISH_USERNAME (or legacy IDRAC_USERNAME)}"
: "${REDFISH_PASSWORD:?set REDFISH_PASSWORD (or legacy IDRAC_PASSWORD)}"

# Pacing defaults keep the crawl gentle on the BMC; override via the environment.
export REDFISH_DISCOVERY_PACE_MS="${REDFISH_DISCOVERY_PACE_MS:-200}"
export REDFISH_DISCOVERY_RETRIES="${REDFISH_DISCOVERY_RETRIES:-3}"
export REDFISH_DISCOVERY_BACKOFF="${REDFISH_DISCOVERY_BACKOFF:-2}"

echo "Redfish discovery crawl -> ~/.json_responses/${REDFISH_IP}/ (pace ${REDFISH_DISCOVERY_PACE_MS}ms, retries ${REDFISH_DISCOVERY_RETRIES})"
# redfish_ctl reads --idrac_ip/--idrac_username/--idrac_password from REDFISH_* (or legacy IDRAC_*).
exec "${REDFISH_CTL_BIN}" discovery
