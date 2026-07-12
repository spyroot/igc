#!/usr/bin/env bash
# Build the canonical M3 goal-planner JSONL from real Redfish captures.
#
# The default roots are the repo LFS-backed igc capture tree plus the
# redfish_ctl GB300 corpus. Override IGC_M3_JSON_ROOTS with a colon-separated
# list when a new real corpus is added.
#
# Author:
# Mus mbayramo@stanford.edu
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_JSONL="${IGC_M3_DATASET_OUT:-${ROOT_DIR}/datasets/m3/m3_goal_plans_full.jsonl}"
TEMPLATES_PER_ACTION="${IGC_M3_TEMPLATES_PER_ACTION:-3}"

DEFAULT_ROOTS=(
  "${ROOT_DIR}/datasets/orig"
  "${ROOT_DIR}/idrac_ctl/tests/supermicro_gb300_corpus/json_responses"
)

if [ -n "${IGC_M3_JSON_ROOTS:-}" ]; then
  IFS=":" read -r -a JSON_ROOTS <<< "${IGC_M3_JSON_ROOTS}"
else
  JSON_ROOTS=("${DEFAULT_ROOTS[@]}")
fi

ARGS=()
for root in "${JSON_ROOTS[@]}"; do
  if [ ! -d "${root}" ]; then
    echo "ERROR: M3 JSON root does not exist: ${root}" >&2
    exit 1
  fi
  ARGS+=(--json-root "${root}")
done

mkdir -p "$(dirname "${OUTPUT_JSONL}")"
PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}" \
  python -m igc.modules.m3_goal_planner_cli build-redfish-ctl-dataset \
    "${ARGS[@]}" \
    --output-jsonl "${OUTPUT_JSONL}" \
    --templates-per-action "${TEMPLATES_PER_ACTION}"

wc -l -c "${OUTPUT_JSONL}"


# Author: Mus mbayramo@stanford.edu
