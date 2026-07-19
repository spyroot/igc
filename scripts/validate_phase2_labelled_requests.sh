#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"

echo "phase2_labelled_requests validation start"
git rev-parse --short HEAD
pytest -q -ra \
    tests/scripts/test_phase2_labelled_requests_cli.py \
    tests/ds/test_phase2_labelled_requests.py \
    tests/ds/test_rest_goal_contract.py \
    tests/modules/test_phase2_labelled_request_metric_keys.py \
    tests/scripts/test_phase3_argument_smoke.py
echo "phase2_labelled_requests validation ok"
