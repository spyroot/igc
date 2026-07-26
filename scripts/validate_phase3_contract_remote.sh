#!/usr/bin/env bash
# Focused validation for the Phase 3 contract in an approved remote environment.
set -euo pipefail

export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

python -m pytest -q \
    tests/ds/test_rest_goal_contract.py \
    tests/scripts/test_phase3_argument_smoke.py

bash -n scripts/validate_phase3_contract_remote.sh

ruff check \
    igc/ds/rest_goal_contract.py \
    scripts/build_phase3_argument_smoke.py \
    tests/ds/test_rest_goal_contract.py \
    tests/scripts/test_phase3_argument_smoke.py

git diff --check
