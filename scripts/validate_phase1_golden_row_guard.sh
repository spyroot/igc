#!/usr/bin/env bash
# Focused CPU-only validation for Phase 1 golden row-key guard changes.
set -euo pipefail

if conda env list | awk '{print $1}' | grep -qx 'igc-dev'; then
    py=(conda run -n igc-dev python)
    lint=(conda run -n igc-dev ruff)
else
    py=(python)
    if command -v ruff >/dev/null 2>&1; then
        lint=(ruff)
    elif command -v uv >/dev/null 2>&1; then
        lint=(uvx ruff)
    else
        echo "BLOCKED: ruff is unavailable in this validation container" >&2
        exit 2
    fi
fi

env \
    PYTHONFAULTHANDLER=1 \
    KMP_DUPLICATE_LIB_OK=TRUE \
    OMP_NUM_THREADS=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    "${py[@]}" -m pytest -q \
    tests/modules/train/test_phase1_golden.py \
    tests/scripts/test_phase1_inference_gate.py \
    -rA

"${lint[@]}" check \
    igc/modules/train/phase1_golden.py \
    tests/modules/train/test_phase1_golden.py \
    scripts/phase1_inference_gate.py \
    tests/scripts/test_phase1_inference_gate.py
