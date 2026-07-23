#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "phase2_labelled_requests validation start"
git rev-parse --short HEAD
scripts/check.sh --profile phase2_labelled_requests --category unit
echo "phase2_labelled_requests validation ok"
