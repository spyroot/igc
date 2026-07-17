#!/usr/bin/env bash
# Focused validation for the slot-side LFS artifact push helper.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

./tests/shell/test_lfs_push_from_slot.sh

if ! command -v bats >/dev/null 2>&1; then
  if command -v apt-get >/dev/null 2>&1 && [ "$(id -u)" = "0" ]; then
    apt-get update -qq
    apt-get install -y --no-install-recommends bats
  fi
fi

command -v bats >/dev/null 2>&1 || {
  echo "ERROR: bats is required for tests/bats/lfs_push_from_slot.bats" >&2
  exit 2
}

bats tests/bats/lfs_push_from_slot.bats
