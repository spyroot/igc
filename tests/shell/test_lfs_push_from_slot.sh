#!/usr/bin/env bash
# Focused regression for scripts/lfs_push_from_slot.sh.
#
# Runs without network or git-lfs: the unsafe-path failure must happen before
# the helper checks for a git-lfs binary, creates a data branch, or stages files.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

bash -n scripts/lfs_push_from_slot.sh

tmp="$(mktemp -d)"
cleanup() { rm -rf "$tmp"; }
trap cleanup EXIT

git -C "$tmp" init -q
git -C "$tmp" config user.email test@example.invalid
git -C "$tmp" config user.name "IGC Test"
printf '*.safetensors filter=lfs diff=lfs merge=lfs -text\n' >"$tmp/.gitattributes"
printf 'payload\n' >"$tmp/raw.bin"
before_branch="$(git -C "$tmp" symbolic-ref --short HEAD)"

out="$tmp/lfs_push.out"
set +e
(
    cd "$tmp"
    IGC_YES=1 IGC_REMOTE=origin "$repo_root/scripts/lfs_push_from_slot.sh" raw.bin
) >"$out" 2>&1
status=$?
set -e

cat "$out"

if [ "$status" -eq 0 ]; then
    echo "expected lfs_push_from_slot.sh to reject non-LFS artifact" >&2
    exit 1
fi

grep -q "not matched by an LFS filter" "$out"
grep -q "Refusing to stage this artifact as a normal Git object." "$out"
test "$(git -C "$tmp" symbolic-ref --short HEAD)" = "$before_branch"

git diff --check
