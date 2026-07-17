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
git -C "$tmp" add .gitattributes
git -C "$tmp" commit -q -m "add lfs attributes"
printf '*.bin filter=lfs diff=lfs merge=lfs -text\n' >>"$tmp/.gitattributes"
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

grep -q "not matched by a committed LFS filter" "$out"
grep -q "Refusing to stage this artifact as a normal Git object." "$out"
test "$(git -C "$tmp" symbolic-ref --short HEAD)" = "$before_branch"

tracked="$tmp/tracked"
mkdir "$tracked"
git -C "$tracked" init -q
git -C "$tracked" config user.email test@example.invalid
git -C "$tracked" config user.name "IGC Test"
printf '*.bin filter=lfs diff=lfs merge=lfs -text\n' >"$tracked/.gitattributes"
git -C "$tracked" add .gitattributes
git -C "$tracked" commit -q -m "add committed lfs attributes"
printf 'payload\n' >"$tracked/raw.bin"
tracked_branch="$(git -C "$tracked" symbolic-ref --short HEAD)"
real_git="$(command -v git)"
mkdir "$tracked/fake-bin"
cat >"$tracked/fake-bin/git" <<EOF
#!/usr/bin/env bash
if [ "\${1:-}" = "lfs" ] && [ "\${2:-}" = "version" ]; then
    exit 127
fi
exec "$real_git" "\$@"
EOF
chmod +x "$tracked/fake-bin/git"

tracked_out="$tracked/lfs_push.out"
set +e
(
    cd "$tracked"
    PATH="$tracked/fake-bin:$PATH" IGC_YES=1 IGC_REMOTE=origin \
        "$repo_root/scripts/lfs_push_from_slot.sh" raw.bin
) >"$tracked_out" 2>&1
tracked_status=$?
set -e

cat "$tracked_out"

if [ "$tracked_status" -eq 0 ]; then
    echo "expected lfs_push_from_slot.sh to stop at the git-lfs availability gate" >&2
    exit 1
fi

grep -q "git-lfs not found" "$tracked_out"
test "$(git -C "$tracked" symbolic-ref --short HEAD)" = "$tracked_branch"

git diff --check
