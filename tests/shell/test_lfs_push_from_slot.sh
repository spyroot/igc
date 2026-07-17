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

# A committed *.bin filter matches raw.bin, so the path check would pass; but a
# git whose `check-attr --source` errors (e.g. git < 2.40, no --source option)
# must die with its own distinct message, not the misleading committed-filter
# refusal. Mirrors the bats "check-attr command failure" regression so the slot
# validator covers it without bats installed.
attrfail="$tmp/attrfail"
mkdir "$attrfail"
git -C "$attrfail" init -q
git -C "$attrfail" config user.email test@example.invalid
git -C "$attrfail" config user.name "IGC Test"
printf '*.bin filter=lfs diff=lfs merge=lfs -text\n' >"$attrfail/.gitattributes"
git -C "$attrfail" add .gitattributes
git -C "$attrfail" commit -q -m "add committed lfs attributes"
printf 'payload\n' >"$attrfail/raw.bin"
attrfail_branch="$(git -C "$attrfail" symbolic-ref --short HEAD)"
mkdir "$attrfail/fake-bin"
cat >"$attrfail/fake-bin/git" <<EOF
#!/usr/bin/env bash
if [ "\${1:-}" = "check-attr" ] && [ "\${2:-}" = "--source" ]; then
    echo "error: unknown option --source" >&2
    exit 129
fi
exec "$real_git" "\$@"
EOF
chmod +x "$attrfail/fake-bin/git"

attrfail_out="$attrfail/lfs_push.out"
set +e
(
    cd "$attrfail"
    PATH="$attrfail/fake-bin:$PATH" IGC_YES=1 IGC_REMOTE=origin \
        "$repo_root/scripts/lfs_push_from_slot.sh" raw.bin
) >"$attrfail_out" 2>&1
attrfail_status=$?
set -e

cat "$attrfail_out"

if [ "$attrfail_status" -eq 0 ]; then
    echo "expected lfs_push_from_slot.sh to fail when check-attr --source errors" >&2
    exit 1
fi

grep -q "cannot read committed LFS attributes" "$attrfail_out"
grep -q "unknown option --source" "$attrfail_out"
if grep -q "not matched by a committed LFS filter" "$attrfail_out"; then
    echo "check-attr command failure must not print the filter-mismatch refusal" >&2
    exit 1
fi
test "$(git -C "$attrfail" symbolic-ref --short HEAD)" = "$attrfail_branch"

# An unborn HEAD (git init, no commit yet) must fail closed before any artifact
# is staged and must not print the misleading filter-mismatch refusal. Mirrors
# the bats "fails closed on an unborn HEAD" regression.
unborn="$tmp/unborn"
mkdir "$unborn"
git -C "$unborn" init -q
git -C "$unborn" config user.email test@example.invalid
git -C "$unborn" config user.name "IGC Test"
printf '*.bin filter=lfs diff=lfs merge=lfs -text\n' >"$unborn/.gitattributes"
printf 'payload\n' >"$unborn/raw.bin"

unborn_out="$unborn/lfs_push.out"
set +e
(
    cd "$unborn"
    IGC_YES=1 IGC_REMOTE=origin \
        "$repo_root/scripts/lfs_push_from_slot.sh" raw.bin
) >"$unborn_out" 2>&1
unborn_status=$?
set -e

cat "$unborn_out"

if [ "$unborn_status" -eq 0 ]; then
    echo "expected lfs_push_from_slot.sh to fail closed on an unborn HEAD" >&2
    exit 1
fi

if grep -q "not matched by a committed LFS filter" "$unborn_out"; then
    echo "unborn HEAD must not print the filter-mismatch refusal" >&2
    exit 1
fi
unborn_staged="$(git -C "$unborn" status --porcelain -- raw.bin)"
case "$unborn_staged" in
    A*)
        echo "unborn HEAD run must not stage the artifact" >&2
        exit 1
        ;;
esac

git diff --check
