#!/usr/bin/env bash
# Verify that a trained weight artifact is represented in Git by a Git LFS
# pointer, without hydrating/downloading the weight blob.
#
# Typical post-run usage on slot2/slot11 after pushing an artifact branch:
#   scripts/verify_lfs_weight_ack.sh \
#       --ref data/model-x-phase1-adapter-20260715T1024Z \
#       --path models/model_x/.../adapter_model.safetensors \
#       --expected-size 323014168 \
#       --expected-sha256 98243e5a7d5311bfac4856fa56684eea38327b5e3d4ab806d6b53f07a764d5e4 \
#       --remote origin
#
# The optional --remote check uses "git lfs fetch --dry-run"; it contacts the
# LFS remote but does not download the object.
set -euo pipefail

die() {
    printf 'BLOCKER: %s\n' "$*" >&2
    exit 1
}

usage() {
    cat <<'EOF'
Usage:
  verify_lfs_weight_ack.sh --ref REF --path PATH [options]
  verify_lfs_weight_ack.sh REF PATH [options]

Required:
  --ref REF                  Git ref/branch/tag containing the artifact pointer
  --path PATH                Artifact path inside REF

Options:
  --expected-size BYTES      Require the LFS pointer size to match BYTES
  --expected-sha256 SHA      Require the LFS pointer oid sha256 to match SHA
  --remote REMOTE            Verify the LFS remote with git lfs fetch --dry-run
  --remote-timeout-seconds N Bound LFS remote dial/activity timeouts (default: 30)
  --max-pointer-bytes N      Max Git blob size allowed for pointer (default: 4096)
  -h, --help                 Show this help

Success output contains one machine-readable line:
  LFS_ARTIFACT_ACK ref=<ref> path=<path> sha256=<sha> size=<bytes> remote=<state>
EOF
}

ref=""
artifact_path=""
expected_size=""
expected_sha=""
remote=""
max_pointer_bytes=4096
remote_timeout_seconds=30

if [ "$#" -ge 2 ] && [ "${1#--}" = "$1" ] && [ "${2#--}" = "$2" ]; then
    ref="$1"
    artifact_path="$2"
    shift 2
fi

while [ "$#" -gt 0 ]; do
    case "$1" in
        --ref)
            shift
            [ "$#" -gt 0 ] || die "--ref requires a value"
            ref="$1"
            ;;
        --path)
            shift
            [ "$#" -gt 0 ] || die "--path requires a value"
            artifact_path="$1"
            ;;
        --expected-size)
            shift
            [ "$#" -gt 0 ] || die "--expected-size requires a value"
            expected_size="$1"
            ;;
        --expected-sha256)
            shift
            [ "$#" -gt 0 ] || die "--expected-sha256 requires a value"
            expected_sha="$1"
            ;;
        --remote)
            shift
            [ "$#" -gt 0 ] || die "--remote requires a value"
            remote="$1"
            ;;
        --max-pointer-bytes)
            shift
            [ "$#" -gt 0 ] || die "--max-pointer-bytes requires a value"
            max_pointer_bytes="$1"
            ;;
        --remote-timeout-seconds)
            shift
            [ "$#" -gt 0 ] || die "--remote-timeout-seconds requires a value"
            remote_timeout_seconds="$1"
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown argument: $1"
            ;;
    esac
    shift
done

[ -n "$ref" ] || die "set --ref REF"
[ -n "$artifact_path" ] || die "set --path PATH"

case "$expected_size" in
    ""|*[!0-9]*) [ -z "$expected_size" ] || die "--expected-size must be an integer" ;;
esac
case "$max_pointer_bytes" in
    ""|*[!0-9]*) die "--max-pointer-bytes must be an integer" ;;
esac
case "$remote_timeout_seconds" in
    ""|*[!0-9]*) die "--remote-timeout-seconds must be an integer" ;;
esac
if [ -n "$expected_sha" ] &&
   ! printf '%s' "$expected_sha" | grep -Eq '^[0-9a-fA-F]{64}$'; then
    die "--expected-sha256 must be a 64-character hex SHA256"
fi

git rev-parse --is-inside-work-tree >/dev/null 2>&1 ||
    die "not inside a git repository"

git rev-parse --verify --quiet "${ref}^{tree}" >/dev/null ||
    die "ref does not resolve to a tree: ${ref}"

object="${ref}:${artifact_path}"
blob_size="$(git cat-file -s "$object" 2>/dev/null)" ||
    die "artifact path not found at ${ref}: ${artifact_path}"

case "$blob_size" in
    ""|*[!0-9]*) die "could not determine Git blob size for ${artifact_path}" ;;
esac
if [ "$blob_size" -gt "$max_pointer_bytes" ]; then
    die "Git blob is ${blob_size} bytes, not an LFS pointer; refusing to read it"
fi

attr="$(git check-attr --source "$ref" filter -- "$artifact_path" 2>/dev/null || true)"
case "$attr" in
    *"filter: lfs"*) ;;
    *) die "artifact is not matched by filter=lfs at ${ref}: ${artifact_path}" ;;
esac

pointer="$(git show "$object")" ||
    die "could not read Git blob for ${artifact_path}"

version_line="$(printf '%s\n' "$pointer" | sed -n '1p')"
oid_line="$(printf '%s\n' "$pointer" | sed -n '2p')"
size_line="$(printf '%s\n' "$pointer" | sed -n '3p')"

[ "$version_line" = "version https://git-lfs.github.com/spec/v1" ] ||
    die "artifact blob is not a Git LFS pointer"

case "$oid_line" in
    oid\ sha256:*) actual_sha="${oid_line#oid sha256:}" ;;
    *) die "LFS pointer is missing oid sha256 line" ;;
esac
printf '%s' "$actual_sha" | grep -Eq '^[0-9a-f]{64}$' ||
    die "LFS pointer SHA256 is malformed"

case "$size_line" in
    size\ *) actual_size="${size_line#size }" ;;
    *) die "LFS pointer is missing size line" ;;
esac
case "$actual_size" in
    ""|*[!0-9]*) die "LFS pointer size is malformed" ;;
esac
[ "$actual_size" -gt 0 ] ||
    die "LFS pointer size must be positive"

if [ -n "$expected_size" ] && [ "$actual_size" != "$expected_size" ]; then
    die "LFS pointer size mismatch: expected ${expected_size}, got ${actual_size}"
fi
if [ -n "$expected_sha" ] &&
   [ "$(printf '%s' "$actual_sha" | tr 'A-F' 'a-f')" != "$(printf '%s' "$expected_sha" | tr 'A-F' 'a-f')" ]; then
    die "LFS pointer SHA256 mismatch: expected ${expected_sha}, got ${actual_sha}"
fi

remote_state="not_checked"
if [ -n "$remote" ]; then
    git lfs version >/dev/null 2>&1 ||
        die "git-lfs is required for --remote"
    fetch_log="$(mktemp -t igc-lfs-fetch.XXXXXX)"
    lfs_storage="$(mktemp -d -t igc-lfs-storage.XXXXXX)"
    if GIT_LFS_SKIP_SMUDGE=1 git \
        -c "lfs.storage=${lfs_storage}" \
        -c "lfs.dialtimeout=${remote_timeout_seconds}" \
        -c "lfs.tlstimeout=${remote_timeout_seconds}" \
        -c "lfs.activitytimeout=${remote_timeout_seconds}" \
        -c "http.lowSpeedLimit=1" \
        -c "http.lowSpeedTime=${remote_timeout_seconds}" \
        lfs fetch \
        --dry-run \
        --include="$artifact_path" \
        "$remote" \
        "$ref" >"$fetch_log" 2>&1; then
        remote_state="dry_run_ok"
    else
        cat "$fetch_log" >&2
        rm -f "$fetch_log"
        rm -rf "$lfs_storage"
        die "remote LFS dry-run failed for ${remote} ${ref} ${artifact_path}"
    fi
    rm -f "$fetch_log"
    rm -rf "$lfs_storage"
fi

printf 'LFS_ARTIFACT_ACK ref=%s path=%s sha256=%s size=%s remote=%s\n' \
    "$ref" "$artifact_path" "$actual_sha" "$actual_size" "$remote_state"
