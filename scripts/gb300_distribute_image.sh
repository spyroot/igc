#!/usr/bin/env bash
# gb300_distribute_image.sh — build the keyless igc-train image once, save it to the
# shared /models filesystem, and docker-load it on all 18 nodes so ANY free node can
# launch with zero pull/build. The NGC base is already cached on the nodes, so each
# `docker load` only materialises the small extra layers (tools + pip deps).
#
# Runs on one node (needs docker + zstd + /models mounted; reaches the others by ssh).
#   scripts/gb300_distribute_image.sh
#   DIST_NODES="<node-ip> <node-ip>" scripts/gb300_distribute_image.sh           # subset
#   DIST_MAX_PARALLEL=8 scripts/gb300_distribute_image.sh                        # bound fanout
#   FORCE_SAVE=1 scripts/gb300_distribute_image.sh                               # re-save tarball
#
# Only the KEYLESS image is distributed here — never the .internal SSH-key image.
#
# Author:
# Mus mbayramo@stanford.edu
set -uo pipefail

IMAGE="${IMAGE:-igc-train}"
TAG="${TAG:-ngc26.03-py3}"
REF="${IMAGE}:${TAG}"
PYTHON="${PYTHON:-python3}"
DATASET_COMPAT_GATE="scripts/gates/dataset_runtime_compat.py"
MODELS_IMAGES="${MODELS_IMAGES:-/models/images}"
# Compression: zstd is smaller/faster (9GB vs ~15GB) but must be on EVERY node for the
# load; gzip is universal (coreutils) but bigger. Default zstd; COMPRESS=gzip when some
# nodes lack zstd so the distribution never fails on a missing decompressor.
COMPRESS="${COMPRESS:-zstd}"
case "$COMPRESS" in
    zstd) EXT="tar.zst"; COMP="zstd -T0 -q"; DECOMP="zstd -dc" ;;
    gzip) EXT="tar.gz";  COMP="gzip";        DECOMP="gzip -dc" ;;
    *) echo "BLOCKER: COMPRESS must be zstd or gzip (got $COMPRESS)" >&2; exit 3 ;;
esac
TARBALL="${MODELS_IMAGES}/${IMAGE}-${TAG}.${EXT}"
IMAGE_FINGERPRINT_FILE="${TARBALL}.fingerprint"
# Node list comes from the environment (or a gitignored nodes file) — the fleet
# addressing is internal and never hardcoded in the public repo.
NODES_FILE="${GB300_NODES_FILE:-.internal/gb300_nodes}"
# shellcheck disable=SC2206  # word-split the space-separated override on purpose
NODES=(${DIST_NODES:-$(cat "$NODES_FILE" 2>/dev/null || true)})
[ "${#NODES[@]}" -gt 0 ] || {
    echo "BLOCKER: set DIST_NODES=\"ip ip ...\" or provide $NODES_FILE" >&2
    exit 3
}
DIST_MAX_PARALLEL="${DIST_MAX_PARALLEL:-${#NODES[@]}}"
case "$DIST_MAX_PARALLEL" in
    ''|*[!0-9]*|0)
        echo "BLOCKER: DIST_MAX_PARALLEL must be a positive integer" >&2
        exit 3
        ;;
esac
if [ "$DIST_MAX_PARALLEL" -gt "${#NODES[@]}" ]; then
    DIST_MAX_PARALLEL="${#NODES[@]}"
fi

declare -A SEEN_NODES=()
for ip in "${NODES[@]}"; do
    if [ -n "${SEEN_NODES[$ip]:-}" ]; then
        echo "BLOCKER: duplicate node in distribution inventory: $ip" >&2
        exit 3
    fi
    SEEN_NODES[$ip]=1
done
SSH="ssh -o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=accept-new"

log() { echo "=== [$(date -u '+%F %T')] $* ==="; }

image_fingerprint() {
    docker image inspect "$1" --format \
        '{{.Architecture}}|{{.Os}}|{{.Created}}|{{json .RootFS.Layers}}|{{json .Config}}' \
        | sha256sum \
        | awk '{print "sha256:" $1}'
}

command -v "${COMPRESS}" >/dev/null || {
    echo "BLOCKER: $COMPRESS missing on $(hostname); use COMPRESS=gzip" >&2
    exit 3
}
command -v "$PYTHON" >/dev/null || {
    echo "BLOCKER: $PYTHON is required to resolve the dataset contract" >&2
    exit 3
}
CONTRACT_SHA="$($PYTHON "$DATASET_COMPAT_GATE" contract-sha --output text)" || exit 3
TRANSFORM_VERSION="$($PYTHON "$DATASET_COMPAT_GATE" transform --output text)" || exit 3

# 1. build (unless already present) then save to /models (read by every node)
needs_build="${FORCE_BUILD:-0}"
if ! docker image inspect "$REF" >/dev/null 2>&1; then
    needs_build=1
elif ! "$PYTHON" "$DATASET_COMPAT_GATE" check-image --image "$REF" >/dev/null; then
    log "$REF is update-required for dataset contract $CONTRACT_SHA"
    needs_build=1
fi
if [ "$needs_build" = "1" ]; then
    log "building $REF from docker/Dockerfile.train"
    docker build \
        --build-arg "IGC_DATASET_CONTRACT_SHA=$CONTRACT_SHA" \
        --build-arg "IGC_DATASET_TRANSFORM_VERSION=$TRANSFORM_VERSION" \
        -f docker/Dockerfile.train \
        -t "$REF" . \
        || { echo "BLOCKER: build failed" >&2; exit 3; }
fi
"$PYTHON" "$DATASET_COMPAT_GATE" check-image --image "$REF" >/dev/null \
    || { echo "BLOCKER: built image does not match the dataset contract" >&2; exit 3; }
SOURCE_FINGERPRINT="$(image_fingerprint "$REF")"
[ -n "$SOURCE_FINGERPRINT" ] || {
    echo "BLOCKER: cannot resolve source image fingerprint for $REF" >&2
    exit 3
}
mkdir -p "$MODELS_IMAGES"
ARCHIVE_FINGERPRINT="$(cat "$IMAGE_FINGERPRINT_FILE" 2>/dev/null || true)"
if [ ! -f "$TARBALL" ] \
    || [ "$ARCHIVE_FINGERPRINT" != "$SOURCE_FINGERPRINT" ] \
    || [ "${FORCE_SAVE:-0}" = "1" ]; then
    log "docker save $REF -> $TARBALL ($COMPRESS)"
    # shellcheck disable=SC2086  # COMP must word-split into "zstd -T0 -q" / "gzip"
    pending="${TARBALL}.pending"
    docker save "$REF" | $COMP > "$pending" \
        || { rm -f "$pending"; echo "BLOCKER: save failed" >&2; exit 3; }
    mv "$pending" "$TARBALL"
    printf '%s\n' "$SOURCE_FINGERPRINT" > "$IMAGE_FINGERPRINT_FILE"
fi
log "tarball ready: $(du -h "$TARBALL" 2>/dev/null | cut -f1) at $TARBALL ($SOURCE_FINGERPRINT)"

# 2. docker load concurrently on every node from the one shared tarball
load_node() {
    local ip="$1"
    local remote_fingerprint
    local loaded_fingerprint

    remote_fingerprint="$($SSH "nvidia@$ip" \
        "docker image inspect '$REF' --format '{{.Architecture}}|{{.Os}}|{{.Created}}|{{json .RootFS.Layers}}|{{json .Config}}' 2>/dev/null | sha256sum | awk '{print \"sha256:\" \$1}'" \
        || true)"
    if [ "$remote_fingerprint" = "$SOURCE_FINGERPRINT" ]; then
        echo "  $ip: already has $REF ($SOURCE_FINGERPRINT)"
        return 0
    fi
    if $SSH "nvidia@$ip" \
        "test -f '$TARBALL' && $DECOMP '$TARBALL' | docker load >/dev/null 2>&1"; then
        loaded_fingerprint="$($SSH "nvidia@$ip" \
            "docker image inspect '$REF' --format '{{.Architecture}}|{{.Os}}|{{.Created}}|{{json .RootFS.Layers}}|{{json .Config}}' 2>/dev/null | sha256sum | awk '{print \"sha256:\" \$1}'" \
            || true)"
    else
        loaded_fingerprint=""
    fi
    if [ "$loaded_fingerprint" = "$SOURCE_FINGERPRINT" ]; then
        echo "  $ip: LOADED $REF ($SOURCE_FINGERPRINT)"
        return 0
    else
        echo \
            "  $ip: FAILED expected=$SOURCE_FINGERPRINT" \
            "observed=${loaded_fingerprint:-missing}" >&2
        return 1
    fi
}

FANOUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/igc-image-fanout.XXXXXX")" || {
    echo "BLOCKER: cannot create fanout status directory" >&2
    exit 3
}
PIDS=()
cleanup_fanout() {
    local exit_status=$?

    trap - EXIT HUP INT TERM
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    for pid in "${PIDS[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
    rm -rf -- "$FANOUT_DIR"
    exit "$exit_status"
}
trap cleanup_fanout EXIT HUP INT TERM

ok=0
fail=0
log "loading $REF from shared storage on ${#NODES[@]} nodes (parallel=$DIST_MAX_PARALLEL)"
for ((batch_start = 0; batch_start < ${#NODES[@]}; batch_start += DIST_MAX_PARALLEL)); do
    batch_end=$((batch_start + DIST_MAX_PARALLEL))
    if [ "$batch_end" -gt "${#NODES[@]}" ]; then
        batch_end="${#NODES[@]}"
    fi

    PIDS=()
    BATCH_INDICES=()
    for ((index = batch_start; index < batch_end; index++)); do
        load_node "${NODES[$index]}" >"$FANOUT_DIR/$index.log" 2>&1 &
        PIDS+=("$!")
        BATCH_INDICES+=("$index")
    done

    for ((offset = 0; offset < ${#PIDS[@]}; offset++)); do
        index="${BATCH_INDICES[$offset]}"
        if wait "${PIDS[$offset]}"; then
            ok=$((ok + 1))
        else
            fail=$((fail + 1))
        fi
        cat "$FANOUT_DIR/$index.log"
    done
    PIDS=()
done

log "image ready on ${ok}/${#NODES[@]} nodes (${fail} failed)"
[ "$fail" = "0" ]

# Author: Mus mbayramo@stanford.edu
