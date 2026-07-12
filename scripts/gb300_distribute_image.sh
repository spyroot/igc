#!/usr/bin/env bash
# gb300_distribute_image.sh — build the keyless igc-train image once, save it to the
# shared /models filesystem, and docker-load it on all 18 nodes so ANY free node can
# launch with zero pull/build. The NGC base is already cached on the nodes, so each
# `docker load` only materialises the small extra layers (tools + pip deps).
#
# Runs on one node (needs docker + zstd + /models mounted; reaches the others by ssh).
#   scripts/gb300_distribute_image.sh
#   DIST_NODES="172.25.230.42 172.25.230.49" scripts/gb300_distribute_image.sh   # subset
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
MODELS_IMAGES="${MODELS_IMAGES:-/models/images}"
TARBALL="${MODELS_IMAGES}/${IMAGE}-${TAG}.tar.zst"
# All 18 GB300 nodes: 172.25.230.40 .. .57
# shellcheck disable=SC2206  # word-split the space-separated override on purpose
NODES=(${DIST_NODES:-$(seq -f "172.25.230.%g" 40 57)})
SSH="ssh -o BatchMode=yes -o ConnectTimeout=8"

log() { echo "=== [$(date -u '+%F %T')] $* ==="; }

command -v zstd >/dev/null || { echo "BLOCKER: zstd missing on $(hostname) — install it (safe-apt-install.sh zstd)" >&2; exit 3; }

# 1. build (unless already present) then save to /models (read by every node)
if ! docker image inspect "$REF" >/dev/null 2>&1; then
    log "building $REF from docker/Dockerfile.train"
    docker build -f docker/Dockerfile.train -t "$REF" . || { echo "BLOCKER: build failed" >&2; exit 3; }
fi
mkdir -p "$MODELS_IMAGES"
if [ ! -f "$TARBALL" ] || [ "${FORCE_SAVE:-0}" = "1" ]; then
    log "docker save $REF -> $TARBALL"
    docker save "$REF" | zstd -T0 -q -o "$TARBALL" || { echo "BLOCKER: save failed" >&2; exit 3; }
fi
log "tarball ready: $(du -h "$TARBALL" 2>/dev/null | cut -f1) at $TARBALL"

# 2. docker load on every node from the shared tarball
ok=0; fail=0
for ip in "${NODES[@]}"; do
    if $SSH "nvidia@$ip" "docker image inspect $REF >/dev/null 2>&1"; then
        echo "  $ip: already has $REF"; ok=$((ok + 1)); continue
    fi
    if $SSH "nvidia@$ip" "test -f '$TARBALL' && zstd -dc '$TARBALL' | docker load >/dev/null 2>&1 && docker image inspect $REF >/dev/null 2>&1"; then
        echo "  $ip: LOADED $REF"; ok=$((ok + 1))
    else
        echo "  $ip: FAILED (unreachable? /models unmounted? zstd missing?)" >&2; fail=$((fail + 1))
    fi
done

log "image ready on ${ok}/${#NODES[@]} nodes (${fail} failed)"
[ "$fail" = "0" ]

# Author: Mus mbayramo@stanford.edu
