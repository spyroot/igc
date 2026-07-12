#!/usr/bin/env bash
# stage_node.sh — prepare a GB300 node for igc training WITHOUT bulk laptop uploads.
#
# HARD RULE (operator, 2026-07-10): the VPN is a shared link — a multi-GB rsync from a
# laptop saturates it and blocks SSH for EVERYONE. Nothing larger than tens of MB may
# originate from the laptop:
#   - CODE           -> the node pulls from GitHub (origin main is current; PR-only flow)
#   - DATASET CACHES -> rebuilt ON the node (--recreate_dataset) or copied node-locally
#                       from the shared /models filesystem at LAN speed
#   - WEIGHTS        -> git-lfs weights repo or /models via publish_checkpoint.sh
#   - CAPTURES       -> the only laptop upload (~tens of MB), bandwidth-capped
#
#   IGC_NODE=<hostname-or-ip>      # required
#   IGC_NODE_USER=nvidia           # ssh user
#   IGC_DEST=igc                   # remote checkout dir under $HOME
#   IGC_REPO_URL=<git url>         # default: this checkout's origin remote
#   IGC_BWLIMIT_KBPS=4000          # captures rsync cap (~4MB/s, link stays usable)
#   IGC_STAGE_DATA=1               # rsync ~/.json_responses captures (small)
#   IGC_STAGE_INTERNAL=0           # rsync .internal/ run config (tiny)
set -euo pipefail

: "${IGC_NODE:?set IGC_NODE to the staging target node (hostname or IP)}"
IGC_NODE_USER="${IGC_NODE_USER:-nvidia}"
IGC_DEST="${IGC_DEST:-igc}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE="${IGC_NODE_USER}@${IGC_NODE}"
IGC_REPO_URL="${IGC_REPO_URL:-$(git -C "${HERE}" remote get-url origin)}"
BW="--bwlimit=${IGC_BWLIMIT_KBPS:-4000}"

echo "code: ${REMOTE} pulls ${IGC_REPO_URL} (no laptop transfer)"
# shellcheck disable=SC2029  # Local values intentionally select the remote checkout command.
ssh "${REMOTE}" "if [ -d '${IGC_DEST}/.git' ]; then git -C '${IGC_DEST}' fetch origin && git -C '${IGC_DEST}' reset --hard origin/main; else git clone '${IGC_REPO_URL}' '${IGC_DEST}'; fi"

if [[ "${IGC_STAGE_DATA:-1}" == "1" ]]; then
    SIZE="$(du -sm "${HOME}/.json_responses" 2>/dev/null | cut -f1 || echo '?')"
    echo "captures: ~${SIZE}MB -> ${REMOTE}:~/.json_responses (bwlimit ${IGC_BWLIMIT_KBPS:-4000}KB/s)"
    rsync -az "${BW}" --exclude '*.tar.gz' --exclude '*.tar' \
        "${HOME}/.json_responses/" "${REMOTE}:.json_responses/"
fi

if [[ "${IGC_STAGE_INTERNAL:-0}" == "1" ]]; then
    echo "run config: .internal/ (tiny; values never printed)"
    rsync -az "${BW}" "${HERE}/.internal/" "${REMOTE}:${IGC_DEST}/.internal/"
fi

echo "dataset caches: NOT transferred — rebuild on the node:"
echo "    ssh ${REMOTE} 'cd ${IGC_DEST} && python igc_main.py --recreate_dataset ...'"
echo "  (or copy a published cache from the shared /models filesystem at LAN speed)"
echo "done. next: ./scripts/submit_train.sh <stage> -w <node>"
