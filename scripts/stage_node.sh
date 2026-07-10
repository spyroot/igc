#!/usr/bin/env bash
# stage_node.sh — stage the igc checkout + captured data onto a GB300 node's local NVMe.
#
# The cluster schedules jobs wherever Slurm likes, but the data lives only where you
# staged it — so stage first, then pin the job to that node (submit_train.sh -w <node>).
# Everything node- or user-specific comes from the environment; nothing is hardcoded:
#
#   IGC_NODE=<hostname-or-ip>   # required: the target node (see your ops notes)
#   IGC_NODE_USER=nvidia        # ssh user (default nvidia)
#   IGC_DEST=igc                # remote checkout dir under $HOME (default igc)
#   IGC_STAGE_DATA=1            # also rsync ~/.json_responses captures (default 1)
#   IGC_STAGE_INTERNAL=0        # also rsync .internal/ (wandb env etc.; default 0)
#
# Usage:
#   IGC_NODE=gb300-poc1-slotN ./scripts/stage_node.sh
#   ./scripts/submit_train.sh m1 -w "${IGC_NODE}"     # then pin the job to it
#
# Excludes keep the transfer lean and internal files off the node image: no .git, no
# tarballs/raw capture dumps from datasets/ (the built caches + tokenizer DO go), no
# experiments/logs, and no agent/ops docs (gitignored but present locally).
set -euo pipefail

: "${IGC_NODE:?set IGC_NODE to the staging target node (hostname or IP)}"
IGC_NODE_USER="${IGC_NODE_USER:-nvidia}"
IGC_DEST="${IGC_DEST:-igc}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE="${IGC_NODE_USER}@${IGC_NODE}"

echo "staging ${HERE} -> ${REMOTE}:~/${IGC_DEST} (checkout, no .git/tarballs/experiments)"
rsync -az \
    --exclude '.git' --exclude '__pycache__' --exclude '*.pyc' \
    --exclude 'experiments/' --exclude 'logs/' --exclude 'imgs/' \
    --exclude '*.tar.gz' --exclude '*.tar' --exclude '*.md5' \
    --exclude 'datasets/raw/' --exclude 'datasets/orig/' \
    --exclude 'datasets/post/' --exclude 'datasets/pre/' \
    --exclude 'CLAUDE.md' --exclude 'TEAM_GUIDE.md' --exclude 'COORDINATION.md' \
    --exclude 'FLASH_BRAIN.md' --exclude 'GPU_ACCESS.md' --exclude 'NV72_MODELS.md' \
    --exclude '.claude/' --exclude '.codex/' --exclude '.internal/' \
    "${HERE}/" "${REMOTE}:${IGC_DEST}/"

if [[ "${IGC_STAGE_DATA:-1}" == "1" ]]; then
    echo "staging ~/.json_responses captures -> ${REMOTE}:~/.json_responses"
    rsync -az "${HOME}/.json_responses/" "${REMOTE}:.json_responses/"
fi

if [[ "${IGC_STAGE_INTERNAL:-0}" == "1" ]]; then
    echo "staging .internal/ (run config; never printed)"
    rsync -az "${HERE}/.internal/" "${REMOTE}:${IGC_DEST}/.internal/"
fi

echo "done. next: ./scripts/submit_train.sh <stage> -w ${IGC_NODE}"
