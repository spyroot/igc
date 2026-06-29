#!/bin/bash
# Publish a trained checkpoint to a shareable destination WITHOUT committing weights
# into the igc repo. Datasets/checkpoints/tokenizers/tarballs stay OUT of git; weights
# live in a separate store (a git-lfs weights repo, a synced path, or Google Drive).
#
# Set the destination via IGC_WEIGHTS_DEST:
#   - a local path or a separate git-lfs repo dir:   IGC_WEIGHTS_DEST=/data/igc-weights
#   - a Google Drive remote via rclone:              IGC_WEIGHTS_DEST=gdrive:igc/weights
#                                                    (needs `rclone config` set up first)
#
# Usage:
#   scripts/publish_checkpoint.sh experiments/<run>/state_encoder
set -euo pipefail

SRC="${1:?usage: publish_checkpoint.sh <checkpoint_dir_or_file>}"
DEST="${IGC_WEIGHTS_DEST:?set IGC_WEIGHTS_DEST (a path / git-lfs repo dir, or an rclone remote like gdrive:igc/weights)}"

[ -e "${SRC}" ] || { echo "no such checkpoint: ${SRC}"; exit 2; }

NAME="$(basename "${SRC}")"
STAMP="$(date +%Y%m%d-%H%M%S)"
TARGET="${NAME}-${STAMP}"
MANIFEST="$(mktemp -t igc-manifest.XXXXXX)"

echo "publishing ${SRC} -> ${DEST}/${TARGET}"
# sha256 manifest for integrity verification on the receiving side
( cd "$(dirname "${SRC}")" && find "${NAME}" -type f -print0 | xargs -0 shasum -a 256 ) > "${MANIFEST}"

case "${DEST}" in
  *:*)  # rclone remote (e.g. gdrive:igc/weights)
        command -v rclone >/dev/null || { echo "rclone not installed; run 'rclone config' or use a path/LFS dest"; exit 2; }
        rclone copy "${SRC}" "${DEST}/${TARGET}" --progress
        rclone copyto "${MANIFEST}" "${DEST}/${TARGET}/SHA256SUMS.txt"
        ;;
  *)    # local path or a separate git-lfs repo dir
        mkdir -p "${DEST}/${TARGET}"
        rsync -a "${SRC}/" "${DEST}/${TARGET}/" 2>/dev/null || rsync -a "${SRC}" "${DEST}/${TARGET}/"
        cp "${MANIFEST}" "${DEST}/${TARGET}/SHA256SUMS.txt"
        if git -C "${DEST}" rev-parse --git-dir >/dev/null 2>&1; then
          echo "git-lfs repo detected at ${DEST}; to share via LFS (NOT the igc repo):"
          echo "  cd ${DEST} && git lfs track '*.pt' '*.safetensors' '*.bin' && git add . \\"
          echo "    && git commit -m 'igc weights ${TARGET}' && git push"
        fi
        ;;
esac

rm -f "${MANIFEST}"
echo "done: ${DEST}/${TARGET} (SHA256SUMS.txt included)"
