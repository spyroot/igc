#!/usr/bin/env bash
# lfs_push_from_slot.sh — push explicitly supplied large artifacts (captured corpora, built
# datasets, model adapters, tarballs)
# from a GB300 slot DIRECTLY to the git-LFS remote over the node's own uplink, so the data
# never crosses the shared VPN. Only the operator has GB300 access; run this ON the slot.
#
# WHY: pulling a multi-GB artifact to the laptop and pushing it from there moves the data over
# the VPN twice and has saturated the shared tunnel before. `git lfs push` from the node uploads
# LFS objects to the remote's LFS store over the datacenter uplink instead.
#
# PR-ONLY: integration is PR-only, so this pushes a `data/<name>` BRANCH and prints the PR URL.
# It never pushes `main`.
#
# NO OS MODIFICATION by default. If git-lfs is missing, this exits with guidance. To auto-install
# it non-destructively, opt in with IGC_LFS_INSTALL=apt (runs the minimal
# `apt-get install -y --no-install-recommends git-lfs` — adds one package, no upgrade/remove) or
# IGC_LFS_INSTALL=docker (runs git-lfs from a container, host untouched — see the docker note).
#
# Usage:
#   scripts/lfs_push_from_slot.sh <artifact_path> [more_paths...]
# Env:
#   IGC_BRANCH=data/<auto>     branch to push (default data/<basename>-<UTC stamp>)
#   IGC_REMOTE=origin          git remote to push to
#   IGC_LFS_INSTALL=           unset (default: fail if git-lfs missing) | apt | docker
#   IGC_COMMIT_MSG="..."       commit subject (default: "Add data artifact <name>")
#   IGC_YES=1                  skip the confirmation prompt
set -euo pipefail

die() { echo "ERROR: $*" >&2; exit 1; }
note() { echo ">> $*"; }

[ "$#" -ge 1 ] || die "usage: $0 <artifact_path> [more_paths...]"

REMOTE="${IGC_REMOTE:-origin}"
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || die "not inside a git repository"
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

# Guard: never operate on a detached/main checkout in a way that could push main.
CUR_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
[ "$CUR_BRANCH" = "main" ] || note "current branch is '$CUR_BRANCH' (a data/* branch will be created)"

# 1. Verify each artifact exists and is LFS-tracked. This happens before the
# git-lfs binary check so a path that would become a normal Git blob fails even
# on hosts where git-lfs is not installed yet.
for path in "$@"; do
  [ -e "$path" ] || die "no such artifact: $path"
  if ! git check-attr filter -- "$path" 2>/dev/null | grep -q 'filter: lfs'; then
    ext="$(basename "$path" | sed 's/^[^.]*//')"
    pattern_note="an exact path"
    if [ -n "$ext" ]; then
      pattern_note="an exact path or extension pattern such as '${ext}'"
    fi
    die "'$path' is not matched by an LFS filter in .gitattributes.
Add ${pattern_note} via PR, then rerun.
Refusing to stage this artifact as a normal Git object."
  fi
done

# 2. Ensure git-lfs is available (no OS change unless opted in).
ensure_lfs() {
  if git lfs version >/dev/null 2>&1; then
    LFS_RUN=(git lfs)
    return
  fi
  case "${IGC_LFS_INSTALL:-}" in
    apt)
      note "installing git-lfs via apt (--no-install-recommends; one package, no OS upgrade)"
      sudo apt-get update -qq
      sudo apt-get install -y --no-install-recommends git-lfs
      git lfs install --local
      LFS_RUN=(git lfs)
      ;;
    docker)
      command -v docker >/dev/null 2>&1 || die "IGC_LFS_INSTALL=docker but docker is not available"
      note "using git-lfs from a container (host OS untouched)"
      # host repo + git config mounted so credentials/remotes resolve inside the container.
      LFS_RUN=(docker run --rm -v "$REPO_ROOT:/repo" -v "$HOME/.gitconfig:/root/.gitconfig:ro"
               -v "$HOME/.git-credentials:/root/.git-credentials:ro" -w /repo
               ghcr.io/git-lfs/git-lfs:latest)
      ;;
    *)
      die "git-lfs not found. Install it non-destructively and retry:
       IGC_LFS_INSTALL=apt    $0 $*     # minimal apt (--no-install-recommends)
       IGC_LFS_INSTALL=docker $0 $*     # containerized git-lfs, host untouched
     or install git-lfs once yourself, then rerun."
      ;;
  esac
}
ensure_lfs "$@"

# 3. Confirm the push (large, outbound).
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
BASENAME="$(basename "${1%/}")"
BRANCH="${IGC_BRANCH:-data/${BASENAME}-${STAMP}}"
TOTAL="$(du -sh "$@" 2>/dev/null | awk '{s=$1} END{print s}')"
note "about to push $# artifact(s) (~${TOTAL:-?}) as LFS objects to ${REMOTE} on branch ${BRANCH}"
if [ "${IGC_YES:-}" != "1" ]; then
  read -r -p "proceed? [y/N] " reply
  [ "$reply" = "y" ] || [ "$reply" = "Y" ] || die "aborted"
fi

# 4. Branch, stage explicitly, commit, LFS-push, push. Never `git add -A`.
git switch -c "$BRANCH"
git add -- "$@"
git commit -m "${IGC_COMMIT_MSG:-Add data artifact ${BASENAME}}"
note "uploading LFS objects to ${REMOTE} over the node uplink..."
"${LFS_RUN[@]}" push "${REMOTE}" "${BRANCH}"
git push -u "${REMOTE}" "${BRANCH}"

# 5. Print the PR URL (PR-only integration).
URL="$(git remote get-url "${REMOTE}")"
SLUG="$(printf '%s' "$URL" | sed -E 's#(git@github.com:|https://github.com/)##; s#\.git$##')"
note "pushed. Open a PR to integrate:"
echo "   https://github.com/${SLUG}/compare/${BRANCH}?expand=1"
