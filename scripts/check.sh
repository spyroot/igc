#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target="${1:-phase2_labelled_requests}"

if [[ -z "${KUBERNETES_SERVICE_HOST:-}" ]]; then
    cat >&2 <<'EOF'
BLOCKER: scripts/check.sh refuses local IGC gate execution.
Observation: this shell is not running inside a Kubernetes pod.
Safe next step: run the homelab-k8s GitLab job or an approved in-cluster job.
EOF
    exit 2
fi

cd "$repo_root"

case "$target" in
    phase2_labelled_requests)
        scripts/validate_phase2_labelled_requests.sh
        ;;
    *)
        echo "ERROR: unknown check target '$target'" >&2
        exit 2
        ;;
esac


# Author: Mus mbayramo@stanford.edu
