#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
profile="merge"
gate=""

usage() {
    cat <<'USAGE'
Usage: scripts/check.sh --profile merge --gate unit.all

Runs an active shared gate binding inside the Internal GitLab Kubernetes runner.
This consumer wrapper never dispatches jobs and refuses laptop execution.
USAGE
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --profile)
            profile="${2:?profile}"
            shift 2
            ;;
        --gate)
            gate="${2:?gate id}"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            printf 'BLOCKER: unexpected argument: %s\n' "$1" >&2
            exit 2
            ;;
    esac
done

[ "$profile" = "merge" ] || {
    printf 'BLOCKER: unsupported gate profile: %s\n' "$profile" >&2
    exit 2
}
[ "$gate" = "unit.all" ] || {
    printf 'BLOCKER: unknown or inactive focused gate: %s\n' "${gate:-missing}" >&2
    exit 2
}
[ "${HOMELAB_IN_CLUSTER:-0}" = "1" ] \
    && [ -n "${KUBERNETES_SERVICE_HOST:-}" ] \
    && [ "${CI_SERVER_HOST:-}" = "gitlab.rnd.embedings.ai" ] \
    && [[ ",${CI_RUNNER_TAGS:-}," == *",homelab-k8s,"* ]] || {
        printf 'BLOCKER: unit.all requires Internal GitLab homelab-k8s execution\n' >&2
        exit 3
    }

cd "$ROOT"
command -v bats >/dev/null 2>&1 || {
    printf 'BLOCKER: shared toolbox is missing bats\n' >&2
    exit 3
}
command -v rg >/dev/null 2>&1 || {
    printf 'BLOCKER: shared toolbox is missing rg\n' >&2
    exit 3
}
command -v jq >/dev/null 2>&1 || {
    printf 'BLOCKER: shared toolbox is missing jq\n' >&2
    exit 3
}

mapfile -t bats_files < <(find tests -type f -name '*.bats' | sort)
[ "${#bats_files[@]}" -gt 0 ] || {
    printf 'BLOCKER: no Bats unit tests found\n' >&2
    exit 3
}
if rg -n --pcre2 '^[[:space:]]*skip([[:space:]]|$)' "${bats_files[@]}"; then
    printf 'BLOCKER: Bats skip directives are not allowed\n' >&2
    exit 3
fi

report_dir="${IGC_GATE_REPORT_DIR:-reports/gates}"
report="${report_dir}/unit.all.json"
pending="${report}.pending"
mkdir -p "$report_dir"

set +e
bats --tap "${bats_files[@]}"
status=$?
set -e

if [ "$status" -eq 0 ]; then
    result="pass"
else
    result="fail"
fi
jq -n \
    --arg schema "igc/gate-result/v1" \
    --arg gate "unit.all" \
    --arg profile "$profile" \
    --arg status "$result" \
    --arg commit "${CI_COMMIT_SHA:-$(git rev-parse HEAD)}" \
    --argjson testFiles "${#bats_files[@]}" \
    '{schema: $schema, gate: $gate, profile: $profile, status: $status,
      commit: $commit, test_files: $testFiles}' >"$pending"
mv "$pending" "$report"
exit "$status"
