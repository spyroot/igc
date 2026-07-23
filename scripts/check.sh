#!/usr/bin/env bash
set -euo pipefail

PROFILE="phase2_labelled_requests"
CATEGORY="unit"
DRY_RUN=0
LOG_FORMAT="text"
LOG_LEVEL="info"
LOG_FILE=""
RUN_ID=""

usage() {
    cat <<'EOF'
Usage:
  scripts/check.sh [--profile phase2_labelled_requests] [--category unit|static|phase2]

Options:
  --help                  Show this help text.
  --dry-run               Print the selected guarded command without running it.
  --profile NAME          Validation profile. Default: phase2_labelled_requests.
  --category NAME         Validation category. Default: unit.
  --log-format text|json  Accepted for the shared script interface.
  --log-level LEVEL       Accepted for the shared script interface.
  --log-file PATH         Accepted for the shared script interface.
  --run-id ID             Accepted for the shared script interface.

This entrypoint refuses laptop execution. Run it inside Kubernetes or inside
the approved GB300/NV72 CPU-only dev container with IGC_REMOTE_VALIDATION=1.
EOF
}

blocker() {
    echo "BLOCKER: $*" >&2
    exit 2
}

parse_args() {
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --help)
                usage
                exit 0
                ;;
            --dry-run)
                DRY_RUN=1
                shift
                ;;
            --profile)
                [ "$#" -ge 2 ] || blocker "--profile requires a value"
                PROFILE="$2"
                shift 2
                ;;
            --category)
                [ "$#" -ge 2 ] || blocker "--category requires a value"
                CATEGORY="$2"
                shift 2
                ;;
            --log-format)
                [ "$#" -ge 2 ] || blocker "--log-format requires a value"
                LOG_FORMAT="$2"
                shift 2
                ;;
            --log-level)
                [ "$#" -ge 2 ] || blocker "--log-level requires a value"
                LOG_LEVEL="$2"
                shift 2
                ;;
            --log-file)
                [ "$#" -ge 2 ] || blocker "--log-file requires a value"
                LOG_FILE="$2"
                shift 2
                ;;
            --run-id)
                [ "$#" -ge 2 ] || blocker "--run-id requires a value"
                RUN_ID="$2"
                shift 2
                ;;
            *)
                blocker "unknown argument: $1"
                ;;
        esac
    done
}

check_interface_values() {
    case "$LOG_FORMAT" in
        text|json) ;;
        *) blocker "--log-format must be text or json" ;;
    esac
    case "$LOG_LEVEL" in
        debug|info|warning|error) ;;
        *) blocker "--log-level must be debug, info, warning, or error" ;;
    esac
    _="$LOG_FILE"
    _="$RUN_ID"
}

allowed_surface() {
    if [ -n "${KUBERNETES_SERVICE_HOST:-}" ]; then
        return 0
    fi
    if [ "${IGC_REMOTE_VALIDATION:-0}" = "1" ] \
        && [ "$(pwd -P)" = "/workspace/igc" ] \
        && [ -n "${IGC_BRANCH:-}" ]; then
        return 0
    fi
    return 1
}

require_allowed_surface() {
    if allowed_surface; then
        return 0
    fi
    blocker \
        "IGC validation must run in Kubernetes or approved GB300/NV72 CPU-only remote dev;" \
        "refusing laptop execution"
}

print_command() {
    printf '%q ' "$@"
    printf '\n'
}

run_phase2_unit() {
    local cmd=(
        python -m pytest -q -ra
        tests/scripts/test_phase2_labelled_requests_cli.py
        tests/ds/test_phase2_labelled_requests.py
        tests/modules/test_phase2_labelled_request_metric_keys.py
        tests/scripts/test_phase2_validation_guards.py
    )
    if [ "$DRY_RUN" = "1" ]; then
        print_command "${cmd[@]}"
        return 0
    fi
    "${cmd[@]}"
}

run_phase2_static() {
    local ruff_cmd=(
        ruff check
        igc/ds/phase2_labelled_requests.py
        scripts/build_phase2_labelled_requests.py
        tests/ds/test_phase2_labelled_requests.py
        tests/scripts/test_phase2_labelled_requests_cli.py
        tests/scripts/test_phase2_validation_guards.py
        tests/modules/test_phase2_labelled_request_metric_keys.py
    )
    local bash_cmd=(
        bash -n
        scripts/check.sh
        scripts/validate_phase2_labelled_requests.sh
    )
    if [ "$DRY_RUN" = "1" ]; then
        print_command "${ruff_cmd[@]}"
        print_command "${bash_cmd[@]}"
        return 0
    fi
    "${ruff_cmd[@]}"
    "${bash_cmd[@]}"
}

main() {
    parse_args "$@"
    check_interface_values
    cd "$(dirname "$0")/.."
    require_allowed_surface

    export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"
    export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
    export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

    case "${PROFILE}:${CATEGORY}" in
        phase2_labelled_requests:unit|phase2_labelled_requests:phase2)
            run_phase2_unit
            ;;
        phase2_labelled_requests:static)
            run_phase2_static
            ;;
        *)
            blocker "unsupported validation profile/category: ${PROFILE}:${CATEGORY}"
            ;;
    esac
}

main "$@"
