#!/usr/bin/env bats

setup() {
    REPO_ROOT="$(cd "${BATS_TEST_DIRNAME}/../.." && pwd)"
    export REPO_ROOT

    mkdir -p "${BATS_TEST_TMPDIR}/bin"
    export PATH="${BATS_TEST_TMPDIR}/bin:${PATH}"

    cat >"${BATS_TEST_TMPDIR}/bin/nvidia-smi" <<'EOF'
#!/usr/bin/env bash
echo "GPU 0: fake"
echo "GPU 1: fake"
EOF
    chmod +x "${BATS_TEST_TMPDIR}/bin/nvidia-smi"

    cat >"${BATS_TEST_TMPDIR}/bin/timeout" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

_timeout_seconds="$1"
shift

world=1
for arg in "$@"; do
    case "$arg" in
        --nproc_per_node=*) world="${arg#--nproc_per_node=}" ;;
    esac
done

case "${FAKE_TIMEOUT_MODE:-expected}" in
    all-pass)
        echo "fake pass world=${world} ds=${DS_SIZE:-?}"
        exit 0
        ;;
    expected)
        if [[ "${DROP_LAST:-}" == "0" && "${DS_SIZE:-}" == "97" && "${world}" != "1" ]]; then
            echo "fake deadlock world=${world} ds=${DS_SIZE:-?}"
            exit 124
        fi
        echo "fake pass world=${world} ds=${DS_SIZE:-?}"
        exit 0
        ;;
    *)
        echo "unknown FAKE_TIMEOUT_MODE=${FAKE_TIMEOUT_MODE}" >&2
        exit 2
        ;;
esac
EOF
    chmod +x "${BATS_TEST_TMPDIR}/bin/timeout"
}

@test "gpu isolation wrapper passes when PASS and DEADLOCK expectations match" {
    run env \
        CASE_TIMEOUT=1 \
        WORLDS="1 2" \
        FAKE_TIMEOUT_MODE=expected \
        bash "${REPO_ROOT}/scripts/gpu_dataset_isolation.sh"

    [ "$status" -eq 0 ]
    [[ "$output" == *"indivisible_NO_drop_last"* ]]
    [[ "$output" == *"DEADLOCK"* ]]
    [[ "$output" == *"MISMATCH=0"* ]]
}

@test "gpu isolation wrapper fails when a known deadlock case unexpectedly passes" {
    run env \
        CASE_TIMEOUT=1 \
        WORLDS="1 2" \
        FAKE_TIMEOUT_MODE=all-pass \
        bash "${REPO_ROOT}/scripts/gpu_dataset_isolation.sh"

    [ "$status" -eq 1 ]
    [[ "$output" == *"MISMATCH"* ]]
    [[ "$output" == *"MISMATCH=1"* ]]
}
