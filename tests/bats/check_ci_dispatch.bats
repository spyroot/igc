#!/usr/bin/env bats

setup() {
    REPO_ROOT="$(cd "${BATS_TEST_DIRNAME}/../.." && pwd)"
    CHECK="${REPO_ROOT}/scripts/check.sh"
    REPORT_DIR="${BATS_TEST_TMPDIR}/reports"
    FAKE_BIN="${BATS_TEST_TMPDIR}/bin"
    mkdir -p "$FAKE_BIN"
}

@test "unit.all refuses laptop execution" {
    run "$CHECK" --profile merge --gate unit.all

    [ "$status" -eq 3 ]
    [[ "$output" == *"requires Internal GitLab homelab-k8s execution"* ]]
}

@test "unknown gate fails before calling bats" {
    cat >"${FAKE_BIN}/bats" <<'EOF'
#!/usr/bin/env bash
exit 97
EOF
    chmod +x "${FAKE_BIN}/bats"

    run env \
        HOMELAB_IN_CLUSTER=1 \
        KUBERNETES_SERVICE_HOST=10.0.0.1 \
        CI_JOB_ID=123 \
        CI_RUNNER_TAGS=homelab-k8s \
        PATH="${FAKE_BIN}:${PATH}" \
        "$CHECK" --profile merge --gate unknown

    [ "$status" -eq 2 ]
    [[ "$output" == *"unknown or inactive focused gate: unknown"* ]]
}

@test "unit.all emits a sanitized exact-commit report" {
    cat >"${FAKE_BIN}/bats" <<'EOF'
#!/usr/bin/env bash
[ "$1" = "--tap" ] || exit 96
printf '1..1\nok 1 focused fixture\n'
EOF
    chmod +x "${FAKE_BIN}/bats"

    run env \
        HOMELAB_IN_CLUSTER=1 \
        KUBERNETES_SERVICE_HOST=10.0.0.1 \
        CI_JOB_ID=123 \
        CI_RUNNER_TAGS=homelab-k8s \
        CI_COMMIT_SHA=1111111111111111111111111111111111111111 \
        IGC_GATE_REPORT_DIR="$REPORT_DIR" \
        PATH="${FAKE_BIN}:${PATH}" \
        "$CHECK" --profile merge --gate unit.all

    [ "$status" -eq 0 ]
    run jq -e \
        '.schema == "igc/gate-result/v1" and
         .gate == "unit.all" and
         .profile == "merge" and
         .status == "pass" and
         .commit == "1111111111111111111111111111111111111111" and
         .test_files > 0' \
        "${REPORT_DIR}/unit.all.json"
    [ "$status" -eq 0 ]
}
