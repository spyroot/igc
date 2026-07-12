#!/usr/bin/env bats

setup() {
    REPO_ROOT="$(cd "${BATS_TEST_DIRNAME}/../.." && pwd)"
    export REPO_ROOT

    mkdir -p "${BATS_TEST_TMPDIR}/bin"
    export PATH="${BATS_TEST_TMPDIR}/bin:${PATH}"
}

@test "preflight requires the dashboard URL" {
    run env -u NV72_FLEET_DASHBOARD_URL \
        bash "${REPO_ROOT}/scripts/preflight_nv72.sh"

    [ "$status" -ne 0 ]
    [[ "$output" == *"NV72_FLEET_DASHBOARD_URL"* ]]
}

@test "preflight pipes fleet state JSON to the Python checker" {
    cat >"${BATS_TEST_TMPDIR}/bin/curl" <<'EOF'
#!/usr/bin/env bash
printf '{"status":"ok"}'
EOF
    chmod +x "${BATS_TEST_TMPDIR}/bin/curl"

    cat >"${BATS_TEST_TMPDIR}/bin/python3" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
cat >"${PREFLIGHT_STDIN_FILE}"
printf '%s\n' "$*" >"${PREFLIGHT_ARGS_FILE}"
printf '%s\n' "${PYTHONPATH:-}" >"${PREFLIGHT_PYTHONPATH_FILE}"
EOF
    chmod +x "${BATS_TEST_TMPDIR}/bin/python3"

    run env \
        NV72_FLEET_DASHBOARD_URL=http://example.invalid \
        PREFLIGHT_STDIN_FILE="${BATS_TEST_TMPDIR}/stdin.json" \
        PREFLIGHT_ARGS_FILE="${BATS_TEST_TMPDIR}/args.txt" \
        PREFLIGHT_PYTHONPATH_FILE="${BATS_TEST_TMPDIR}/pythonpath.txt" \
        bash "${REPO_ROOT}/scripts/preflight_nv72.sh" --require-endpoint pro

    [ "$status" -eq 0 ]
    [ "$(cat "${BATS_TEST_TMPDIR}/stdin.json")" = '{"status":"ok"}' ]
    [ "$(cat "${BATS_TEST_TMPDIR}/args.txt")" = "-m igc.shared.nv72_preflight --require-endpoint pro" ]
    [ "$(cat "${BATS_TEST_TMPDIR}/pythonpath.txt")" = "${REPO_ROOT}" ]
}

@test "preflight reports a blocker when the dashboard request fails" {
    cat >"${BATS_TEST_TMPDIR}/bin/curl" <<'EOF'
#!/usr/bin/env bash
exit 22
EOF
    chmod +x "${BATS_TEST_TMPDIR}/bin/curl"

    run env \
        NV72_FLEET_DASHBOARD_URL=http://example.invalid \
        bash "${REPO_ROOT}/scripts/preflight_nv72.sh"

    [ "$status" -eq 1 ]
    [[ "$output" == *"BLOCKER: fleet dashboard unreachable"* ]]
}
