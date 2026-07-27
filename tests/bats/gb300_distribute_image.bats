#!/usr/bin/env bats

setup() {
    REPO_ROOT="$(cd "${BATS_TEST_DIRNAME}/../.." && pwd)"
    SCRIPT="${REPO_ROOT}/scripts/gb300_distribute_image.sh"
    FAKE_BIN="${BATS_TEST_TMPDIR}/bin"
    CALL_LOG="${BATS_TEST_TMPDIR}/calls.log"
    mkdir -p "$FAKE_BIN"
    export SCRIPT FAKE_BIN CALL_LOG
}

install_forbidden_docker_ssh_fakes() {
    for tool in docker ssh; do
        cat >"${FAKE_BIN}/${tool}" <<'EOF'
#!/usr/bin/env bash
{
    printf '%s' "$(basename "$0")"
    printf ' %q' "$@"
    printf '\n'
} >>"${CALL_LOG}"
exit 97
EOF
        chmod +x "${FAKE_BIN}/${tool}"
    done
}

assert_no_docker_or_ssh_calls() {
    [ ! -e "$CALL_LOG" ]
}

@test "zero DIST_MAX_PARALLEL fails before docker or ssh" {
    install_forbidden_docker_ssh_fakes

    run env \
        DIST_NODES="node-a node-b" \
        DIST_MAX_PARALLEL=0 \
        PATH="${FAKE_BIN}:/usr/bin:/bin" \
        bash "$SCRIPT"

    [ "$status" -eq 3 ]
    [[ "$output" == *"BLOCKER: DIST_MAX_PARALLEL must be a positive integer"* ]]
    assert_no_docker_or_ssh_calls
}

@test "duplicate DIST_NODES fails before docker or ssh" {
    install_forbidden_docker_ssh_fakes

    run env \
        DIST_NODES="node-a node-b node-a" \
        PATH="${FAKE_BIN}:/usr/bin:/bin" \
        bash "$SCRIPT"

    [ "$status" -eq 3 ]
    [[ "$output" == *"BLOCKER: duplicate node in distribution inventory: node-a"* ]]
    assert_no_docker_or_ssh_calls
}

@test "source keeps strict ssh, eighteen-node cap, and backgrounded default fanout" {
    run grep -F 'MAX_FANOUT_NODES=18' "$SCRIPT"
    [ "$status" -eq 0 ]

    run grep -F 'StrictHostKeyChecking=yes' "$SCRIPT"
    [ "$status" -eq 0 ]

    run grep -F 'DIST_MAX_PARALLEL="${DIST_MAX_PARALLEL:-${#NODES[@]}}"' "$SCRIPT"
    [ "$status" -eq 0 ]

    launch_line="$(
        grep -nF 'load_node "${NODES[$index]}" >"$FANOUT_DIR/$index.log" 2>&1 &' "$SCRIPT" \
            | cut -d: -f1 \
            | head -n1
    )"
    pid_line="$(
        grep -nF 'PIDS+=("$!")' "$SCRIPT" \
            | cut -d: -f1 \
            | head -n1
    )"
    wait_line="$(
        grep -nF 'if wait "${PIDS[$offset]}"; then' "$SCRIPT" \
            | cut -d: -f1 \
            | head -n1
    )"

    [[ "$launch_line" =~ ^[0-9]+$ ]]
    [[ "$pid_line" =~ ^[0-9]+$ ]]
    [[ "$wait_line" =~ ^[0-9]+$ ]]
    [ "$launch_line" -lt "$pid_line" ]
    [ "$pid_line" -lt "$wait_line" ]
}
