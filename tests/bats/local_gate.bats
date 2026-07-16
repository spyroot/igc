#!/usr/bin/env bats
# Offline tests for scripts/local_gate.sh: the machine-wide gate lock that
# serializes heavy local test suites across agent sessions. Uses the
# IGC_GATE_CMD override seam so no pytest suite actually runs.
#
# Author:
# Mus mbayramo@stanford.edu

setup() {
    REPO_ROOT="$(cd "$(dirname "$BATS_TEST_FILENAME")/../.." && pwd)"
    SCRIPT="$REPO_ROOT/scripts/local_gate.sh"
    LOCK="$BATS_TEST_TMPDIR/gate.lock"
}

@test "runs the override command and propagates success" {
    run env IGC_GATE_LOCK_DIR="$LOCK" IGC_GATE_CMD="echo gated-ok" bash "$SCRIPT"
    [ "$status" -eq 0 ]
    [[ "$output" == *gated-ok* ]]
}

@test "propagates the override command's failure exit code" {
    run env IGC_GATE_LOCK_DIR="$LOCK" IGC_GATE_CMD="exit 7" bash "$SCRIPT"
    [ "$status" -eq 7 ]
}

@test "releases the lock after the run" {
    env IGC_GATE_LOCK_DIR="$LOCK" IGC_GATE_CMD="true" bash "$SCRIPT"
    [ ! -d "$LOCK" ]
}

@test "steals a stale lock left by a dead process" {
    mkdir -p "$LOCK"
    echo 99999999 > "$LOCK/pid"   # certainly-dead pid
    run env IGC_GATE_LOCK_DIR="$LOCK" IGC_GATE_CMD="echo stole-it" bash "$SCRIPT"
    [ "$status" -eq 0 ]
    [[ "$output" == *stole-it* ]]
}

@test "kills a wedged command at the timeout instead of hanging" {
    run env IGC_GATE_LOCK_DIR="$LOCK" IGC_GATE_TIMEOUT_SECS=2 \
        IGC_GATE_CMD="sleep 60" bash "$SCRIPT"
    [ "$status" -ne 0 ]
    [ ! -d "$LOCK" ]
}
