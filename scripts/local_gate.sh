#!/usr/bin/env bash
# local_gate.sh — run the offline gate serialized across agent sessions.
#
# Many agents (Claude passes, Codex workers) each running `pytest -q` at the
# same time starve the laptop: suites wedge (the fake-nvidia-smi fixture under
# contention), shells hang, and the desktop goes laggy. This wrapper makes the
# heavy local gate a one-at-a-time, low-priority operation:
#
#   * one gate at a time machine-wide (mkdir lock in $TMPDIR; waits its turn),
#   * stale locks from dead holders are stolen automatically (pid liveness),
#   * the suite runs under `nice -n 19` so the UI stays responsive,
#   * a hard timeout (default 30 min, IGC_GATE_TIMEOUT_SECS) kills a wedged
#     suite instead of blocking every later gate.
#
# Usage (defaults to the standard offline gate on the CURRENT checkout):
#   bash scripts/local_gate.sh                     # full offline gate
#   bash scripts/local_gate.sh tests/ds -q         # extra pytest args pass through
#   IGC_GATE_PYTHON=/path/to/python bash scripts/local_gate.sh
#
# Env knobs:
#   IGC_GATE_PYTHON        python to use (default: python3 on PATH)
#   IGC_GATE_TIMEOUT_SECS  kill a wedged suite after this many seconds (1800)
#   IGC_GATE_LOCK_DIR      exact lock location override
#   IGC_GATE_CMD           full override of the gated command (tests use this)
#
# Used by: agent coordination passes as the default local gate entry point
# (AGENT_HANDOFF.md points sessions here); safe for humans too. CI does not use
# it — runners are isolated. Exercised offline by tests/bats/local_gate.bats.
#
# Author:
# Mus mbayramo@stanford.edu
set -uo pipefail

LOCK_OWNER="$(id -u 2>/dev/null || printf '%s' "${USER:-unknown}")"
LOCK_DIR="${IGC_GATE_LOCK_DIR:-${TMPDIR:-/tmp}/igc-local-gate.${LOCK_OWNER}.lock}"
TIMEOUT_SECS="${IGC_GATE_TIMEOUT_SECS:-1800}"
PYTHON_BIN="${IGC_GATE_PYTHON:-python3}"

log() { echo "[local_gate] $*" >&2; }

acquire_lock() {
    # mkdir is atomic on macOS and Linux; the pid file lets a later caller
    # detect a dead holder and steal the lock instead of waiting forever.
    while ! mkdir "$LOCK_DIR" 2>/dev/null; do
        holder="$(cat "$LOCK_DIR/pid" 2>/dev/null || true)"
        if [ -n "$holder" ] && ! kill -0 "$holder" 2>/dev/null; then
            log "stealing stale lock from dead pid $holder"
            rm -rf "$LOCK_DIR"
            continue
        fi
        log "gate busy (held by pid ${holder:-unknown}) — waiting 15s"
        sleep 15
    done
    echo $$ > "$LOCK_DIR/pid"
    trap 'rm -rf "$LOCK_DIR"' EXIT INT TERM
}

run_with_timeout() {
    # prefer coreutils gtimeout/timeout; fall back to a watchdog subshell.
    if command -v gtimeout >/dev/null 2>&1; then
        gtimeout --signal=TERM "$TIMEOUT_SECS" "$@"
    elif command -v timeout >/dev/null 2>&1; then
        timeout --signal=TERM "$TIMEOUT_SECS" "$@"
    else
        "$@" &
        cmd_pid=$!
        ( sleep "$TIMEOUT_SECS" && kill "$cmd_pid" 2>/dev/null ) &
        watchdog=$!
        wait "$cmd_pid"; rc=$?
        kill "$watchdog" 2>/dev/null
        return "$rc"
    fi
}

acquire_lock

if [ -n "${IGC_GATE_CMD:-}" ]; then
    # test seam: run an arbitrary command under the lock + timeout machinery
    log "running override command under gate lock"
    run_with_timeout bash -c "$IGC_GATE_CMD"
    exit $?
fi

log "running offline gate (nice -n 19, timeout ${TIMEOUT_SECS}s)"
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
    run_with_timeout nice -n 19 "$PYTHON_BIN" -m pytest -q "$@"
rc=$?
log "gate finished rc=$rc"
exit "$rc"

# Author: Mus mbayramo@stanford.edu
