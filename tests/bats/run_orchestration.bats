#!/usr/bin/env bats

setup() {
    REPO_ROOT="$(cd "${BATS_TEST_DIRNAME}/../.." && pwd)"
    export REPO_ROOT

    mkdir -p "${BATS_TEST_TMPDIR}/bin" "${BATS_TEST_TMPDIR}/out"
    export PATH="${BATS_TEST_TMPDIR}/bin:${PATH}"
    PYTHON_BIN="${PYTHON_BIN:-python3}"
    export PYTHON_BIN
}

write_slurm_spec() {
    cat >"${BATS_TEST_TMPDIR}/slurm.yaml" <<EOF
version: 1
name: bats-slurm
backend: slurm
image:
  ref: igc-train:test
runtime:
  command: [python, igc_main.py, --help]
paths:
  output: ${BATS_TEST_TMPDIR}/out
resources:
  gpus: 1
  nodes: 1
slurm:
  output: ${BATS_TEST_TMPDIR}/out/slurm-%j.out
sanity:
  slurm:
    enabled: true
    command: "printf IGC_SLURM_SANITY_OK"
    sentinel: IGC_SLURM_SANITY_OK
    timeout_seconds: 3
    poll_seconds: 1
EOF
}

install_completed_slurm_fakes() {
    cat >"${BATS_TEST_TMPDIR}/bin/sbatch" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
out=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        --output=*) out="${1#--output=}" ;;
        --output) shift; out="$1" ;;
    esac
    shift || true
done
out="${out//%j/123}"
mkdir -p "$(dirname "$out")"
printf 'IGC_SLURM_SANITY_OK\n' >"$out"
printf '123\n'
EOF
    chmod +x "${BATS_TEST_TMPDIR}/bin/sbatch"

    cat >"${BATS_TEST_TMPDIR}/bin/sacct" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "${FAKE_SLURM_STATE:-COMPLETED}"
EOF
    chmod +x "${BATS_TEST_TMPDIR}/bin/sacct"

    cat >"${BATS_TEST_TMPDIR}/bin/scancel" <<'EOF'
#!/usr/bin/env bash
exit 0
EOF
    chmod +x "${BATS_TEST_TMPDIR}/bin/scancel"
}

@test "render_run renders committed docker example" {
    run "${PYTHON_BIN}" "${REPO_ROOT}/scripts/render_run.py" \
        --spec "${REPO_ROOT}/configs/run/examples/docker-smoke.yaml"

    [ "$status" -eq 0 ]
    [[ "$output" == *"backend: docker"* ]]
    [[ "$output" == *"docker run"* ]]
    [[ "$output" == *"\${IGC_DOCKER_ENV_FILE}"* ]]
}

@test "launch_run refuses live execution without dry-run" {
    run "${PYTHON_BIN}" "${REPO_ROOT}/scripts/launch_run.py" \
        --spec "${REPO_ROOT}/configs/run/examples/docker-smoke.yaml"

    [ "$status" -eq 2 ]
    [[ "$output" == *"refusing live launch"* ]]
}

@test "slurm sanity dry-run renders sbatch without calling scheduler" {
    write_slurm_spec

    run "${PYTHON_BIN}" "${REPO_ROOT}/scripts/slurm_sanity_from_spec.py" \
        --spec "${BATS_TEST_TMPDIR}/slurm.yaml" --dry-run

    [ "$status" -eq 0 ]
    [[ "$output" == *"sbatch"* ]]
    [[ "$output" == *"IGC_SLURM_SANITY_OK"* ]]
}

@test "slurm sanity live mode passes with mocked completed job" {
    write_slurm_spec
    install_completed_slurm_fakes

    run env IGC_SLURM_SANITY_TEST=1 "${PYTHON_BIN}" "${REPO_ROOT}/scripts/slurm_sanity_from_spec.py" \
        --spec "${BATS_TEST_TMPDIR}/slurm.yaml" --live

    [ "$status" -eq 0 ]
    [[ "$output" == *"slurm sanity OK"* ]]
}

@test "slurm sanity live mode reports failed job gracefully" {
    write_slurm_spec
    install_completed_slurm_fakes

    run env IGC_SLURM_SANITY_TEST=1 FAKE_SLURM_STATE=FAILED \
        "${PYTHON_BIN}" "${REPO_ROOT}/scripts/slurm_sanity_from_spec.py" \
        --spec "${BATS_TEST_TMPDIR}/slurm.yaml" --live

    [ "$status" -eq 1 ]
    [[ "$output" == *"BLOCKER"* ]]
    [[ "$output" == *"FAILED"* ]]
}

@test "docker image sync dry-run renders inspect and pull" {
    run "${PYTHON_BIN}" "${REPO_ROOT}/scripts/docker_image_sync.py" \
        --spec "${REPO_ROOT}/configs/run/examples/docker-smoke.yaml" --dry-run

    [ "$status" -eq 0 ]
    [[ "$output" == *"docker image inspect igc-train:latest"* ]]
    [[ "$output" == *"docker pull igc-train:latest"* ]]
    [[ "$output" != *"docker build"* ]]
}

@test "docker image sync reports pull failure gracefully" {
    cat >"${BATS_TEST_TMPDIR}/bin/docker" <<'EOF'
#!/usr/bin/env bash
if [[ "$1 $2 $3" == "image inspect igc-train:latest" ]]; then
    exit 1
fi
if [[ "$1 $2" == "pull igc-train:latest" ]]; then
    echo "registry unavailable" >&2
    exit 42
fi
exit 99
EOF
    chmod +x "${BATS_TEST_TMPDIR}/bin/docker"

    run "${PYTHON_BIN}" "${REPO_ROOT}/scripts/docker_image_sync.py" \
        --spec "${REPO_ROOT}/configs/run/examples/docker-smoke.yaml"

    [ "$status" -eq 1 ]
    [[ "$output" == *"BLOCKER: docker pull failed"* ]]
    [[ "$output" == *"registry unavailable"* ]]
}
