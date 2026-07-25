#!/usr/bin/env bats

setup() {
    REPO_ROOT="$(cd "${BATS_TEST_DIRNAME}/../.." && pwd)"
    export REPO_ROOT
    OPS="${REPO_ROOT}/scripts/docker_igc_ops.sh"
    export OPS
    export IGC_INTERNAL_ROOT="${BATS_TEST_TMPDIR}/internal"
    mkdir -p "${IGC_INTERNAL_ROOT}/.internal/docker"
    : >"${IGC_INTERNAL_ROOT}/.internal/docker/Dockerfile.test"
    : >"${IGC_INTERNAL_ROOT}/.internal/docker/Dockerfile.train"
    : >"${IGC_INTERNAL_ROOT}/.internal/docker/id_rsa"
    chmod 600 "${IGC_INTERNAL_ROOT}/.internal/docker/id_rsa"
}

@test "deploy requires an explicit HOST" {
    run bash "$OPS" deploy

    [ "$status" -ne 0 ]
    [[ "$output" == *"BLOCKER: set HOST"* ]]
}

@test "stop requires an explicit HOST" {
    run bash "$OPS" stop

    [ "$status" -ne 0 ]
    [[ "$output" == *"BLOCKER: set HOST"* ]]
}

@test "upload dry-run builds the private SSH image on each node and verifies conda" {
    run env IGC_DRY_RUN=1 IGC_NODES="127.0.0.1" IGC_INTERNAL_ROOT="$IGC_INTERNAL_ROOT" bash "$OPS" upload

    [ "$status" -eq 0 ]
    [[ "$output" == *"docker build -f docker/Dockerfile.train"* ]]
    [[ "$output" == *"docker build -f .internal/docker/Dockerfile.train"* ]]
    [[ "$output" == *"command -v conda"* ]]
    [[ "$output" == *"conda --version"* ]]
    [[ "$output" == *"command -v git-lfs"* ]]
    [[ "$output" == *"private NV72 node image"* ]]
    [[ "$output" != *"docker save igc-train-ssh:local"* ]]
    [[ "$output" != *"docker push igc-train-ssh:local"* ]]
}

@test "deploy dry-run targets one resolved host and refuses implicit recreate" {
    run env IGC_DRY_RUN=1 HOST=127.0.0.1 bash "$OPS" deploy

    [ "$status" -eq 0 ]
    [[ "$output" == *"nvidia@127.0.0.1"* ]]
    [[ "$output" == *"docker inspect igc-phase1-pretrain"* ]]
    [[ "$output" == *"RECREATE=1"* ]]
    [[ "$output" != *"docker rm -f igc-phase1-pretrain"* ]]
}

@test "deploy dry-run can explicitly recreate only the named container" {
    run env IGC_DRY_RUN=1 HOST=127.0.0.1 RECREATE=1 bash "$OPS" deploy

    [ "$status" -eq 0 ]
    [[ "$output" == *"nvidia@127.0.0.1"* ]]
    [[ "$output" == *"docker rm -f igc-phase1-pretrain"* ]]
    [[ "$output" == *"docker run -d --name igc-phase1-pretrain"* ]]
    [[ "$output" != *"docker image prune"* ]]
}

@test "stop dry-run stops only the named container on the resolved host" {
    run env IGC_DRY_RUN=1 HOST=127.0.0.1 bash "$OPS" stop

    [ "$status" -eq 0 ]
    [[ "$output" == *"nvidia@127.0.0.1"* ]]
    [[ "$output" == *"docker stop igc-phase1-pretrain"* ]]
    [[ "$output" != *"docker rm"* ]]
    [[ "$output" != *"docker image prune"* ]]
}

@test "dev-run requires an explicit HOST and command" {
    run env DEV_CMD="python -V" bash "$OPS" dev-run

    [ "$status" -ne 0 ]
    [[ "$output" == *"BLOCKER: set HOST"* ]]

    run env HOST=slot2 bash "$OPS" dev-run

    [ "$status" -ne 0 ]
    [[ "$output" == *"BLOCKER: set DEV_CMD"* ]]
}

@test "dev-create dry-run starts a CPU-only per-agent container" {
    run env IGC_DRY_RUN=1 HOST=127.0.0.1 DEV_NAME=worker-a IGC_BRANCH=feature/example bash "$OPS" dev-create

    [ "$status" -eq 0 ]
    [[ "$output" == *"nvidia@127.0.0.1"* ]]
    [[ "$output" == *"docker image inspect igc-train-ssh:local"* ]]
    [[ "$output" == *"clone/pull feature/example into /models/igc/dev/worker-a/igc"* ]]
    [[ "$output" == *"docker run -d --name igc-dev-worker-a"* ]]
    [[ "$output" == *"--label igc.role=cpu-dev"* ]]
    [[ "$output" == *"CPU-only dev container"* ]]
    [[ "$output" != *"docker run -d --name igc-dev-worker-a --gpus"* ]]
}

@test "dev-run dry-run executes only through remote docker exec in dev container" {
    run env IGC_DRY_RUN=1 HOST=127.0.0.1 DEV_NAME=worker-a IGC_BRANCH=feature/example DEV_CMD="python -m pytest -q tests/example.py" bash "$OPS" dev-run

    [ "$status" -eq 0 ]
    [[ "$output" == *"nvidia@127.0.0.1"* ]]
    [[ "$output" == *"docker inspect igc-dev-worker-a"* ]]
    [[ "$output" == *"docker exec igc-dev-worker-a"* ]]
    [[ "$output" == *"git fetch/pull feature/example"* ]]
    [[ "$output" == *"no laptop docker or laptop pytest"* ]]
    [[ "$output" != *"docker build"* ]]
}
