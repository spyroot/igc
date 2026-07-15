#!/usr/bin/env bats

setup() {
    REPO_ROOT="$(cd "${BATS_TEST_DIRNAME}/../.." && pwd)"
    export REPO_ROOT
}

@test "model adapter safetensors files are tracked by Git LFS" {
    run git -C "${REPO_ROOT}" check-attr filter -- models/model_x/adapter.safetensors

    [ "$status" -eq 0 ]
    [[ "$output" == *"models/model_x/adapter.safetensors: filter: lfs"* ]]
}
