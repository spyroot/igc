#!/usr/bin/env bats

setup() {
    REPO_ROOT="$(cd "${BATS_TEST_DIRNAME}/../.." && pwd)"
    export REPO_ROOT
}

@test "slot LFS push refuses artifacts not covered by an LFS filter" {
    tmp="$(mktemp -d)"
    git -C "${tmp}" init -q
    git -C "${tmp}" config user.email test@example.invalid
    git -C "${tmp}" config user.name "IGC Test"
    printf '*.safetensors filter=lfs diff=lfs merge=lfs -text\n' >"${tmp}/.gitattributes"
    git -C "${tmp}" add .gitattributes
    git -C "${tmp}" commit -q -m "add lfs attributes"
    printf 'payload\n' >"${tmp}/raw.bin"
    before_branch="$(git -C "${tmp}" symbolic-ref --short HEAD)"

    run bash -c 'cd "$1" && env IGC_YES=1 IGC_REMOTE=origin "$2/scripts/lfs_push_from_slot.sh" raw.bin' \
        _ "${tmp}" "${REPO_ROOT}"

    [ "$status" -ne 0 ]
    [[ "$output" == *"not matched by an LFS filter"* ]]
    [[ "$output" == *"Refusing to stage this artifact as a normal Git object."* ]]
    [ "$(git -C "${tmp}" symbolic-ref --short HEAD)" = "${before_branch}" ]
}

@test "model adapter safetensors files are tracked by Git LFS" {
    run git -C "${REPO_ROOT}" check-attr filter -- models/model_x/adapter.safetensors

    [ "$status" -eq 0 ]
    [[ "$output" == *"models/model_x/adapter.safetensors: filter: lfs"* ]]
}
