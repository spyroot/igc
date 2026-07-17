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
    [[ "$output" == *"not matched by a committed LFS filter"* ]]
    [[ "$output" == *"Refusing to stage this artifact as a normal Git object."* ]]
    [ "$(git -C "${tmp}" symbolic-ref --short HEAD)" = "${before_branch}" ]
}

@test "slot LFS push ignores uncommitted gitattributes filters" {
    tmp="$(mktemp -d)"
    git -C "${tmp}" init -q
    git -C "${tmp}" config user.email test@example.invalid
    git -C "${tmp}" config user.name "IGC Test"
    printf '*.safetensors filter=lfs diff=lfs merge=lfs -text\n' >"${tmp}/.gitattributes"
    git -C "${tmp}" add .gitattributes
    git -C "${tmp}" commit -q -m "add committed lfs attributes"
    printf '*.bin filter=lfs diff=lfs merge=lfs -text\n' >>"${tmp}/.gitattributes"
    printf 'payload\n' >"${tmp}/raw.bin"
    before_branch="$(git -C "${tmp}" symbolic-ref --short HEAD)"

    run bash -c 'cd "$1" && env IGC_YES=1 IGC_REMOTE=origin "$2/scripts/lfs_push_from_slot.sh" raw.bin' \
        _ "${tmp}" "${REPO_ROOT}"

    [ "$status" -ne 0 ]
    [[ "$output" == *"not matched by a committed LFS filter"* ]]
    [[ "$output" == *"Refusing to stage this artifact as a normal Git object."* ]]
    [ "$(git -C "${tmp}" symbolic-ref --short HEAD)" = "${before_branch}" ]
}

@test "slot LFS push accepts committed gitattributes filters before git-lfs gate" {
    tmp="$(mktemp -d)"
    git -C "${tmp}" init -q
    git -C "${tmp}" config user.email test@example.invalid
    git -C "${tmp}" config user.name "IGC Test"
    printf '*.bin filter=lfs diff=lfs merge=lfs -text\n' >"${tmp}/.gitattributes"
    git -C "${tmp}" add .gitattributes
    git -C "${tmp}" commit -q -m "add committed lfs attributes"
    printf 'payload\n' >"${tmp}/raw.bin"
    before_branch="$(git -C "${tmp}" symbolic-ref --short HEAD)"
    real_git="$(command -v git)"
    mkdir "${tmp}/fake-bin"
    cat >"${tmp}/fake-bin/git" <<EOF
#!/usr/bin/env bash
if [ "\${1:-}" = "lfs" ] && [ "\${2:-}" = "version" ]; then
    exit 127
fi
exec "${real_git}" "\$@"
EOF
    chmod +x "${tmp}/fake-bin/git"

    run bash -c 'cd "$1" && env PATH="$1/fake-bin:$PATH" IGC_YES=1 IGC_REMOTE=origin "$2/scripts/lfs_push_from_slot.sh" raw.bin' \
        _ "${tmp}" "${REPO_ROOT}"

    [ "$status" -ne 0 ]
    [[ "$output" == *"git-lfs not found"* ]]
    [[ "$output" != *"not matched by a committed LFS filter"* ]]
    [ "$(git -C "${tmp}" symbolic-ref --short HEAD)" = "${before_branch}" ]
}

@test "slot LFS push reports check-attr command failure distinctly" {
    tmp="$(mktemp -d)"
    git -C "${tmp}" init -q
    git -C "${tmp}" config user.email test@example.invalid
    git -C "${tmp}" config user.name "IGC Test"
    printf '*.bin filter=lfs diff=lfs merge=lfs -text\n' >"${tmp}/.gitattributes"
    git -C "${tmp}" add .gitattributes
    git -C "${tmp}" commit -q -m "add committed lfs attributes"
    printf 'payload\n' >"${tmp}/raw.bin"
    before_branch="$(git -C "${tmp}" symbolic-ref --short HEAD)"
    real_git="$(command -v git)"
    mkdir "${tmp}/fake-bin"
    # Mimic a git without check-attr --source: usage error on stderr, exit 129.
    cat >"${tmp}/fake-bin/git" <<EOF
#!/usr/bin/env bash
if [ "\${1:-}" = "check-attr" ] && [ "\${2:-}" = "--source" ]; then
    echo "error: unknown option \\\`source'" >&2
    exit 129
fi
exec "${real_git}" "\$@"
EOF
    chmod +x "${tmp}/fake-bin/git"

    run bash -c 'cd "$1" && env PATH="$1/fake-bin:$PATH" IGC_YES=1 IGC_REMOTE=origin "$2/scripts/lfs_push_from_slot.sh" raw.bin' \
        _ "${tmp}" "${REPO_ROOT}"

    [ "$status" -ne 0 ]
    [[ "$output" == *"cannot read committed LFS attributes"* ]]
    [[ "$output" == *"unknown option"* ]]
    [[ "$output" != *"not matched by a committed LFS filter"* ]]
    [ "$(git -C "${tmp}" symbolic-ref --short HEAD)" = "${before_branch}" ]
}

@test "slot LFS push fails closed on an unborn HEAD repository" {
    # An unborn HEAD dies earlier, at the branch-name guard (rev-parse
    # --abbrev-ref HEAD exits 128 under set -e), so no artifact is staged and
    # the misleading filter-mismatch refusal is never printed.
    tmp="$(mktemp -d)"
    git -C "${tmp}" init -q
    git -C "${tmp}" config user.email test@example.invalid
    git -C "${tmp}" config user.name "IGC Test"
    printf '*.bin filter=lfs diff=lfs merge=lfs -text\n' >"${tmp}/.gitattributes"
    printf 'payload\n' >"${tmp}/raw.bin"

    run bash -c 'cd "$1" && env IGC_YES=1 IGC_REMOTE=origin "$2/scripts/lfs_push_from_slot.sh" raw.bin' \
        _ "${tmp}" "${REPO_ROOT}"

    [ "$status" -ne 0 ]
    [[ "$output" != *"not matched by a committed LFS filter"* ]]
    staged="$(git -C "${tmp}" status --porcelain -- raw.bin)"
    [[ "${staged}" != A* ]]
}

@test "model adapter safetensors files are tracked by Git LFS" {
    run git -C "${REPO_ROOT}" check-attr filter -- models/model_x/adapter.safetensors

    [ "$status" -eq 0 ]
    [[ "$output" == *"models/model_x/adapter.safetensors: filter: lfs"* ]]
}
