#!/usr/bin/env bats

setup() {
    REPO_ROOT="$(cd "${BATS_TEST_DIRNAME}/../.." && pwd)"
    export REPO_ROOT
    SCRIPT="${REPO_ROOT}/scripts/verify_lfs_weight_ack.sh"
    export SCRIPT

    TMP_REPO="${BATS_TEST_TMPDIR}/repo"
    export TMP_REPO
    mkdir -p "${TMP_REPO}/models/model_x/run"
    git -C "${TMP_REPO}" init -q
    git -C "${TMP_REPO}" config user.email "test@example.invalid"
    git -C "${TMP_REPO}" config user.name "IGC Test"

    # Store synthetic pointer text as-is even when the host has global LFS filters.
    git -C "${TMP_REPO}" config filter.lfs.clean cat
    git -C "${TMP_REPO}" config filter.lfs.smudge cat
    git -C "${TMP_REPO}" config --unset-all filter.lfs.process 2>/dev/null || true
    git -C "${TMP_REPO}" config filter.lfs.required false

    cat >"${TMP_REPO}/.gitattributes" <<'EOF'
*.safetensors filter=lfs diff=lfs merge=lfs -text
EOF
}

commit_fixture() {
    git -C "${TMP_REPO}" add .gitattributes
    blob="$(
        cd "${TMP_REPO}"
        git hash-object -w --no-filters -- models/model_x/run/adapter_model.safetensors
    )"
    git -C "${TMP_REPO}" update-index \
        --add \
        --cacheinfo 100644 "$blob" models/model_x/run/adapter_model.safetensors
    git -C "${TMP_REPO}" commit -q -m "fixture"
}

write_pointer() {
    cat >"${TMP_REPO}/models/model_x/run/adapter_model.safetensors" <<'EOF'
version https://git-lfs.github.com/spec/v1
oid sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef
size 123456
EOF
}

@test "ack succeeds for a Git LFS pointer with expected size and sha" {
    write_pointer
    commit_fixture

    run bash -c '
        cd "$TMP_REPO"
        bash "$SCRIPT" \
            --ref HEAD \
            --path models/model_x/run/adapter_model.safetensors \
            --expected-size 123456 \
            --expected-sha256 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef
    '

    [ "$status" -eq 0 ]
    [[ "$output" == *"LFS_ARTIFACT_ACK"* ]]
    [[ "$output" == *"size=123456"* ]]
    [[ "$output" == *"remote=not_checked"* ]]
}

@test "ack rejects a non-pointer blob even when path has LFS attributes" {
    printf 'not a pointer\n' >"${TMP_REPO}/models/model_x/run/adapter_model.safetensors"
    commit_fixture

    run bash -c '
        cd "$TMP_REPO"
        bash "$SCRIPT" HEAD models/model_x/run/adapter_model.safetensors
    '

    [ "$status" -eq 1 ]
    [[ "$output" == *"BLOCKER"* ]]
    [[ "$output" == *"not a Git LFS pointer"* ]]
}

@test "ack rejects expected size mismatches" {
    write_pointer
    commit_fixture

    run bash -c '
        cd "$TMP_REPO"
        bash "$SCRIPT" \
            --ref HEAD \
            --path models/model_x/run/adapter_model.safetensors \
            --expected-size 99
    '

    [ "$status" -eq 1 ]
    [[ "$output" == *"BLOCKER"* ]]
    [[ "$output" == *"size mismatch"* ]]
}

@test "ack refuses to read oversized Git blobs" {
    yes raw-weight-bytes | head -400 >"${TMP_REPO}/models/model_x/run/adapter_model.safetensors"
    commit_fixture

    run bash -c '
        cd "$TMP_REPO"
        bash "$SCRIPT" \
            --ref HEAD \
            --path models/model_x/run/adapter_model.safetensors \
            --max-pointer-bytes 128
    '

    [ "$status" -eq 1 ]
    [[ "$output" == *"BLOCKER"* ]]
    [[ "$output" == *"not an LFS pointer; refusing to read it"* ]]
}
