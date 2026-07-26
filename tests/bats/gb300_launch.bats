#!/usr/bin/env bats
# Offline contract tests for scripts/gb300_launch.sh.
#
# These tests inspect dry-run rendering only. They must not execute Docker,
# nvidia-smi, GPUs, model loads, or training commands on the operator laptop.

setup() {
    REPO_ROOT="$(cd "${BATS_TEST_DIRNAME}/../.." && pwd)"
    export REPO_ROOT
    LAUNCH="${REPO_ROOT}/scripts/gb300_launch.sh"
    export LAUNCH
    export IGC_CODE_DIR="${BATS_TEST_TMPDIR}/igc"
    export IGC_MODELS_DIR="${BATS_TEST_TMPDIR}/models"
    export IGC_OUTPUT_DIR="${IGC_MODELS_DIR}/runs/phase1_gpt2_smoke"
    export IGC_DRY_RUN=1
    mkdir -p "$IGC_CODE_DIR" "$IGC_MODELS_DIR"
}

@test "gb300_launch requires IGC_PROFILE" {
    run env -u IGC_PROFILE \
        IGC_MODELS_DIR="$IGC_MODELS_DIR" \
        IGC_OUTPUT_DIR="$IGC_OUTPUT_DIR" \
        IGC_DRY_RUN=1 \
        bash "$LAUNCH"

    [ "$status" -ne 0 ]
    [[ "$output" == *"set IGC_PROFILE"* ]]
}

@test "gb300_launch requires explicit durable IGC_OUTPUT_DIR" {
    run env -u IGC_OUTPUT_DIR \
        IGC_PROFILE=phase1_gpt2_smoke \
        IGC_MODELS_DIR="$IGC_MODELS_DIR" \
        IGC_DRY_RUN=1 \
        bash "$LAUNCH"

    [ "$status" -ne 0 ]
    [[ "$output" == *"set IGC_OUTPUT_DIR to a durable path under IGC_MODELS_DIR"* ]]
}

@test "gb300_launch rejects output outside IGC_MODELS_DIR" {
    run env \
        IGC_PROFILE=phase1_gpt2_smoke \
        IGC_MODELS_DIR="$IGC_MODELS_DIR" \
        IGC_OUTPUT_DIR="${BATS_TEST_TMPDIR}/not-models/out" \
        IGC_DRY_RUN=1 \
        bash "$LAUNCH"

    [ "$status" -eq 3 ]
    [[ "$output" == *"BLOCKER: IGC_OUTPUT_DIR must be under IGC_MODELS_DIR"* ]]
}

@test "gb300_launch dry-run delegates to scripts/run_profile.sh with profile env" {
    run env \
        IGC_PROFILE=phase1_gpt2_smoke \
        IGC_GPUS=2 \
        IGC_CODE_DIR="$IGC_CODE_DIR" \
        IGC_MODELS_DIR="$IGC_MODELS_DIR" \
        IGC_OUTPUT_DIR="$IGC_OUTPUT_DIR" \
        IGC_DRY_RUN=1 \
        bash "$LAUNCH"

    [ "$status" -eq 0 ]
    [[ "$output" == *"profile=phase1_gpt2_smoke gpus=2"* ]]
    [[ "$output" == *"--gpus 2"* ]]
    [[ "$output" == *"-e IGC_PROFILE=phase1_gpt2_smoke"* ]]
    [[ "$output" == *"-e IGC_OUTPUT_DIR=${IGC_OUTPUT_DIR}"* ]]
    [[ "$output" == *"bash scripts/run_profile.sh"* ]]
    [[ "$output" != *"igc_main.py"* ]]
    [[ "$output" != *"--train"* ]]
    [[ "$output" != *"--llm"* ]]
}

@test "gb300_launch dry-run performs no docker or GPU probe action" {
    fake_bin="${BATS_TEST_TMPDIR}/bin"
    mkdir -p "$fake_bin"
    cat >"${fake_bin}/docker" <<'EOF'
#!/usr/bin/env bash
echo "docker must not execute in dry-run" >&2
exit 91
EOF
    cat >"${fake_bin}/nvidia-smi" <<'EOF'
#!/usr/bin/env bash
echo "nvidia-smi must not execute in dry-run" >&2
exit 92
EOF
    chmod +x "${fake_bin}/docker" "${fake_bin}/nvidia-smi"

    run env \
        IGC_PROFILE=phase1_gpt2_smoke \
        IGC_CODE_DIR="$IGC_CODE_DIR" \
        IGC_MODELS_DIR="$IGC_MODELS_DIR" \
        IGC_OUTPUT_DIR="$IGC_OUTPUT_DIR" \
        IGC_DRY_RUN=1 \
        PATH="${fake_bin}:/usr/bin:/bin" \
        bash "$LAUNCH"

    [ "$status" -eq 0 ]
    [[ "$output" == *"docker "* ]]
    [[ "$output" != *"must not execute"* ]]
}

@test "gb300_launch contains no retired stage aliases or direct training flags" {
    run grep -Eq 'IGC_RUNG|IGC_STAGE|smoke1|smoke4|run4|fsdp4|train_igc\.sbatch' "$LAUNCH"
    [ "$status" -ne 0 ]

    run grep -Eq 'igc_main\.py|--train|--llm|--max_steps|--num_train_epochs' "$LAUNCH"
    [ "$status" -ne 0 ]
}
