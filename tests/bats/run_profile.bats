#!/usr/bin/env bats

setup() {
    REPO_ROOT="$(cd "${BATS_TEST_DIRNAME}/../.." && pwd)"
    export REPO_ROOT

    mkdir -p "${BATS_TEST_TMPDIR}/bin" "${BATS_TEST_TMPDIR}/data"
    export PATH="${BATS_TEST_TMPDIR}/bin:${PATH}"
}

install_python_stub() {
    cat >"${BATS_TEST_TMPDIR}/bin/python" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-m" && "${2:-}" == "igc.modules.train.launch" ]]; then
    printf '%s\n' "$*" >>"${RUN_PROFILE_CALLS_FILE}"
    for arg in "$@"; do
        if [[ "$arg" == "--print-argv" ]]; then
            echo "--train llm --corpus_objective phase1_pretrain --batch_size 2"
            exit 0
        fi
    done
    exit 0
fi

if [[ "${1:-}" == "igc_main.py" ]]; then
    printf '%s\n' "$*" >"${RUN_PROFILE_FINAL_ARGS_FILE}"
    exit 0
fi

echo "unexpected python call: $*" >&2
exit 2
EOF
    chmod +x "${BATS_TEST_TMPDIR}/bin/python"
}

install_accelerate_stub() {
    cat >"${BATS_TEST_TMPDIR}/bin/accelerate" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

printf '%s\n' "$*" >"${RUN_PROFILE_ACCELERATE_ARGS_FILE}"
EOF
    chmod +x "${BATS_TEST_TMPDIR}/bin/accelerate"
}

@test "run_profile requires IGC_PROFILE" {
    run env -u IGC_PROFILE bash "${REPO_ROOT}/scripts/run_profile.sh"

    [ "$status" -ne 0 ]
    [[ "$output" == *"set IGC_PROFILE"* ]]
}

@test "run_profile resolves profile argv and launches igc_main with paths, corpus, and extras" {
    install_python_stub

    run env \
        IGC_PROFILE=phase1_gpt2_smoke \
        IGC_DATA_DIR="${BATS_TEST_TMPDIR}/data" \
        IGC_CORPUS_DIR="${BATS_TEST_TMPDIR}/corpus" \
        IGC_CORPUS_EVAL_DIR="${BATS_TEST_TMPDIR}/corpus-eval" \
        IGC_OUTPUT_DIR="${BATS_TEST_TMPDIR}/out" \
        IGC_METRIC_REPORT=tensorboard \
        IGC_SET="batch_size=2 lr=1e-4" \
        RUN_PROFILE_CALLS_FILE="${BATS_TEST_TMPDIR}/calls.txt" \
        RUN_PROFILE_FINAL_ARGS_FILE="${BATS_TEST_TMPDIR}/final-args.txt" \
        bash "${REPO_ROOT}/scripts/run_profile.sh" -- --recreate_dataset

    [ "$status" -eq 0 ]
    [ -d "${BATS_TEST_TMPDIR}/out" ]

    calls="$(cat "${BATS_TEST_TMPDIR}/calls.txt")"
    [[ "$calls" == *"--profile phase1_gpt2_smoke"* ]]
    [[ "$calls" == *"--set batch_size=2"* ]]
    [[ "$calls" == *"--set lr=1e-4"* ]]
    [[ "$calls" == *"--print-argv"* ]]

    final_args="$(cat "${BATS_TEST_TMPDIR}/final-args.txt")"
    [[ "$final_args" == *"igc_main.py --train llm --corpus_objective phase1_pretrain --batch_size 2"* ]]
    [[ "$final_args" == *"--json_data_dir ${BATS_TEST_TMPDIR}/data"* ]]
    [[ "$final_args" == *"--corpus_dir ${BATS_TEST_TMPDIR}/corpus"* ]]
    [[ "$final_args" == *"--corpus_eval_dir ${BATS_TEST_TMPDIR}/corpus-eval"* ]]
    [[ "$final_args" == *"--corpus_objective phase1_pretrain"* ]]
    [[ "$final_args" == *"--output_dir ${BATS_TEST_TMPDIR}/out"* ]]
    [[ "$final_args" == *"--metric_report tensorboard -- --recreate_dataset"* ]]
}

@test "run_profile requires held-out corpus directory with train corpus directory" {
    install_python_stub

    run env \
        IGC_PROFILE=phase1_gpt2_smoke \
        IGC_CORPUS_DIR="${BATS_TEST_TMPDIR}/corpus" \
        IGC_OUTPUT_DIR="${BATS_TEST_TMPDIR}/out" \
        IGC_METRIC_REPORT=tensorboard \
        RUN_PROFILE_CALLS_FILE="${BATS_TEST_TMPDIR}/calls.txt" \
        RUN_PROFILE_FINAL_ARGS_FILE="${BATS_TEST_TMPDIR}/final-args.txt" \
        bash "${REPO_ROOT}/scripts/run_profile.sh"

    [ "$status" -eq 2 ]
    [[ "$output" == *"IGC_CORPUS_EVAL_DIR is required with IGC_CORPUS_DIR"* ]]
}

@test "run_profile uses plain python when IGC_GPUS is 1" {
    install_python_stub

    run env \
        IGC_PROFILE=phase1_gpt2_smoke \
        IGC_GPUS=1 \
        IGC_OUTPUT_DIR="${BATS_TEST_TMPDIR}/out" \
        IGC_METRIC_REPORT=tensorboard \
        RUN_PROFILE_CALLS_FILE="${BATS_TEST_TMPDIR}/calls.txt" \
        RUN_PROFILE_FINAL_ARGS_FILE="${BATS_TEST_TMPDIR}/final-args.txt" \
        bash "${REPO_ROOT}/scripts/run_profile.sh"

    [ "$status" -eq 0 ]
    final_args="$(cat "${BATS_TEST_TMPDIR}/final-args.txt")"
    [[ "$final_args" == igc_main.py\ --train\ llm* ]]
    [[ "$final_args" != *"accelerate launch"* ]]
    [ ! -e "${BATS_TEST_TMPDIR}/accelerate-args.txt" ]
}

@test "run_profile uses accelerate launch only when IGC_GPUS is greater than 1" {
    install_python_stub
    install_accelerate_stub

    run env \
        IGC_PROFILE=phase1_gpt2_smoke \
        IGC_GPUS=2 \
        IGC_MASTER_PORT=29611 \
        IGC_OUTPUT_DIR="${BATS_TEST_TMPDIR}/out" \
        IGC_METRIC_REPORT=tensorboard \
        RUN_PROFILE_CALLS_FILE="${BATS_TEST_TMPDIR}/calls.txt" \
        RUN_PROFILE_FINAL_ARGS_FILE="${BATS_TEST_TMPDIR}/final-args.txt" \
        RUN_PROFILE_ACCELERATE_ARGS_FILE="${BATS_TEST_TMPDIR}/accelerate-args.txt" \
        bash "${REPO_ROOT}/scripts/run_profile.sh"

    [ "$status" -eq 0 ]
    [ ! -e "${BATS_TEST_TMPDIR}/final-args.txt" ]
    accelerate_args="$(cat "${BATS_TEST_TMPDIR}/accelerate-args.txt")"
    [[ "$accelerate_args" == launch\ --num_processes\ 2* ]]
    [[ "$accelerate_args" == *"--main_process_ip 127.0.0.1"* ]]
    [[ "$accelerate_args" == *"--main_process_port 29611"* ]]
    [[ "$accelerate_args" == *"igc_main.py --use_accelerator --train llm"* ]]
    [[ "$accelerate_args" == *"--output_dir ${BATS_TEST_TMPDIR}/out"* ]]
    [[ "$accelerate_args" == *"--metric_report tensorboard"* ]]
}

@test "run_profile rejects retired IGC_CORPUS_OBJECTIVE env override" {
    run env \
        IGC_PROFILE=phase1_gpt2_smoke \
        IGC_CORPUS_OBJECTIVE=legacy \
        bash "${REPO_ROOT}/scripts/run_profile.sh"

    [ "$status" -eq 2 ]
    [[ "$output" == *"IGC_CORPUS_OBJECTIVE is retired"* ]]
}
