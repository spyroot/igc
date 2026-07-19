#!/usr/bin/env bash
# Run bounded IGC profiling inside an already-running GB300/NV72 slot container.
#
# This wrapper is a coordinator: it verifies the remote slot, records
# Docker/NVIDIA evidence, creates a /models profile bundle, and delegates the
# real dataset-to-CUDA timing to scripts/profile_dataset_to_cuda.py.
set -euo pipefail

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 2
}

quote() {
    printf '%q' "$1"
}

: "${HOST:?set HOST to slot2 or slot11 before remote profiling}"

PROFILE_MODE="${PROFILE_MODE:-snapshot}"
case "${PROFILE_MODE}" in
    snapshot|cpu|cuda|all) ;;
    *) die "PROFILE_MODE must be snapshot, cpu, cuda, or all" ;;
esac

if [[ "${PROFILE_MODE}" == "cuda" || "${PROFILE_MODE}" == "all" ]]; then
    [[ "${ALLOW_GPU_PROFILE:-}" == "1" ]] ||
        die "set ALLOW_GPU_PROFILE=1 for CUDA profiling"
    case "${HOST}" in
        slot2|slot11) ;;
        *)
            [[ "${IGC_ALLOW_PROFILE_HOST_OVERRIDE:-}" == "1" ]] ||
                die "CUDA profiling is restricted to HOST=slot2 or HOST=slot11"
            ;;
    esac
    if [[ -z "${PROFILE_DATASET_ARGS:-}" ]]; then
        CUDA_CORPUS_TAR="${CUDA_CORPUS_TAR:-/models/igc/full_corpus/supermicro_gb300_full_corpus.tar.gz}"
        CUDA_SAMPLE_ROWS="${CUDA_SAMPLE_ROWS:-4096}"
        CUDA_MODEL="${CUDA_MODEL:-gpt2}"
        CUDA_TOKENIZER="${CUDA_TOKENIZER:-${CUDA_MODEL}}"
        CUDA_CORPUS_OBJECTIVE="${CUDA_CORPUS_OBJECTIVE:-phase1_pretrain}"
        CUDA_BATCH_SIZE="${CUDA_BATCH_SIZE:-4}"
        CUDA_SEQ_LEN="${CUDA_SEQ_LEN:-1024}"
        CUDA_NUM_WORKERS="${CUDA_NUM_WORKERS:-8}"
        CUDA_STEPS="${CUDA_STEPS:-10}"
        CUDA_WARMUP="${CUDA_WARMUP:-3}"
        CUDA_PRECISION="${CUDA_PRECISION:-bf16}"
        CUDA_DEVICE="${CUDA_DEVICE:-cuda:0}"
        profile_args=(
            --corpus-tar "${CUDA_CORPUS_TAR}"
            --trust-npy-pickle
            --sample-rows "${CUDA_SAMPLE_ROWS}"
            --model-type "${CUDA_MODEL}"
            --tokenizer "${CUDA_TOKENIZER}"
            --corpus-objective "${CUDA_CORPUS_OBJECTIVE}"
            --batch-size "${CUDA_BATCH_SIZE}"
            --seq-len "${CUDA_SEQ_LEN}"
            --num-workers "${CUDA_NUM_WORKERS}"
            --pin-memory
            --non-blocking
            --steps "${CUDA_STEPS}"
            --warmup "${CUDA_WARMUP}"
            --precision "${CUDA_PRECISION}"
            --device "${CUDA_DEVICE}"
        )
        if [[ "${CUDA_TRACE:-0}" == "1" ]]; then
            profile_args+=(--trace)
        fi
        if [[ "${CUDA_ALLOW_TOKENIZER_DOWNLOAD:-0}" == "1" ]]; then
            profile_args+=(--allow-tokenizer-download)
        fi
        PROFILE_DATASET_ARGS=""
        for arg in "${profile_args[@]}"; do
            PROFILE_DATASET_ARGS+=" $(quote "${arg}")"
        done
        PROFILE_DATASET_ARGS="${PROFILE_DATASET_ARGS# }"
    fi
    [[ " ${PROFILE_DATASET_ARGS} " != *" --output-dir "* ]] ||
        die "do not pass --output-dir in PROFILE_DATASET_ARGS; wrapper owns the bundle path"
fi

SSH_BIN="${SSH_BIN:-ssh}"
CONTAINER="${CONTAINER:-igc-phase1-pretrain}"
IGC_CODE_DIR="${IGC_CODE_DIR:-/workspace/igc}"
GPU_COUNT="${GPU_COUNT:-1}"
GPU_IDS="${GPU_IDS:-}"
GPU_MAX_MEM_MB="${GPU_MAX_MEM_MB:-1024}"
GPU_MAX_UTIL="${GPU_MAX_UTIL:-5}"
GPU_IDLE_SAMPLES="${GPU_IDLE_SAMPLES:-3}"
GPU_IDLE_INTERVAL="${GPU_IDLE_INTERVAL:-5}"
MIN_MODELS_FREE_GB="${MIN_MODELS_FREE_GB:-20}"
RUN_ID="${RUN_ID:-igc-${PROFILE_MODE}-$(date -u +%Y%m%dT%H%M%SZ)}"
REMOTE_TIMEOUT="${REMOTE_TIMEOUT:-10}"
BRAIN_READY_URL="${BRAIN_READY_URL:-}"
FLEET_STATE_URL="${FLEET_STATE_URL:-}"

[[ "${RUN_ID}" =~ ^[A-Za-z0-9._+-]+$ ]] ||
    die "RUN_ID may contain only letters, digits, dot, underscore, plus, and dash"
[[ "${GPU_COUNT}" =~ ^[0-9]+$ ]] && [[ "${GPU_COUNT}" -ge 1 ]] ||
    die "GPU_COUNT must be a positive integer"
[[ "${GPU_IDLE_SAMPLES}" =~ ^[0-9]+$ ]] && [[ "${GPU_IDLE_SAMPLES}" -ge 1 ]] ||
    die "GPU_IDLE_SAMPLES must be a positive integer"
[[ "${GPU_IDLE_INTERVAL}" =~ ^[0-9]+$ ]] ||
    die "GPU_IDLE_INTERVAL must be a non-negative integer"
[[ "${MIN_MODELS_FREE_GB}" =~ ^[0-9]+$ ]] ||
    die "MIN_MODELS_FREE_GB must be a non-negative integer"

remote_env=(
    "PROFILE_MODE=$(quote "${PROFILE_MODE}")"
    "CONTAINER=$(quote "${CONTAINER}")"
    "IGC_CODE_DIR=$(quote "${IGC_CODE_DIR}")"
    "GPU_COUNT=$(quote "${GPU_COUNT}")"
    "GPU_IDS=$(quote "${GPU_IDS}")"
    "GPU_MAX_MEM_MB=$(quote "${GPU_MAX_MEM_MB}")"
    "GPU_MAX_UTIL=$(quote "${GPU_MAX_UTIL}")"
    "GPU_IDLE_SAMPLES=$(quote "${GPU_IDLE_SAMPLES}")"
    "GPU_IDLE_INTERVAL=$(quote "${GPU_IDLE_INTERVAL}")"
    "MIN_MODELS_FREE_GB=$(quote "${MIN_MODELS_FREE_GB}")"
    "RUN_ID=$(quote "${RUN_ID}")"
    "PROFILE_DATASET_ARGS=$(quote "${PROFILE_DATASET_ARGS:-}")"
    "BRAIN_READY_URL=$(quote "${BRAIN_READY_URL}")"
    "FLEET_STATE_URL=$(quote "${FLEET_STATE_URL}")"
)

remote_prefix="${remote_env[*]}"

"${SSH_BIN}" \
    -o BatchMode=yes \
    -o ConnectTimeout="${REMOTE_TIMEOUT}" \
    "${HOST}" \
    "${remote_prefix} bash -s" <<'REMOTE'
set -euo pipefail

die() {
    SUMMARY_STATUS="failed"
    SUMMARY_MESSAGE="$*"
    printf 'ERROR: %s\n' "$*" >&2
    exit 2
}

q() {
    printf '%q' "$1"
}

json_string() {
    local value="$1"
    value="${value//\\/\\\\}"
    value="${value//\"/\\\"}"
    value="${value//$'\n'/\\n}"
    value="${value//$'\r'/\\r}"
    value="${value//$'\t'/\\t}"
    printf '"%s"' "${value}"
}

json_bool_for_file() {
    local path="$1"
    if [[ -s "${path}" ]]; then
        printf 'true'
    else
        printf 'false'
    fi
}

write_run_env() {
    {
        printf 'RUN_ID=%q\n' "${RUN_ID}"
        printf 'PROFILE_MODE=%q\n' "${PROFILE_MODE}"
        printf 'BUNDLE=%q\n' "${BUNDLE}"
        printf 'HOST=%q\n' "${REMOTE_HOST}"
        printf 'CONTAINER=%q\n' "${CONTAINER}"
        printf 'IGC_CODE_DIR=%q\n' "${IGC_CODE_DIR}"
        printf 'GPU_COUNT=%q\n' "${GPU_COUNT}"
        printf 'GPU_IDS=%q\n' "${GPU_IDS}"
        printf 'GPU_MAX_MEM_MB=%q\n' "${GPU_MAX_MEM_MB}"
        printf 'GPU_MAX_UTIL=%q\n' "${GPU_MAX_UTIL}"
        printf 'GPU_IDLE_SAMPLES=%q\n' "${GPU_IDLE_SAMPLES}"
        printf 'GPU_IDLE_INTERVAL=%q\n' "${GPU_IDLE_INTERVAL}"
        printf 'MIN_MODELS_FREE_GB=%q\n' "${MIN_MODELS_FREE_GB}"
        printf 'STARTED_AT=%q\n' "${STARTED_AT}"
        printf 'PROFILE_DATASET_ARGS=%q\n' "${PROFILE_DATASET_ARGS}"
    } > "${BUNDLE}/run.env"
}

write_summary() {
    local status="$1"
    local exit_code="$2"
    local finished_at
    finished_at="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    {
        printf '{\n'
        printf '  "status": %s,\n' "$(json_string "${status}")"
        printf '  "exit_code": %s,\n' "${exit_code}"
        printf '  "run_id": %s,\n' "$(json_string "${RUN_ID}")"
        printf '  "profile_mode": %s,\n' "$(json_string "${PROFILE_MODE}")"
        printf '  "bundle": %s,\n' "$(json_string "${BUNDLE}")"
        printf '  "host": %s,\n' "$(json_string "${REMOTE_HOST}")"
        printf '  "container": %s,\n' "$(json_string "${CONTAINER}")"
        printf '  "igc_code_dir": %s,\n' "$(json_string "${IGC_CODE_DIR}")"
        printf '  "gpu_count": %s,\n' "${GPU_COUNT}"
        printf '  "gpu_ids": %s,\n' "$(json_string "${GPU_IDS}")"
        printf '  "started_at": %s,\n' "$(json_string "${STARTED_AT}")"
        printf '  "finished_at": %s,\n' "$(json_string "${finished_at}")"
        printf '  "message": %s,\n' "$(json_string "${SUMMARY_MESSAGE:-}")"
        printf '  "artifacts": {\n'
        printf '    "run_env": %s,\n' "$(json_bool_for_file "${BUNDLE}/run.env")"
        printf '    "wrapper_log": %s,\n' "$(json_bool_for_file "${BUNDLE}/wrapper.log")"
        printf '    "docker_ps": %s,\n' "$(json_bool_for_file "${BUNDLE}/docker_ps.txt")"
        printf '    "docker_metadata": %s,\n' "$(json_bool_for_file "${BUNDLE}/docker_metadata.txt")"
        printf '    "container_git_state": %s,\n' "$(json_bool_for_file "${BUNDLE}/container_git_state.txt")"
        printf '    "nvidia_smi_before": %s,\n' "$(json_bool_for_file "${BUNDLE}/nvidia_smi_before.csv")"
        printf '    "nvidia_smi_after": %s,\n' "$(json_bool_for_file "${BUNDLE}/nvidia_smi_after.csv")"
        printf '    "gpu_selection": %s,\n' "$(json_bool_for_file "${BUNDLE}/gpu_selection.txt")"
        printf '    "profile_cpu_log": %s,\n' "$(json_bool_for_file "${BUNDLE}/profile_cpu.log")"
        printf '    "profile_dataset_cuda_log": %s,\n' "$(json_bool_for_file "${BUNDLE}/profile_dataset_cuda.log")"
        printf '    "profile_summary": %s\n' "$(json_bool_for_file "${BUNDLE}/profile_summary.json")"
        printf '  }\n'
        printf '}\n'
    } > "${BUNDLE}/summary.json"
}

finish_summary() {
    local exit_code="$?"
    if [[ -n "${BUNDLE:-}" && -d "${BUNDLE}" ]]; then
        if ((exit_code == 0)); then
            write_summary "ok" "${exit_code}"
        else
            write_summary "${SUMMARY_STATUS:-failed}" "${exit_code}"
        fi
    fi
    return "${exit_code}"
}

require_models_space() {
    local available_kb required_kb
    available_kb="$(
        df -Pk /models |
            awk 'NR == 2 {print $4}'
    )"
    [[ "${available_kb}" =~ ^[0-9]+$ ]] ||
        die "could not determine free space under /models"
    required_kb=$((MIN_MODELS_FREE_GB * 1024 * 1024))
    if ((available_kb < required_kb)); then
        die "/models free space below MIN_MODELS_FREE_GB=${MIN_MODELS_FREE_GB}"
    fi
}

trim_spaces() {
    local value="$1"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    printf '%s' "${value}"
}

gpu_requested() {
    local idx="$1"
    [[ -z "${GPU_IDS}" || ",${GPU_IDS}," == *",${idx},"* ]]
}

select_idle_gpus() {
    command -v nvidia-smi >/dev/null 2>&1 ||
        die "nvidia-smi required to select GPUs"

    local -A idle_counts=()
    local sample idx mem util snapshot
    for sample in $(seq 1 "${GPU_IDLE_SAMPLES}"); do
        snapshot="$(
            nvidia-smi \
                --query-gpu=index,memory.used,utilization.gpu \
                --format=csv,noheader,nounits
        )"
        printf '%s\n' "${snapshot}" |
            tee "${BUNDLE}/nvidia_smi_idle_sample_${sample}.csv"
        while IFS=, read -r idx mem util; do
            idx="$(trim_spaces "${idx}")"
            mem="$(trim_spaces "${mem}")"
            util="$(trim_spaces "${util}")"
            [[ "${idx}" =~ ^[0-9]+$ ]] || continue
            [[ "${mem}" =~ ^[0-9]+$ ]] || continue
            [[ "${util}" =~ ^[0-9]+$ ]] || continue
            gpu_requested "${idx}" || continue
            if ((mem <= GPU_MAX_MEM_MB && util <= GPU_MAX_UTIL)); then
                idle_counts["${idx}"]=$(( ${idle_counts["${idx}"]:-0} + 1 ))
            fi
        done <<< "${snapshot}"
        if ((sample < GPU_IDLE_SAMPLES && GPU_IDLE_INTERVAL > 0)); then
            sleep "${GPU_IDLE_INTERVAL}"
        fi
    done

    local selected=()
    for idx in "${!idle_counts[@]}"; do
        if ((idle_counts["${idx}"] == GPU_IDLE_SAMPLES)); then
            selected+=("${idx}")
        fi
    done
    mapfile -t selected < <(printf '%s\n' "${selected[@]}" | sort -n)
    if ((${#selected[@]} < GPU_COUNT)); then
        die "not enough GPUs stayed idle for ${GPU_IDLE_SAMPLES} samples"
    fi
    (IFS=,; printf '%s' "${selected[*]:0:GPU_COUNT}")
}

BUNDLE="/models/igc/profile_runs/${RUN_ID}"
LOCK="/models/igc/profile_runs/.${PROFILE_MODE}-${HOSTNAME:-slot}.lock"

[[ -d /models ]] || die "/models is not mounted"
[[ -w /models ]] || die "/models is not writable"
mkdir -p /models/igc/profile_runs
[[ ! -e "${BUNDLE}" ]] || die "profile bundle already exists: ${BUNDLE}"
mkdir -p "${BUNDLE}"
STARTED_AT="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
REMOTE_HOST="$(hostname)"
SUMMARY_STATUS="running"
SUMMARY_MESSAGE=""
trap finish_summary EXIT
write_run_env

exec > >(tee -a "${BUNDLE}/wrapper.log") 2>&1

if command -v flock >/dev/null 2>&1; then
    exec 9>"${LOCK}"
    flock -n 9 || die "another profile wrapper holds ${LOCK}"
fi

printf 'run_id=%s\n' "${RUN_ID}"
printf 'profile_mode=%s\n' "${PROFILE_MODE}"
printf 'host=%s\n' "${REMOTE_HOST}"
printf 'started_at=%s\n' "${STARTED_AT}"
df -h /models | tee "${BUNDLE}/models_df.txt"

if [[ "${PROFILE_MODE}" == "cuda" || "${PROFILE_MODE}" == "all" ]]; then
    require_models_space
fi

if [[ -n "${BRAIN_READY_URL}" ]] && command -v curl >/dev/null 2>&1; then
    curl --noproxy '*' -fsS --connect-timeout 3 --max-time 5 \
        "${BRAIN_READY_URL}" > "${BUNDLE}/brain_ready.json" ||
        printf 'Brain readiness snapshot unavailable\n' \
            > "${BUNDLE}/brain_ready.unavailable.txt"
fi
if [[ -n "${FLEET_STATE_URL}" ]] && command -v curl >/dev/null 2>&1; then
    curl --noproxy '*' -fsS --connect-timeout 3 --max-time 5 \
        "${FLEET_STATE_URL}" > "${BUNDLE}/fleet_state.json" ||
        printf 'Fleet state snapshot unavailable\n' \
            > "${BUNDLE}/fleet_state.unavailable.txt"
fi

docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}' |
    tee "${BUNDLE}/docker_ps.txt"

running="$(docker inspect -f '{{.State.Running}}' "${CONTAINER}" 2>/dev/null || true)"
[[ "${running}" == "true" ]] || die "container ${CONTAINER} is not running"

docker inspect \
    -f 'name={{.Name}} image={{.Config.Image}} status={{.State.Status}} pid={{.State.Pid}}' \
    "${CONTAINER}" > "${BUNDLE}/docker_metadata.txt"
docker exec "${CONTAINER}" bash -lc \
    "cd $(q "${IGC_CODE_DIR}") && git rev-parse HEAD && git status --short --branch" \
    > "${BUNDLE}/container_git_state.txt"

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi \
        --query-gpu=index,memory.used,utilization.gpu \
        --format=csv,noheader,nounits |
        tee "${BUNDLE}/nvidia_smi_before.csv"
    nvidia-smi pmon -c 1 > "${BUNDLE}/nvidia_smi_pmon_before.txt" 2>&1 || true
else
    printf 'nvidia-smi unavailable on host\n' |
        tee "${BUNDLE}/nvidia_smi_before.csv"
fi

if [[ "${PROFILE_MODE}" == "snapshot" ]]; then
    printf 'snapshot bundle: %s\n' "${BUNDLE}"
    exit 0
fi

if [[ "${PROFILE_MODE}" == "cpu" || "${PROFILE_MODE}" == "all" ]]; then
    docker exec "${CONTAINER}" bash -lc \
        "cd $(q "${IGC_CODE_DIR}") && make profile" |
        tee "${BUNDLE}/profile_cpu.log"
fi

if [[ "${PROFILE_MODE}" == "cuda" || "${PROFILE_MODE}" == "all" ]]; then
    GPU_IDS="$(select_idle_gpus)"
    printf 'cuda_visible_devices=%s\n' "${GPU_IDS}" |
        tee "${BUNDLE}/gpu_selection.txt"
    full_args="${PROFILE_DATASET_ARGS} --output-dir $(q "${BUNDLE}")"
    docker exec \
        -e CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
        -e PYTHONUNBUFFERED=1 \
        "${CONTAINER}" \
        bash -lc \
        "cd $(q "${IGC_CODE_DIR}") && PROFILE_DATASET_ARGS=${full_args@Q} make profile-dataset-cuda" |
        tee "${BUNDLE}/profile_dataset_cuda.log"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi \
        --query-gpu=index,memory.used,utilization.gpu \
        --format=csv,noheader,nounits |
        tee "${BUNDLE}/nvidia_smi_after.csv"
    nvidia-smi pmon -c 1 > "${BUNDLE}/nvidia_smi_pmon_after.txt" 2>&1 || true
fi
printf 'profile bundle: %s\n' "${BUNDLE}"
REMOTE
