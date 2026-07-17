#!/usr/bin/env bash
# Run bounded IGC profiling inside an already-running GB300/NV72 slot container.
#
# This wrapper is intentionally a coordinator: it verifies the remote slot,
# records Docker/NVIDIA evidence, creates a /models profile bundle, and then
# delegates the actual dataset-to-CUDA timing to scripts/profile_dataset_to_cuda.py.
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
    [[ "${ALLOW_GPU_PROFILE:-}" == "1" ]] || die "set ALLOW_GPU_PROFILE=1 for CUDA profiling"
    case "${HOST}" in
        slot2|slot11) ;;
        *)
            [[ "${IGC_ALLOW_PROFILE_HOST_OVERRIDE:-}" == "1" ]] ||
                die "CUDA profiling is restricted to HOST=slot2 or HOST=slot11"
            ;;
    esac
    [[ -n "${PROFILE_DATASET_ARGS:-}" ]] || die "set PROFILE_DATASET_ARGS for CUDA profiling"
    [[ " ${PROFILE_DATASET_ARGS} " != *" --output-dir "* ]] ||
        die "do not pass --output-dir in PROFILE_DATASET_ARGS; wrapper owns the bundle path"
fi

SSH_BIN="${SSH_BIN:-ssh}"
CONTAINER="${CONTAINER:-igc}"
IGC_CODE_DIR="${IGC_CODE_DIR:-/workspace/igc}"
GPU_COUNT="${GPU_COUNT:-1}"
GPU_IDS="${GPU_IDS:-}"
GPU_MAX_MEM_MB="${GPU_MAX_MEM_MB:-256}"
GPU_MAX_UTIL="${GPU_MAX_UTIL:-5}"
RUN_ID="${RUN_ID:-igc-${PROFILE_MODE}-$(date -u +%Y%m%dT%H%M%SZ)}"
REMOTE_TIMEOUT="${REMOTE_TIMEOUT:-10}"

[[ "${RUN_ID}" =~ ^[A-Za-z0-9._+-]+$ ]] ||
    die "RUN_ID may contain only letters, digits, dot, underscore, plus, and dash"
[[ "${GPU_COUNT}" =~ ^[0-9]+$ ]] && [[ "${GPU_COUNT}" -ge 1 ]] ||
    die "GPU_COUNT must be a positive integer"

remote_env=(
    "PROFILE_MODE=$(quote "${PROFILE_MODE}")"
    "CONTAINER=$(quote "${CONTAINER}")"
    "IGC_CODE_DIR=$(quote "${IGC_CODE_DIR}")"
    "GPU_COUNT=$(quote "${GPU_COUNT}")"
    "GPU_IDS=$(quote "${GPU_IDS}")"
    "GPU_MAX_MEM_MB=$(quote "${GPU_MAX_MEM_MB}")"
    "GPU_MAX_UTIL=$(quote "${GPU_MAX_UTIL}")"
    "RUN_ID=$(quote "${RUN_ID}")"
    "PROFILE_DATASET_ARGS=$(quote "${PROFILE_DATASET_ARGS:-}")"
)

remote_prefix="${remote_env[*]}"

"${SSH_BIN}" \
    -o BatchMode=yes \
    -o ConnectTimeout="${REMOTE_TIMEOUT}" \
    "${HOST}" \
    "${remote_prefix} bash -s" <<'REMOTE'
set -euo pipefail

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 2
}

q() {
    printf '%q' "$1"
}

BUNDLE="/models/igc/profile_runs/${RUN_ID}"
LOCK="/models/igc/profile_runs/.${PROFILE_MODE}-${HOSTNAME:-slot}.lock"

[[ -d /models ]] || die "/models is not mounted"
[[ -w /models ]] || die "/models is not writable"
mkdir -p /models/igc/profile_runs
[[ ! -e "${BUNDLE}" ]] || die "profile bundle already exists: ${BUNDLE}"
mkdir -p "${BUNDLE}"

exec > >(tee -a "${BUNDLE}/wrapper.log") 2>&1

if command -v flock >/dev/null 2>&1; then
    exec 9>"${LOCK}"
    flock -n 9 || die "another profile wrapper holds ${LOCK}"
fi

printf 'run_id=%s\n' "${RUN_ID}"
printf 'profile_mode=%s\n' "${PROFILE_MODE}"
printf 'host=%s\n' "$(hostname)"
date -u '+started_at=%Y-%m-%dT%H:%M:%SZ'
df -h /models | tee "${BUNDLE}/models_df.txt"

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
else
    printf 'nvidia-smi unavailable on host\n' | tee "${BUNDLE}/nvidia_smi_before.csv"
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
    if [[ -z "${GPU_IDS}" ]]; then
        command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi required to select GPUs"
        GPU_IDS="$(
            nvidia-smi \
                --query-gpu=index,memory.used,utilization.gpu \
                --format=csv,noheader,nounits |
            awk -F, \
                -v count="${GPU_COUNT}" \
                -v max_mem="${GPU_MAX_MEM_MB}" \
                -v max_util="${GPU_MAX_UTIL}" '
                    function trim(v) {
                        gsub(/^[[:space:]]+|[[:space:]]+$/, "", v)
                        return v
                    }
                    {
                        idx = trim($1)
                        mem = trim($2) + 0
                        util = trim($3) + 0
                        if (mem <= max_mem && util <= max_util) {
                            selected[n++] = idx
                        }
                    }
                    END {
                        if (n < count) {
                            exit 3
                        }
                        for (i = 0; i < count; i++) {
                            printf "%s%s", (i ? "," : ""), selected[i]
                        }
                    }'
        )" || die "not enough idle GPUs for GPU_COUNT=${GPU_COUNT}"
    fi
    printf 'cuda_visible_devices=%s\n' "${GPU_IDS}" | tee "${BUNDLE}/gpu_selection.txt"
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
fi
printf 'profile bundle: %s\n' "${BUNDLE}"
REMOTE
