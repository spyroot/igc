#!/usr/bin/env bash
# Remote-safe IGC profiler runner for GB300/NV72 slots.
#
# This wrapper is for bugfix and slow-path workers that need repeatable profile
# bundles without using the operator laptop for tests or Docker. It runs through
# SSH on an approved slot, executes inside an existing IGC Docker container, and
# writes evidence/logs under /models/igc/profile_runs/<run_id>.
#
# Author:
# Mus mbayramo@stanford.edu
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST="${HOST:-}"
NV72_USER="${NV72_USER:-nvidia}"
SSH_OPTS="${SSH_OPTS:--o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=accept-new}"
CONTAINER="${CONTAINER:-igc-phase1-pretrain}"
CODE_DIR="${CODE_DIR:-/workspace/igc}"
RUN_ROOT="${RUN_ROOT:-/models/igc/profile_runs}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-${RANDOM:-0}}"
PROFILE_MODE="${PROFILE_MODE:-snapshot}" # snapshot|cpu|cuda|all
ALLOW_GPU_PROFILE="${ALLOW_GPU_PROFILE:-0}"
ALLOW_OTHER_GPU_HOSTS="${ALLOW_OTHER_GPU_HOSTS:-0}"
ALLOW_OVERWRITE="${ALLOW_OVERWRITE:-0}"
GPU_IDS="${GPU_IDS:-auto}"
GPU_COUNT="${GPU_COUNT:-1}"
GPU_MAX_MEM_MB="${GPU_MAX_MEM_MB:-1024}"
GPU_MAX_UTIL="${GPU_MAX_UTIL:-5}"
GPU_IDLE_SAMPLES="${GPU_IDLE_SAMPLES:-3}"
GPU_IDLE_INTERVAL="${GPU_IDLE_INTERVAL:-5}"
MIN_MODELS_FREE_GB="${MIN_MODELS_FREE_GB:-20}"
CPU_SECTION="${CPU_SECTION:-rl}"
CPU_PROFILE_ARGS="${CPU_PROFILE_ARGS:---section ${CPU_SECTION} --profile}"
PROFILE_PYTHON="${PROFILE_PYTHON:-python}"
CUDA_MODEL="${CUDA_MODEL:-gpt2}"
CUDA_BATCH_SIZE="${CUDA_BATCH_SIZE:-4}"
CUDA_SEQ_LEN="${CUDA_SEQ_LEN:-1024}"
CUDA_PRECISION="${CUDA_PRECISION:-bf16}"
CUDA_STEPS="${CUDA_STEPS:-10}"
CUDA_WARMUP="${CUDA_WARMUP:-3}"
CUDA_TRACE="${CUDA_TRACE:-1}"
SYNC_BRANCH="${SYNC_BRANCH:-0}"
IGC_BRANCH="${IGC_BRANCH:-}"
IGC_DRY_RUN="${IGC_DRY_RUN:-0}"
NV72_BRAIN_URL="${NV72_BRAIN_URL:-}"
NV72_FLEET_DASHBOARD_URL="${NV72_FLEET_DASHBOARD_URL:-}"
NV72_HOST_MAP_FILE="${NV72_HOST_MAP_FILE:-${REPO_ROOT}/.internal/nv72_hosts}"
NV72_APPROVED_HOSTS="${NV72_APPROVED_HOSTS:-slot2 slot11}"

usage() {
    cat <<EOF
Usage:
  HOST=slot11 $0
  HOST=slot11 PROFILE_MODE=cpu $0
  HOST=slot11 PROFILE_MODE=cuda ALLOW_GPU_PROFILE=1 GPU_COUNT=2 $0
  HOST=slot11 PROFILE_MODE=all ALLOW_GPU_PROFILE=1 GPU_IDS=1,2 RUN_ID=bugfix-a $0

Modes:
  snapshot  Capture Fleet/SSH/Docker/NVIDIA evidence only. Default; no profiler.
  cpu       Run scripts/bench_hot_paths.py inside CONTAINER. No GPU launch.
  cuda      Run scripts/gpu_train_profile.py inside CONTAINER on selected GPUs.
  all       Run cpu then cuda.

Important env:
  HOST                  slot name/IP, e.g. slot11, n11, 11, or a mapped host.
  CONTAINER             existing IGC container, default igc-phase1-pretrain.
  RUN_ROOT/RUN_ID       output path: RUN_ROOT/RUN_ID.
  ALLOW_GPU_PROFILE=1   required for PROFILE_MODE=cuda|all.
  ALLOW_OTHER_GPU_HOSTS=1
                        allow CUDA profiling outside approved slot2/slot11.
  GPU_IDS               comma-separated physical IDs, or auto.
  GPU_COUNT             number of idle GPUs to auto-select.
  GPU_MAX_MEM_MB        auto-select memory threshold, default 1024.
  GPU_MAX_UTIL          auto-select utilization threshold, default 5.
  GPU_IDLE_SAMPLES      consecutive idle samples required, default 3.
  GPU_IDLE_INTERVAL     seconds between idle samples, default 5.
  MIN_MODELS_FREE_GB    required free space under /models, default 20.
  NV72_BRAIN_URL        Private Brain URL from the NV72 restart runbook.
  NV72_FLEET_DASHBOARD_URL
                        Private Fleet dashboard URL from the NV72 restart runbook.
  NV72_HOST_MAP_FILE    Private '<slot> <host>' map for slot aliases.
  NV72_APPROVED_HOSTS   Space-separated approved slot aliases for CUDA profiling.
  SYNC_BRANCH=1         optional git fetch/checkout/pull IGC_BRANCH in container.
  IGC_DRY_RUN=1         print the remote plan only.
EOF
}

blocker() {
    echo "BLOCKER: $*" >&2
    exit 2
}

log() {
    echo "=== $* ==="
}

resolve_host() {
    local host="${1:-}"
    local slot=""
    local mapped=""

    case "$host" in
        "") return 1 ;;
        slot*) slot="${host#slot}" ;;
        n*) slot="${host#n}" ;;
        [0-9]*) slot="$host" ;;
        *)
            if [[ "$host" =~ ^[[:alnum:].:-]+$ ]]; then
                printf '%s\n' "$host"
                return 0
            fi
            return 1
            ;;
    esac

    case "$slot" in
        ''|*[!0-9]*) return 1 ;;
    esac

    slot=$((10#$slot))
    if [ "$slot" -lt 0 ]; then
        return 1
    fi
    [ -r "$NV72_HOST_MAP_FILE" ] || return 1
    mapped="$(awk -v slot="$slot" \
        '$1 == slot || $1 == ("slot" slot) || $1 == ("n" sprintf("%02d", slot)) {print $2; exit}' \
        "$NV72_HOST_MAP_FILE")"
    [ -n "$mapped" ] || return 1
    printf '%s\n' "$mapped"
}

ssh_dest() {
    printf '%s@%s' "$NV72_USER" "$1"
}

run_ssh() {
    local ip="$1"
    shift
    # shellcheck disable=SC2086 # SSH_OPTS is intentionally a shell-style option string.
    # shellcheck disable=SC2029 # Remote command strings are intentionally rendered locally.
    ssh ${SSH_OPTS} "$(ssh_dest "$ip")" "$@"
}

quote_remote_command() {
    local cmd="$1"
    shift
    local arg
    for arg in "$@"; do
        cmd+=" $(printf '%q' "$arg")"
    done
    printf '%s\n' "$cmd"
}

validate_mode() {
    case "$PROFILE_MODE" in
        snapshot|cpu|cuda|all) ;;
        *) blocker "PROFILE_MODE must be snapshot, cpu, cuda, or all" ;;
    esac

    if { [ "$PROFILE_MODE" = "cuda" ] || [ "$PROFILE_MODE" = "all" ]; } &&
        [ "$ALLOW_GPU_PROFILE" != "1" ]; then
        blocker "PROFILE_MODE=${PROFILE_MODE} requires ALLOW_GPU_PROFILE=1"
    fi

    case "$GPU_COUNT" in
        ''|*[!0-9]*) blocker "GPU_COUNT must be an integer" ;;
    esac

    [ -n "$NV72_BRAIN_URL" ] || blocker "set NV72_BRAIN_URL from the private NV72 runbook"
    [ -n "$NV72_FLEET_DASHBOARD_URL" ] ||
        blocker "set NV72_FLEET_DASHBOARD_URL from the private NV72 runbook"
}

remote_script() {
    cat <<'REMOTE'
set -euo pipefail

CONTAINER="$1"
CODE_DIR="$2"
RUN_ROOT="$3"
RUN_ID="$4"
PROFILE_MODE="$5"
GPU_IDS="$6"
GPU_COUNT="$7"
GPU_MAX_MEM_MB="$8"
GPU_MAX_UTIL="$9"
CPU_PROFILE_ARGS="${10}"
PROFILE_PYTHON="${11}"
CUDA_MODEL="${12}"
CUDA_BATCH_SIZE="${13}"
CUDA_SEQ_LEN="${14}"
CUDA_PRECISION="${15}"
CUDA_STEPS="${16}"
CUDA_WARMUP="${17}"
CUDA_TRACE="${18}"
SYNC_BRANCH="${19}"
IGC_BRANCH="${20}"
NV72_BRAIN_URL="${21}"
NV72_FLEET_DASHBOARD_URL="${22}"
ALLOW_OVERWRITE="${23}"
GPU_IDLE_SAMPLES="${24}"
GPU_IDLE_INTERVAL="${25}"
MIN_MODELS_FREE_GB="${26}"

host_python() {
    if command -v python3 >/dev/null 2>&1; then
        python3 "$@"
        return
    fi
    if command -v python >/dev/null 2>&1; then
        python "$@"
        return
    fi
    echo "BLOCKER: remote host needs python3 or python for profiler bookkeeping" >&2
    exit 2
}

out="${RUN_ROOT}/${RUN_ID}"
mkdir -p "$RUN_ROOT"

lock_root="${RUN_ROOT}/.locks"
mkdir -p "$lock_root"
lock_dir="${lock_root}/$(hostname)-profile.lock"
if ! mkdir "$lock_dir" 2>/dev/null; then
    echo "BLOCKER: profiler lock exists for $(hostname): ${lock_dir}" >&2
    exit 2
fi
cleanup_lock() {
    rmdir "$lock_dir" 2>/dev/null || true
}
trap cleanup_lock EXIT

if [ -e "$out" ] && [ "$ALLOW_OVERWRITE" != "1" ]; then
    echo "BLOCKER: output directory already exists: ${out}; set ALLOW_OVERWRITE=1 only if intentional" >&2
    exit 2
fi

free_mb="$(df -Pm "$RUN_ROOT" | awk 'NR==2 {print $4}')"
required_mb=$((MIN_MODELS_FREE_GB * 1024))
if [ "${free_mb:-0}" -lt "$required_mb" ]; then
    echo "BLOCKER: ${RUN_ROOT} has ${free_mb:-0} MiB free, need at least ${required_mb} MiB" >&2
    exit 2
fi

mkdir -p "$out"
chmod 770 "$out" 2>/dev/null || true

record() {
    local name="$1"
    shift
    {
        printf '### %s\n' "$name"
        printf '### command:'
        printf ' %q' "$@"
        printf '\n'
        "$@"
    } >"${out}/${name}.log" 2>&1 || {
        rc=$?
        printf 'rc=%s\n' "$rc" >>"${out}/${name}.log"
        return "$rc"
    }
}

write_json_summary() {
    local selected_gpus="$1"
    local status="$2"
    host_python - "$out/summary.json" "$RUN_ID" "$PROFILE_MODE" "$selected_gpus" "$status" <<'PY'
import json
import os
import sys

path, run_id, mode, gpu_ids, status = sys.argv[1:6]
payload = {
    "run_id": run_id,
    "profile_mode": mode,
    "selected_gpu_ids": gpu_ids,
    "status": status,
    "output_dir": os.path.dirname(path),
}
with open(path, "w", encoding="utf-8") as fh:
    json.dump(payload, fh, indent=2, sort_keys=True)
    fh.write("\n")
PY
}

select_gpus() {
    if [ "$GPU_IDS" != "auto" ]; then
        printf '%s\n' "$GPU_IDS"
        return 0
    fi

    samples="${out}/gpu_idle_samples.tsv"
    : >"$samples"
    i=1
    while [ "$i" -le "$GPU_IDLE_SAMPLES" ]; do
        nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
            --format=csv,noheader,nounits |
            awk -F, -v sample="$i" '
                {
                    gsub(/ /, "", $1)
                    gsub(/ /, "", $2)
                    gsub(/ /, "", $3)
                    printf "%s\t%s\t%s\t%s\n", sample, $1, $2, $3
                }
            ' >>"$samples"
        if [ "$i" -lt "$GPU_IDLE_SAMPLES" ]; then
            sleep "$GPU_IDLE_INTERVAL"
        fi
        i=$((i + 1))
    done

    host_python - "$samples" "$GPU_COUNT" "$GPU_IDLE_SAMPLES" "$GPU_MAX_MEM_MB" "$GPU_MAX_UTIL" <<'PY'
import collections
import sys

path, need_s, samples_s, max_mem_s, max_util_s = sys.argv[1:6]
need = int(need_s)
samples = int(samples_s)
max_mem = int(max_mem_s)
max_util = int(max_util_s)
counts = collections.Counter()
order = []

with open(path, encoding="utf-8") as fh:
    for line in fh:
        _sample, gpu, mem, util = line.rstrip("\n").split("\t")
        if gpu not in order:
            order.append(gpu)
        if int(mem) <= max_mem and int(util) <= max_util:
            counts[gpu] += 1

selected = [gpu for gpu in order if counts[gpu] >= samples]
if len(selected) < need:
    sys.exit(2)
print(",".join(selected[:need]))
PY
}

{
    printf 'run_id=%s\n' "$RUN_ID"
    printf 'host=%s\n' "$(hostname)"
    printf 'started_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'profile_mode=%s\n' "$PROFILE_MODE"
    printf 'container=%s\n' "$CONTAINER"
    printf 'code_dir=%s\n' "$CODE_DIR"
} >"${out}/run.env"

curl -fsS --connect-timeout 2 --max-time 5 "${NV72_BRAIN_URL}/ready" \
    >"${out}/brain_ready.json" 2>"${out}/brain_ready.err" || true
curl -fsS --connect-timeout 2 --max-time 5 "${NV72_FLEET_DASHBOARD_URL}/api/v1/state" \
    >"${out}/fleet_state.json" 2>"${out}/fleet_state.err" || true

record hostname hostname || true
record df_models df -h /models || true
record nvidia_smi_query nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
    --format=csv,noheader,nounits || true
record nvidia_smi_pmon nvidia-smi pmon -c 1 || true
record docker_ps docker ps --format '{{.Names}} {{.Status}} {{.Image}}' || true

docker inspect "$CONTAINER" >"${out}/docker_inspect_raw.json"
host_python - "${out}/docker_inspect_raw.json" "${out}/docker_inspect_sanitized.json" <<'PY'
import json
import sys

src, dst = sys.argv[1:3]
with open(src, encoding="utf-8") as fh:
    payload = json.load(fh)
safe = []
for item in payload:
    safe.append({
        "Id": item.get("Id", "")[:12],
        "Name": item.get("Name"),
        "Image": item.get("Image", "")[:12],
        "State": {
            "Status": item.get("State", {}).get("Status"),
            "Running": item.get("State", {}).get("Running"),
            "StartedAt": item.get("State", {}).get("StartedAt"),
        },
        "Mounts": [
            {
                "Type": m.get("Type"),
                "Destination": m.get("Destination"),
                "RW": m.get("RW"),
            }
            for m in item.get("Mounts", [])
        ],
        "DeviceRequests": item.get("HostConfig", {}).get("DeviceRequests"),
    })
with open(dst, "w", encoding="utf-8") as fh:
    json.dump(safe, fh, indent=2, sort_keys=True)
    fh.write("\n")
PY
rm -f "${out}/docker_inspect_raw.json"

docker exec "$CONTAINER" bash -lc "
    set -euo pipefail
    cd '$CODE_DIR'
    git config --global --add safe.directory '$CODE_DIR' 2>/dev/null || true
    git status --short --branch
    git rev-parse HEAD
    command -v python || true
    command -v accelerate || true
    command -v nsys || true
    command -v ncu || true
    python - <<'PY'
import json
import torch
print(json.dumps({
    'torch': getattr(torch, '__version__', None),
    'cuda_available': torch.cuda.is_available(),
    'cuda_device_count': torch.cuda.device_count(),
    'cuda_version': torch.version.cuda,
}, sort_keys=True))
PY
" >"${out}/container_preflight.log" 2>&1

if [ "$SYNC_BRANCH" = "1" ]; then
    [ -n "$IGC_BRANCH" ] || {
        echo "SYNC_BRANCH=1 requires IGC_BRANCH" >"${out}/sync_branch.err"
        exit 2
    }
    docker exec "$CONTAINER" bash -lc "
        set -euo pipefail
        cd '$CODE_DIR'
        git fetch origin '$IGC_BRANCH'
        if git show-ref --verify --quiet 'refs/heads/$IGC_BRANCH'; then
            git checkout '$IGC_BRANCH'
        else
            git checkout -B '$IGC_BRANCH' 'origin/$IGC_BRANCH'
        fi
        git pull --ff-only origin '$IGC_BRANCH'
        git rev-parse HEAD
    " >"${out}/sync_branch.log" 2>&1
fi

selected_gpus=""
if [ "$PROFILE_MODE" = "cuda" ] || [ "$PROFILE_MODE" = "all" ]; then
    if ! selected_gpus="$(select_gpus)"; then
        {
            echo "BLOCKER: not enough idle GPUs for GPU_COUNT=${GPU_COUNT}"
            echo "thresholds: memory<=${GPU_MAX_MEM_MB}MB util<=${GPU_MAX_UTIL}%"
            cat "${out}/nvidia_smi_query.log" 2>/dev/null || true
        } >"${out}/gpu_selection.err"
        write_json_summary "" "blocked_gpu_selection"
        exit 2
    fi
    printf '%s\n' "$selected_gpus" >"${out}/selected_gpus.txt"
fi

if [ "$PROFILE_MODE" = "cpu" ] || [ "$PROFILE_MODE" = "all" ]; then
    docker exec "$CONTAINER" bash -lc "
        set -euo pipefail
        cd '$CODE_DIR'
        export KMP_DUPLICATE_LIB_OK=TRUE
        export OMP_NUM_THREADS=\${OMP_NUM_THREADS:-1}
        export TRANSFORMERS_OFFLINE=1
        export HF_DATASETS_OFFLINE=1
        ${PROFILE_PYTHON} scripts/bench_hot_paths.py ${CPU_PROFILE_ARGS}
    " >"${out}/cpu_hot_paths.log" 2>&1
fi

if [ "$PROFILE_MODE" = "cuda" ] || [ "$PROFILE_MODE" = "all" ]; then
    cuda_out="${out}/cuda_train_profile"
    mkdir -p "$cuda_out"
    trace_arg=""
    if [ "$CUDA_TRACE" = "1" ]; then
        trace_arg="--trace"
    fi
    if [ "$GPU_COUNT" = "1" ]; then
        docker exec \
            -e CUDA_VISIBLE_DEVICES="$selected_gpus" \
            -e TRANSFORMERS_OFFLINE=1 \
            -e HF_DATASETS_OFFLINE=1 \
            "$CONTAINER" \
            bash -lc "
                set -euo pipefail
                cd '$CODE_DIR'
                ${PROFILE_PYTHON} scripts/gpu_train_profile.py \
                    --model_type '$CUDA_MODEL' \
                    --batch_size '$CUDA_BATCH_SIZE' \
                    --seq_len '$CUDA_SEQ_LEN' \
                    --precision '$CUDA_PRECISION' \
                    --steps '$CUDA_STEPS' \
                    --warmup '$CUDA_WARMUP' \
                    ${trace_arg} \
                    --output_dir '$cuda_out'
            " >"${out}/cuda_train_profile.log" 2>&1
    else
        docker exec \
            -e CUDA_VISIBLE_DEVICES="$selected_gpus" \
            -e TRANSFORMERS_OFFLINE=1 \
            -e HF_DATASETS_OFFLINE=1 \
            "$CONTAINER" \
            bash -lc "
                set -euo pipefail
                cd '$CODE_DIR'
                accelerate launch --multi_gpu \
                    --num_processes '$GPU_COUNT' \
                    --num_machines 1 \
                    --mixed_precision '$CUDA_PRECISION' \
                    scripts/gpu_train_profile.py \
                    --model_type '$CUDA_MODEL' \
                    --batch_size '$CUDA_BATCH_SIZE' \
                    --seq_len '$CUDA_SEQ_LEN' \
                    --precision '$CUDA_PRECISION' \
                    --steps '$CUDA_STEPS' \
                    --warmup '$CUDA_WARMUP' \
                    ${trace_arg} \
                    --output_dir '$cuda_out'
            " >"${out}/cuda_train_profile.log" 2>&1
    fi
fi

write_json_summary "$selected_gpus" "ok"
printf 'profile_out=%s\n' "$out"
REMOTE
}

is_approved_gpu_host() {
    local candidate
    for candidate in $NV72_APPROVED_HOSTS; do
        if [ "$(resolve_host "$candidate" 2>/dev/null || true)" = "$1" ]; then
            return 0
        fi
    done
    return 1
}

main() {
    if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
        usage
        return 0
    fi

    [ -n "$HOST" ] || blocker "set HOST=slot11|slot2|IP"
    validate_mode

    local ip
    ip="$(resolve_host "$HOST")" || blocker "invalid HOST: ${HOST}"

    if { [ "$PROFILE_MODE" = "cuda" ] || [ "$PROFILE_MODE" = "all" ]; } &&
        [ "$ALLOW_OTHER_GPU_HOSTS" != "1" ]; then
        is_approved_gpu_host "$ip" ||
            blocker "CUDA profiling is restricted to NV72_APPROVED_HOSTS; set ALLOW_OTHER_GPU_HOSTS=1 only with explicit operator approval"
    fi

    if [ "$IGC_DRY_RUN" = "1" ]; then
        echo "DRY-RUN profile $(ssh_dest "$ip")"
    echo "  mode=${PROFILE_MODE} container=${CONTAINER} run=${RUN_ROOT}/${RUN_ID}"
    echo "  cpu command: ${PROFILE_PYTHON} scripts/bench_hot_paths.py ${CPU_PROFILE_ARGS}"
    echo "  cuda allowed=${ALLOW_GPU_PROFILE} gpu_ids=${GPU_IDS} gpu_count=${GPU_COUNT}"
    echo "  idle gate: samples=${GPU_IDLE_SAMPLES} interval=${GPU_IDLE_INTERVAL}s mem<=${GPU_MAX_MEM_MB}MB util<=${GPU_MAX_UTIL}%"
    echo "  free-space gate: ${MIN_MODELS_FREE_GB} GiB under ${RUN_ROOT}"
    echo "  remote command writes evidence/logs under ${RUN_ROOT}/${RUN_ID}"
    return 0
    fi

    log "remote profile ${PROFILE_MODE} on $(ssh_dest "$ip") container=${CONTAINER}"
    local remote_cmd
    remote_cmd="$(quote_remote_command "bash -s --" \
        "$CONTAINER" "$CODE_DIR" "$RUN_ROOT" "$RUN_ID" "$PROFILE_MODE" "$GPU_IDS" "$GPU_COUNT" \
        "$GPU_MAX_MEM_MB" "$GPU_MAX_UTIL" "$CPU_PROFILE_ARGS" "$PROFILE_PYTHON" "$CUDA_MODEL" \
        "$CUDA_BATCH_SIZE" "$CUDA_SEQ_LEN" "$CUDA_PRECISION" "$CUDA_STEPS" "$CUDA_WARMUP" \
        "$CUDA_TRACE" "$SYNC_BRANCH" "$IGC_BRANCH" "$NV72_BRAIN_URL" "$NV72_FLEET_DASHBOARD_URL" \
        "$ALLOW_OVERWRITE" "$GPU_IDLE_SAMPLES" "$GPU_IDLE_INTERVAL" "$MIN_MODELS_FREE_GB")"
    run_ssh "$ip" "$remote_cmd" \
        <<<"$(remote_script)"
}

main "$@"


# Author: Mus mbayramo@stanford.edu
