#!/usr/bin/env bash
# Launch one YAML-profiled Phase 1/2/3 SFT run in the prepared GB300 image.
# Run this script on the allocated GB300 host, never on the operator laptop.
set -euo pipefail

PROFILE="${IGC_PROFILE:?set IGC_PROFILE to a name from configs/training/profiles.yaml}"
GPUS="${IGC_GPUS:-1}"
IMAGE="${IGC_IMAGE:-igc-train:ngc26.03-py3}"
CODE_DIR="${IGC_CODE_DIR:-${HOME}/igc}"
MODELS_DIR="${IGC_MODELS_DIR:-/models}"
OUTPUT_DIR="${IGC_OUTPUT_DIR:?set IGC_OUTPUT_DIR to a durable path under IGC_MODELS_DIR}"
RUN_NAME="${IGC_RUN_NAME:-${PROFILE}}"
DRY_RUN="${IGC_DRY_RUN:-0}"

blocker() {
    printf 'BLOCKER: %s\n' "$*" >&2
    exit 3
}

is_positive_int() {
    case "${1:-}" in
        ''|*[!0-9]*) return 1 ;;
        *) [ "$1" -ge 1 ] ;;
    esac
}

case "$DRY_RUN" in
    0|1) ;;
    *) blocker "IGC_DRY_RUN must be 0 or 1" ;;
esac
is_positive_int "$GPUS" || blocker "IGC_GPUS must be an integer >= 1"
case "$OUTPUT_DIR" in
    "${MODELS_DIR}"/*) ;;
    *) blocker "IGC_OUTPUT_DIR must be under IGC_MODELS_DIR (${MODELS_DIR})" ;;
esac

for value in "$PROFILE" "$IMAGE" "$CODE_DIR" "$MODELS_DIR" "$OUTPUT_DIR" "$RUN_NAME"; do
    case "$value" in
        *$'\n'*|*$'\r'*) blocker "launcher values cannot contain newlines" ;;
    esac
done

container_name="igc-${RUN_NAME//[^A-Za-z0-9_.-]/-}"
docker_args=(
    run --rm
    --gpus "$GPUS"
    --ipc=host
    --ulimit memlock=-1
    --ulimit stack=67108864
    --name "$container_name"
    -v "${CODE_DIR}:/workspace/igc"
    -v "${MODELS_DIR}:${MODELS_DIR}"
    -w /workspace/igc
    -e "IGC_PROFILE=${PROFILE}"
    -e "IGC_GPUS=${GPUS}"
    -e "IGC_OUTPUT_DIR=${OUTPUT_DIR}"
)

# Pass only named runtime inputs. Their values are never rendered by this script.
for name in \
    IGC_CORPUS_DIR \
    IGC_CORPUS_EVAL_DIR \
    IGC_SFT_DATA_PATH \
    IGC_SFT_EVAL_DATA_PATH \
    IGC_SFT_DATA_MANIFEST \
    IGC_SFT_EVAL_MANIFEST \
    IGC_METRIC_REPORT \
    IGC_SET \
    IGC_MASTER_PORT \
    IGC_FOUNDATION_MODEL_SHA \
    IGC_TOKENIZER_SHA \
    IGC_MODEL_DIR \
    IGC_MODEL_X_ADAPTER_DIR \
    IGC_MODEL_X_SHA \
    IGC_GOAL_EXTRACTOR_ADAPTER_DIR \
    IGC_GOAL_EXTRACTOR_SHA \
    NCCL_CUMEM_ENABLE \
    NCCL_MNNVL_ENABLE
do
    if [ -n "${!name:-}" ]; then
        docker_args+=(-e "$name")
    fi
done

inner='set -euo pipefail
for env_file in .internal/hf.env .internal/wandb.env; do
    if [ -f "$env_file" ]; then
        set -a
        . "$env_file"
        set +a
    fi
done
exec bash scripts/run_profile.sh'

if [ "$DRY_RUN" = "1" ]; then
    printf 'profile=%s gpus=%s image=%s output=%s\n' \
        "$PROFILE" "$GPUS" "$IMAGE" "$OUTPUT_DIR"
    printf 'docker'
    printf ' %q' "${docker_args[@]}" "$IMAGE" bash -lc "$inner"
    printf '\n'
    exit 0
fi

command -v docker >/dev/null 2>&1 || blocker "docker is unavailable on the GB300 host"
command -v nvidia-smi >/dev/null 2>&1 || blocker "nvidia-smi is unavailable on the GB300 host"
[ -d "$CODE_DIR/.git" ] || blocker "IGC_CODE_DIR is not a git checkout: ${CODE_DIR}"
[ -d "$MODELS_DIR" ] || blocker "IGC_MODELS_DIR does not exist: ${MODELS_DIR}"
docker image inspect "$IMAGE" >/dev/null 2>&1 \
    || blocker "prepared training image is missing: ${IMAGE}"

available_gpus="$(nvidia-smi -L 2>/dev/null | /usr/bin/grep -c '^GPU ' || true)"
[ "$GPUS" -le "${available_gpus:-0}" ] \
    || blocker "requested ${GPUS} GPUs but the host exposes ${available_gpus:-0}"
busy_processes="$(
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
        | /usr/bin/grep -c . || true
)"
[ "$busy_processes" -eq 0 ] || blocker "GPU compute processes are already active"

mkdir -p "$OUTPUT_DIR"
exec docker "${docker_args[@]}" "$IMAGE" bash -lc "$inner"
