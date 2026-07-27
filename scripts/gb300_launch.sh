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
DATASET_COMPAT_GATE="scripts/gates/dataset_runtime_compat.py"

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
    IGC_CORPUS_SUMMARY \
    IGC_SOURCE_REGISTRY \
    IGC_REDFISH_CORPUS_MANIFEST \
    IGC_DSP2043_CORPUS_MANIFEST \
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

# The variables in this script are intentionally expanded inside the container.
# shellcheck disable=SC2016
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

# Compare the image label to the exact checked-out data contract without
# exposing the host Docker socket to the container.
if ! docker image inspect "$IMAGE" \
    | docker run --rm -i \
        -v "${CODE_DIR}:/workspace/igc:ro" \
        -w /workspace/igc \
        "$IMAGE" \
        python "$DATASET_COMPAT_GATE" \
        check-image --image "$IMAGE" --labels-json -
then
    blocker "training image is update-required for the checked-out dataset contract"
fi

if [[ "$PROFILE" == phase1_* ]]; then
    CORPUS_DIR="${IGC_CORPUS_DIR:?set IGC_CORPUS_DIR for Phase 1}"
    CORPUS_SUMMARY="${IGC_CORPUS_SUMMARY:?set IGC_CORPUS_SUMMARY for Phase 1}"
    SOURCE_REGISTRY="${IGC_SOURCE_REGISTRY:-${CODE_DIR}/configs/data/redfish_sources.yaml}"
    REDFISH_MANIFEST="${IGC_REDFISH_CORPUS_MANIFEST:?set IGC_REDFISH_CORPUS_MANIFEST for Phase 1}"
    DSP2043_MANIFEST="${IGC_DSP2043_CORPUS_MANIFEST:?set IGC_DSP2043_CORPUS_MANIFEST for Phase 1}"

    for path in \
        "$CORPUS_DIR" \
        "$CORPUS_SUMMARY" \
        "$REDFISH_MANIFEST" \
        "$DSP2043_MANIFEST"
    do
        case "$path" in
            "${MODELS_DIR}"/*) ;;
            *) blocker "Phase 1 data and manifests must be under IGC_MODELS_DIR: $path" ;;
        esac
    done
    case "$SOURCE_REGISTRY" in
        "${CODE_DIR}"/*|"${MODELS_DIR}"/*) ;;
        *)
            blocker \
                "Phase 1 source registry is outside code/models mounts: $SOURCE_REGISTRY"
            ;;
    esac
    [ -d "$CORPUS_DIR" ] || blocker "Phase 1 corpus release is missing: $CORPUS_DIR"
    for path in \
        "$CORPUS_SUMMARY" \
        "$SOURCE_REGISTRY" \
        "$REDFISH_MANIFEST" \
        "$DSP2043_MANIFEST"
    do
        [ -f "$path" ] || blocker "Phase 1 compatibility input is missing: $path"
    done

    container_source_registry="$SOURCE_REGISTRY"
    case "$SOURCE_REGISTRY" in
        "${CODE_DIR}"/*)
            container_source_registry="/workspace/igc/${SOURCE_REGISTRY#"${CODE_DIR}"/}"
            ;;
    esac
    if ! docker run --rm \
        -v "${CODE_DIR}:/workspace/igc:ro" \
        -v "${MODELS_DIR}:${MODELS_DIR}:ro" \
        -w /workspace/igc \
        -e "IGC_FOUNDATION_MODEL_SHA=${IGC_FOUNDATION_MODEL_SHA:-}" \
        -e "IGC_TOKENIZER_SHA=${IGC_TOKENIZER_SHA:-}" \
        "$IMAGE" \
        python "$DATASET_COMPAT_GATE" \
        check-release \
        --release-root "$CORPUS_DIR" \
        --summary "$CORPUS_SUMMARY" \
        --source-registry "$container_source_registry" \
        --training-profile "$PROFILE" \
        --source-manifest "redfish_ctl_full_corpus=$REDFISH_MANIFEST" \
        --source-manifest "dsp2043_redfish_ctl_corpus=$DSP2043_MANIFEST"
    then
        blocker "Phase 1 corpus release is incompatible with its sources or profile"
    fi
fi

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
