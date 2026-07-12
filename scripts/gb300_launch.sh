#!/usr/bin/env bash
# gb300_launch.sh — launch ONE igc training run on ONE GB300 node via docker.
#
# Runs ON the node (docker runs locally; the cluster's Slurm is only driveable from
# slot0 because munge is down on the compute nodes, so we launch directly and must
# guard the node ourselves). It ALWAYS re-checks that the GPUs are actually free in
# the seconds right before launch — the fleet churns and a node can be grabbed while
# you set up, so an earlier "free" reading is never trusted.
#
# For >1 GPU it runs a single-GPU pre-build FIRST (serialises the dataset build, which
# races across ranks) with no experiment tracking, then the multi-GPU accelerate/FSDP
# run — that second run is the ONE clean W&B run (rank-gated in metric_factory, so one
# run per job not one per GPU). For 1 GPU it is a single plain-python run.
#
# Nothing host- or path-specific is baked in; everything is an env knob so the same
# script runs on any node and against a node-local OR a shared /models checkout.
#
# Usage (on the node):
#   IGC_GPUS=4 IGC_MODEL=gpt2 scripts/gb300_launch.sh
#   IGC_GPUS=1 IGC_MODEL=<hf-repo-id> IGC_USE_PEFT=1 EPOCHS=3 scripts/gb300_launch.sh
#   IGC_CODE_DIR=/models/igc IGC_DATA_DIR=/models/igc_data scripts/gb300_launch.sh
#
# Detached so it survives an ssh drop — redirect from the CALLER; never add
# `exec > >(tee ...)` inside this script (process substitution wedges a setsid launch):
#   nohup setsid bash scripts/gb300_launch.sh > ~/igc-run.log 2>&1 </dev/null & exit 0
#
# Used by: run by hand / by scripts/gb300_stage.sh on a free node; not imported by
# Python. Iterate on code with `git -C "$IGC_CODE_DIR" pull` (the checkout is bind-
# mounted, so no image rebuild) then re-run — no container teardown needed.
#
# Author:
# Mus mbayramo@stanford.edu
set -uo pipefail

# --- knobs (env-overridable; no hardcoded hosts, paths, or endpoints) ---------
IGC_GPUS="${IGC_GPUS:-4}"                                   # 1 = plain python; >1 = accelerate FSDP
IGC_STAGE="${IGC_STAGE:-m1}"                                # curriculum stage (see scripts/train_igc.sbatch)
IGC_MODEL="${IGC_MODEL:-gpt2}"                              # gpt2 smoke, or an HF repo id to scale up
IGC_CODE_DIR="${IGC_CODE_DIR:-$HOME/igc}"                   # igc checkout (node-local or /models)
IGC_DATA_DIR="${IGC_DATA_DIR:-$HOME/.json_responses}"       # captured Redfish responses (mounted read-only)
IGC_IMAGE="${IGC_IMAGE:-nvcr.io/nvidia/pytorch:26.03-py3}"  # bare NGC (runtime pip) or igc-train:ngc26.03 (deps baked)
IGC_RUN="${IGC_RUN:-verify}"                                # experiment name / output subdir / container name
EPOCHS="${EPOCHS:-3}"
IGC_MIN_FREE_GB="${IGC_MIN_FREE_GB:-100}"                   # HF pulls + dataset + checkpoints need headroom
CONTAINER="igc-${IGC_RUN}"

log() { echo "=== [$(date -u '+%F %T')] $* ==="; }
blocker() { echo "BLOCKER: $*" >&2; exit 3; }

# --- 1. pre-flight: the node must be genuinely free RIGHT NOW ------------------
log "pre-flight on $(hostname)"
command -v nvidia-smi >/dev/null || blocker "nvidia-smi missing — not a GPU node?"
command -v docker >/dev/null || blocker "docker missing"

busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)
if [ "$busy" -ne 0 ]; then
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader >&2
    blocker "$busy GPU compute process(es) already on $(hostname) — another team is here, refusing to launch"
fi
ngpu=$(nvidia-smi -L 2>/dev/null | grep -c . || true)
[ "$IGC_GPUS" -le "$ngpu" ] || blocker "asked $IGC_GPUS GPUs, node has $ngpu"

avail=$(df -BG --output=avail "$HOME" 2>/dev/null | tail -1 | tr -dc '0-9' || echo 0)
[ "${avail:-0}" -ge "$IGC_MIN_FREE_GB" ] || blocker "only ${avail}G free under \$HOME (< ${IGC_MIN_FREE_GB}G)"

[ -f "$IGC_CODE_DIR/igc_main.py" ] || blocker "no igc checkout at $IGC_CODE_DIR (clone it or point IGC_CODE_DIR at /models)"
[ -d "$IGC_DATA_DIR" ] || blocker "no captured data at $IGC_DATA_DIR"
docker image inspect "$IGC_IMAGE" >/dev/null 2>&1 || blocker "image $IGC_IMAGE not present — pull it or build igc-train from docker/Dockerfile.train"

# --- 2. run in the container --------------------------------------------------
# Shared igc_main.py flags for both phases (dataset args identical => pre-build's
# cache is reused by the multi-GPU run instead of rebuilt).
COMMON="--train llm --llm latent --model_type ${IGC_MODEL} --json_data_dir /root/.json_responses --tf32 --seed 42 --log_level info --llm_log_level info"
[ "${IGC_USE_PEFT:-0}" = "1" ] && COMMON="${COMMON} --use_peft --lora_r ${LORA_R:-16} --lora_alpha ${LORA_ALPHA:-32}"

docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
log "docker run ${IGC_IMAGE} | ${IGC_GPUS} GPU | code=${IGC_CODE_DIR} data=${IGC_DATA_DIR}"
docker run --rm --gpus all --ipc=host --shm-size=32g --name "$CONTAINER" \
    -v "${IGC_CODE_DIR}:/workspace/igc" \
    -v "${IGC_DATA_DIR}:/root/.json_responses:ro" \
    -w /workspace/igc \
    -e IGC_GPUS -e IGC_STAGE -e IGC_MODEL -e IGC_RUN -e EPOCHS -e COMMON \
    "$IGC_IMAGE" bash -lc '
        set -uo pipefail
        cd /workspace/igc
        # deps are baked into igc-train:ngc26.03; only the bare NGC image needs runtime pip.
        python -c "import transformers, accelerate, wandb" 2>/dev/null \
            || pip install --no-cache-dir -r docker/requirements-train.txt 2>&1 | tail -3
        [ -f .internal/wandb.env ] && { set -a; . .internal/wandb.env; set +a; }
        export WANDB_PROJECT="${WANDB_PROJECT:-igc}"
        nvidia-smi -L

        if [ "$IGC_GPUS" -gt 1 ]; then
            echo "--- PHASE A: single-GPU pre-build (serialise dataset build, no W&B) ---"
            WANDB_MODE=disabled python igc_main.py --device cuda:0 $COMMON \
                --num_train_epochs 1 --output_dir /tmp/prebuild --metric_report tensorboard \
                || { echo "PRE-BUILD FAILED rc=$?"; exit 20; }
        fi

        echo "--- TRAIN: ${IGC_GPUS} GPU -> W&B (one rank-gated run) ---"
        export WANDB_NAME="${IGC_RUN}-${IGC_GPUS}gpu-${IGC_MODEL//\//_}"
        OUT="/workspace/igc/experiments/${IGC_RUN}"; mkdir -p "$OUT"
        if [ "$IGC_GPUS" -gt 1 ]; then
            accelerate launch --num_processes "$IGC_GPUS" --num_machines 1 --machine_rank 0 \
                --main_process_ip 127.0.0.1 --main_process_port 29500 \
                igc_main.py --use_accelerator --sharding fsdp $COMMON \
                --num_train_epochs "$EPOCHS" --output_dir "$OUT" --metric_report wandb
        else
            python igc_main.py --device cuda:0 $COMMON \
                --num_train_epochs "$EPOCHS" --output_dir "$OUT" --metric_report wandb
        fi
    '
rc=$?
log "run exited rc=${rc}"
exit "$rc"

# Author: Mus mbayramo@stanford.edu
