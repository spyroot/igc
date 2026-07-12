#!/bin/bash
# Unified GB300/Slurm launcher for the igc training curriculum. ONE Slurm job per
# stage, selected by IGC_STAGE; runs igc_main.py inside the NGC PyTorch container on
# one GB300. Runbook + cautions: GPU_ACCESS.md. Full guide: docs/TRAINING.md.
#
# Stages (curriculum order — each later stage loads the earlier artifacts):
#   m1 | state_encoder   --train llm  --llm latent     state encoder (backbone SFT; --use_peft for LoRA)
#   m2 | autoencoder     --train llm  --llm encoder     state autoencoder (compresses the state)
#   m3 | goal            --train llm  --llm goal         goal extractor (planner)
#   m3p| parameter       --train llm  --llm parameter    goal+parameter extractor (arg head)
#   m6 | agent           --train agent --rl all          RL agent (DQN+HER) — needs m1+m2 first
#   all                  --train all  --llm all --rl all  everything in one job
#
#   export WANDB_API_KEY=...            # in your shell, NOT here (rotate if ever shared)
#   IGC_STAGE=m1 sbatch scripts/train_igc.sbatch                 # gpt2 smoke -> wandb
#   IGC_STAGE=m1 IGC_MODEL=<hf-repo-id> IGC_USE_PEFT=1 EPOCHS=3 sbatch scripts/train_igc.sbatch
#   IGC_STAGE=m6 EPOCHS=50 sbatch scripts/train_igc.sbatch       # RL agent (after m1+m2)
#
#SBATCH --job-name=igc-train
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
# NOTE: submit_train.sh overrides --gres to gpu:${IGC_GPUS} at submit time.
#SBATCH --exclude=gb300-poc1-slot2,gb300-poc1-slot15,gb300-poc1-slot16
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out
#SBATCH --requeue
#SBATCH --signal=USR1@90

set -euo pipefail

# Disk preflight: a run that starts on a nearly-full node-local NVMe dies mid-training
# (HF pulls + dataset + checkpoints). Fail fast, with an explicit override knob.
IGC_MIN_FREE_GB="${IGC_MIN_FREE_GB:-100}"
AVAIL_GB="$(df -BG --output=avail "${HOME}" 2>/dev/null | tail -1 | tr -dc '0-9' || echo 0)"
if [[ "${AVAIL_GB}" -lt "${IGC_MIN_FREE_GB}" ]]; then
    echo "BLOCKER: only ${AVAIL_GB}G free under \$HOME (< IGC_MIN_FREE_GB=${IGC_MIN_FREE_GB}G) — clean up or lower the floor deliberately." >&2
    exit 1
fi

# Fabric update (operator, 2026-07-10): NVLink/RoCE healthy — NVLS on by default;
# override with IGC_NCCL_NVLS=0 if nccl_smoke (the R4 gate) says otherwise.
export NCCL_NVLS_ENABLE="${IGC_NCCL_NVLS:-1}"
export NCCL_MNNVL_ENABLE="${NCCL_MNNVL_ENABLE:-1}"
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-1}"
export HF_HOME="${HF_HOME:-/models/hf-cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export PYTHONUNBUFFERED=1

# Overridable knobs (env). Stage code + captured data on the node's local NVMe.
IGC_STAGE="${IGC_STAGE:-m1}"                     # m1|m2|m3|m3p|m6|all (see header)
IGC_GPUS="${IGC_GPUS:-1}"                        # per-node GPUs; >1 or multi-node = accelerate+FSDP2
# multi-node rendezvous: first job node is the accelerate main process host.
MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST:-$(hostname)}" 2>/dev/null | head -n1 || hostname)"
MASTER_PORT="${MASTER_PORT:-29500}"
export MASTER_ADDR MASTER_PORT
IGC_SHARDING="${IGC_SHARDING:-fsdp}"             # sharding mode for multi-GPU (fsdp = torch-native)
IGC_MODEL="${IGC_MODEL:-gpt2}"                   # small for the smoke; large HF decoder to scale
IGC_DIR="${IGC_DIR:-$HOME/igc}"                  # node-local igc checkout
DATA_DIR="${DATA_DIR:-$HOME/.json_responses}"    # node-local captured Redfish responses
EPOCHS="${EPOCHS:-3}"
NGC_IMAGE="${NGC_IMAGE:-nvcr.io/nvidia/pytorch:26.03-py3}"
SEED="${SEED:-42}"
# SHARED curriculum artifact root: every stage of one run reads/writes experiments/${IGC_RUN}
# (per-module subdirs: state_encoder/, state_autoencoder/, ...), so m6 can find m1's checkpoint.
# Use the SAME IGC_RUN across m1->m2->m3->m6; do NOT make it per-job.
IGC_RUN="${IGC_RUN:-default}"

# Map the stage to igc_main.py flags. The agent stage counts epochs with --rl_num_train_epochs
# (the RL knob); the language stages use --num_train_epochs.
case "${IGC_STAGE}" in
    m1|state_encoder) TRAIN_ARGS="--train llm --llm latent"      ; EPOCH_FLAG="--num_train_epochs"    ;;
    m2|autoencoder)   TRAIN_ARGS="--train llm --llm encoder"     ; EPOCH_FLAG="--num_train_epochs"    ;;
    m3|goal)          TRAIN_ARGS="--train llm --llm goal"        ; EPOCH_FLAG="--num_train_epochs"    ;;
    m3p|parameter)    TRAIN_ARGS="--train llm --llm parameter"   ; EPOCH_FLAG="--num_train_epochs"    ;;
    m6|agent|rl)      TRAIN_ARGS="--train agent --rl all"        ; EPOCH_FLAG="--rl_num_train_epochs" ;;
    all)              TRAIN_ARGS="--train all --llm all --rl all"; EPOCH_FLAG="--num_train_epochs"    ;;
    *) echo "ERROR: unknown IGC_STAGE='${IGC_STAGE}' (use m1|m2|m3|m3p|m6|all)" >&2; exit 2 ;;
esac

# The combined `all` stage also runs the RL agent, whose epochs come from --rl_num_train_epochs
# (default 1000) — --num_train_epochs does NOT cap it. Bound it explicitly for `all`.
RL_EPOCH_ARG=""
[ "${IGC_STAGE}" = "all" ] && RL_EPOCH_ARG="--rl_num_train_epochs ${RL_EPOCHS:-${EPOCHS}}"

# LoRA/PEFT for a large backbone (the gpt2 smoke uses full FT). Set IGC_USE_PEFT=1 to enable.
EXTRA_ARGS=""
[ "${IGC_USE_PEFT:-0}" = "1" ] && EXTRA_ARGS="--use_peft --lora_r ${LORA_R:-16} --lora_alpha ${LORA_ALPHA:-32}"

# Experiment tracking. Default to wandb; the API key MUST come from your environment
# (export WANDB_API_KEY=...). Never hardcode or commit it. WANDB_PROJECT is safe to set.
IGC_REPORT="${IGC_REPORT:-wandb}"
RUN_NAME="${WANDB_NAME:-${IGC_STAGE}-${IGC_MODEL//\//_}-${SLURM_JOB_ID:-local}}"
if [ "${IGC_REPORT}" = "wandb" ]; then
    # shellcheck source=/dev/null
    [ -f "${IGC_DIR}/.internal/hf.env" ] && . "${IGC_DIR}/.internal/hf.env"
    # convenience: source a node-local gitignored key file if present (.internal/ is never committed)
    # shellcheck source=/dev/null
    [ -f "${IGC_DIR}/.internal/wandb.env" ] && . "${IGC_DIR}/.internal/wandb.env"
    : "${WANDB_API_KEY:?set WANDB_API_KEY in your env or ${IGC_DIR}/.internal/wandb.env to use wandb}"
    export WANDB_API_KEY
    export WANDB_PROJECT="${WANDB_PROJECT:-igc}"
    export WANDB_NAME="${RUN_NAME}"
fi

echo "igc train | stage=${IGC_STAGE} args='${TRAIN_ARGS}' model=${IGC_MODEL} epochs=${EPOCHS} report=${IGC_REPORT} node=$(hostname)"

# Export the few values the inner heredoc needs (avoids brittle nested quoting).
export IGC_MODEL EPOCHS SEED IGC_REPORT TRAIN_ARGS EPOCH_FLAG EXTRA_ARGS RUN_NAME IGC_RUN RL_EPOCH_ARG IGC_GPUS IGC_SHARDING MASTER_ADDR MASTER_PORT

# shellcheck disable=SC2016  # The single-quoted payload is evaluated inside the container.
srun --container-image="${NGC_IMAGE}" \
     --container-mounts="${IGC_DIR}:/workspace/igc,${DATA_DIR}:/root/.json_responses" \
     bash -lc '
        set -euo pipefail
        cd /workspace/igc
        [ -f .internal/hf.env ] && { set -a; . .internal/hf.env; set +a; }
        export HF_HOME="${HF_HOME:-/models/hf-cache}"
        export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
        export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
        export NCCL_MNNVL_ENABLE="${NCCL_MNNVL_ENABLE:-1}"
        export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-1}"
        # auto-load W&B creds (gitignored env file) so the run streams live
        [ -f .internal/wandb.env ] && { set -a; . .internal/wandb.env; set +a; }
        pip install --no-cache-dir -r docker/requirements-train.txt
        nvidia-smi -L
        # Shared curriculum root so later stages find earlier checkpoints (per-module subdirs).
        OUT="/workspace/igc/experiments/${IGC_RUN}"
        mkdir -p "${OUT}"
        # 1 GPU: plain python. >1 GPU: accelerate launch with torch-native FSDP2
        # (per-rank device resolution + collective checkpoint gather are wired in igc);
        # a sharded config in a single-process launch fails loudly by design.
        NNODES="${SLURM_NNODES:-1}"
        NODE_RANK="${SLURM_NODEID:-0}"
        TOTAL_PROCS=$(( IGC_GPUS * NNODES ))
        if [[ "${TOTAL_PROCS}" -gt 1 ]]; then
            LAUNCH=(accelerate launch
                    --num_processes "${TOTAL_PROCS}"
                    --num_machines "${NNODES}"
                    --machine_rank "${NODE_RANK}"
                    --main_process_ip "${MASTER_ADDR}"
                    --main_process_port "${MASTER_PORT}"
                    igc_main.py --use_accelerator --sharding "${IGC_SHARDING}")
        else
            LAUNCH=(python igc_main.py --device cuda:0)
        fi
        # shellcheck disable=SC2086
        "${LAUNCH[@]}" \
            ${TRAIN_ARGS} \
            --model_type "${IGC_MODEL}" \
            ${EPOCH_FLAG} "${EPOCHS}" ${RL_EPOCH_ARG} \
            --json_data_dir /root/.json_responses \
            --output_dir "${OUT}" \
            --tf32 --seed "${SEED}" \
            --metric_report "${IGC_REPORT}" \
            ${EXTRA_ARGS} \
            --log_level info --llm_log_level info
     '

# End-of-job publish: node-local NVMe is not durable storage — when IGC_WEIGHTS_DEST is
# set (e.g. a path under the shared /models filesystem), copy the run's artifacts off
# the node with a sha256 manifest. Opt-in: publish_checkpoint.sh requires the dest.
if [[ -n "${IGC_WEIGHTS_DEST:-}" ]]; then
    "${IGC_DIR}/scripts/publish_checkpoint.sh" "${IGC_DIR}/experiments/${IGC_RUN}" \
        || echo "WARNING: checkpoint publish failed — weights remain ONLY on this node" >&2
else
    echo "NOTE: IGC_WEIGHTS_DEST unset — weights stay on node-local NVMe (set it to publish, e.g. a /models path)." >&2
fi
