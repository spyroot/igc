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
#SBATCH --exclude=gb300-poc1-slot2,gb300-poc1-slot15,gb300-poc1-slot16
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out
#SBATCH --requeue
#SBATCH --signal=USR1@90

set -euo pipefail

# 1-GPU first: the NVLink fabric is flaky under multi-GPU (disable SHARP/multicast).
export NCCL_NVLS_ENABLE=0
export PYTHONUNBUFFERED=1

# Overridable knobs (env). Stage code + captured data on the node's local NVMe.
IGC_STAGE="${IGC_STAGE:-m1}"                     # m1|m2|m3|m3p|m6|all (see header)
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
    # convenience: source a node-local gitignored key file if present (.internal/ is never committed)
    [ -f "${IGC_DIR}/.internal/wandb.env" ] && . "${IGC_DIR}/.internal/wandb.env"
    : "${WANDB_API_KEY:?set WANDB_API_KEY in your env or ${IGC_DIR}/.internal/wandb.env to use wandb}"
    export WANDB_API_KEY
    export WANDB_PROJECT="${WANDB_PROJECT:-igc}"
    export WANDB_NAME="${RUN_NAME}"
fi

echo "igc train | stage=${IGC_STAGE} args='${TRAIN_ARGS}' model=${IGC_MODEL} epochs=${EPOCHS} report=${IGC_REPORT} node=$(hostname)"

# Export the few values the inner heredoc needs (avoids brittle nested quoting).
export IGC_MODEL EPOCHS SEED IGC_REPORT TRAIN_ARGS EPOCH_FLAG EXTRA_ARGS RUN_NAME IGC_RUN RL_EPOCH_ARG

srun --container-image="${NGC_IMAGE}" \
     --container-mounts="${IGC_DIR}:/workspace/igc,${DATA_DIR}:/root/.json_responses" \
     bash -lc '
        set -euo pipefail
        cd /workspace/igc
        pip install --no-cache-dir -r docker/requirements-train.txt
        nvidia-smi -L
        # Shared curriculum root so later stages find earlier checkpoints (per-module subdirs).
        OUT="/workspace/igc/experiments/${IGC_RUN}"
        mkdir -p "${OUT}"
        # shellcheck disable=SC2086
        python igc_main.py \
            ${TRAIN_ARGS} \
            --model_type "${IGC_MODEL}" \
            ${EPOCH_FLAG} "${EPOCHS}" ${RL_EPOCH_ARG} \
            --json_data_dir /root/.json_responses \
            --output_dir "${OUT}" \
            --device cuda:0 --seed "${SEED}" \
            --metric_report "${IGC_REPORT}" \
            ${EXTRA_ARGS} \
            --log_level info --llm_log_level info
     '
