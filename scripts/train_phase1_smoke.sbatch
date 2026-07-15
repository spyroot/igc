#!/bin/bash
# First end-to-end GPU loop: Phase 1 Redfish JSON pretraining
# (igc/modules/llm_train_state_encoder.py LlmEmbeddingsTrainer) on ONE GB300.
# Validate on a small model first, then set IGC_MODEL to a large decoder and scale.
# Runbook + cautions: GPU_ACCESS.md. Full guide: docs/TRAINING.md.
#
#   export WANDB_API_KEY=...            # in your shell, NOT here (rotate if ever shared)
#   sbatch scripts/train_phase1_smoke.sbatch                          # gpt2 smoke -> wandb
#   IGC_MODEL=<hf-repo-id> EPOCHS=3 sbatch scripts/train_phase1_smoke.sbatch
#
#SBATCH --job-name=igc-phase1
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --exclude=gb300-poc1-slot2,gb300-poc1-slot15,gb300-poc1-slot16
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#SBATCH --requeue
#SBATCH --signal=USR1@90

set -euo pipefail

# 1-GPU first: the NVLink fabric is flaky under multi-GPU (disable SHARP/multicast).
export NCCL_NVLS_ENABLE=0
export PYTHONUNBUFFERED=1

# Overridable knobs (env). Stage code + captured data on the node's local NVMe.
IGC_MODEL="${IGC_MODEL:-gpt2}"                   # small for the smoke; large HF decoder to scale
IGC_DIR="${IGC_DIR:-$HOME/igc}"                  # node-local igc checkout
DATA_DIR="${DATA_DIR:-$HOME/.json_responses}"    # node-local captured Redfish responses
EPOCHS="${EPOCHS:-3}"
NGC_IMAGE="${NGC_IMAGE:-nvcr.io/nvidia/pytorch:26.03-py3}"
SEED="${SEED:-42}"

# LoRA/PEFT for a large backbone (the gpt2 smoke uses full FT). Set IGC_USE_PEFT=1 to enable.
export EXTRA_ARGS=""
[ "${IGC_USE_PEFT:-0}" = "1" ] && EXTRA_ARGS="--use_peft --lora_r ${LORA_R:-16} --lora_alpha ${LORA_ALPHA:-32}"

# Experiment tracking. Default to wandb; the API key MUST come from your environment
# (export WANDB_API_KEY=...). Never hardcode or commit it. WANDB_PROJECT is safe to set.
IGC_REPORT="${IGC_REPORT:-wandb}"
if [ "${IGC_REPORT}" = "wandb" ]; then
    # convenience: source a node-local gitignored key file if present (.internal/ is never committed)
    # shellcheck source=/dev/null
    [ -f "${IGC_DIR}/.internal/wandb.env" ] && . "${IGC_DIR}/.internal/wandb.env"
    : "${WANDB_API_KEY:?set WANDB_API_KEY in your env or ${IGC_DIR}/.internal/wandb.env to use wandb}"
    export WANDB_API_KEY
    export WANDB_PROJECT="${WANDB_PROJECT:-igc}"
    export WANDB_NAME="${WANDB_NAME:-phase1-${IGC_MODEL//\//_}-${SLURM_JOB_ID:-local}}"
fi

echo "igc Phase 1 SFT | model=${IGC_MODEL} epochs=${EPOCHS} report=${IGC_REPORT} node=$(hostname)"

# shellcheck disable=SC2016  # The single-quoted payload is evaluated inside the container.
srun --container-image="${NGC_IMAGE}" \
     --container-mounts="${IGC_DIR}:/workspace/igc,${DATA_DIR}:/root/.json_responses" \
     bash -lc '
        set -euo pipefail
        cd /workspace/igc
        pip install --no-cache-dir -r docker/requirements-train.txt
        nvidia-smi -L
        python igc_main.py \
            --train llm --llm latent \
            --weights_role model_x \
            --corpus_objective phase1_pretrain \
            --model_type "'"${IGC_MODEL}"'" \
            --num_train_epochs "'"${EPOCHS}"'" \
            --json_data_dir /root/.json_responses \
            --device cuda:0 --seed "'"${SEED}"'" \
            --metric_report "'"${IGC_REPORT}"'" \
            ${EXTRA_ARGS} \
            --log_level info --llm_log_level info
     '
