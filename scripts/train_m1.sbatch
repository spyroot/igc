#!/bin/bash
# First end-to-end GPU loop: train M1, the state-encoder (causal/masked-LM SFT,
# igc/modules/llm_train_state_encoder.py LlmEmbeddingsTrainer) on ONE GB300.
# Validate on a small model first, then set IGC_MODEL to a large decoder and scale.
# Runbook + cautions: GPU_ACCESS.md. Launch from a cluster login node:
#
#   sbatch scripts/train_m1.sbatch                          # gpt2 smoke
#   IGC_MODEL=<hf-repo-id> EPOCHS=3 sbatch scripts/train_m1.sbatch
#
#SBATCH --job-name=igc-m1
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

echo "igc M1 SFT | model=${IGC_MODEL} epochs=${EPOCHS} node=$(hostname)"

srun --container-image="${NGC_IMAGE}" \
     --container-mounts="${IGC_DIR}:/workspace/igc,${DATA_DIR}:/root/.json_responses" \
     bash -lc '
        set -euo pipefail
        cd /workspace/igc
        pip install --no-cache-dir -r docker/requirements-train.txt
        nvidia-smi -L
        python igc_main.py \
            --train llm --llm latent \
            --model_type "'"${IGC_MODEL}"'" \
            --num_train_epochs "'"${EPOCHS}"'" \
            --json_data_dir /root/.json_responses \
            --device cuda:0 \
            --log_level info --llm_log_level info
     '
