#!/bin/bash
# First end-to-end GPU loop: Phase 1 Redfish JSON pretraining
# (igc/modules/train/sft.py SFTTrainer) on ONE GB300.
# This is only a Slurm/container wrapper around the canonical YAML profile runner.
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

# Paths are deployment inputs. Model and training knobs come only from the profile.
IGC_DIR="${IGC_DIR:?set IGC_DIR to the remote checkout}"
IGC_CORPUS_ROOT="${IGC_CORPUS_ROOT:?set IGC_CORPUS_ROOT to the materialized Phase 1 corpus}"
IGC_PROFILE="${IGC_PROFILE:-phase1_gpt2_smoke}"
IGC_OUTPUT_ROOT="${IGC_OUTPUT_ROOT:?set IGC_OUTPUT_ROOT to a remote artifact directory}"
NGC_IMAGE="${NGC_IMAGE:-nvcr.io/nvidia/pytorch:26.03-py3}"

echo "igc Phase 1 SFT | profile=${IGC_PROFILE} node=$(hostname)"

# shellcheck disable=SC2016  # The single-quoted payload is evaluated inside the container.
srun --container-image="${NGC_IMAGE}" \
     --container-mounts="${IGC_DIR}:/workspace/igc,${IGC_CORPUS_ROOT}:/data/phase1,${IGC_OUTPUT_ROOT}:/outputs" \
     bash -lc '
        set -euo pipefail
        cd /workspace/igc
        nvidia-smi -L
        IGC_PROFILE="'"${IGC_PROFILE}"'" \
        IGC_CORPUS_DIR=/data/phase1/train \
        IGC_CORPUS_EVAL_DIR=/data/phase1/heldout \
        IGC_OUTPUT_DIR="/outputs/'"${IGC_PROFILE}"'-${SLURM_JOB_ID}" \
        bash scripts/run_profile.sh
     '
