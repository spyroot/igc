#!/usr/bin/env bats
# Offline render/validation tests for scripts/gb300_launch.sh.
#
# Everything runs in DRY-RUN mode (IGC_DRY_RUN=1): the launcher prints the exact docker
# and igc_main.py command(s) and exits without touching docker, a GPU, or a node — so these
# assert the flag mapping, the ladder rungs, fail-fast validation, and that no secret or
# fleet host leaks, all on a plain CPU box. No docker/nvidia-smi is on PATH here on purpose.
#
# Author:
# Mus mbayramo@stanford.edu

setup() {
    REPO_ROOT="$(cd "${BATS_TEST_DIRNAME}/../.." && pwd)"
    export REPO_ROOT
    LAUNCH="${REPO_ROOT}/scripts/gb300_launch.sh"
    export LAUNCH
    # keep paths off $HOME so the tests are hermetic
    export IGC_CODE_DIR="${BATS_TEST_TMPDIR}/igc"
    export IGC_DATA_DIR="${BATS_TEST_TMPDIR}/data"
    export IGC_DRY_RUN=1
}

# --- ladder rungs -------------------------------------------------------------

@test "smoke1: 1 GPU plain-python, steps capped, W&B off (tensorboard)" {
    run env IGC_RUNG=smoke1 bash "$LAUNCH"
    [ "$status" -eq 0 ]
    [[ "$output" == *"gpus=1 sharding=none smoke=1"* ]]
    [[ "$output" == *"python igc_main.py --device cuda:0"* ]]
    [[ "$output" == *"--max_steps 20"* ]]
    [[ "$output" == *"--metric_report tensorboard"* ]]
    [[ "$output" == *"name=verify-m1-1gpu-none-gpt2-smoke mode=disabled"* ]]
    # a 1-GPU run must NOT wrap in accelerate or pass a sharding/mixed_precision flag
    [[ "$output" != *"accelerate launch"* ]]
}

@test "smoke4: 4 GPU DDP, capped steps, W&B off, has the pre-build phase" {
    run env IGC_RUNG=smoke4 bash "$LAUNCH"
    [ "$status" -eq 0 ]
    [[ "$output" == *"gpus=4 sharding=ddp smoke=1"* ]]
    [[ "$output" == *"accelerate launch --num_processes 4"* ]]
    [[ "$output" == *"--sharding ddp"* ]]
    [[ "$output" == *"PHASE A"* ]]
    [[ "$output" == *"--max_steps 20"* ]]
    [[ "$output" == *"mode=disabled"* ]]
}

@test "run4: 4 GPU DDP real short run -> W&B, TRAIN not step-capped" {
    run env IGC_RUNG=run4 bash "$LAUNCH"
    [ "$status" -eq 0 ]
    [[ "$output" == *"--sharding ddp"* ]]
    [[ "$output" == *"--num_train_epochs 3"* ]]
    [[ "$output" == *"--metric_report wandb"* ]]
    [[ "$output" == *"mode=online"* ]]
    # the real TRAIN command carries no --max_steps (the pre-build line legitimately caps at 1)
    train_line="$(printf '%s\n' "$output" | grep 'TRAIN ')"
    [[ "$train_line" != *"--max_steps"* ]]
}

@test "fsdp4: 4 GPU FSDP2 only when memory requires it" {
    run env IGC_RUNG=fsdp4 bash "$LAUNCH"
    [ "$status" -eq 0 ]
    [[ "$output" == *"--sharding fsdp"* ]]
    [[ "$output" == *"--metric_report wandb"* ]]
}

@test "DDP is the default for multi-GPU (no rung, no sharding)" {
    run env IGC_GPUS=4 bash "$LAUNCH"
    [ "$status" -eq 0 ]
    [[ "$output" == *"--sharding ddp"* ]]
    [[ "$output" != *"--sharding fsdp"* ]]
}

# --- env-knob -> igc_main.py flag mapping -------------------------------------

@test "LoRA/rsLoRA knobs map to --use_peft/--adapter_method/--lora_*" {
    run env IGC_GPUS=4 IGC_USE_PEFT=1 IGC_ADAPTER=rslora LORA_R=32 LORA_ALPHA=64 bash "$LAUNCH"
    [ "$status" -eq 0 ]
    [[ "$output" == *"--use_peft --adapter_method rslora --lora_r 32 --lora_alpha 64 --lora_dropout 0.05"* ]]
}

@test "grad-accum, precision, batch and model map to real flags" {
    run env IGC_GPUS=4 IGC_GRAD_ACCUM=8 IGC_PRECISION=fp8 IGC_BATCH=64 IGC_MODEL=Qwen/Qwen2.5-7B bash "$LAUNCH"
    [ "$status" -eq 0 ]
    [[ "$output" == *"--gradient_accumulation_steps 8"* ]]
    [[ "$output" == *"--mixed_precision fp8"* ]]
    [[ "$output" == *"--per_device_train_batch_size 64"* ]]
    [[ "$output" == *"--model_type Qwen/Qwen2.5-7B"* ]]
    # HF repo id with '/' is sanitized in the W&B run name
    [[ "$output" == *"name=verify-m1-4gpu-ddp-Qwen_Qwen2.5-7B"* ]]
}

@test "NCCL defaults: CUMEM on, MNNVL off; both overridable" {
    run env IGC_RUNG=run4 bash "$LAUNCH"
    [[ "$output" == *"-e NCCL_CUMEM_ENABLE=1 -e NCCL_MNNVL_ENABLE=0"* ]]

    run env IGC_RUNG=run4 NCCL_MNNVL_ENABLE=1 NCCL_CUMEM_ENABLE=0 bash "$LAUNCH"
    [[ "$output" == *"-e NCCL_CUMEM_ENABLE=0 -e NCCL_MNNVL_ENABLE=1"* ]]
}

@test "explicit knobs override a rung preset" {
    run env IGC_RUNG=smoke4 IGC_SHARDING=fsdp IGC_SMOKE=0 bash "$LAUNCH"
    [ "$status" -eq 0 ]
    [[ "$output" == *"--sharding fsdp"* ]]
    [[ "$output" == *"--metric_report wandb"* ]]
}

# --- fail-fast validation (no silent fallback) --------------------------------

@test "invalid IGC_SHARDING is rejected" {
    run env IGC_SHARDING=zero9 bash "$LAUNCH"
    [ "$status" -ne 0 ]
    [[ "$output" == *"BLOCKER: IGC_SHARDING='zero9' invalid"* ]]
}

@test "invalid IGC_PRECISION is rejected" {
    run env IGC_PRECISION=int4 bash "$LAUNCH"
    [ "$status" -ne 0 ]
    [[ "$output" == *"BLOCKER: IGC_PRECISION='int4' invalid"* ]]
}

@test "invalid IGC_ADAPTER is rejected" {
    run env IGC_ADAPTER=qlora bash "$LAUNCH"
    [ "$status" -ne 0 ]
    [[ "$output" == *"BLOCKER: IGC_ADAPTER='qlora' invalid"* ]]
}

@test "non-integer / zero IGC_GPUS is rejected" {
    run env IGC_GPUS=abc bash "$LAUNCH"
    [ "$status" -ne 0 ]
    [[ "$output" == *"IGC_GPUS must be an integer >= 1"* ]]

    run env IGC_GPUS=0 bash "$LAUNCH"
    [ "$status" -ne 0 ]
}

@test "unknown IGC_RUNG is rejected" {
    run env IGC_RUNG=mega8 bash "$LAUNCH"
    [ "$status" -ne 0 ]
    [[ "$output" == *"unknown IGC_RUNG='mega8'"* ]]
}

@test "negative batch / bad max_steps are rejected" {
    run env IGC_BATCH=-1 bash "$LAUNCH"
    [ "$status" -ne 0 ]
    run env IGC_MAX_STEPS=x bash "$LAUNCH"
    [ "$status" -ne 0 ]
}

# --- safety: no secret / no fleet host in the render --------------------------

@test "render leaks no fleet IP and references creds by file, not value" {
    run env IGC_RUNG=run4 bash "$LAUNCH"
    [ "$status" -eq 0 ]
    # single-node rendezvous is localhost only; no internal 172.25.x.x fleet address
    [[ "$output" != *"172.25."* ]]
    [[ "$output" == *"127.0.0.1"* ]]
    # W&B/HF creds are sourced from a gitignored file path, never printed inline
    [[ "$output" == *".internal/wandb.env"* ]]
    [[ "$output" != *"WANDB_API_KEY="* ]]
    [[ "$output" != *"HF_TOKEN="* ]]
}

# --- IGC_STAGE is a real selector, not just a label -------------------------

@test "IGC_STAGE maps to the real --train/--llm flags and labels the run" {
    run env IGC_GPUS=4 IGC_STAGE=m2 bash "$LAUNCH"
    [ "$status" -eq 0 ]
    [[ "$output" == *"--train llm --llm encoder"* ]]
    [[ "$output" == *"stage=m2"* ]]
    [[ "$output" == *"name=verify-m2-"* ]]
    # m2 must NOT silently emit m1's --llm latent
    [[ "$output" != *"--llm latent"* ]]

    run env IGC_GPUS=4 IGC_STAGE=m3 bash "$LAUNCH"
    [[ "$output" == *"--llm goal"* ]]
}

@test "RL/combined stages are rejected here (they belong in train_igc.sbatch)" {
    run env IGC_STAGE=m6 bash "$LAUNCH"
    [ "$status" -ne 0 ]
    [[ "$output" == *"not wired here"* ]]
    [[ "$output" == *"train_igc.sbatch"* ]]
}

@test "invalid IGC_STAGE is rejected" {
    run env IGC_STAGE=m9 bash "$LAUNCH"
    [ "$status" -ne 0 ]
    [[ "$output" == *"IGC_STAGE='m9' invalid"* ]]
}

# --- pre-build is a capped dataset build, not a full single-GPU epoch --------

@test "multi-GPU pre-build is capped at --max_steps 1 (build, not a whole epoch)" {
    run env IGC_RUNG=run4 bash "$LAUNCH"
    [ "$status" -eq 0 ]
    # the PHASE A line must carry the cap so it exits after the dataset cache is written
    [[ "$output" == *"PHASE A"*"--max_steps 1"* ]]
}

@test "IGC_PREBUILD=0 skips the pre-build phase entirely" {
    run env IGC_GPUS=4 IGC_PREBUILD=0 bash "$LAUNCH"
    [ "$status" -eq 0 ]
    [[ "$output" != *"PHASE A"* ]]
}

# --- fail-fast on argv-breaking whitespace ----------------------------------

@test "whitespace in IGC_RUN or IGC_MODEL is rejected" {
    run env IGC_RUN="my run" bash "$LAUNCH"
    [ "$status" -ne 0 ]
    [[ "$output" == *"IGC_RUN must contain no whitespace"* ]]

    run env IGC_MODEL="a b" bash "$LAUNCH"
    [ "$status" -ne 0 ]
    [[ "$output" == *"IGC_MODEL must contain no whitespace"* ]]
}

# --- render tells the truth about 1-GPU precision ---------------------------

@test "1-GPU render marks IGC_PRECISION as ignored (tf32 only), not applied" {
    run env IGC_RUNG=smoke1 IGC_PRECISION=bf16 bash "$LAUNCH"
    [ "$status" -eq 0 ]
    [[ "$output" == *"tf32 only (IGC_PRECISION=bf16 ignored on 1 GPU)"* ]]
    [[ "$output" != *"--mixed_precision"* ]]
}

@test "dry-run needs neither docker nor nvidia-smi on PATH" {
    # a minimal PATH with no docker / nvidia-smi still renders (proves no launch happens)
    run env -u IGC_DRY_RUN HOME="${BATS_TEST_TMPDIR}" IGC_DRY_RUN=1 IGC_RUNG=smoke1 \
        PATH="/usr/bin:/bin" IGC_CODE_DIR="${IGC_CODE_DIR}" IGC_DATA_DIR="${IGC_DATA_DIR}" \
        bash "$LAUNCH"
    [ "$status" -eq 0 ]
    [[ "$output" == *"DRY RUN — nothing launched"* ]]
}
