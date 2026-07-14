#!/usr/bin/env bash
# gb300_launch.sh — launch ONE igc training run on ONE GB300 node via docker.
#
# The "boring and repeatable" training path. Runs ON the node (docker runs locally;
# the cluster's Slurm is only driveable from slot0 because munge is down on the compute
# nodes, so we launch directly and guard the node ourselves). It ALWAYS re-checks that
# the GPUs are free in the seconds right before launch — the fleet churns and a node can
# be grabbed while you set up, so an earlier "free" reading is never trusted.
#
# Parallelism follows docs/DISTRIBUTED_PLAN.md: for models that fit on one GPU (the ones
# igc actually fine-tunes) DDP is the default and ~3x faster than FSDP2 at 4 GPU; reach
# for --sharding fsdp only when the backbone genuinely does not fit. For >1 GPU it runs a
# single-GPU pre-build FIRST (serialises the dataset build, which races across ranks) with
# no experiment tracking, then the multi-GPU accelerate run — that second run is the ONE
# clean W&B run (rank-gated in metric_factory, so one run per job not one per GPU).
#
# Nothing host-, path-, or credential-specific is baked in: every setting is an env knob,
# W&B/HF creds are sourced INSIDE the container from gitignored env files (never printed),
# and there are no hardcoded fleet hosts (single-node accelerate rendezvouses on localhost).
#
# THE LADDER (prove the substrate cheaply, then scale) — one rung per invocation:
#   IGC_RUNG=smoke1 scripts/gb300_launch.sh   # 1 GPU, ~20 steps, W&B off — is the path alive?
#   IGC_RUNG=smoke4 scripts/gb300_launch.sh   # 4 GPU DDP, ~20 steps, W&B off — does DDP wire up?
#   IGC_RUNG=run4   scripts/gb300_launch.sh   # 4 GPU DDP, real short run -> W&B
#   IGC_RUNG=fsdp4  scripts/gb300_launch.sh   # 4 GPU FSDP2, real short run -> W&B (only if it must shard)
# or drive every knob yourself (IGC_GPUS / IGC_SHARDING / IGC_SMOKE / ...).
#
# DRY RUN — print the EXACT docker + training command(s) without launching (no docker,
# no GPU, no node needed; safe on a laptop or in CI):
#   IGC_DRY_RUN=1 IGC_RUNG=run4 IGC_MODEL=gpt2 scripts/gb300_launch.sh
#
# Detached so it survives an ssh drop — redirect from the CALLER; never add
# `exec > >(tee ...)` inside this script (process substitution wedges a setsid launch):
#   nohup setsid bash scripts/gb300_launch.sh > ~/igc-run.log 2>&1 </dev/null & exit 0
#
# Used by: run by hand on a free reserved slot (GPU_ACCESS.md §3), or after the 2-node
# DDP sanity gate (scripts/gb300_sanity_check.sh) proves the fabric. Not imported by Python.
# Iterate on code with `git -C "$IGC_CODE_DIR" pull` (the checkout is bind-mounted, so no
# image rebuild) then re-run. Rendering/validation is covered by tests/bats/gb300_launch.bats.
#
# Author:
# Mus mbayramo@stanford.edu
set -uo pipefail

# --- knobs (env-overridable; no hardcoded hosts, paths, or endpoints) ---------
# Ladder / topology
IGC_RUNG="${IGC_RUNG:-}"                                    # smoke1|smoke4|run4|fsdp4|"" (preset the ladder rung; overrides gpus/sharding/smoke defaults)
IGC_GPUS="${IGC_GPUS:-}"                                    # GPUs for THIS run (1=plain python; >1=accelerate). Empty => rung/default 4
IGC_SHARDING="${IGC_SHARDING:-}"                            # none|ddp|zero2|zero3|zero3_offload|fsdp (>1 GPU). Empty => ddp (the default per DISTRIBUTED_PLAN)
IGC_SMOKE="${IGC_SMOKE:-}"                                  # 1=fast check (cap steps, W&B off, 1 epoch); 0=real run. Empty => rung/0
# Model / data / output
IGC_STAGE="${IGC_STAGE:-m1}"                                # legacy LLM stage -> --train/--llm (m1=latent|m2=encoder|m3=goal|m3p=parameter). Phase profiles use scripts/run_profile.sh
IGC_MODEL="${IGC_MODEL:-gpt2}"                              # gpt2 smoke, or an HF repo id / local path to scale up
IGC_RUN="${IGC_RUN:-verify}"                                # run name -> W&B name, output subdir, container name
IGC_CODE_DIR="${IGC_CODE_DIR:-${HOME:-/root}/igc}"          # igc checkout (node-local or /models/igc)
IGC_DATA_DIR="${IGC_DATA_DIR:-${HOME:-/root}/.json_responses}"  # captured Redfish responses (mounted read-only)
IGC_MODELS_DIR="${IGC_MODELS_DIR:-/models}"                 # shared 240TB BeeGFS (large/durable artifacts)
IGC_OUTPUT_DIR="${IGC_OUTPUT_DIR:-experiments/${IGC_RUN}}"  # checkpoints + tensorboard land here (relative path resolved under the mounted checkout /workspace/igc)
IGC_IMAGE="${IGC_IMAGE:-nvcr.io/nvidia/pytorch:26.03-py3}"  # bare NGC (runtime pip) or igc-train:ngc26.03 (deps baked)
# Training hyper-params (all map to real igc_main.py flags; see igc/shared/shared_arg_parser.py)
EPOCHS="${EPOCHS:-3}"                                       # --num_train_epochs (ignored in smoke: forced to 1)
IGC_BATCH="${IGC_BATCH:-128}"                               # --per_device_train_batch_size (256 OOMs a large-vocab backbone; 128 is the safe start)
IGC_GRAD_ACCUM="${IGC_GRAD_ACCUM:-1}"                       # --gradient_accumulation_steps (raise to grow effective batch without more HBM)
IGC_WORKERS="${IGC_WORKERS:-16}"                            # --num_workers (default 1 left 143/144 Grace cores idle)
IGC_PRECISION="${IGC_PRECISION:-bf16}"                      # --mixed_precision on the accelerate path: no|fp16|bf16|fp8 (bf16 recommended)
IGC_MAX_STEPS="${IGC_MAX_STEPS:-}"                          # --max_steps cap; empty => none (real run). Smoke defaults to 20 if unset
# LoRA / rsLoRA (the supported large-backbone path)
IGC_USE_PEFT="${IGC_USE_PEFT:-0}"                          # 1 => --use_peft (LoRA/PEFT fine-tune of a bf16 base)
IGC_ADAPTER="${IGC_ADAPTER:-lora}"                         # --adapter_method: lora|rslora|dora (rslora = rank-stabilized)
LORA_R="${LORA_R:-16}"                                     # --lora_r (rank)
LORA_ALPHA="${LORA_ALPHA:-32}"                             # --lora_alpha (scaling)
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"                       # --lora_dropout
# W&B / HF env (creds live in gitignored files, sourced INSIDE the container, never printed)
IGC_WANDB_ENV="${IGC_WANDB_ENV:-.internal/wandb.env}"      # gitignored W&B creds file (WANDB_API_KEY/ENTITY/PROJECT)
IGC_HF_ENV="${IGC_HF_ENV:-.internal/hf.env}"               # gitignored HF cache/token file (HF_HOME/HF_HUB_CACHE/HF_TOKEN)
WANDB_PROJECT="${WANDB_PROJECT:-igc}"                       # W&B project (non-secret)
WANDB_MODE="${WANDB_MODE:-}"                                # online|offline|disabled; empty => online for a real run, disabled for a smoke
# NCCL (GB300 defaults per TEAM_GUIDE: CUMEM on, MNNVL off unless a preflight proves the IMEX channels)
IGC_NCCL_CUMEM="${IGC_NCCL_CUMEM:-${NCCL_CUMEM_ENABLE:-1}}" # preserve the launcher decision across private env sourcing
NCCL_CUMEM_ENABLE="${IGC_NCCL_CUMEM}"                       # default 1 (CUMEM helps NCCL memory registration)
IGC_NCCL_MNNVL="${IGC_NCCL_MNNVL:-${NCCL_MNNVL_ENABLE:-0}}" # preserve the launcher decision across private env sourcing
NCCL_MNNVL_ENABLE="${IGC_NCCL_MNNVL}"                       # default 0 (multi-node MNNVL off until a node preflight proves it works)
# Control
IGC_PREBUILD="${IGC_PREBUILD:-}"                            # multi-GPU dataset pre-build: 1=run, 0=skip. Empty => 1 for >1 GPU. Skip (0) for a model too large to load on ONE GPU (pre-build the dataset separately first)
IGC_DRY_RUN="${IGC_DRY_RUN:-0}"                             # 1 => print the exact command(s) and exit; no docker, no GPU
IGC_MASTER_PORT="${IGC_MASTER_PORT:-29500}"                 # single-node accelerate rendezvous port (localhost)
IGC_MIN_FREE_GB="${IGC_MIN_FREE_GB:-100}"                  # HF pulls + dataset + checkpoints need headroom (real run only)
IGC_ALLOW_RUNTIME_PIP="${IGC_ALLOW_RUNTIME_PIP:-0}"        # 1 => allow in-container `pip install` if deps missing; 0 => fail loudly (no silent fallback)

log()     { echo "=== [$(date -u '+%F %T')] $* ==="; }
blocker() { echo "BLOCKER: $*" >&2; exit 3; }
is_int()  { case "${1:-}" in ''|*[!0-9]*) return 1 ;; *) return 0 ;; esac; }
is_decimal() { [[ "${1:-}" =~ ^[0-9]+([.][0-9]+)?$ ]]; }
is_safe_shell_atom() { case "${1:-}" in ''|*[!A-Za-z0-9_./:@+=,-]*) return 1 ;; *) return 0 ;; esac; }
require_safe_shell_atom() {
    local name="$1" value="${!1}"
    is_safe_shell_atom "$value" || blocker "$name contains whitespace or shell-control characters (got '${value}')"
}

# --- 1. resolve the ladder rung (a preset is just defaults; explicit knobs win) ----
case "$IGC_RUNG" in
    smoke1) : "${IGC_GPUS:=1}" "${IGC_SHARDING:=none}" "${IGC_SMOKE:=1}" ;;
    smoke4) : "${IGC_GPUS:=4}" "${IGC_SHARDING:=ddp}"  "${IGC_SMOKE:=1}" ;;
    run4)   : "${IGC_GPUS:=4}" "${IGC_SHARDING:=ddp}"  "${IGC_SMOKE:=0}" ;;
    fsdp4)  : "${IGC_GPUS:=4}" "${IGC_SHARDING:=fsdp}" "${IGC_SMOKE:=0}" ;;
    "")     : ;;                                            # no rung: fall through to per-knob defaults
    *)      blocker "unknown IGC_RUNG='${IGC_RUNG}' (use smoke1|smoke4|run4|fsdp4 or set the knobs directly)" ;;
esac
# per-knob defaults for anything the rung did not set
: "${IGC_GPUS:=4}" "${IGC_SMOKE:=0}"
if [ "$IGC_GPUS" -gt 1 ] 2>/dev/null; then : "${IGC_SHARDING:=ddp}"; else : "${IGC_SHARDING:=none}"; fi

# --- 2. validate knobs (fail fast, explicit — runs in dry-run too) --------------
{ is_int "$IGC_GPUS" && [ "$IGC_GPUS" -ge 1 ]; } || blocker "IGC_GPUS must be an integer >= 1 (got '${IGC_GPUS}')"
for v in IGC_BATCH IGC_GRAD_ACCUM IGC_WORKERS EPOCHS; do
    { is_int "${!v}" && [ "${!v}" -ge 1 ]; } || blocker "$v must be a positive integer (got '${!v}')"
done
for v in IGC_RUN IGC_MODEL IGC_OUTPUT_DIR IGC_IMAGE IGC_CODE_DIR IGC_DATA_DIR IGC_MODELS_DIR IGC_WANDB_ENV IGC_HF_ENV; do
    require_safe_shell_atom "$v"
done
[ -z "$IGC_MAX_STEPS" ] || { is_int "$IGC_MAX_STEPS" && [ "$IGC_MAX_STEPS" -ge 1 ]; } || blocker "IGC_MAX_STEPS must be empty or a positive integer (got '${IGC_MAX_STEPS}')"
{ is_int "$IGC_MASTER_PORT" && [ "$IGC_MASTER_PORT" -ge 1 ] && [ "$IGC_MASTER_PORT" -le 65535 ]; } || blocker "IGC_MASTER_PORT must be an integer TCP port 1-65535 (got '${IGC_MASTER_PORT}')"
case "$IGC_SHARDING" in none|ddp|zero2|zero3|zero3_offload|fsdp) ;; *) blocker "IGC_SHARDING='${IGC_SHARDING}' invalid (none|ddp|zero2|zero3|zero3_offload|fsdp)" ;; esac
case "$IGC_PRECISION" in no|fp16|bf16|fp8) ;; *) blocker "IGC_PRECISION='${IGC_PRECISION}' invalid (no|fp16|bf16|fp8)" ;; esac
case "$IGC_ADAPTER" in lora|rslora|dora) ;; *) blocker "IGC_ADAPTER='${IGC_ADAPTER}' invalid (lora|rslora|dora)" ;; esac
case "$IGC_USE_PEFT" in 0|1) ;; *) blocker "IGC_USE_PEFT must be 0 or 1 (got '${IGC_USE_PEFT}')" ;; esac
case "$IGC_SMOKE" in 0|1) ;; *) blocker "IGC_SMOKE must be 0 or 1 (got '${IGC_SMOKE}')" ;; esac
case "$IGC_PREBUILD" in ''|0|1) ;; *) blocker "IGC_PREBUILD must be empty, 0, or 1 (got '${IGC_PREBUILD}')" ;; esac
case "$IGC_DRY_RUN" in 0|1) ;; *) blocker "IGC_DRY_RUN must be 0 or 1 (got '${IGC_DRY_RUN}')" ;; esac
case "$WANDB_MODE" in ''|online|offline|disabled) ;; *) blocker "WANDB_MODE must be empty, online, offline, or disabled (got '${WANDB_MODE}')" ;; esac
case "$NCCL_CUMEM_ENABLE" in 0|1) ;; *) blocker "NCCL_CUMEM_ENABLE must be 0 or 1 (got '${NCCL_CUMEM_ENABLE}')" ;; esac
case "$NCCL_MNNVL_ENABLE" in 0|1) ;; *) blocker "NCCL_MNNVL_ENABLE must be 0 or 1 (got '${NCCL_MNNVL_ENABLE}')" ;; esac
{ is_int "$LORA_R" && [ "$LORA_R" -ge 1 ]; } || blocker "LORA_R must be a positive integer (got '${LORA_R}')"
{ is_int "$LORA_ALPHA" && [ "$LORA_ALPHA" -ge 1 ]; } || blocker "LORA_ALPHA must be a positive integer (got '${LORA_ALPHA}')"
is_decimal "$LORA_DROPOUT" || blocker "LORA_DROPOUT must be a decimal value (got '${LORA_DROPOUT}')"

# Resolve legacy LLM stages -> the real --train/--llm flags. Phase 2/3 profile
# work is additionally available through scripts/run_profile.sh and the shared
# specs in configs/phase_training/profiles.yaml; these aliases stay live for
# direct compatibility with igc_main.py --llm goal/parameter.
case "$IGC_STAGE" in
    m1|state_encoder) STAGE_ARGS="--train llm --llm latent" ;;
    m2|autoencoder)   STAGE_ARGS="--train llm --llm encoder" ;;
    m3|goal)          STAGE_ARGS="--train llm --llm goal" ;;
    m3p|parameter)    STAGE_ARGS="--train llm --llm parameter" ;;
    m6|agent|rl|all)  blocker "IGC_STAGE='${IGC_STAGE}' (RL/combined) is not wired here — use scripts/train_igc.sbatch; this launcher covers the legacy LLM stages m1|m2|m3|m3p" ;;
    *)                blocker "IGC_STAGE='${IGC_STAGE}' invalid (m1|m2|m3|m3p; Phase profiles use scripts/run_profile.sh)" ;;
esac

# --- 3. resolve smoke- and run-shaped settings ---------------------------------
if [ "$IGC_SMOKE" = "1" ]; then
    RUN_EPOCHS=1                                            # a smoke never needs more than one epoch
    MAX_STEPS="${IGC_MAX_STEPS:-20}"                        # cap steps so it finishes in minutes
    RUN_WANDB_MODE="${WANDB_MODE:-disabled}"               # a smoke does not pollute the dashboard
else
    RUN_EPOCHS="$EPOCHS"
    MAX_STEPS="${IGC_MAX_STEPS:-}"                          # no cap for a real run unless asked
    RUN_WANDB_MODE="${WANDB_MODE:-online}"
fi
if [ "$RUN_WANDB_MODE" = "disabled" ]; then METRIC_REPORT=tensorboard; else METRIC_REPORT=wandb; fi
MODEL_TAG="${IGC_MODEL//\//_}"                              # HF repo ids contain '/', unsafe in a W&B name/subdir
SMOKE_SUFFIX=""; [ "$IGC_SMOKE" = "1" ] && SMOKE_SUFFIX="-smoke"
WANDB_NAME="${IGC_RUN}-${IGC_STAGE}-${IGC_GPUS}gpu-${IGC_SHARDING}-${MODEL_TAG}${SMOKE_SUFFIX}"
CONTAINER="igc-${IGC_RUN}"

# --- 4. build the igc_main.py flag set (grounded in shared_arg_parser.py) -------
COMMON="${STAGE_ARGS} --model_type ${IGC_MODEL} --json_data_dir /root/.json_responses --tf32 --seed 42 --log_level info --llm_log_level info --per_device_train_batch_size ${IGC_BATCH} --gradient_accumulation_steps ${IGC_GRAD_ACCUM} --num_workers ${IGC_WORKERS}"
[ "$IGC_USE_PEFT" = "1" ] && COMMON="${COMMON} --use_peft --adapter_method ${IGC_ADAPTER} --lora_r ${LORA_R} --lora_alpha ${LORA_ALPHA} --lora_dropout ${LORA_DROPOUT}"
TAIL="--num_train_epochs ${RUN_EPOCHS} --output_dir ${IGC_OUTPUT_DIR} --metric_report ${METRIC_REPORT}"
[ -n "$MAX_STEPS" ] && TAIL="${TAIL} --max_steps ${MAX_STEPS}"

# the exact per-rank training command (accelerate wraps it for >1 GPU; --mixed_precision
# is an accelerate-path flag, so it is only added there — a 1-GPU run uses --tf32 only).
# The >1-GPU pre-build serialises the dataset build (which races across ranks): the dataset
# is constructed BEFORE the train loop, so --max_steps 1 produces the cache and exits without
# training a whole epoch. It still loads the model on ONE GPU, so set IGC_PREBUILD=0 for a
# model too large to fit single-GPU (pre-build the dataset separately first).
if [ "$IGC_GPUS" -gt 1 ]; then
    : "${IGC_PREBUILD:=1}"
    if [ "$IGC_PREBUILD" = "1" ]; then
        PREBUILD_CMD="WANDB_MODE=disabled python igc_main.py --device cuda:0 ${COMMON} --num_train_epochs 1 --max_steps 1 --output_dir /tmp/prebuild --metric_report tensorboard"
    else
        PREBUILD_CMD=""                                    # skipped: dataset assumed already built (IGC_PREBUILD=0)
    fi
    TRAIN_CMD="accelerate launch --num_processes ${IGC_GPUS} --num_machines 1 --machine_rank 0 --main_process_ip 127.0.0.1 --main_process_port ${IGC_MASTER_PORT} igc_main.py --use_accelerator --sharding ${IGC_SHARDING} --mixed_precision ${IGC_PRECISION} ${COMMON} ${TAIL}"
else
    : "${IGC_PREBUILD:=0}"
    PREBUILD_CMD=""                                        # 1 GPU: the single run also builds the dataset, no separate pre-build
    TRAIN_CMD="python igc_main.py --device cuda:0 ${COMMON} ${TAIL}"
fi

# --- 5. render the exact command(s) (used by dry-run AND logged before a real run) --
render() {
    local peft_desc="off" prec_desc="${IGC_PRECISION} (--mixed_precision, accelerate path)"
    [ "$IGC_USE_PEFT" = "1" ] && peft_desc="${IGC_ADAPTER} r=${LORA_R} a=${LORA_ALPHA}"
    # --mixed_precision is an accelerate-path flag; a 1-GPU plain-python run honours --tf32 only
    [ "$IGC_GPUS" -gt 1 ] || prec_desc="tf32 only (IGC_PRECISION=${IGC_PRECISION} ignored on 1 GPU)"
    echo "# ---------------------------------------------------------------------------"
    echo "# igc training render : rung=${IGC_RUNG:-<custom>} stage=${IGC_STAGE} gpus=${IGC_GPUS} sharding=${IGC_SHARDING} smoke=${IGC_SMOKE}"
    echo "#   model=${IGC_MODEL}  precision=${prec_desc}  peft=${peft_desc}"
    echo "#   batch=${IGC_BATCH} grad_accum=${IGC_GRAD_ACCUM} epochs=${RUN_EPOCHS} max_steps=${MAX_STEPS:-none} workers=${IGC_WORKERS} prebuild=${IGC_PREBUILD}"
    echo "#   output=${IGC_OUTPUT_DIR}  metric=${METRIC_REPORT}"
    echo "#   W&B: project=${WANDB_PROJECT} name=${WANDB_NAME} mode=${RUN_WANDB_MODE}  (creds sourced in-container from ${IGC_WANDB_ENV}; HF from ${IGC_HF_ENV})"
    echo "#   NCCL: NCCL_CUMEM_ENABLE=${NCCL_CUMEM_ENABLE} NCCL_MNNVL_ENABLE=${NCCL_MNNVL_ENABLE}"
    echo "docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \\"
    echo "    --name ${CONTAINER} \\"
    echo "    -v ${IGC_CODE_DIR}:/workspace/igc \\"
    echo "    -v ${IGC_DATA_DIR}:/root/.json_responses:ro \\"
    echo "    -v ${IGC_MODELS_DIR}:/models   # (only when ${IGC_MODELS_DIR} exists) \\"
    echo "    -e NCCL_CUMEM_ENABLE=${NCCL_CUMEM_ENABLE} -e NCCL_MNNVL_ENABLE=${NCCL_MNNVL_ENABLE} \\"
    echo "    -w /workspace/igc ${IGC_IMAGE} bash -lc '<inner>'"
    echo "# inner (in-container):"
    [ -n "$PREBUILD_CMD" ] && echo "#   PHASE A (pre-build, W&B off) : ${PREBUILD_CMD}"
    echo "#   TRAIN                        : ${TRAIN_CMD}"
    echo "# ---------------------------------------------------------------------------"
}

# --- 6. dry-run: render and stop (no docker, no GPU, no node needed) ------------
if [ "$IGC_DRY_RUN" = "1" ]; then
    render
    log "DRY RUN — nothing launched"
    exit 0
fi

# --- 7. pre-flight: the node must be genuinely free RIGHT NOW -------------------
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

# check free space where the run actually writes: IGC_OUTPUT_DIR is relative, resolved under the
# mounted checkout IGC_CODE_DIR (e.g. /models/igc), NOT necessarily $HOME. (HF cache lives wherever
# the private hf.env points HF_HOME — size that mount separately.)
avail=$(df -BG --output=avail "$IGC_CODE_DIR" 2>/dev/null | tail -1 | tr -dc '0-9' || echo 0)
[ "${avail:-0}" -ge "$IGC_MIN_FREE_GB" ] || blocker "only ${avail}G free on the checkpoint filesystem (${IGC_CODE_DIR}) (< ${IGC_MIN_FREE_GB}G)"

[ -f "$IGC_CODE_DIR/igc_main.py" ] || blocker "no igc checkout at $IGC_CODE_DIR (clone it or point IGC_CODE_DIR at /models/igc)"
[ -d "$IGC_DATA_DIR" ] || blocker "no captured data at $IGC_DATA_DIR"
docker image inspect "$IGC_IMAGE" >/dev/null 2>&1 || blocker "image $IGC_IMAGE not present — pull it or build igc-train from docker/Dockerfile.train"

# --- 8. launch --------------------------------------------------------------------
log "launch: rung=${IGC_RUNG:-<custom>} ${IGC_GPUS}GPU sharding=${IGC_SHARDING} smoke=${IGC_SMOKE} model=${IGC_MODEL}"
render
MODELS_MOUNT=""
[ -d "$IGC_MODELS_DIR" ] && MODELS_MOUNT="-v ${IGC_MODELS_DIR}:/models"

# The inner script is a literal with resolved values (so what dry-run prints is what runs);
# creds are sourced from the gitignored env files by PATH, never interpolated as values.
INNER=$(cat <<INNEREOF
set -uo pipefail
cd /workspace/igc
if ! python -c "import transformers, accelerate, wandb" 2>/dev/null; then
    if [ "${IGC_ALLOW_RUNTIME_PIP}" = "1" ]; then
        echo "deps missing in ${IGC_IMAGE} — IGC_ALLOW_RUNTIME_PIP=1, installing"
        pip install --no-cache-dir -r docker/requirements-train.txt 2>&1 | tail -3
    else
        echo "BLOCKER: training deps (transformers/accelerate/wandb) missing in ${IGC_IMAGE}." >&2
        echo "Use igc-train:ngc26.03 (deps baked) or re-run with IGC_ALLOW_RUNTIME_PIP=1." >&2
        exit 4
    fi
fi
# creds: source the gitignored env files if present; NEVER echo their values
[ -f "${IGC_WANDB_ENV}" ] && { set -a; . "${IGC_WANDB_ENV}"; set +a; }
[ -f "${IGC_HF_ENV}" ]    && { set -a; . "${IGC_HF_ENV}";    set +a; }
export NCCL_CUMEM_ENABLE="${IGC_NCCL_CUMEM}"
export NCCL_MNNVL_ENABLE="${IGC_NCCL_MNNVL}"
export WANDB_PROJECT="${WANDB_PROJECT}"
export WANDB_NAME="${WANDB_NAME}"
export WANDB_MODE="${RUN_WANDB_MODE}"
HF_TOKEN_STATUS=none
[ -n "\${HF_TOKEN:-}" ] && HF_TOKEN_STATUS=available
echo "run: W&B project=\${WANDB_PROJECT} name=\${WANDB_NAME} mode=\${WANDB_MODE} | HF cache=\${HF_HOME:-<default>} token=\${HF_TOKEN_STATUS}"
mkdir -p "${IGC_OUTPUT_DIR}"
nvidia-smi -L
INNEREOF
)
if [ -n "$PREBUILD_CMD" ]; then
    INNER="${INNER}
echo '--- PHASE A: single-GPU pre-build (serialise dataset build, no W&B) ---'
${PREBUILD_CMD} || { echo \"PRE-BUILD FAILED rc=\$?\"; exit 20; }"
fi
INNER="${INNER}
echo '--- TRAIN: ${IGC_GPUS} GPU sharding=${IGC_SHARDING} -> ${METRIC_REPORT} ---'
${TRAIN_CMD}"

docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
# shellcheck disable=SC2086  # MODELS_MOUNT must word-split into a -v arg (or be empty)
docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    --name "$CONTAINER" \
    -v "${IGC_CODE_DIR}:/workspace/igc" \
    -v "${IGC_DATA_DIR}:/root/.json_responses:ro" \
    $MODELS_MOUNT \
    -e "NCCL_CUMEM_ENABLE=${NCCL_CUMEM_ENABLE}" \
    -e "NCCL_MNNVL_ENABLE=${NCCL_MNNVL_ENABLE}" \
    -w /workspace/igc \
    "$IGC_IMAGE" bash -lc "$INNER"
rc=$?
log "run exited rc=${rc} (output=${IGC_OUTPUT_DIR}, W&B name=${WANDB_NAME}, mode=${RUN_WANDB_MODE})"
exit "$rc"

# Author: Mus mbayramo@stanford.edu
