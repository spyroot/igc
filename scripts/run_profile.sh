#!/usr/bin/env bash
# Run a Phase 1, 2, or 3 SFT experiment by profile name.
#
# Public-safe: this script contains no endpoints, hosts, credentials, or private
# paths. Data and output locations come from the environment, and training flags
# come from the named profile in configs/training/profiles.yaml.
#
# Usage:
#   IGC_PROFILE=phase1_7b_rslora_r32 \
#   IGC_DATA_DIR=~/.json_responses \
#   IGC_OUTPUT_DIR=experiments/phase1_7b_rslora_r32 \
#   bash scripts/run_profile.sh
#
#   # override a profile field, or pass extra igc_main flags after --:
#   IGC_PROFILE=phase1_3b_lora IGC_SET="batch_size=16 lr=2e-4" bash scripts/run_profile.sh -- --recreate_dataset
#   IGC_PROFILE=phase1_gpt2_smoke IGC_SET="max_steps=10" bash scripts/run_profile.sh
#
# Profile names are defined only in configs/training/profiles.yaml.
set -euo pipefail

PROFILE="${IGC_PROFILE:?set IGC_PROFILE (e.g. phase1_7b_rslora_r32)}"
GPUS="${IGC_GPUS:-1}"
MASTER_PORT="${IGC_MASTER_PORT:-29500}"
DATA_DIR="${IGC_DATA_DIR:-$HOME/.json_responses}"
CORPUS_DIR="${IGC_CORPUS_DIR:-}"
CORPUS_EVAL_DIR="${IGC_CORPUS_EVAL_DIR:-}"
SFT_DATA_PATH="${IGC_SFT_DATA_PATH:-}"
SFT_EVAL_DATA_PATH="${IGC_SFT_EVAL_DATA_PATH:-}"
SFT_DATA_MANIFEST="${IGC_SFT_DATA_MANIFEST:-}"
SFT_EVAL_MANIFEST="${IGC_SFT_EVAL_MANIFEST:-}"
OUT_DIR="${IGC_OUTPUT_DIR:-experiments/${PROFILE}}"
METRIC_REPORT="${IGC_METRIC_REPORT:-wandb}"

case "$GPUS" in
    ''|*[!0-9]*) echo "IGC_GPUS must be an integer >= 1" >&2; exit 2 ;;
    *) [ "$GPUS" -ge 1 ] || { echo "IGC_GPUS must be an integer >= 1" >&2; exit 2; } ;;
esac
case "$MASTER_PORT" in
    ''|*[!0-9]*) echo "IGC_MASTER_PORT must be an integer TCP port" >&2; exit 2 ;;
esac
[ "$MASTER_PORT" -ge 1 ] && [ "$MASTER_PORT" -le 65535 ] \
    || { echo "IGC_MASTER_PORT must be an integer TCP port" >&2; exit 2; }

if [ -n "${IGC_CORPUS_OBJECTIVE:-}" ]; then
    echo "IGC_CORPUS_OBJECTIVE is retired; use IGC_SET=\"corpus_objective=...\" so the resolved profile emits one objective flag." >&2
    exit 2
fi

if [ "$METRIC_REPORT" = "wandb" ]; then
    : "${WANDB_API_KEY:?the remote runtime must inject WANDB_API_KEY for W&B reporting}"
fi

# Optional per-run overrides: IGC_SET="batch_size=16 lr=2e-4"
SET_ARGS=()
for kv in ${IGC_SET:-}; do SET_ARGS+=(--set "$kv"); done
CORPUS_ARGS=()
if [ -n "$CORPUS_DIR" ]; then
    if [ -z "$CORPUS_EVAL_DIR" ]; then
        echo "IGC_CORPUS_EVAL_DIR is required with IGC_CORPUS_DIR" >&2
        exit 2
    fi
    CORPUS_ARGS+=(--corpus_dir "$CORPUS_DIR")
    CORPUS_ARGS+=(--corpus_eval_dir "$CORPUS_EVAL_DIR")
fi
if [ -n "$SFT_DATA_PATH" ]; then
    if [ -z "$SFT_EVAL_DATA_PATH" ]; then
        echo "IGC_SFT_EVAL_DATA_PATH is required with IGC_SFT_DATA_PATH" >&2
        exit 2
    fi
    if [ -z "$SFT_DATA_MANIFEST" ] || [ -z "$SFT_EVAL_MANIFEST" ]; then
        echo "IGC_SFT_DATA_MANIFEST and IGC_SFT_EVAL_MANIFEST are required with IGC_SFT_DATA_PATH" >&2
        exit 2
    fi
    CORPUS_ARGS+=(--sft_data_path "$SFT_DATA_PATH")
    CORPUS_ARGS+=(--sft_eval_data_path "$SFT_EVAL_DATA_PATH")
    CORPUS_ARGS+=(--sft_data_manifest "$SFT_DATA_MANIFEST")
    CORPUS_ARGS+=(--sft_eval_manifest "$SFT_EVAL_MANIFEST")
fi

echo "== resolved profile: ${PROFILE} =="
python -m igc.modules.train.launch --profile "$PROFILE" "${SET_ARGS[@]}"

# shellcheck disable=SC2046  # intentional word-splitting of the resolved argv
ARGV=$(python -m igc.modules.train.launch --profile "$PROFILE" "${SET_ARGS[@]}" --print-argv)

mkdir -p "$OUT_DIR"
echo "== launching igc_main.py (data=${DATA_DIR}, out=${OUT_DIR}) =="
if [ "$GPUS" -gt 1 ]; then
    RUNNER=(
        accelerate launch
        --num_processes "$GPUS"
        --num_machines 1
        --machine_rank 0
        --main_process_ip 127.0.0.1
        --main_process_port "$MASTER_PORT"
        igc_main.py
        --use_accelerator
    )
else
    RUNNER=(python igc_main.py)
fi
# shellcheck disable=SC2086  # ARGV is a shell-form argv emitted by the launcher.
exec "${RUNNER[@]}" $ARGV \
  --json_data_dir "$DATA_DIR" \
  "${CORPUS_ARGS[@]}" \
  --output_dir "$OUT_DIR" \
  --metric_report "$METRIC_REPORT" \
  "$@"
