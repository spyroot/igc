#!/usr/bin/env bash
# Run an IGC training experiment by PROFILE NAME.
#
# Public-safe: this script contains no endpoints, hosts, credentials, or private
# paths. Data and output locations come from the environment, and the training flags
# come from the named profile in igc.modules.train.profiles (the single source of truth,
# per docs/TRAINING_OPTIMIZATION_PLAN.md).
#
# Usage:
#   IGC_PROFILE=m1_7b_rslora_r32 \
#   IGC_DATA_DIR=~/.json_responses \
#   IGC_OUTPUT_DIR=experiments/m1_7b_rslora_r32 \
#   bash scripts/run_profile.sh
#
#   # override a profile field, or pass extra igc_main flags after --:
#   IGC_PROFILE=m1_3b_lora IGC_SET="batch_size=16 lr=2e-4" bash scripts/run_profile.sh -- --recreate_dataset
#
# Profiles include M1 names from igc.modules.train.profiles and Phase 1/2/3 names
# from configs/phase_training/profiles.yaml.
set -euo pipefail

PROFILE="${IGC_PROFILE:?set IGC_PROFILE (e.g. m1_7b_rslora_r32)}"
DATA_DIR="${IGC_DATA_DIR:-$HOME/.json_responses}"
OUT_DIR="${IGC_OUTPUT_DIR:-experiments/${PROFILE}}"
METRIC_REPORT="${IGC_METRIC_REPORT:-wandb}"

# Auto-load W&B credentials so the run streams to the dashboard with no manual export.
# The env file is gitignored (creds never committed); override the path via IGC_WANDB_ENV.
WANDB_ENV="${IGC_WANDB_ENV:-.internal/wandb.env}"
if [ -f "$WANDB_ENV" ]; then
    set -a
    # shellcheck source=/dev/null
    . "$WANDB_ENV"
    set +a
    echo "== W&B: streaming to \${WANDB_ENTITY:-?}/\${WANDB_PROJECT:-?} (real-time) =="
elif [ "$METRIC_REPORT" = "wandb" ]; then
    echo "== W&B: no \$WANDB_API_KEY and no $WANDB_ENV — set IGC_METRIC_REPORT=tensorboard or provide creds ==" >&2
fi

# Optional per-run overrides: IGC_SET="batch_size=16 lr=2e-4"
SET_ARGS=()
for kv in ${IGC_SET:-}; do SET_ARGS+=(--set "$kv"); done

echo "== resolved profile: ${PROFILE} =="
python -m igc.modules.train.launch --profile "$PROFILE" "${SET_ARGS[@]}"

# shellcheck disable=SC2046  # intentional word-splitting of the resolved argv
ARGV=$(python -m igc.modules.train.launch --profile "$PROFILE" "${SET_ARGS[@]}" --print-argv)

mkdir -p "$OUT_DIR"
echo "== launching igc_main.py (data=${DATA_DIR}, out=${OUT_DIR}) =="
# shellcheck disable=SC2086  # ARGV is a shell-form argv emitted by the launcher.
exec python igc_main.py $ARGV \
  --json_data_dir "$DATA_DIR" \
  --output_dir "$OUT_DIR" \
  --metric_report "$METRIC_REPORT" \
  "$@"
