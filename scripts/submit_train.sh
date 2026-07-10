#!/usr/bin/env bash
# Preflight + submit helper for igc training on the GB300 Slurm lab. Run this FROM a
# cluster node (e.g. ssh nvidia@172.25.230.40). It checks the queue, finds a genuinely
# idle node that is NOT one of the hands-off nodes, warns if Flash is using its GPUs,
# then submits scripts/train_igc.sbatch for the chosen stage. See GPU_ACCESS.md.
#
#   ./scripts/submit_train.sh m1                 # smoke: gpt2 state encoder -> wandb
#   IGC_MODEL=<hf-id> IGC_USE_PEFT=1 EPOCHS=3 ./scripts/submit_train.sh m1
#   ./scripts/submit_train.sh m6 -w gb300-poc1-slot7   # pin to a node where data is staged
#
# Never targets slot2 (Flash), slot15/.55 (Pro, TP=4 — busy), slot16 (down). slot1/.41 is also down.

set -euo pipefail

STAGE="${1:-m1}"
shift || true
EXTRA_SBATCH_ARGS=("$@")          # e.g. -w gb300-poc1-slot7
EXCLUDE="gb300-poc1-slot2,gb300-poc1-slot15,gb300-poc1-slot16"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SBATCH_FILE="${HERE}/scripts/train_igc.sbatch"

case "${STAGE}" in
    m1|m2|m3|m3p|m6|agent|state_encoder|autoencoder|goal|parameter|rl|all) : ;;
    *) echo "ERROR: stage '${STAGE}' is not one of m1|m2|m3|m3p|m6|all" >&2; exit 2 ;;
esac

# Fleet-health gate (TEAM_GUIDE blocker rule): never submit against an unhealthy fleet.
# Requires NV72_FLEET_DASHBOARD_URL in the environment; IGC_SKIP_PREFLIGHT=1 is the
# explicit, logged escape hatch for operator judgment calls.
if [[ "${IGC_SKIP_PREFLIGHT:-0}" != "1" ]]; then
    "${HERE}/scripts/preflight_nv72.sh" || {
        echo "BLOCKER: fleet preflight failed — fix the fleet or set IGC_SKIP_PREFLIGHT=1 deliberately." >&2
        exit 1
    }
else
    echo "WARNING: IGC_SKIP_PREFLIGHT=1 — submitting without the fleet-health gate." >&2
fi

command -v sbatch >/dev/null || { echo "ERROR: sbatch not found — run this from a GB300 node (ssh nvidia@172.25.230.40)." >&2; exit 1; }
[ -f "${SBATCH_FILE}" ] || { echo "ERROR: ${SBATCH_FILE} not found." >&2; exit 1; }

echo "== queue (empty = wide open) =="
squeue || true
echo "== per-node GRES + state =="
sinfo -o "%n %G %t" || true

echo
echo "Reminder: Slurm cannot see the out-of-band Flash/Pro containers. Before a long run, confirm your"
echo "target node is truly free:  ssh nvidia@<node> nvidia-smi   (a Flash replica sits on its GPU 0)."
echo "Excluding: ${EXCLUDE}"
echo

if [ "${WANDB_API_KEY:-}" = "" ] && [ ! -f "${HERE}/.internal/wandb.env" ]; then
    echo "WARN: no WANDB_API_KEY in env and no ${HERE}/.internal/wandb.env — set one or pass IGC_REPORT=tensorboard." >&2
fi

set -x
IGC_STAGE="${STAGE}" sbatch \
    --job-name="igc-${STAGE}" \
    --exclude="${EXCLUDE}" \
    "${EXTRA_SBATCH_ARGS[@]}" \
    "${SBATCH_FILE}"
set +x

echo
echo "Watch:   squeue --me ;  tail -f igc-${STAGE}-*.out"
echo "Cancel:  scancel <jobid>"
echo "W&B:     https://wandb.ai/spyroot/igc  (run name: ${STAGE}-...)"
