#!/usr/bin/env bash
# gb300_sanity_check.sh — distributed backward-pass sanity BEFORE committing 72 GPUs.
#
# Runs scripts/dist_sanity.py under torchrun in the NGC/igc-train container for a ladder
# of GPU counts. A GB300 node has 4 GPUs, so: 1 = single proc, 4 = single node, 8 = TWO
# nodes (multi-node — validates the RoCE fabric + cross-node NCCL, the thing that will
# make or break a 72-GPU run). It refuses to run on a node another team is using, and
# prints PASS/FAIL per config; any failure means "do NOT start full training".
#
# Usage (on a node, or from a control host with ssh to the nodes):
#   SANITY_NODES="172.25.230.42 172.25.230.49" scripts/gb300_sanity_check.sh   # runs 1,4,8
#   SANITY_GPUS="1 4" scripts/gb300_sanity_check.sh                            # single-node only
#   IGC_IMAGE=igc-train:ngc26.03-py3 SANITY_NODES="…" scripts/gb300_sanity_check.sh
#
# For the 8-GPU (multi-node) test the code must be visible on BOTH nodes — point
# IGC_CODE_DIR at the shared checkout (/models/igc) so every node sees the same file.
#
# Author:
# Mus mbayramo@stanford.edu
set -uo pipefail

IMAGE="${IGC_IMAGE:-nvcr.io/nvidia/pytorch:26.03-py3}"
IGC_CODE_DIR="${IGC_CODE_DIR:-$HOME/igc}"          # /models/igc for the multi-node test
MODELS_DIR="${MODELS_DIR:-/models}"
GPUS_LADDER="${SANITY_GPUS:-1 4 8}"
MODES="${SANITY_MODES:-ddp fsdp2}"                 # parallelism to confirm, one pass each
SCRIPT="${SANITY_SCRIPT:-scripts/dist_sanity.py}"  # or scripts/dataset_sanity.py (dataset feed)
IGC_MODEL="${IGC_MODEL:-gpt2}"                      # tokenizer key for the dataset pass
IGC_DATASET_DIR="${IGC_DATASET_DIR:-/workspace/igc/datasets}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
PORT="${SANITY_PORT:-29511}"
# shellcheck disable=SC2206  # word-split the space-separated node list on purpose
NODES=(${SANITY_NODES:-$(hostname -I 2>/dev/null | awk '{print $1}')})

# NVIDIA-recommended container flags (from the NGC startup banner) + dual mounts:
# shared /models (240 TB) and node-local code, so a run can write either place.
NV_FLAGS="--gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864"
MOUNTS="-v ${IGC_CODE_DIR}:/workspace/igc -v ${MODELS_DIR}:/models"
SSH="ssh -o BatchMode=yes -o ConnectTimeout=6"

log() { echo "=== [$(date -u '+%F %T')] $* ==="; }

node_free() {  # ip -> 0 if no GPU compute process is running
    local busy
    # grep -c exits 1 on zero matches (a FREE node) while still printing "0", so `|| true`
    # keeps that "0"; an empty result means the ssh itself failed -> treat as busy.
    busy=$($SSH "nvidia@$1" 'nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true' 2>/dev/null)
    [ -n "$busy" ] && [ "$busy" = "0" ]
}

# shellcheck disable=SC2086  # NV_FLAGS/MOUNTS must word-split into separate docker args
run_on() {  # node_ip nnodes node_rank master_ip nproc mode
    $SSH "nvidia@$1" \
        "docker run --rm --network host $NV_FLAGS $MOUNTS \
             -e SANITY_MODE=$6 -e SANITY_STEPS=1 -e SANITY_HUGE_GB=${SANITY_HUGE_GB:-0} \
             -e NCCL_MNNVL_ENABLE=${NCCL_MNNVL_ENABLE:-0} -e IGC_MODEL=$IGC_MODEL -e IGC_DATASET_DIR=$IGC_DATASET_DIR \
             -w /workspace/igc $IMAGE \
             torchrun --nnodes=$2 --node_rank=$3 --nproc_per_node=$5 \
                      --master_addr=$4 --master_port=$PORT $SCRIPT"
}

overall=0
for g in $GPUS_LADDER; do
    need=$(( (g + GPUS_PER_NODE - 1) / GPUS_PER_NODE ))   # nodes required
    nproc=$(( g < GPUS_PER_NODE ? g : GPUS_PER_NODE ))    # procs per node

    if [ "${#NODES[@]}" -lt "$need" ]; then
        echo "  SKIP ${g}-GPU: need ${need} node(s), have ${#NODES[@]} — set SANITY_NODES='ip1 ip2'" >&2
        continue
    fi
    use=("${NODES[@]:0:$need}")

    busy=0
    for ip in "${use[@]}"; do node_free "$ip" || { echo "  BLOCKER: node $ip is busy — skipping ${g}-GPU" >&2; busy=1; }; done
    [ "$busy" = "0" ] || { overall=1; continue; }
    master="${use[0]}"

    # one SINGLE gradient pass per parallelism mode — confirm DDP and FSDP2 both work
    for mode in $MODES; do
        log "sanity: ${g} GPU  mode=${mode}  (${need} node(s), ${nproc}/node, 1 pass)"
        pids=(); rank=0
        for ip in "${use[@]}"; do
            run_on "$ip" "$need" "$rank" "$master" "$nproc" "$mode" &
            pids+=("$!"); rank=$((rank + 1))
        done
        rc=0
        for p in "${pids[@]}"; do wait "$p" || rc=1; done
        if [ "$rc" = "0" ]; then echo "  ${g}-GPU ${mode}: PASS"; else echo "  ${g}-GPU ${mode}: FAIL"; overall=1; fi
    done
done

if [ "$overall" = "0" ]; then
    log "ALL SANITY PASSED — fabric + distributed stack ready for full training"
else
    log "SANITY FAILURES — do NOT start full training until resolved"
fi
exit "$overall"

# Author: Mus mbayramo@stanford.edu
