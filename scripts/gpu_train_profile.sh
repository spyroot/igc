#!/usr/bin/env bash
# Sweep the per-section training-step profiler across GPU counts / batch sizes on a GB300 node.
#
# Mirrors scripts/gpu_dataset_isolation.sh: one profiled configuration per accelerate launch, each
# writing profile_summary.json under a per-config dir, so you can compare the forward/backward/
# optimizer/comms split and samples/sec as you scale 4 -> 8 GPUs or change batch/precision. This is
# the "profile each section on multi-GPU, see where we can do better" pass before a real run.
#
# Usage (on the node, inside the igc container/env):
#   scripts/gpu_train_profile.sh                       # default sweep: {4,8} GPU x {4,8} batch, bf16
#   GPUS="4" BATCHES="8 16" MODEL=gpt2 scripts/gpu_train_profile.sh
#
# Env knobs:
#   GPUS      space-separated world sizes to try   (default "4 8")
#   BATCHES   space-separated per-rank batch sizes (default "4 8")
#   MODEL     backbone model_type/id               (default gpt2)
#   SEQ_LEN   sequence length                      (default 1024)
#   PRECISION fp32|fp16|bf16                        (default bf16)
#   STEPS / WARMUP                                  (default 20 / 5)
#   OUT       output root                          (default ./profile_out)
#
# Author:
# Mus mbayramo@stanford.edu
set -euo pipefail

GPUS="${GPUS:-4 8}"
BATCHES="${BATCHES:-4 8}"
MODEL="${MODEL:-gpt2}"
SEQ_LEN="${SEQ_LEN:-1024}"
PRECISION="${PRECISION:-bf16}"
STEPS="${STEPS:-20}"
WARMUP="${WARMUP:-5}"
OUT="${OUT:-./profile_out}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ngpu="$(python -c 'import torch;print(torch.cuda.device_count())' 2>/dev/null || echo 0)"
echo "node has ${ngpu} visible CUDA device(s); sweeping GPUS='${GPUS}' BATCHES='${BATCHES}'"

for g in $GPUS; do
    if [ "$g" -gt "$ngpu" ]; then
        echo "SKIP world=${g}: node has only ${ngpu} GPU(s)"
        continue
    fi
    for b in $BATCHES; do
        tag="w${g}_b${b}_${PRECISION}"
        odir="${OUT}/${tag}"
        echo "===== profile ${tag} (${g} GPU x batch ${b}) ====="
        accelerate launch --multi_gpu --num_processes "$g" --num_machines 1 --mixed_precision "$PRECISION" \
            "${here}/scripts/gpu_train_profile.py" \
            --model_type "$MODEL" --batch_size "$b" --seq_len "$SEQ_LEN" \
            --precision "$PRECISION" --steps "$STEPS" --warmup "$WARMUP" \
            --trace --output_dir "$odir" || echo "FAILED ${tag} (see output above)"
    done
done

echo ""
echo "===== sweep summary (samples/sec + section split) ====="
python - "$OUT" <<'PY'
import json, os, sys
root = sys.argv[1]
rows = []
for d in sorted(os.listdir(root)) if os.path.isdir(root) else []:
    p = os.path.join(root, d, "profile_summary.json")
    if not os.path.exists(p):
        continue
    s = json.load(open(p))
    sec = s["sections"]
    rows.append((d, s["samples_per_sec"], s["step_ms"]["mean_ms"],
                 sec["forward"]["mean_ms"], sec["backward"]["mean_ms"],
                 sec["optimizer"]["mean_ms"], s.get("peak_mem_gb")))
hdr = ("config", "samp/s", "step_ms", "fwd_ms", "bwd_ms", "opt_ms", "mem_gb")
print("{:<16}{:>10}{:>10}{:>9}{:>9}{:>9}{:>9}".format(*hdr))
for r in rows:
    mem = f"{r[6]:.2f}" if r[6] is not None else "-"
    print("{:<16}{:>10.1f}{:>10.2f}{:>9.2f}{:>9.2f}{:>9.2f}{:>9}".format(
        r[0], r[1], r[2], r[3], r[4], r[5], mem))
PY
