#!/usr/bin/env bash
# gpu_dataset_isolation.sh — sweep the dataloader/parallelism edge cases in ISOLATION on GPU.
#
# Runs ON a GB300 node. Each case is one `timeout <T> torchrun ... gpu_dataset_isolation.py`
# invocation: a hang (the epoch-boundary save-collective deadlock) becomes a bounded, attributable
# DEADLOCK result instead of a mysterious stall deep in a real 72-GPU run. The headline case is the
# regression: drop_last=0 + an INDIVISIBLE dataset + world>1 must reproduce the deadlock, and
# drop_last=1 must PASS — proving the train-loader drop_last fix.
#
# Pre-flight the fleet dashboard before running (RoCE/BeeGFS/GPU availability). Usage:
#   bash scripts/gpu_dataset_isolation.sh            # default sweep on available GPUs
#   CASE_TIMEOUT=60 WORLDS="1 2 4" bash scripts/gpu_dataset_isolation.sh
set -u

HARNESS="$(dirname "$0")/gpu_dataset_isolation.py"
CASE_TIMEOUT="${CASE_TIMEOUT:-90}"        # seconds; a hang past this = DEADLOCK
ngpu="$(nvidia-smi -L 2>/dev/null | grep -c . || echo 1)"
WORLDS="${WORLDS:-$(for w in 1 2 4; do [ "$w" -le "$ngpu" ] && echo -n "$w "; done)}"

# case: label|world|DS_SIZE|BATCH|DROP_LAST|expected(PASS|DEADLOCK)
CASES=(
  "even_divisible_drop|@|128|8|1|PASS"
  "indivisible_drop_last|@|97|8|1|PASS"          # the FIX: even batches -> no deadlock
  "indivisible_NO_drop_last|@|97|8|0|DEADLOCK"   # the BUG: uneven batches -> save-collective hang
  "tiny_dataset_drop|@|3|4|1|PASS"               # world may exceed samples
  "odd_batch_drop|@|1000|7|1|PASS"
)

pass=0; fail=0; dead=0; mismatch=0
printf "%-26s %-6s %-8s %-9s %s\n" CASE WORLD EXPECT GOT DETAIL
for world in $WORLDS; do
  for spec in "${CASES[@]}"; do
    IFS='|' read -r label _ ds batch drop expect <<<"$spec"
    [ "$world" -eq 1 ] && [ "$expect" = "DEADLOCK" ] && continue   # single-rank can't deadlock
    out=$(DS_SIZE="$ds" BATCH="$batch" DROP_LAST="$drop" \
          timeout "$CASE_TIMEOUT" torchrun --nproc_per_node="$world" --nnodes=1 \
            --rdzv-backend=c10d --rdzv-endpoint=localhost:0 "$HARNESS" 2>&1)
    rc=$?
    if [ "$rc" -eq 124 ]; then got="DEADLOCK"
    elif [ "$rc" -eq 0 ]; then got="PASS"
    else got="FAIL"; fi
    ok="ok"; [ "$got" != "$expect" ] && ok="MISMATCH"
    [ "$ok" = "MISMATCH" ] && mismatch=$((mismatch+1))
    [ "$got" = "PASS" ] && pass=$((pass+1))
    [ "$got" = "DEADLOCK" ] && dead=$((dead+1))
    [ "$got" = "FAIL" ] && fail=$((fail+1))
    printf "%-26s %-6s %-8s %-9s %s\n" "$label" "$world" "$expect" "$got" "$ok"
    if [ "$ok" = "MISMATCH" ] || [ "$got" = "FAIL" ]; then
      echo "    last: $(printf '%s' "$out" | tail -1)"
    fi
  done
done
echo "----"
echo "summary: PASS=$pass DEADLOCK(reproduced)=$dead FAIL=$fail MISMATCH=$mismatch"
# Exit nonzero only if an expectation was violated (a PASS-case deadlocked, or a bug-case passed).
if [ "$mismatch" -ne 0 ]; then
  exit 1
fi
