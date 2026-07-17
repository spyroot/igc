# Smoke ladder — from offline gate to a full GB300 run

Each rung is a gate: a run may not move up until the rung below passes. The ladder exists
because the expensive failures (broken resume, corrupt sharded checkpoints, a mis-matched
tokenizer) are cheap to catch low and ruinous to discover hours into a cluster job.

Conventions: the offline gate runs `make gate` from the repo root (configured by
`pytest.ini`, which registers and excludes the `gpu`/`slow`/`download`/`dataset`/`live`
markers and sets the macOS OpenMP guard used by Torch tests). Cluster steps read the
fleet dashboard address from `NV72_FLEET_DASHBOARD_URL`
(an internal URL kept out of this repo; see the team's ops notes) and are gated by
`scripts/preflight_nv72.sh`, which pipes `/api/v1/state` through the offline-tested
checker in `igc/shared/nv72_preflight.py`.

## R0 — offline gate (every change)

```bash
make gate PYTHON=python
```

## R1 — CPU mini-train (before any GPU time)

A 20-step Phase 1 RedfishBackbone run on the tiny dataset (`--device cpu`, `--num_workers 2`).
The checkpoint role is `model_x`; launch profiles use `phase1_*` names, not historical aliases.
Gates: loss decreases; a checkpoint is written; **the run resumes from its own checkpoint**
(kill it, restart, confirm the epoch advances instead of restarting at zero).

## R2 — 1 GPU, gpt2 smoke profile

`phase1_gpt2_smoke` via `scripts/run_profile.sh` on one GB300. Gates: fleet preflight passes;
the W&B run appears (single run — not one per rank); `kill -USR1` produces a resumable
checkpoint; throughput is recorded.

## R3 — 1 GPU, modern backbone (LoRA)

The `phase1_local` profile (weights dir from `IGC_MODEL_DIR`) or a Qwen2.5 profile, 50 steps,
**on a dataset rebuilt for that backbone** (`--recreate_dataset`; the tokenizer-provenance
guard in the dataset loader refuses mismatched caches). Gates: the vocabulary assertion
passes (no `safe_resize` refusal); loss decreases; checkpoint round-trips.

## R4 — 4 GPU, FSDP2 sharded

`accelerate launch --num_processes 4` with `--sharding fsdp` (multi-node: `IGC_NODES=N`,
rendezvous via the first job node). The fabric is reported healthy (2026-07-10); NVLS is on
by default and `IGC_NCCL_NVLS=0` is the fallback if the microbench disagrees. Gates: an NCCL all-reduce microbench completes
first; the sharded run saves AND reloads a checkpoint (the collective-gather path);
single-process launches of a sharded config must fail loudly (by design).

## R5 — full run

Only after R4: the real profile, full step budget, checkpoint rotation on, end-of-job
weight publishing to shared storage, disk-space preflight green.

Author:
Mus mbayramo@stanford.edu
