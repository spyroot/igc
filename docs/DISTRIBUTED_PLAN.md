# igc distributed-training plan (GB300 NVL72)

Decision record for scaling igc training on the 18-node GB300 fleet. Synthesised from a
DeepSeek-V4-Pro max-reasoning review (2026-07-12) against the verified cluster inventory
and the observed 4-GPU sanity results. Turn this into the launch/train scripts under
`scripts/`; it is the "golden" contract a run must satisfy.

## Cluster facts (verified 2026-07-12)

18 nodes, each: **4× GB300 (~280 GB HBM/GPU), 144 Grace cores, ~2 TB RAM, 1.8 TB local
NVMe**. 72 GPUs total. RoCE (RDMA) + intra-node NVLink. Shared BeeGFS `/models` (240 TB,
metadata-sensitive → prefer packed files). No Slurm from compute nodes (launch via docker
+ torchrun). torch 2.11 (NGC `pytorch:26.03`) + accelerate.

Sanity (single gradient pass): DDP and FSDP2 both pass on 1 and 4 GPU. At 4 GPU, **DDP
~3.0 s vs FSDP2 ~9.0 s** for a model that fits on one GPU — FSDP2 sharding is pure
overhead when the model fits.

## A. Parallelism — the decision

Per-GPU memory ≈ **model-state + activations + peak-gather**. Model-state = **14 bytes/param**
(2 bf16 param + 4 fp32 grad + 8 fp32 Adam m,v). Activations with gradient-checkpointing
≈ `B·S·H·34` bytes. FSDP per-layer gather ≈ `6·12·H²`.

**State-encoder LLM (full fine-tune):**

| Params | Strategy | Why |
| --- | --- | --- |
| ≤ 7 B | **DDP** | 7 B×14 = 98 GB, ~180 GB headroom; fastest, simplest |
| 7–200 B | **FSDP2** (`FULL_SHARD`, per-layer wrap) | model-state ÷ world-size; no offload needed at 72 GPU |
| 200–500 B | **FSDP2 + CPU-offload** (optimizer states → 2 TB RAM) | frees HBM for activations |
| > 500 B | **Pipeline-parallel + FSDP2** | per-GPU param further reduced |

**LoRA:** trainable params are tiny (adapters only); frozen backbone is bf16 (2 bytes/param).
This does **not** make DDP viable to 120 B (120 B bf16 = 240 GB leaves no room for
activations — a think-high error, corrected). For a large LoRA backbone, still **FSDP2** to
shard the frozen weights; skip CPU-offload (no optimizer state to offload).

**Default for igc today:** models we actually fine-tune fit on one GPU → **DDP** (the 3×
speedup is real). Reach for FSDP2 only when the backbone genuinely does not fit.

**RL agent — a different problem.** The DQN+HER Q-net is < 5 M params; gradient
data-parallel of it is pointless (the update is trivial). Scale RL by **environment
throughput, not model sharding**:

* Each GPU runs **vectorised envs** (16–32/GPU, `gymnasium` async) + batched frozen-LLM
  state-encoding on that GPU.
* Transitions go to a **per-node in-RAM replay buffer** (2 TB RAM → ~2 M transitions/node).
* Synchronous DDP all-reduce of the tiny Q-net gradient (~30 MB) across ranks — hidden
  behind replay sampling. Actor-learner **inside** each GPU; `wait_for_everyone()` at each
  training boundary. (If LLM-encoder forward dominates, an async Ape-X actor-learner with a
  distributed replay becomes worth the complexity — revisit with a throughput measurement.)

## B. Dataset / data pipeline

* **Storage:** copy the tokenized dataset to **node-local NVMe** for training reads (local
  SSD speed, zero BeeGFS metadata pressure). `/models` holds the master copy + checkpoints,
  not live training reads.
* **Format:** one **memory-mapped token array + `index.npy`** (`(start, length)` per sample),
  not thousands of small `.pt` files, not WebDataset. Zero metadata stress, random access,
  fits 1.8 TB.
* **Build ONCE, single-process, then distribute** (rsync/`docker load`-style to each node).
  Never let multiple ranks build/write concurrently — that is the dataset build race we hit
  (`igc/ds/redfish_dataset` — fixed to tolerate it, but pre-build is still the rule).
* **DataLoader (per GPU process):** `num_workers=8`, `pin_memory=True`,
  `persistent_workers=True`, `prefetch_factor=4`.

## C. The deadlock invariants (what makes a 72-GPU run survive the night)

Every one of these must hold; each is a one-line guard:

1. **Even batches per rank.** `DistributedSampler(drop_last=True)` **and**
   `DataLoader(drop_last=True)` — ranks must see the same number of batches or the last
   collective is missing a rank. Call `sampler.set_epoch(epoch)` every epoch.
2. **No rank-conditional collective.** A state-dict gather / all-reduce / barrier reached by
   only some ranks deadlocks. Wrap any `if is_main_process:` block that could hit a
   collective with `accelerator.wait_for_everyone()` on both sides, or use
   `accelerator.save_state()`. (This was the epoch-2 hang — no barrier after the rank-0
   save; fixed 2026-07-12.)
3. **Barrier after every manual save.** All ranks meet before the next epoch's first
   collective while rank 0 finishes writing.
4. **Symmetric validation.** `validate()` runs an FSDP-collective forward per eval batch;
   the eval loader must be evenly sharded too, and the accuracy all-reduced (not per-rank).
   Guard the 0-batch shard (return 0.0, never divide by zero).
5. **Fixed grad-accumulation steps** across ranks — no dynamic early-exit.
6. **Checkpoint to `/models`.** Sharded (FSDP) or full, every 500–1000 steps, keep last 3,
   include model+optimizer+scheduler+step. Load with all ranks participating.
7. **Deterministic seeding.** `set_seed(seed, device_specific=True)` and
   `DistributedSampler(seed=...)`.
8. **Elastic launch for recovery.** `torchrun --rdzv_backend=c10d --max-restarts=5`; on
   restart, resume from the latest `/models` checkpoint at the right step.

Verified pieces so far: single-pass DDP + FSDP2 both green on 1/4 GPU (`scripts/dist_sanity.py`);
the epoch-2 barrier fix landed; the dataset feed pass is `scripts/dataset_sanity.py`.
Open: an 8-GPU (2-node) fabric pass, and an FSDP2 pass on a model larger than one GPU.
