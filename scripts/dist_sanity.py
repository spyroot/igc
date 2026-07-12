"""Distributed backward-pass sanity check for the GB300 fabric.

Run under torchrun/accelerate for 1, 4, 8, … ranks (8 = 2 nodes, i.e. multi-node).
It validates that the WHOLE distributed training stack works end to end — NCCL init,
forward, backward, gradient all-reduce (DDP), optimiser step — BEFORE 72 GPUs are
committed to a full run. It is pure torch with no igc imports on purpose: a failure
here points at the fabric / NCCL / RoCE / launch wiring, not at the model or dataset.

Used by: scripts/gb300_sanity_check.sh, which launches this under torchrun in the NGC
container for 1 / 4 / 8 GPU. Not imported anywhere else.

Checks, per rank, then collectively:
  * forward + backward run and the loss is finite;
  * every gradient is finite (a NaN/Inf grad fails loudly);
  * a known all-reduce (sum of ranks) returns the exact expected value — a hung or
    mis-wired fabric fails HERE instead of silently deadlocking a real run;
  * throughput (samples/s) is printed so a slow interconnect is visible.

Author:
Mus mbayramo@stanford.edu
"""

import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


def main() -> None:
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world > 1

    if not torch.cuda.is_available():
        raise SystemExit("[sanity] FAIL: CUDA not available in the container")

    if distributed:
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # Non-trivial params so backward moves real all-reduce traffic over the fabric.
    dim = int(os.environ.get("SANITY_DIM", "4096"))
    batch = int(os.environ.get("SANITY_BATCH", "32"))
    steps = int(os.environ.get("SANITY_STEPS", "20"))

    model = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim)).to(device)
    if distributed:
        model = DDP(model, device_ids=[local_rank])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x = torch.randn(batch, dim, device=device)

    torch.cuda.synchronize()
    t0 = time.time()
    last_loss = float("nan")
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = model(x).pow(2).mean()
        loss.backward()
        bad = [n for n, p in model.named_parameters()
               if p.grad is not None and not torch.isfinite(p.grad).all()]
        if bad:
            raise SystemExit(f"[sanity] FAIL rank {rank}: non-finite grad in {bad[:3]}")
        opt.step()
        last_loss = loss.item()
    torch.cuda.synchronize()
    dt = time.time() - t0

    # A collective with a KNOWN answer: sum(0..world-1). A hung/mis-wired fabric
    # deadlocks or returns the wrong value HERE, not 20 minutes into a real run.
    if distributed:
        probe = torch.tensor([float(rank)], device=device)
        dist.all_reduce(probe, op=dist.ReduceOp.SUM)
        expected = world * (world - 1) / 2.0
        if abs(probe.item() - expected) > 1e-3:
            raise SystemExit(f"[sanity] FAIL rank {rank}: all_reduce={probe.item()} expected={expected}")

    thr = steps * batch / dt if dt > 0 else 0.0
    print(f"[sanity] rank={rank}/{world} local={local_rank} {torch.cuda.get_device_name(local_rank)} "
          f"loss={last_loss:.4f} {steps}steps {dt:.2f}s {thr:.0f}samp/s allreduce=OK", flush=True)

    if distributed:
        dist.barrier()
        if rank == 0:
            print(f"[sanity] PASS world={world}: backward + all_reduce healthy on all ranks", flush=True)
        dist.destroy_process_group()
    else:
        print("[sanity] PASS world=1: single-GPU backward healthy", flush=True)


if __name__ == "__main__":
    main()


# Author: Mus mbayramo@stanford.edu
