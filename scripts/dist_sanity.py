"""Distributed backward-pass sanity check for the GB300 fabric.

Confirms the parallelism we intend to scale with — **DDP** and **FSDP2** — actually
works on this fabric, with a SINGLE gradient pass per mode (one forward → backward →
optimiser step is enough: it exercises NCCL init, the wrap, the backward collective
[DDP all-reduce / FSDP2 reduce-scatter], and the optimiser step). Run under torchrun
for 1, 4, 8 … ranks (8 = 2 nodes, i.e. multi-node — the RoCE/NCCL cross-node path).

Pure torch, no igc imports: a failure here points at the fabric / NCCL / launch wiring
(or a broken FSDP2 install), not at the model or dataset. This is the go/no-go gate
before 72 GPUs, and it tells us which parallelism to commit to.

Used by: scripts/gb300_sanity_check.sh (loops SANITY_MODE over ddp,fsdp2 × 1/4/8 GPU).

Env knobs: SANITY_MODE=ddp|fsdp2 (default ddp), SANITY_STEPS (default 1 = single pass),
SANITY_DIM, SANITY_BATCH.

Author:
Mus mbayramo@stanford.edu
"""

import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


def _wrap_fsdp2(model: nn.Module) -> nn.Module:
    """Shard the model with the FSDP2 (`fully_shard`) API, tolerant of torch version."""
    try:
        from torch.distributed.fsdp import fully_shard
    except ImportError:  # older torch keeps it under the composable namespace
        from torch.distributed._composable.fsdp import fully_shard
    for layer in model:  # shard each param-bearing submodule, then the root
        if any(True for _ in layer.parameters(recurse=False)):
            fully_shard(layer)
    fully_shard(model)
    return model


def main() -> None:
    mode = os.environ.get("SANITY_MODE", "ddp").lower()
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = "RANK" in os.environ or world > 1

    if not torch.cuda.is_available():
        raise SystemExit("[sanity] FAIL: CUDA not available in the container")
    if mode not in ("ddp", "fsdp2"):
        raise SystemExit(f"[sanity] FAIL: unknown SANITY_MODE={mode} (ddp|fsdp2)")

    if distributed:
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    dim = int(os.environ.get("SANITY_DIM", "4096"))
    batch = int(os.environ.get("SANITY_BATCH", "32"))
    steps = int(os.environ.get("SANITY_STEPS", "1"))  # one pass confirms the mechanism

    model = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim)).to(device)
    if mode == "ddp":
        model = DDP(model, device_ids=[local_rank]) if distributed else model
    else:  # fsdp2
        model = _wrap_fsdp2(model)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x = torch.randn(batch, dim, device=device)

    torch.cuda.synchronize()
    t0 = time.time()
    last_loss = float("nan")
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = model(x).pow(2).mean()
        loss.backward()  # DDP all-reduce / FSDP2 reduce-scatter happens here
        bad = [n for n, p in model.named_parameters()
               if p.grad is not None and not torch.isfinite(p.grad).all()]
        if bad:
            raise SystemExit(f"[sanity] FAIL rank {rank} mode={mode}: non-finite grad in {bad[:3]}")
        opt.step()
        last_loss = float(loss)
    torch.cuda.synchronize()
    dt = time.time() - t0

    # A collective with a KNOWN answer: sum(0..world-1). A hung/mis-wired fabric
    # deadlocks or returns the wrong value HERE, not deep into a real run.
    if distributed and world > 1:
        probe = torch.tensor([float(rank)], device=device)
        dist.all_reduce(probe, op=dist.ReduceOp.SUM)
        expected = world * (world - 1) / 2.0
        if abs(probe.item() - expected) > 1e-3:
            raise SystemExit(f"[sanity] FAIL rank {rank} mode={mode}: all_reduce={probe.item()} expected={expected}")

    print(f"[sanity] mode={mode} rank={rank}/{world} local={local_rank} "
          f"{torch.cuda.get_device_name(local_rank)} loss={last_loss:.4f} "
          f"{steps}pass {dt*1000:.0f}ms collective=OK", flush=True)

    if distributed:
        dist.barrier()
        if rank == 0:
            print(f"[sanity] PASS mode={mode} world={world}: {mode} backward + collective healthy", flush=True)
        dist.destroy_process_group()
    else:
        print(f"[sanity] PASS mode={mode} world=1: single-GPU backward healthy", flush=True)


if __name__ == "__main__":
    main()


# Author: Mus mbayramo@stanford.edu
