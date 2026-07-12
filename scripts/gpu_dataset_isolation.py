"""GPU isolation harness: reproduce the epoch-boundary save-collective deadlock in ONE case.

The epoch-2 hang only manifests with real multi-GPU + NCCL: with an indivisible dataset and a
train DataLoader WITHOUT drop_last, ranks process a different number of batches, so one rank
reaches the epoch-end collective (broadcast_flag + accelerator.get_state_dict, mirrored here by
an all_reduce) while another is still iterating -> NCCL hangs. This script runs exactly ONE
configuration and EXITS, so the orchestrator's `timeout` turns a hang into a clean, attributable
"DEADLOCK" result instead of a mysterious training stall.

It exercises the real code path — an accelerate-prepared DataLoader built the same way as
``igc/modules/llm_train_state_encoder.py`` (custom sampler optional, drop_last configurable) —
then loops one epoch and hits an epoch-end collective. Pure enough that a failure points at the
dataloader/parallelism wiring, not the model.

Run one case (under torchrun):
    DS_SIZE=97 BATCH=8 DROP_LAST=0 SAMPLER=custom \
      torchrun --nproc_per_node=4 scripts/gpu_dataset_isolation.py

Env: DS_SIZE (dataset length), BATCH, DROP_LAST (0/1), SAMPLER (none|custom), EPOCHS (default 2).
Used by: scripts/gpu_dataset_isolation.sh (sweeps the edge-case matrix with per-case timeouts).

Author:
Mus mbayramo@stanford.edu
"""
import os
import sys

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def main() -> None:
    ds_size = _env_int("DS_SIZE", 97)
    batch = _env_int("BATCH", 8)
    drop_last = _env_int("DROP_LAST", 1) == 1
    epochs = _env_int("EPOCHS", 2)
    sampler_kind = os.environ.get("SAMPLER", "none").lower()

    rank = _env_int("RANK", 0)
    world = _env_int("WORLD_SIZE", 1)
    local_rank = _env_int("LOCAL_RANK", 0)
    distributed = "RANK" in os.environ or world > 1

    if distributed:
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    dataset = TensorDataset(torch.arange(ds_size))

    # Mirror the igc trainer: a DataLoader (with a distributed sampler when multi-rank), then
    # accelerate.prepare wraps it. We build the distributed sampler explicitly so the case is
    # deterministic; drop_last is the variable under test.
    if distributed and world > 1:
        from torch.utils.data import DistributedSampler
        sampler = DistributedSampler(dataset, num_replicas=world, rank=rank,
                                     shuffle=(sampler_kind == "custom"), drop_last=drop_last)
        loader = DataLoader(dataset, batch_size=batch, sampler=sampler, drop_last=drop_last)
    else:
        loader = DataLoader(dataset, batch_size=batch, shuffle=False, drop_last=drop_last)

    try:
        from accelerate import Accelerator
        acc = Accelerator()
        loader = acc.prepare(loader)
    except Exception as exc:  # accelerate optional; the collective below still reproduces the hang
        print(f"[iso] rank {rank}: accelerate.prepare skipped ({exc})", flush=True)

    for epoch in range(epochs):
        nb = 0
        for _ in loader:
            nb += 1
        # THE COLLECTIVE every rank must reach together (mirrors broadcast_flag + get_state_dict
        # at the igc epoch boundary). If ranks saw a different nb, they arrive at different times;
        # a truly divergent count makes this all_reduce hang -> the orchestrator times out.
        if distributed:
            counts = torch.tensor([nb], device=device)
            dist.all_reduce(counts, op=dist.ReduceOp.MAX)
            cmin = torch.tensor([nb], device=device)
            dist.all_reduce(cmin, op=dist.ReduceOp.MIN)
            if rank == 0 and counts.item() != cmin.item():
                print(f"[iso] rank0 epoch {epoch}: UNEVEN batch counts across ranks "
                      f"(min={cmin.item()} max={counts.item()}) — deadlock risk", flush=True)
        print(f"[iso] rank={rank}/{world} epoch={epoch} batches={nb} ds={ds_size} "
              f"batch={batch} drop_last={drop_last}", flush=True)

    if distributed:
        dist.barrier()
        dist.destroy_process_group()
    if rank == 0:
        print(f"[iso] PASS ds={ds_size} batch={batch} world={world} drop_last={drop_last} "
              f"— all ranks reached the epoch-end collective every epoch", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()


# Author: Mus mbayramo@stanford.edu
