"""Distributed dataset-feed sanity for igc.

Loads the built igc dataset, splits it across ranks with a DistributedSampler, iterates
EVERY batch, and uploads each to the GPU — exactly what training does
(``llm_train_state_encoder`` moves ``batch['input_ids']`` / ``['attention_mask']``
``.to(device)`` each step). It confirms multi-GPU data feeding — load, split, concurrent
read, GPU upload — works in ISOLATION on 1/4 GPU before a real fine-tune.

Yes, we DO upload to GPU: the ``.pt`` dataset cache is read into CPU RAM, and each batch's
tensors are moved to ``cuda:LOCAL_RANK`` here, per step, same as the trainer. Pure data
path (no model / backward) so a failure points at loading / splitting / reading, not at
training.

Used by: scripts/gb300_sanity_check.sh with SANITY_SCRIPT=scripts/dataset_sanity.py.
Needs a pre-built dataset cache at IGC_DATASET_DIR (default /workspace/igc/datasets).

Env: IGC_DATASET_DIR, IGC_MODEL (tokenizer key), SANITY_BATCH, SANITY_WORKERS, SANITY_SEQ.

Author:
Mus mbayramo@stanford.edu
"""

import os
import sys
import time

# torchrun runs this from scripts/, so scripts/ (not the repo root) is on sys.path;
# add the repo root so `import igc` resolves inside the container.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler


def _upload(batch, device) -> int:
    """Move every tensor in the batch to the GPU (as the trainer does); return count moved."""
    moved = 0
    if isinstance(batch, dict):
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(device, non_blocking=True)
                moved += 1
    elif isinstance(batch, (list, tuple)):
        for v in batch:
            if torch.is_tensor(v):
                v.to(device, non_blocking=True)
                moved += 1
    elif torch.is_tensor(batch):
        batch.to(device, non_blocking=True)
        moved = 1
    return moved


def _batch_size(batch) -> int:
    t = None
    if isinstance(batch, dict):
        t = next((v for v in batch.values() if torch.is_tensor(v)), None)
    elif isinstance(batch, (list, tuple)):
        t = next((v for v in batch if torch.is_tensor(v)), None)
    elif torch.is_tensor(batch):
        t = batch
    return int(t.shape[0]) if t is not None else 0


def main() -> None:
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = "RANK" in os.environ or world > 1

    if not torch.cuda.is_available():
        raise SystemExit("[ds-sanity] FAIL: CUDA not available in the container")
    if distributed:
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    from igc.ds.redfish_masked_dataset import MaskedJSONDataset
    ds = MaskedJSONDataset(
        os.environ.get("IGC_DATASET_DIR", "/workspace/igc/datasets"),
        default_tokenize=os.environ.get("IGC_MODEL", "gpt2"),
        max_len=int(os.environ.get("SANITY_SEQ", "1024")),
        recreate_dataset=False,
        do_consistency_check=False,
    )

    bs = int(os.environ.get("SANITY_BATCH", "8"))
    workers = int(os.environ.get("SANITY_WORKERS", "4"))
    if distributed and world > 1:
        sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True, drop_last=True)
        loader = DataLoader(ds, batch_size=bs, sampler=sampler, num_workers=workers, pin_memory=True)
    else:
        loader = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=workers, pin_memory=True)

    torch.cuda.synchronize()
    t0 = time.time()
    nb = ns = tensors = 0
    for batch in loader:
        tensors = _upload(batch, device)   # UPLOAD to GPU, exactly like the trainer
        ns += _batch_size(batch)
        nb += 1
    torch.cuda.synchronize()
    dt = time.time() - t0

    thr = ns / dt if dt > 0 else 0.0
    print(f"[ds-sanity] rank={rank}/{world} dataset_len={len(ds)} batches={nb} samples={ns} "
          f"tensors/batch={tensors} {dt:.2f}s {thr:.0f}samp/s upload=OK", flush=True)

    if distributed and world > 1:
        agg = torch.tensor([float(nb), float(ns)], device=device)
        dist.all_reduce(agg, op=dist.ReduceOp.SUM)
        dist.barrier()
        if rank == 0:
            print(f"[ds-sanity] PASS world={world}: {int(agg[0])} batches / {int(agg[1])} samples "
                  f"read+uploaded across all ranks (dataset_len={len(ds)}, split OK)", flush=True)
        dist.destroy_process_group()
    else:
        print(f"[ds-sanity] PASS world=1: {nb} batches / {ns} samples read+uploaded "
              f"(dataset_len={len(ds)})", flush=True)


if __name__ == "__main__":
    main()


# Author: Mus mbayramo@stanford.edu
