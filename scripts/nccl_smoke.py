"""NCCL all-reduce microbench — the entry gate for any multi-GPU rung.

The GB300 fabric has a documented history of NVLink flakiness, so a 4-GPU
training job must not be the first thing to exercise collective comms. Run
this first (30 seconds, R4 rung of docs/SMOKE_LADDER.md):

    torchrun --nproc_per_node 4 scripts/nccl_smoke.py   # IGC_NCCL_NVLS=0 only if this fails

Each rank all-reduces a tensor repeatedly and rank 0 prints the effective
bus bandwidth; any fabric fault surfaces here as a hang or NCCL error in
seconds instead of hours into training. Pure helpers below are offline-
unit-tested; the collective path needs GPUs and torchrun.

Author:
Mus mbayramo@stanford.edu
"""

import argparse
import os
import time


def tensor_megabytes(numel: int, bytes_per_element: int = 4) -> float:
    """Size of the reduced tensor in MiB.

    :param numel: number of elements.
    :param bytes_per_element: element width (fp32 default).
    :return: size in MiB.
    """
    return numel * bytes_per_element / (1024 ** 2)


def busbw_gbps(numel: int, iters: int, seconds: float, world_size: int) -> float:
    """Effective all-reduce bus bandwidth in GB/s (ring algorithm accounting).

    :param numel: elements per all-reduce.
    :param iters: number of all-reduce calls timed.
    :param seconds: total wall time for the timed calls.
    :param world_size: participating ranks.
    :return: bus bandwidth in GB/s (0.0 when inputs are degenerate).
    """
    if seconds <= 0 or world_size <= 1 or iters <= 0:
        return 0.0
    bytes_moved = numel * 4 * iters * 2 * (world_size - 1) / world_size
    return bytes_moved / seconds / 1e9


def main() -> int:
    """Run the collective loop under torchrun; print rank-0 bandwidth."""
    parser = argparse.ArgumentParser(description="NCCL all-reduce microbench")
    parser.add_argument("--numel", type=int, default=64 * 1024 * 1024,
                        help="elements per all-reduce (fp32; default 256 MiB).")
    parser.add_argument("--iters", type=int, default=20, help="timed iterations.")
    parser.add_argument("--warmup", type=int, default=3, help="untimed iterations.")
    args = parser.parse_args()

    import torch
    import torch.distributed as dist

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    payload = torch.ones(args.numel, dtype=torch.float32, device="cuda")

    for _ in range(args.warmup):
        dist.all_reduce(payload)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(args.iters):
        dist.all_reduce(payload)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    if rank == 0:
        print(f"nccl_smoke OK: world={world} "
              f"size={tensor_megabytes(args.numel):.0f}MiB iters={args.iters} "
              f"busbw={busbw_gbps(args.numel, args.iters, elapsed, world):.1f} GB/s")
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# Author: Mus mbayramo@stanford.edu
