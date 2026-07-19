"""
Multi-GPU per-section profiler for the Phase 1 RedfishBackbone training step.

igc has offline CPU hot-path benchmarks (``scripts/bench_hot_paths.py`` — the RL decision path)
and a distributed dataset-isolation harness (``scripts/gpu_dataset_isolation.py`` — the epoch-2
deadlock guard), but nothing that answers "where does the GPU time actually go in one training
step, on 4 or 8 GPUs?" This script fills that gap: it reconstructs the exact training step of
``LlmEmbeddingsTrainer._train`` (data move -> forward -> backward incl. gradient sync -> optimizer)
with the same ``Accelerator.prepare`` / ``accelerator.backward`` path, and times each section with
CUDA events over N steps, plus a ``torch.profiler`` op-level trace.

It is offline-safe: the backbone is built from an ``AutoConfig`` (random weights) so no HF download
is needed — profiling times the compute+comms, which do not depend on weight values. The batch is
synthetic (real shape ``batch x seq_len``); this isolates the compute/comms step from the data
pipeline, whose throughput is measured separately by the dataset-isolation harness / a real run.

Used by: run on a GB300 node before a real training run to size batch/precision/grad-accum and to
see the forward/backward/optimizer/comms split. Not imported by any runtime module; a standalone
operator tool. See ``scripts/gpu_train_profile.sh`` for the 4- and 8-GPU sweep recipe.

Run (one process, smoke): ``python scripts/gpu_train_profile.py --steps 5 --warmup 2``
Run (4 GPU, faithful):    ``accelerate launch --multi_gpu --num_processes 4 \
                              scripts/gpu_train_profile.py --model_type gpt2 --batch_size 8 \
                              --seq_len 1024 --precision bf16 --steps 20 --warmup 5``

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from contextlib import nullcontext
from typing import Dict, List, Optional

import torch


# The training step decomposes into exactly these sections (mirrors _train's inner loop). "backward"
# includes the gradient all-reduce (DDP) / reduce-scatter (FSDP) since that comm overlaps backward.
SECTIONS = ("data", "forward", "backward", "optimizer")


def _dtype(precision: str) -> torch.dtype:
    """Map a --precision string to a torch dtype (fp32 default)."""
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]


def build_backbone(model_type: str, seq_len: int):
    """Build a CausalLM backbone from config only (random weights, no download).

    Timing the forward/backward is independent of the trained weight values, so a config-built
    model gives a faithful compute profile while staying fully offline.

    :param model_type: HF model type, e.g. ``gpt2`` (or a repo id whose config is cached).
    :param seq_len: max position embeddings the profiled sequence needs to fit.
    :return: an ``AutoModelForCausalLM`` on CPU (the caller / accelerate moves it to device).
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    # Build from a known type so no network is touched; widen positions to cover seq_len for the
    # legacy learned-positional backbones (a GPT-2 with n_positions < seq_len would index-error —
    # exactly the 1024 limit D-003 retires; here we just size the config to the requested seq_len).
    try:
        config = AutoConfig.from_pretrained(model_type, local_files_only=True)
    except Exception:
        config = AutoConfig.for_model(model_type)
    for attr in ("n_positions", "max_position_embeddings"):
        if hasattr(config, attr) and getattr(config, attr) < seq_len:
            setattr(config, attr, seq_len)
    return AutoModelForCausalLM.from_config(config)


def _synthetic_batch(batch_size: int, seq_len: int, vocab_size: int, device) -> Dict[str, torch.Tensor]:
    """A real-shape synthetic batch (input_ids/attention_mask/labels), all on device.

    Deterministic (fixed generator) so the profile is reproducible across ranks and runs.
    """
    gen = torch.Generator().manual_seed(0)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), generator=gen)
    batch = {
        "input_ids": input_ids.to(device),
        "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long, device=device),
        "labels": input_ids.to(device),
    }
    return batch


class _Timers:
    """Per-section CUDA-event timers; falls back to wall clock on CPU."""

    def __init__(self, use_cuda: bool):
        self.use_cuda = use_cuda
        self.samples: Dict[str, List[float]] = {s: [] for s in SECTIONS}
        self.step_samples: List[float] = []

    def time(self, section: str):
        return _Section(self, section)


class _Section:
    def __init__(self, timers: _Timers, name: str):
        self.t, self.name = timers, name

    def __enter__(self):
        if self.t.use_cuda:
            self.start = torch.cuda.Event(enable_timing=True)
            self.end = torch.cuda.Event(enable_timing=True)
            self.start.record()
        else:
            self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self.t.use_cuda:
            self.end.record()
            torch.cuda.synchronize()
            self.t.samples[self.name].append(self.start.elapsed_time(self.end))  # ms
        else:
            self.t.samples[self.name].append((time.perf_counter() - self.start) * 1e3)
        return False


def run(args) -> Optional[dict]:
    """Profile ``args.steps`` training steps and return the rank-0 summary dict (None off rank 0)."""
    from accelerate import Accelerator
    from accelerate.utils import set_seed

    set_seed(0)
    accelerator = Accelerator(mixed_precision=None if args.precision == "fp32" else args.precision)
    device = accelerator.device
    use_cuda = device.type == "cuda"
    is_main = accelerator.is_main_process
    world = accelerator.num_processes

    model = build_backbone(args.model_type, args.seq_len)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model, optimizer = accelerator.prepare(model, optimizer)
    vocab_size = getattr(getattr(model, "config", None), "vocab_size", 50257)
    batch = _synthetic_batch(args.batch_size, args.seq_len, vocab_size, device)

    if is_main:
        print(f"[profile] world={world} device={device} precision={args.precision} "
              f"model={args.model_type} batch={args.batch_size} seq_len={args.seq_len} "
              f"warmup={args.warmup} steps={args.steps}", flush=True)

    def one_step(timers: Optional[_Timers]):
        """A single train step mirroring _train (accumulate -> fwd -> backward -> step)."""
        ctx = timers.time if timers else (lambda _n: nullcontext())
        model.train()
        with accelerator.accumulate(model):
            with ctx("data"):
                moved = {k: v for k, v in batch.items()}  # already on device; marks the section
            with ctx("forward"):
                loss = model(**moved).loss
            with ctx("backward"):
                accelerator.backward(loss)
            with ctx("optimizer"):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        return loss

    # Warmup — allocator, cudnn autotune, and (FSDP) the first all-gather are not representative.
    for _ in range(args.warmup):
        one_step(None)
    if use_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)

    timers = _Timers(use_cuda)
    profiler_ctx = nullcontext()
    prof = None
    if args.trace and is_main:
        from torch.profiler import ProfilerActivity, profile
        acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if use_cuda else [])
        prof = profile(activities=acts, record_shapes=False, profile_memory=True, with_stack=False)
        profiler_ctx = prof

    with profiler_ctx:
        for _ in range(args.steps):
            step_start = time.perf_counter()
            one_step(timers)
            if use_cuda:
                torch.cuda.synchronize()
            timers.step_samples.append((time.perf_counter() - step_start) * 1e3)
            if prof is not None:
                prof.step()

    if not is_main:
        return None

    # Per-section + throughput summary (rank 0). Throughput counts the whole cluster's samples.
    def stats(xs: List[float]) -> Dict[str, float]:
        return {"mean_ms": statistics.mean(xs), "p50_ms": statistics.median(xs),
                "max_ms": max(xs), "min_ms": min(xs)}

    step_ms = statistics.mean(timers.step_samples)
    global_bs = args.batch_size * world
    summary = {
        "world": world, "precision": args.precision, "model_type": args.model_type,
        "batch_size_per_rank": args.batch_size, "global_batch_size": global_bs,
        "seq_len": args.seq_len, "steps": args.steps,
        "step_ms": stats(timers.step_samples),
        "samples_per_sec": global_bs / (step_ms / 1e3),
        "sections": {s: stats(timers.samples[s]) for s in SECTIONS},
        "peak_mem_gb": (torch.cuda.max_memory_allocated(device) / 1e9) if use_cuda else None,
    }

    print("\n===== per-section timing (mean ms/step, rank 0) =====")
    for s in SECTIONS:
        m = summary["sections"][s]["mean_ms"]
        pct = 100.0 * m / sum(summary["sections"][x]["mean_ms"] for x in SECTIONS)
        print(f"  {s:10s} {m:8.2f} ms  ({pct:4.1f}%)")
    print(f"  {'STEP':10s} {step_ms:8.2f} ms/step")
    print(f"\n  throughput : {summary['samples_per_sec']:.1f} samples/sec (global, {world} rank(s))")
    if summary["peak_mem_gb"] is not None:
        print(f"  peak mem   : {summary['peak_mem_gb']:.2f} GB/rank")

    if prof is not None:
        sort_key = "cuda_time_total" if use_cuda else "cpu_time_total"
        print("\n===== torch.profiler top ops =====")
        print(prof.key_averages().table(sort_by=sort_key, row_limit=15))
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            prof.export_chrome_trace(os.path.join(args.output_dir, "trace.json"))

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "profile_summary.json"), "w") as fh:
            json.dump(summary, fh, indent=2)
        print(f"\n  wrote {args.output_dir}/profile_summary.json", flush=True)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model_type", default="gpt2", help="HF model type/id (config-built, no download)")
    ap.add_argument("--batch_size", type=int, default=8, help="per-rank batch size")
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    ap.add_argument("--steps", type=int, default=20, help="profiled optimizer steps")
    ap.add_argument("--warmup", type=int, default=5, help="unprofiled warmup steps")
    ap.add_argument("--trace", action="store_true", help="also capture a torch.profiler op trace")
    ap.add_argument("--output_dir", default="", help="write profile_summary.json (+ trace.json)")
    run(ap.parse_args())


if __name__ == "__main__":
    main()


# Author: Mus mbayramo@stanford.edu
