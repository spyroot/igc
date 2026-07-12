"""Training loop for the M3 goal planner.

This is a small supervised fine-tuning wrapper for causal language models. It
expects :class:`M3GoalPlanJsonlDataset` records and trains the model to emit
structured M3 goal-plan JSON from instruction plus context.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any

import torch
import torch.distributed as dist
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from igc.ds.m3_goal_plan_dataset import M3GoalPlanJsonlDataset, M3GoalPlanSFTCollator
from igc.modules.m3_goal_planner import M3ModelGoalPlanner


@dataclass
class M3GoalPlannerSFTConfig:
    """Supervised fine-tuning config for M3."""

    output_dir: str
    epochs: int = 1
    batch_size: int = 1
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    optimizer_name: str = "adamw"
    scheduler: str = "constant"
    warmup_ratio: float = 0.0
    warmup_steps: int = 0
    max_grad_norm: Optional[float] = None
    max_length: int = 2048
    gradient_accumulation_steps: int = 1
    device: Optional[str] = None
    log_every: int = 10
    max_steps: Optional[int] = None
    dataloader_num_workers: int = 0
    precision: str = "fp32"
    seed: int = 42
    tf32: bool = True
    gradient_checkpointing: bool = False
    torch_compile: bool = False
    ddp_find_unused_parameters: bool = False
    metric_prefix: str = "03_m3_goal_planner"
    metric_names: Optional[dict[str, str]] = None


class M3GoalPlannerSFTTrainer:
    """Causal-LM SFT trainer for M3 goal-plan JSON."""

    def __init__(
        self,
        model,
        tokenizer,
        train_dataset: M3GoalPlanJsonlDataset,
        config: M3GoalPlannerSFTConfig,
        *,
        eval_dataset: Optional[M3GoalPlanJsonlDataset] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        metric_logger=None,
    ):
        """Create a trainer."""
        _seed_everything(config.seed)
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.config = config
        self.distributed = _DistributedContext.from_env()
        self.device = _resolve_device(config.device, self.distributed.local_rank)
        _configure_torch_runtime(config)
        _enable_gradient_checkpointing(self.model, config.gradient_checkpointing)
        self.model.to(self.device)
        if config.torch_compile and hasattr(torch, "compile"):
            self.model = torch.compile(self.model)
        if self.distributed.enabled:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.device.index] if self.device.type == "cuda" else None,
                find_unused_parameters=config.ddp_find_unused_parameters,
            )
        self.optimizer = optimizer or _build_optimizer(self.model, config, self.device)
        self.collator = M3GoalPlanSFTCollator(tokenizer, max_length=config.max_length)
        self.scheduler = None
        self.metric_logger = metric_logger

    def train(self) -> list[dict[str, float]]:
        """Run supervised fine-tuning and return loss history."""
        sampler = (
            DistributedSampler(
                self.train_dataset,
                num_replicas=self.distributed.world_size,
                rank=self.distributed.rank,
                shuffle=True,
                seed=self.config.seed,
            )
            if self.distributed.enabled
            else None
        )
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            collate_fn=self.collator,
            num_workers=self.config.dataloader_num_workers,
            pin_memory=self.device.type == "cuda",
        )
        self.scheduler = _build_scheduler(self.optimizer, self.config, len(loader))
        history: list[dict[str, float]] = []
        global_step = 0
        accum = max(1, int(self.config.gradient_accumulation_steps))
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        for epoch in range(self.config.epochs):
            if sampler is not None:
                sampler.set_epoch(epoch)
            epoch_loss = 0.0
            epoch_steps = 0
            epoch_tokens = 0
            epoch_positions = 0
            epoch_samples = 0
            epoch_start = time.monotonic()
            last_grad_norm: Optional[float] = None
            for batch_idx, batch in enumerate(loader, start=1):
                batch = {
                    key: value.to(self.device, non_blocking=self.device.type == "cuda")
                    for key, value in batch.items()
                }
                epoch_tokens += int(batch["attention_mask"].sum().detach().cpu().item())
                epoch_positions += int(batch["attention_mask"].numel())
                epoch_samples += int(batch["input_ids"].shape[0])
                with _autocast(self.device, self.config.precision):
                    outputs = self.model(**batch)
                loss = outputs.loss / accum
                loss.backward()
                detached_loss = _mean_across_ranks(loss.detach() * accum, self.distributed)
                epoch_loss += float(detached_loss.cpu().item())
                epoch_steps += 1
                if batch_idx % accum == 0 or batch_idx == len(loader):
                    if self.config.max_grad_norm:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            float(self.config.max_grad_norm),
                        )
                        last_grad_norm = float(grad_norm.detach().cpu().item())
                    self.optimizer.step()
                    if self.scheduler is not None:
                        self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    if self.config.max_steps and global_step >= self.config.max_steps:
                        self._append_history(
                            history,
                            epoch,
                            global_step,
                            epoch_loss,
                            epoch_steps,
                            epoch_tokens,
                            epoch_positions,
                            epoch_samples,
                            epoch_start,
                            last_grad_norm,
                        )
                        return history
                if (
                    self.config.log_every
                    and global_step > 0
                    and global_step % self.config.log_every == 0
                ):
                    self._append_history(
                        history,
                        epoch,
                        global_step,
                        epoch_loss,
                        epoch_steps,
                        epoch_tokens,
                        epoch_positions,
                        epoch_samples,
                        epoch_start,
                        last_grad_norm,
                    )
            self._append_history(
                history,
                epoch,
                global_step,
                epoch_loss,
                epoch_steps,
                epoch_tokens,
                epoch_positions,
                epoch_samples,
                epoch_start,
                last_grad_norm,
            )
        return history

    def save(self, output_dir: Optional[str] = None) -> None:
        """Save model and tokenizer in HuggingFace format."""
        if not self.distributed.is_main_process:
            _barrier(self.distributed)
            return
        path = Path(output_dir or self.config.output_dir)
        path.mkdir(parents=True, exist_ok=True)
        model = self.model.module if hasattr(self.model, "module") else self.model
        model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        _barrier(self.distributed)

    def planner(self) -> M3ModelGoalPlanner:
        """Return an inference planner backed by the trained model."""
        model = self.model.module if hasattr(self.model, "module") else self.model
        return M3ModelGoalPlanner(model, self.tokenizer)

    def _append_history(
        self,
        history: list[dict[str, float]],
        epoch: int,
        global_step: int,
        epoch_loss: float,
        epoch_steps: int,
        epoch_tokens: int,
        epoch_positions: int,
        epoch_samples: int,
        epoch_start: float,
        grad_norm: Optional[float],
    ) -> None:
        """Append rank-zero training metrics."""
        elapsed = max(time.monotonic() - epoch_start, 1e-9)
        total_tokens = _sum_number_across_ranks(epoch_tokens, self.distributed)
        total_positions = _sum_number_across_ranks(epoch_positions, self.distributed)
        total_samples = _sum_number_across_ranks(epoch_samples, self.distributed)
        if not self.distributed.is_main_process:
            return
        loss = epoch_loss / max(epoch_steps, 1)
        lr = self.optimizer.param_groups[0]["lr"]
        record = {
            "epoch": float(epoch),
            "step": float(global_step),
            "loss": loss,
            "ppl": float(math.exp(min(loss, 20.0))),
            "lr": float(lr),
            "tokens_per_second": float(total_tokens / elapsed),
            "samples_per_second": float(total_samples / elapsed),
            "padding_ratio": float(1.0 - (total_tokens / max(total_positions, 1))),
        }
        if self.device.type == "cuda":
            record["gpu_memory_allocated_gb"] = float(
                torch.cuda.memory_allocated(self.device) / (1024 ** 3)
            )
            record["gpu_memory_reserved_gb"] = float(
                torch.cuda.memory_reserved(self.device) / (1024 ** 3)
            )
        if grad_norm is not None:
            record["grad_norm"] = float(grad_norm)
        history.append(record)
        if self.metric_logger is not None:
            metrics = history[-1]
            self.metric_logger.log_metric(
                _metric_name(self.config, "train_loss", "train_loss"),
                metrics["loss"],
                global_step,
            )
            self.metric_logger.log_metric(
                _metric_name(self.config, "train_ppl", "train_ppl"),
                metrics["ppl"],
                global_step,
            )
            self.metric_logger.log_metric(
                _metric_name(self.config, "train_lr", "train_lr"),
                metrics["lr"],
                global_step,
            )
            if "grad_norm" in metrics:
                self.metric_logger.log_metric(
                    _metric_name(self.config, "train_grad_norm", "train_grad_norm"),
                    metrics["grad_norm"],
                    global_step,
                )
            self.metric_logger.log_metric(
                _metric_name(self.config, "train_tokens_per_second", "tokens_per_second"),
                metrics["tokens_per_second"],
                global_step,
            )
            self.metric_logger.log_metric(
                _metric_name(self.config, "train_samples_per_second", "samples_per_second"),
                metrics["samples_per_second"],
                global_step,
            )
            self.metric_logger.log_metric(
                _metric_name(self.config, "train_padding_ratio", "padding_ratio"),
                metrics["padding_ratio"],
                global_step,
            )
            if "gpu_memory_allocated_gb" in metrics:
                self.metric_logger.log_metric(
                    _metric_name(
                        self.config,
                        "train_gpu_memory_allocated_gb",
                        "gpu_memory_allocated_gb",
                    ),
                    metrics["gpu_memory_allocated_gb"],
                    global_step,
                )
            if "gpu_memory_reserved_gb" in metrics:
                self.metric_logger.log_metric(
                    _metric_name(
                        self.config,
                        "train_gpu_memory_reserved_gb",
                        "gpu_memory_reserved_gb",
                    ),
                    metrics["gpu_memory_reserved_gb"],
                    global_step,
                )
            self.metric_logger.log_metric(
                _metric_name(self.config, "train_epoch", "epoch"),
                float(epoch),
                global_step,
            )
            self.metric_logger.log_metric(
                _metric_name(self.config, "train_optimizer_step", "optimizer_step"),
                float(global_step),
                global_step,
            )


@dataclass
class _DistributedContext:
    """Minimal torch.distributed state inferred from launcher env vars."""

    enabled: bool
    rank: int
    world_size: int
    local_rank: int

    @property
    def is_main_process(self) -> bool:
        """Whether this process should log and save artifacts."""
        return self.rank == 0

    @classmethod
    def from_env(cls) -> "_DistributedContext":
        """Initialize process group when launched under ``torchrun``."""
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        enabled = world_size > 1
        if enabled and not dist.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend=backend)
        return cls(enabled=enabled, rank=rank, world_size=world_size, local_rank=local_rank)


def _resolve_device(device: Optional[str], local_rank: int) -> torch.device:
    """Resolve rank-local training device."""
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank)
    return torch.device("cpu")


def _configure_torch_runtime(config: M3GoalPlannerSFTConfig) -> None:
    """Enable runtime flags requested by the training spec."""
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = bool(config.tf32)
        torch.backends.cudnn.allow_tf32 = bool(config.tf32)


def _seed_everything(seed: int) -> None:
    """Seed local RNGs for reproducible shuffling and dropout."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _enable_gradient_checkpointing(model, enabled: bool) -> None:
    """Enable gradient checkpointing when the model supports it."""
    if not enabled:
        return
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "config"):
        model.config.use_cache = False


def _build_optimizer(model, config: M3GoalPlannerSFTConfig, device: torch.device):
    """Build the optimizer requested by the spec."""
    kwargs = {
        "lr": config.learning_rate,
        "weight_decay": config.weight_decay,
        "betas": config.betas,
        "eps": config.eps,
    }
    if config.optimizer_name == "adamw_torch_fused" and device.type == "cuda":
        try:
            return AdamW(model.parameters(), fused=True, **kwargs)
        except TypeError:
            pass
    return AdamW(model.parameters(), **kwargs)


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: M3GoalPlannerSFTConfig,
    batches_per_epoch: int,
):
    """Build a lightweight scheduler without adding launcher dependencies."""
    steps_per_epoch = math.ceil(batches_per_epoch / max(1, config.gradient_accumulation_steps))
    total_steps = int(config.max_steps or max(1, steps_per_epoch * config.epochs))
    warmup_steps = int(config.warmup_steps or round(total_steps * config.warmup_ratio))
    scheduler_name = (config.scheduler or "constant").lower()

    def lr_lambda(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max(float(step) / float(max(1, warmup_steps)), 1e-8)
        if scheduler_name == "cosine":
            progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress))))
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _autocast(device: torch.device, precision: str):
    """Return autocast context for configured precision."""
    precision = (precision or "fp32").lower()
    if device.type != "cuda" or precision == "fp32":
        return torch.autocast(device_type="cpu", enabled=False)
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _mean_across_ranks(
    value: torch.Tensor,
    distributed: _DistributedContext,
) -> torch.Tensor:
    """Average a scalar tensor across ranks for rank-zero metrics."""
    if not distributed.enabled:
        return value
    clone = value.detach().clone()
    dist.all_reduce(clone, op=dist.ReduceOp.SUM)
    clone /= distributed.world_size
    return clone


def _sum_number_across_ranks(value: int, distributed: _DistributedContext) -> int:
    """Sum an integer counter across distributed ranks."""
    if not distributed.enabled:
        return int(value)
    tensor = torch.tensor(value, device=torch.device("cuda") if torch.cuda.is_available() else "cpu")
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return int(tensor.item())


def _barrier(distributed: _DistributedContext) -> None:
    """Synchronize ranks when distributed training is active."""
    if distributed.enabled:
        dist.barrier()


def _metric_name(config: M3GoalPlannerSFTConfig, key: str, fallback: str) -> str:
    """Return configured metric name or a prefix-scoped fallback."""
    names: dict[str, Any] = config.metric_names or {}
    if key in names:
        return str(names[key])
    prefix = config.metric_prefix.rstrip("/")
    return f"{prefix}/{fallback}"


# Author: Mus mbayramo@stanford.edu
