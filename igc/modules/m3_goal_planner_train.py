"""Training loop for the M3 goal planner.

This is a small supervised fine-tuning wrapper for causal language models. It
expects :class:`M3GoalPlanJsonlDataset` records and trains the model to emit
structured M3 goal-plan JSON from instruction plus context.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from igc.ds.m3_goal_plan_dataset import M3GoalPlanJsonlDataset, M3GoalPlanSFTCollator
from igc.modules.m3_goal_planner import M3ModelGoalPlanner


@dataclass
class M3GoalPlannerSFTConfig:
    """Supervised fine-tuning config for M3."""

    output_dir: str
    epochs: int = 1
    batch_size: int = 1
    learning_rate: float = 5e-5
    max_length: int = 2048
    gradient_accumulation_steps: int = 1
    device: Optional[str] = None
    log_every: int = 10
    max_steps: Optional[int] = None


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
    ):
        """Create a trainer."""
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.config = config
        self.device = torch.device(
            config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(self.device)
        self.optimizer = optimizer or AdamW(self.model.parameters(), lr=config.learning_rate)
        self.collator = M3GoalPlanSFTCollator(tokenizer, max_length=config.max_length)

    def train(self) -> list[dict[str, float]]:
        """Run supervised fine-tuning and return loss history."""
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=self.collator,
        )
        history: list[dict[str, float]] = []
        global_step = 0
        accum = max(1, int(self.config.gradient_accumulation_steps))
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        for epoch in range(self.config.epochs):
            epoch_loss = 0.0
            epoch_steps = 0
            for batch_idx, batch in enumerate(loader, start=1):
                batch = {key: value.to(self.device) for key, value in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss / accum
                loss.backward()
                epoch_loss += float(loss.detach().cpu().item()) * accum
                epoch_steps += 1
                if batch_idx % accum == 0 or batch_idx == len(loader):
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    if self.config.max_steps and global_step >= self.config.max_steps:
                        history.append({
                            "epoch": float(epoch),
                            "step": float(global_step),
                            "loss": epoch_loss / max(epoch_steps, 1),
                        })
                        return history
                if (
                    self.config.log_every
                    and global_step > 0
                    and global_step % self.config.log_every == 0
                ):
                    history.append({
                        "epoch": float(epoch),
                        "step": float(global_step),
                        "loss": epoch_loss / max(epoch_steps, 1),
                    })
            history.append({
                "epoch": float(epoch),
                "step": float(global_step),
                "loss": epoch_loss / max(epoch_steps, 1),
            })
        return history

    def save(self, output_dir: Optional[str] = None) -> None:
        """Save model and tokenizer in HuggingFace format."""
        path = Path(output_dir or self.config.output_dir)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def planner(self) -> M3ModelGoalPlanner:
        """Return an inference planner backed by the trained model."""
        return M3ModelGoalPlanner(self.model, self.tokenizer)


# Author: Mus mbayramo@stanford.edu
