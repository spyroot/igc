"""Dataset and collator for M3 goal-planner supervised fine-tuning.

JSONL records contain an operator instruction, context, and the structured M3
goal plan target. The collator turns those records into causal-LM SFT batches:
the prompt is masked out and only the target JSON contributes to loss.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import torch
from torch.utils.data import Dataset

from igc.modules.m3_goal_planner import (
    M3GoalPlan,
    M3GoalPlanJsonCodec,
    M3GoalPlannerContext,
    build_redfish_action_goal_plan,
)


@dataclass
class M3GoalPlanRecord:
    """One supervised M3 training example."""

    instruction: str
    context: M3GoalPlannerContext
    target_plan: M3GoalPlan
    source: str = "jsonl"

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "M3GoalPlanRecord":
        """Parse a training record from a JSON dictionary."""
        target_value = value.get("target_plan") or value.get("target") or value.get("goal_plan")
        if target_value is None:
            raise ValueError("M3 record requires target_plan, target, or goal_plan")
        return cls(
            instruction=value["instruction"],
            context=M3GoalPlannerContext.from_mapping(value.get("context")),
            target_plan=M3GoalPlan.from_dict(target_value),
            source=str(value.get("source", "jsonl")),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize a record to JSONL-friendly shape."""
        return {
            "instruction": self.instruction,
            "context": self.context.to_dict(),
            "target_plan": self.target_plan.to_dict(),
            "source": self.source,
        }


class M3GoalPlanJsonlDataset(Dataset):
    """JSONL dataset for M3 goal-planner SFT."""

    def __init__(self, records: Iterable[M3GoalPlanRecord]):
        """Create a dataset from records."""
        self.records = list(records)

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "M3GoalPlanJsonlDataset":
        """Load records from a JSONL path."""
        records = []
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(M3GoalPlanRecord.from_dict(json.loads(line)))
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"invalid M3 JSONL record at line {line_number}: {exc}") from exc
        return cls(records)

    @classmethod
    def from_redfish_action_catalog(
        cls,
        action_catalog: Iterable[dict[str, Any]],
        *,
        context: Optional[M3GoalPlannerContext] = None,
        templates_per_action: int = 3,
    ) -> "M3GoalPlanJsonlDataset":
        """Build supervised M3 records from all discovered Redfish actions.

        :param action_catalog: Iterable of action records with ``action`` and
            ``target`` fields. Optional fields include ``method``,
            ``allowed_methods``, and ``arguments``.
        :param context: Shared planner context for every record.
        :param templates_per_action: Number of NLP templates emitted per action.
        :return: Dataset with one or more records per action.
        """
        base_context = context or M3GoalPlannerContext()
        records = []
        for action_record in action_catalog:
            record_context = _context_for_action(base_context, action_record)
            for instruction in _redfish_action_instructions(
                action_record,
                limit=templates_per_action,
            ):
                records.append(M3GoalPlanRecord(
                    instruction=instruction,
                    context=record_context,
                    target_plan=build_redfish_action_goal_plan(
                        instruction,
                        action_record,
                        record_context,
                    ),
                    source="redfish_action_catalog",
                ))
        return cls(records)

    @classmethod
    def from_json_dataset_actions(
        cls,
        dataset,
        *,
        templates_per_action: int = 3,
    ) -> "M3GoalPlanJsonlDataset":
        """Build M3 records from a ``JSONDataset``-like ``action_to_rest`` map."""
        action_to_rest = dict(getattr(dataset, "action_to_rest", {}) or {})
        action_catalog = [
            {"action": action, "target": target}
            for action, target in sorted(action_to_rest.items())
        ]
        return cls.from_redfish_action_catalog(
            action_catalog,
            templates_per_action=templates_per_action,
        )

    def write_jsonl(self, path: str | Path) -> None:
        """Write records to JSONL."""
        with Path(path).open("w", encoding="utf-8") as handle:
            for record in self.records:
                handle.write(json.dumps(record.to_dict(), sort_keys=True))
                handle.write("\n")

    def __len__(self) -> int:
        """Return number of examples."""
        return len(self.records)

    def __getitem__(self, index: int) -> M3GoalPlanRecord:
        """Return a record by index."""
        return self.records[index]


class M3GoalPlanSFTCollator:
    """Causal-LM collator for M3 goal-planner SFT."""

    def __init__(self, tokenizer, max_length: int = 2048):
        """Create a collator.

        :param tokenizer: HuggingFace tokenizer.
        :param max_length: Maximum sequence length after prompt+target packing.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, records: list[M3GoalPlanRecord]) -> dict[str, torch.Tensor]:
        """Collate records into ``input_ids``, ``attention_mask``, and ``labels``."""
        packed = [self._encode_record(record) for record in records]
        max_len = max(len(item["input_ids"]) for item in packed)
        input_ids = []
        labels = []
        attention_mask = []
        for item in packed:
            pad = max_len - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [self.tokenizer.pad_token_id] * pad)
            labels.append(item["labels"] + [-100] * pad)
            attention_mask.append([1] * len(item["input_ids"]) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def _encode_record(self, record: M3GoalPlanRecord) -> dict[str, list[int]]:
        """Encode one record and mask prompt tokens from loss."""
        prompt = M3GoalPlanJsonCodec.build_prompt(record.instruction, record.context)
        target = M3GoalPlanJsonCodec.dumps(record.target_plan)
        eos = self.tokenizer.eos_token or ""
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        target_ids = self.tokenizer(target + eos, add_special_tokens=False)["input_ids"]
        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + target_ids
        if len(input_ids) > self.max_length:
            overflow = len(input_ids) - self.max_length
            prompt_trim = min(overflow, len(prompt_ids))
            input_ids = input_ids[prompt_trim:]
            labels = labels[prompt_trim:]
            input_ids = input_ids[-self.max_length:]
            labels = labels[-self.max_length:]
        return {"input_ids": input_ids, "labels": labels}


def load_m3_goal_plan_dataset(
    jsonl_path: str | Path,
) -> M3GoalPlanJsonlDataset:
    """Load an M3 dataset from a real JSONL path."""
    return M3GoalPlanJsonlDataset.from_jsonl(jsonl_path)


def _redfish_action_instructions(
    action_record: dict[str, Any],
    *,
    limit: int,
) -> list[str]:
    """Generate NLP instructions for one Redfish action catalog entry."""
    action = str(
        action_record.get("action")
        or action_record.get("name")
        or action_record.get("op")
        or "redfish action"
    )
    target = str(
        action_record.get("target")
        or action_record.get("uri")
        or action_record.get("endpoint")
        or ""
    )
    readable = action.replace(".", " ").replace("#", " ").strip()
    templates = [
        f"execute redfish action {readable}",
        f"run {readable} on {target}",
        f"make the server perform {readable}",
        f"invoke {readable} and verify the result",
        f"use redfish to {readable}",
    ]
    return templates[:max(1, limit)]


def _context_for_action(
    context: M3GoalPlannerContext,
    action_record: dict[str, Any],
) -> M3GoalPlannerContext:
    """Return compact per-record context for one discovered action."""
    return M3GoalPlannerContext(
        state_summary=dict(context.state_summary),
        tool_catalog=[dict(action_record)],
        approved_images=list(context.approved_images),
        safety_policy=dict(context.safety_policy),
    )


# Author: Mus mbayramo@stanford.edu
