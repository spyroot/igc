"""Goal conditioning bridge between M3 output, M1/M2 state, and M6 pointer scoring.

M3 answers "what should be true?" as a typed :class:`igc.core.types.Goal`.
M1/M2 answer "what is true now?" as a structured state latent from Redfish
state text and graph/candidate features. M6 combines both: it encodes the Goal
to ``goal_h``, combines it with ``state_h``, and scores legal candidates.

This module is deliberately adapter-shaped. It does not train M3 and does not
use the legacy one-hot goal tensor; it serializes typed Goals and feeds the
existing pointer policy contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Dict, Iterable, List, Mapping, Optional

import torch

from igc.core.types import Goal, RiskLevel, ToolAction
from igc.modules.llm.llm_encoder import TextEncoder
from igc.modules.policy.pointer_policy import Igc_PointerQNetwork


@dataclass(frozen=True)
class GoalEncoding:
    """Serializable goal-conditioning payload used by RL-facing examples."""

    goal_id: str
    instruction: str
    spec: Dict[str, Any]
    constraints: List[Any]
    plan: List[Dict[str, Any]]
    text: str
    embedding_text: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal_id": self.goal_id,
            "instruction": self.instruction,
            "spec": self.spec,
            "constraints": self.constraints,
            "plan": self.plan,
            "text": self.text,
            "embedding_text": self.embedding_text,
        }


def goal_from_dict(payload: Mapping[str, Any]) -> Goal:
    """Build a typed Goal from an M3-style dict payload."""
    plan = [
        _tool_action_from_any(item)
        for item in (payload.get("plan") or [])
    ]
    return Goal(
        instruction=str(payload.get("instruction", "")),
        spec=dict(payload.get("spec") or {}),
        env_name=str(payload.get("env_name", "")),
        constraints=list(payload.get("constraints") or []),
        plan=plan or None,
    )


def encode_goal_contract(goal: Goal | Mapping[str, Any]) -> GoalEncoding:
    """Serialize a Goal into stable text/spec/plan form for goal conditioning."""
    typed_goal = goal_from_dict(goal) if isinstance(goal, Mapping) else goal
    plan = [
        _tool_action_to_dict(item)
        for item in (typed_goal.plan or [])
    ]
    payload = {
        "instruction": typed_goal.instruction,
        "spec": typed_goal.spec,
        "constraints": typed_goal.constraints,
        "plan": plan,
    }
    text = "\n".join([
        f"GOAL_INSTRUCTION {typed_goal.instruction}",
        f"GOAL_SPEC {_canonical(typed_goal.spec)}",
        f"GOAL_CONSTRAINTS {_canonical(typed_goal.constraints)}",
        f"GOAL_PLAN {_canonical(plan)}",
    ])
    # The latent key is semantic/canonical, not current-state text and not raw operator
    # text alone. Different paraphrases with the same spec/constraints/plan collapse to
    # the same baseline embedding; representation training can later add contrastive
    # pressure over these same fields.
    embedding_text = "\n".join([
        f"GOAL_SPEC {_canonical(typed_goal.spec)}",
        f"GOAL_CONSTRAINTS {_canonical(typed_goal.constraints)}",
        f"GOAL_PLAN {_canonical(plan)}",
    ])
    return GoalEncoding(
        goal_id="goal:" + _stable_hex(payload)[:16],
        instruction=typed_goal.instruction,
        spec=dict(typed_goal.spec),
        constraints=list(typed_goal.constraints),
        plan=plan,
        text=text,
        embedding_text=embedding_text,
    )


class GoalConditioningAdapter:
    """Encode typed Goals with the shared text encoder.

    Inputs are typed Goals or M3 Goal dictionaries. Current resource state is
    intentionally not accepted here; state/goal interaction belongs to M6's
    ``f_state_goal(state_latent, goal_latent)`` query path.
    """

    def __init__(self, encoder: TextEncoder):
        self.encoder = encoder

    def encode(self, goals: Iterable[Goal | Mapping[str, Any]]) -> torch.Tensor:
        """Return ``[B, H]`` goal latents from typed Goals or M3 dicts."""
        encodings = [encode_goal_contract(goal) for goal in goals]
        return self.encoder.encode([encoding.embedding_text for encoding in encodings])


class GoalEncoder(GoalConditioningAdapter):
    """Named M6 adapter: ``Goal(instruction, spec, constraints, plan) -> goal_latent``."""


@dataclass(frozen=True)
class GoalEncoderTrainingContract:
    """Public training contract for goal extraction and goal representation stages."""

    extraction_input: str = "operator_text"
    extraction_target: str = "Goal(instruction, spec, constraints, plan)"
    extraction_metrics: tuple = (
        "exact_goal_json_match",
        "field_level_spec_match",
        "plan_step_match",
        "argument_validity",
    )
    representation_input: str = "canonical Goal render, never current state"
    representation_objective: str = "contrastive/triplet/retrieval over same-goal positives and incompatible negatives"
    rl_update_stage: str = "after typed goal baseline, through BC pointer warm-start then DQN/HER"


def pointer_scores_from_goal(
    pointer: Igc_PointerQNetwork,
    state_h: torch.Tensor,
    goals: Iterable[Goal | Mapping[str, Any]],
    candidate_h: torch.Tensor,
    candidate_mask: Optional[torch.Tensor],
    goal_encoder: GoalConditioningAdapter,
) -> torch.Tensor:
    """Score legal candidates using both M1/M2 state latent and M3 Goal latent."""
    goal_h = goal_encoder.encode(goals).to(state_h.device)
    return pointer(state_h, goal_h, candidate_h, candidate_mask)


def _tool_action_from_any(value: Any) -> ToolAction:
    if isinstance(value, ToolAction):
        return value
    if isinstance(value, Mapping):
        payload = dict(value)
        risk = payload.get("risk_level", "READ_ONLY")
        if isinstance(risk, str):
            payload["risk_level"] = RiskLevel[risk]
        return ToolAction(
            tool_name=str(payload["tool_name"]),
            op=str(payload["op"]),
            arguments=dict(payload.get("arguments") or {}),
            target=payload.get("target"),
            risk_level=payload.get("risk_level", RiskLevel.READ_ONLY),
            schema_id=payload.get("schema_id"),
        )
    raise TypeError(f"plan entries must be ToolAction or dict, got {type(value).__name__}")


def _tool_action_to_dict(value: Any) -> Dict[str, Any]:
    return _tool_action_from_any(value).to_dict()


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _stable_hex(value: Any) -> str:
    import hashlib

    return hashlib.blake2b(_canonical(value).encode("utf-8"), digest_size=16).hexdigest()
