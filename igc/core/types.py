"""Core type definitions for the igc goal-conditioned tool-use framework.

Structured, JSON-serializable representations shared across every environment
adapter, the trajectory recorder, and the learning/eval layer. The keystone
type is :class:`ToolAction`, which replaces the legacy one-hot action vector
(a fixed ``Box(num_urls + 6)``) with a structured, hashable tool call that
carries arguments — the slot the one-hot encoding never had.

Pure standard library on purpose (no torch/numpy/transformers): these types are
imported everywhere and must stay cheap and CPU/offline-testable.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Optional


class RiskLevel(IntEnum):
    """Monotone risk level. A guardrail denies an action when its level exceeds
    the allowed ceiling, so ordering matters: READ_ONLY < MUTATING < DESTRUCTIVE.
    """

    READ_ONLY = 0
    MUTATING = 1
    DESTRUCTIVE = 2


@dataclass(frozen=True)
class ToolAction:
    """A structured, hashable, JSON-serializable tool call (the keystone type).

    Replaces a one-hot action vector with a named tool, an operation/verb, an
    ``arguments`` payload, an optional ``target`` (url/path/table, kept separate
    from ``tool_name`` so an enumerated codec can still index it), a risk level,
    and an optional ``schema_id`` for argument validation.
    """

    tool_name: str
    op: str
    arguments: dict = field(default_factory=dict)
    target: Optional[str] = None
    risk_level: RiskLevel = RiskLevel.READ_ONLY
    schema_id: Optional[str] = None

    def __hash__(self) -> int:
        """Hash over the scalar fields plus a canonical JSON of ``arguments``,
        so the (frozen) instance is hashable despite the dict field and equal
        actions hash equally.
        """
        args_json = json.dumps(self.arguments, sort_keys=True, default=str)
        return hash(
            (self.tool_name, self.op, args_json, self.target, self.risk_level, self.schema_id)
        )

    def to_dict(self) -> dict:
        """Serialize to a plain dict (risk_level by ``.name``). ``arguments`` is
        copied so callers cannot mutate the frozen instance's payload.
        """
        return {
            "tool_name": self.tool_name,
            "op": self.op,
            "arguments": dict(self.arguments),
            "target": self.target,
            "risk_level": self.risk_level.name,
            "schema_id": self.schema_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ToolAction":
        """Reconstruct from :meth:`to_dict` output: ``from_dict(a.to_dict()) == a``."""
        return cls(
            tool_name=d["tool_name"],
            op=d["op"],
            arguments=dict(d.get("arguments") or {}),
            target=d.get("target"),
            risk_level=RiskLevel[d["risk_level"]],
            schema_id=d.get("schema_id"),
        )


@dataclass
class Observation:
    """What an environment returns: a serialized ``text`` form the LLM encoder
    tokenizes, the raw ``structured`` payload, a normalized ``status`` (HTTP code
    or a synthesized equivalent such as an exit code / SQLSTATE), and room for a
    cached ``encoded`` embedding (kept ``Any`` to avoid a torch import here).
    """

    text: str
    structured: Any = None
    status: int = 200
    error: bool = False
    encoded: Any = None
    info: dict = field(default_factory=dict)


@dataclass
class Goal:
    """A goal: a natural-language ``instruction``, a machine-checkable ``spec``
    the Evaluator reads (replacing embedding ``allclose``), optional
    ``constraints``, and an optional ordered ``plan`` (a list of
    :class:`ToolAction` sub-goals filled by the planner / casting layer).
    """

    instruction: str
    spec: dict = field(default_factory=dict)
    env_name: str = ""
    plan: Optional[list] = None
    constraints: list = field(default_factory=list)
    encoded: Any = None


@dataclass
class Transition:
    """One MDP step. Five fields including the ``terminated``/``truncated`` done
    flags the legacy 4-tuple replay buffer lacked; ``info`` may carry an
    ``achieved_goal`` for hindsight relabeling.
    """

    observation: Observation
    reward: float
    terminated: bool
    truncated: bool
    info: dict = field(default_factory=dict)


@dataclass
class SimResult:
    """A simulator's raw result, generalizing the existing ``MockResponse``
    (``json_data``/``status_code``/``error``/``new_state``) to any backend.
    """

    body: Any
    status: int
    error: bool = False
    new_state: Any = None


@dataclass
class ToolSpec:
    """Per-tool declaration in an environment's catalog: the ``ops`` it supports,
    a per-op ``arg_schema`` (validates :class:`ToolAction` arguments and sizes the
    policy head), a ``risk_level``, and a ``target_space`` — ``"enumerated"`` (a
    codec one-hot, e.g. discovered Redfish URLs) or ``"freeform"`` (a generated
    string such as a file path or SQL text).
    """

    tool_name: str
    ops: list
    arg_schema: dict = field(default_factory=dict)
    risk_level: RiskLevel = RiskLevel.READ_ONLY
    target_space: str = "enumerated"
    description: str = ""


# Author: Mus mbayramo@stanford.edu
