"""Structural interfaces (Protocols) for the igc goal-conditioned tool-use framework.

These are the seams every environment plugin implements. They are
``@runtime_checkable`` so tests and the registry can do structural ``isinstance``
checks (which verify member names, exactly what we want for duck-typed plugins).

Pure stdlib + typing, so this stays cheap and CPU/offline-testable. The concrete
data records live in :mod:`igc.core.types`.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable

from igc.core.types import Goal, Observation, SimResult, ToolAction, ToolSpec, Transition


@runtime_checkable
class Simulator(Protocol):
    """Backend that executes an operation and returns a raw result. The existing
    ``MockServer`` satisfies this; sqlite/filesystem back it with a real library.
    ``snapshot``/``restore`` give dry-run (e.g. sqlite ``BEGIN``/``ROLLBACK``).
    """

    def reset(self, seed: Optional[int] = None) -> None: ...

    def execute(self, target: str, op: str, args: Optional[dict] = None) -> SimResult: ...

    def snapshot(self) -> Any: ...

    def restore(self, snap: Any) -> None: ...

    def is_live(self) -> bool: ...


@runtime_checkable
class ToolCatalog(Protocol):
    """Declares the tools/ops available in an environment, derives the legal actions
    for an observation, and validates an action's arguments against its schema.
    """

    def specs(self) -> list[ToolSpec]: ...

    def available_actions(self, obs: Observation) -> list[ToolAction]: ...

    def validate(self, action: ToolAction) -> bool: ...


@runtime_checkable
class Evaluator(Protocol):
    """Pluggable goal-satisfaction + dense reward, replacing the hardcoded
    ``torch.allclose`` goal check and the module-level reward function.
    """

    def verify(self, goal: Goal, obs: Observation) -> tuple[bool, float]: ...


@runtime_checkable
class GoalEnvironment(Protocol):
    """The one interface every backend (redfish/filesystem/sql/github) implements."""

    name: str

    def reset(self, goal: Goal) -> Observation: ...

    def available_actions(self, obs: Observation) -> list[ToolAction]: ...

    def step(self, action: ToolAction) -> Transition: ...

    def verify(self, obs: Observation, goal: Goal) -> tuple[bool, float]: ...
