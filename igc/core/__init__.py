"""Core contracts for the igc goal-conditioned tool-use framework.

Pure-stdlib types shared by every environment adapter, the trajectory layer, and
the learning/eval layer. Heavier protocols (Simulator, GoalEnvironment) live in
``igc.core.protocols``; keep this package import-light.

Author:
Mus mbayramo@stanford.edu
"""
from igc.core.types import (
    Goal,
    Observation,
    RiskLevel,
    SimResult,
    ToolAction,
    ToolSpec,
    Transition,
)

__all__ = [
    "Goal",
    "Observation",
    "RiskLevel",
    "SimResult",
    "ToolAction",
    "ToolSpec",
    "Transition",
]
