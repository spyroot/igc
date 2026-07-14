"""Concrete verification for latent sub-goal training and HER.

The policy can condition on ``z_sub_goal``; success is still measured against
the hidden concrete :class:`igc.ds.goal_dataset.GoalSurface`. This module is the
first deterministic verifier for state equality and membership-style goals.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from igc.ds.goal_dataset import GoalSurface


@dataclass(frozen=True)
class GoalVerificationResult:
    """Result of checking one concrete sub-goal against an observation."""

    satisfied: bool
    reward: float
    reason: str
    observed_value: Any = None
    target_value: Any = None


def _get_path(body: Mapping[str, Any], path: str) -> Any:
    """Read a dotted path from a nested mapping.

    :param body: observation body.
    :param path: dotted property path.
    :return: observed value or ``None`` when absent.
    """
    current: Any = body
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


class GoalVerifier:
    """Verify hidden concrete sub-goals against observations.

    This deliberately does not inspect embeddings. A successful POST/PATCH can
    help debug an action, but state goals require measured state in the next
    observation.
    """

    def verify(
        self,
        surface: GoalSurface,
        observation: Mapping[str, Any],
        action_result: Mapping[str, Any] | None = None,
    ) -> GoalVerificationResult:
        """Verify one goal surface.

        :param surface: hidden concrete verifier payload.
        :param observation: next observation body.
        :param action_result: optional action result metadata, currently not
            trusted as state-goal success.
        :return: verification result.
        """
        del action_result  # accepted requests are not proof of final state.
        kind = str(surface.verifier.get("kind", "state_eq"))
        if kind == "contains":
            return self._verify_contains(surface, observation)
        if kind == "transition":
            return GoalVerificationResult(
                satisfied=False,
                reward=0.0,
                reason="transition_requires_followup_observation",
                observed_value=None,
                target_value=surface.target_value,
            )
        return self._verify_state_eq(surface, observation)

    def _verify_state_eq(
        self,
        surface: GoalSurface,
        observation: Mapping[str, Any],
    ) -> GoalVerificationResult:
        """Verify a state-equality sub-goal."""
        property_path = str(surface.verifier.get("property_path") or surface.fact_path)
        target = surface.verifier.get("target_value", surface.target_value)
        observed = _get_path(observation, property_path)
        satisfied = observed == target
        return GoalVerificationResult(
            satisfied=satisfied,
            reward=1.0 if satisfied else 0.0,
            reason="state_eq",
            observed_value=observed,
            target_value=target,
        )

    def _verify_contains(
        self,
        surface: GoalSurface,
        observation: Mapping[str, Any],
    ) -> GoalVerificationResult:
        """Verify a membership sub-goal for list-valued facts."""
        property_path = str(surface.verifier.get("property_path") or surface.fact_path)
        target = surface.verifier.get("target_value", surface.target_value)
        observed = _get_path(observation, property_path)
        satisfied = isinstance(observed, list) and target in observed
        return GoalVerificationResult(
            satisfied=satisfied,
            reward=1.0 if satisfied else 0.0,
            reason="contains",
            observed_value=observed,
            target_value=target,
        )


# Author: Mus mbayramo@stanford.edu
