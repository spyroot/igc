"""GoalEncoder contracts and tiny deterministic baseline.

Production training will replace the hashing baseline with learned
``TextGoalEncoder`` and ``GoalSurfaceEncoder`` modules. The interface already
uses the corrected semantics: encode one atomic sub-goal into ``z_sub_goal``;
compound instructions remain a set of those vectors plus dependency metadata.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable

from igc.ds.goal_dataset import GoalRef, GoalSurface, GoalTextExample


@dataclass(frozen=True)
class GoalLatent:
    """A deterministic stand-in for a learned ``z_sub_goal`` vector."""

    goal_id: str
    values: tuple[float, ...]


class GoalEncoder:
    """Tiny deterministic encoder with the learned encoder's public shape."""

    def __init__(self, dim: int = 16):
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim

    def encode_ref(self, goal_ref: GoalRef) -> GoalLatent:
        """Encode one atomic goal ref into a stable vector.

        :param goal_ref: atomic sub-goal label.
        :return: deterministic latent for tests and offline plumbing.
        """
        return GoalLatent(goal_ref.goal_id, self._hash(goal_ref.goal_id))

    def encode_surface(self, surface: GoalSurface) -> GoalLatent:
        """Encode a concrete surface by its semantic atomic goal ref."""
        return self.encode_ref(surface.goal_ref)

    def encode_text_example(self, example: GoalTextExample) -> tuple[GoalLatent, ...]:
        """Encode every atomic sub-goal in a text example.

        A compound instruction returns multiple ``z_sub_goal`` items rather
        than one opaque plan vector.
        """
        return tuple(self.encode_ref(ref) for ref in example.goal_refs)

    def _hash(self, key: str) -> tuple[float, ...]:
        """Hash a key into a small deterministic float tuple."""
        values = []
        counter = 0
        while len(values) < self.dim:
            digest = hashlib.sha256(f"{key}:{counter}".encode("utf-8")).digest()
            for byte in digest:
                values.append((byte / 127.5) - 1.0)
                if len(values) == self.dim:
                    break
            counter += 1
        return tuple(values)


def encode_goal_set(encoder: GoalEncoder, refs: Iterable[GoalRef]) -> tuple[GoalLatent, ...]:
    """Encode a set/list of atomic sub-goal refs.

    :param encoder: goal encoder instance.
    :param refs: atomic sub-goals.
    :return: one latent per sub-goal.
    """
    return tuple(encoder.encode_ref(ref) for ref in refs)


# Author: Mus mbayramo@stanford.edu
