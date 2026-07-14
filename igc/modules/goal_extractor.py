"""GoalExtractor contract for text-to-atomic-sub-goal extraction.

The extractor is a learned or injected model that maps operator text into the
same target envelope written by the dataset builder: atomic :class:`GoalRef`
rows plus optional text-stated dependency hints. This module intentionally has
no Redfish keyword rules or hardcoded goal IDs; deterministic labels come from
``igc.ds.goal_dataset_builder`` and trained models consume those rows later.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from igc.ds.goal_dataset import GoalDependency, GoalRef, GoalTextExample


@dataclass(frozen=True)
class GoalExtraction:
    """Structured extractor output.

    :param text: original operator text.
    :param atomic_goal_refs: unordered atomic sub-goals found in text.
    :param relations: text-stated partial-order hints.
    :param evidence: optional text spans or decoder evidence keyed by goal id.
    """

    text: str
    atomic_goal_refs: Sequence[GoalRef]
    relations: Sequence[GoalDependency]
    evidence: Mapping[str, Any]

    @property
    def dependency_hints(self) -> Sequence[GoalDependency]:
        """Backward-compatible alias for code that still says dependencies."""
        return self.relations

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the JSON target shape used by extractor training."""
        return {
            "text": self.text,
            "atomic_goal_refs": [ref.to_dict() for ref in self.atomic_goal_refs],
            "relations": [relation.to_dict() for relation in self.relations],
            "evidence": dict(self.evidence),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GoalExtraction":
        """Rebuild from :meth:`to_dict` output."""
        return cls(
            text=str(data.get("text", "")),
            atomic_goal_refs=tuple(
                GoalRef.from_dict(item) for item in data.get("atomic_goal_refs", ())
            ),
            relations=tuple(
                GoalDependency.from_dict(item) for item in data.get("relations", ())
            ),
            evidence=dict(data.get("evidence") or {}),
        )


def extraction_from_text_example(example: GoalTextExample) -> GoalExtraction:
    """Convert one supervised dataset row into the extractor target envelope.

    :param example: text row with deterministic JSON-derived labels.
    :return: target envelope for model training or evaluation.
    """
    return GoalExtraction(
        text=example.text,
        atomic_goal_refs=tuple(example.goal_refs),
        relations=tuple(example.dependencies),
        evidence=dict(example.metadata.get("evidence") or {}),
    )


def parse_goal_extraction(value: GoalExtraction | Mapping[str, Any] | str) -> GoalExtraction:
    """Parse model or adapter output into a :class:`GoalExtraction`.

    :param value: already parsed extraction, mapping, or JSON string.
    :return: structured extraction.
    """
    if isinstance(value, GoalExtraction):
        return value
    if isinstance(value, str):
        value = json.loads(value)
    return GoalExtraction.from_dict(value)


class GoalExtractor:
    """Pluggable extractor wrapper around a trained decoder or test adapter."""

    def __init__(
        self,
        decoder: Callable[[str], GoalExtraction | Mapping[str, Any] | str] | None = None,
    ):
        self.decoder = decoder

    def extract(self, text: str) -> GoalExtraction:
        """Extract atomic sub-goals from text through the configured decoder.

        :param text: operator instruction.
        :return: structured extraction result.
        :raises RuntimeError: when no trained/injected decoder is configured.
        """
        if self.decoder is None:
            raise RuntimeError("GoalExtractor requires a trained or injected decoder")
        extraction = parse_goal_extraction(self.decoder(text))
        if not extraction.text:
            return GoalExtraction(
                text=text,
                atomic_goal_refs=tuple(extraction.atomic_goal_refs),
                relations=tuple(extraction.relations),
                evidence=dict(extraction.evidence),
            )
        return extraction


# Author: Mus mbayramo@stanford.edu
