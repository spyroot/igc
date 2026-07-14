"""GoalExtractor / GoalEncoder dataset records.

The key contract is intentionally narrow: operator text maps to an unordered
set of atomic sub-goals. Each atomic sub-goal becomes an active ``z_sub_goal``
for policy conditioning; compound instructions may carry dependency hints, but
they are not execution plans. Hidden vendor-specific verifier payloads stay
with :class:`GoalSurface` for reward and HER.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class GoalRef:
    """Vendor-neutral atomic sub-goal label.

    :param goal_id: stable semantic identifier used as the class/slot label.
    :param family: broad family such as ``power`` / ``boot`` / ``network``.
    :param resource_type: vendor-neutral resource class.
    :param property_path: dotted Redfish property path, or ``""`` for actions.
    :param operator: comparison operator; initially ``"eq"``.
    :param target_value: target value for state goals.
    :param mode: ``"state"`` or ``"transition"``.
    :param action_name: Redfish action name for transition goals.
    :param arguments: action arguments for transition goals.
    :param constraints: optional semantic constraints that are not policy-visible.
    """
    goal_id: str
    family: str
    resource_type: str
    property_path: str = ""
    operator: str = "eq"
    target_value: Any = None
    mode: str = "state"
    action_name: str = ""
    arguments: Mapping[str, Any] = field(default_factory=dict)
    constraints: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict.

        :return: plain dict representation.
        """
        data = asdict(self)
        data["arguments"] = dict(self.arguments)
        data["constraints"] = dict(self.constraints)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GoalRef":
        """Rebuild from :meth:`to_dict` output.

        :param data: serialized goal ref.
        :return: reconstructed :class:`GoalRef`.
        """
        return cls(
            goal_id=str(data["goal_id"]),
            family=str(data["family"]),
            resource_type=str(data["resource_type"]),
            property_path=str(data.get("property_path", "")),
            operator=str(data.get("operator", "eq")),
            target_value=data.get("target_value"),
            mode=str(data.get("mode", "state")),
            action_name=str(data.get("action_name", "")),
            arguments=dict(data.get("arguments") or {}),
            constraints=dict(data.get("constraints") or {}),
        )


@dataclass(frozen=True)
class GoalDependency:
    """Explicit text-level partial order between atomic sub-goals.

    This is metadata for compound instructions such as "set NTP then boot"; it
    is not a concrete API sequence and does not turn the goal set into a plan.

    :param before_goal_id: goal that text says must happen first.
    :param after_goal_id: goal that text says comes after.
    :param relation: relation label, currently ``"before"``.
    :param evidence: text cue such as ``"then"`` / ``"after"``.
    """
    before_goal_id: str
    after_goal_id: str
    relation: str = "before"
    evidence: str = ""

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GoalDependency":
        """Rebuild from :meth:`to_dict` output."""
        return cls(
            before_goal_id=str(data["before_goal_id"]),
            after_goal_id=str(data["after_goal_id"]),
            relation=str(data.get("relation", "before")),
            evidence=str(data.get("evidence", "")),
        )


@dataclass(frozen=True)
class GoalSurface:
    """Vendor/world-specific evidence for one atomic :class:`GoalRef`.

    The policy should never need this full payload. The verifier and HER logic
    use it to compare concrete observations and recompute rewards.
    """
    goal_ref: GoalRef
    vendor: str
    source: str
    resource_uri: str
    resource_type: str
    fact_path: str
    target_value: Any
    current_value: Any = None
    allowed_values: Sequence[Any] = field(default_factory=tuple)
    verifier: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        return {
            "goal_ref": self.goal_ref.to_dict(),
            "vendor": self.vendor,
            "source": self.source,
            "resource_uri": self.resource_uri,
            "resource_type": self.resource_type,
            "fact_path": self.fact_path,
            "target_value": self.target_value,
            "current_value": self.current_value,
            "allowed_values": list(self.allowed_values),
            "verifier": dict(self.verifier),
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GoalSurface":
        """Rebuild from :meth:`to_dict` output."""
        return cls(
            goal_ref=GoalRef.from_dict(data["goal_ref"]),
            vendor=str(data.get("vendor", "")),
            source=str(data.get("source", "")),
            resource_uri=str(data["resource_uri"]),
            resource_type=str(data.get("resource_type", "")),
            fact_path=str(data.get("fact_path", "")),
            target_value=data.get("target_value"),
            current_value=data.get("current_value"),
            allowed_values=tuple(data.get("allowed_values") or ()),
            verifier=dict(data.get("verifier") or {}),
            provenance=dict(data.get("provenance") or {}),
        )


@dataclass(frozen=True)
class GoalTextExample:
    """Text-to-atomic-sub-goal training row.

    ``goal_refs`` is a tuple so equality is deterministic for tests and JSONL
    round trips. Callers can treat it as an unordered set unless dependencies
    explicitly say otherwise.
    """
    text: str
    goal_refs: Sequence[GoalRef]
    dependencies: Sequence[GoalDependency] = field(default_factory=tuple)
    text_source: str = "template"
    split: str = "train"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        return {
            "text": self.text,
            "goal_refs": [ref.to_dict() for ref in self.goal_refs],
            "dependencies": [dep.to_dict() for dep in self.dependencies],
            "text_source": self.text_source,
            "split": self.split,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GoalTextExample":
        """Rebuild from :meth:`to_dict` output."""
        return cls(
            text=str(data["text"]),
            goal_refs=tuple(GoalRef.from_dict(item) for item in data.get("goal_refs", ())),
            dependencies=tuple(
                GoalDependency.from_dict(item) for item in data.get("dependencies", ())
            ),
            text_source=str(data.get("text_source", "template")),
            split=str(data.get("split", "train")),
            metadata=dict(data.get("metadata") or {}),
        )

    @staticmethod
    def write_jsonl(path: Path | str, rows: Iterable["GoalTextExample"]) -> None:
        """Write text examples as JSON Lines.

        :param path: output JSONL path.
        :param rows: examples to write.
        """
        target = Path(path)
        with target.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row.to_dict(), sort_keys=True) + "\n")


def read_goal_text_examples(path: Path | str) -> tuple[GoalTextExample, ...]:
    """Read :class:`GoalTextExample` rows from JSON Lines.

    :param path: input JSONL path.
    :return: loaded examples.
    """
    source = Path(path)
    rows = []
    with source.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped:
                rows.append(GoalTextExample.from_dict(json.loads(stripped)))
    return tuple(rows)


def write_goal_surfaces(path: Path | str, rows: Iterable[GoalSurface]) -> None:
    """Write goal surfaces as JSON Lines.

    :param path: output JSONL path.
    :param rows: surfaces to write.
    """
    target = Path(path)
    with target.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row.to_dict(), sort_keys=True) + "\n")


def read_goal_surfaces(path: Path | str) -> tuple[GoalSurface, ...]:
    """Read :class:`GoalSurface` rows from JSON Lines.

    :param path: input JSONL path.
    :return: loaded surfaces.
    """
    source = Path(path)
    rows = []
    with source.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped:
                rows.append(GoalSurface.from_dict(json.loads(stripped)))
    return tuple(rows)


# Author: Mus mbayramo@stanford.edu
