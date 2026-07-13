"""Paraphrase generation helpers for GoalExtractor datasets.

The generator proposes operator text ``X`` only. Deterministic code owns
``true_y`` by attaching JSON-derived :class:`igc.ds.goal_dataset.GoalRef` labels
before any model call. A trained or injected extractor may later validate
candidate text, but the bootstrap path intentionally does not depend on a
handwritten parser.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Protocol, Sequence

from igc.ds.goal_dataset import GoalDependency, GoalRef, GoalTextExample
from igc.ds.goal_dataset_builder import make_goal_text_example


class ParaphraseProvider(Protocol):
    """Provider interface for candidate text generation."""

    def generate(self, prompt: str) -> Sequence[str]: ...


@dataclass(frozen=True)
class StaticParaphraseProvider:
    """Offline provider used by tests and dry runs."""

    texts: Sequence[str]

    def generate(self, prompt: str) -> Sequence[str]:
        """Return static texts, ignoring the prompt."""
        del prompt
        return tuple(self.texts)


class OpenAICompatibleParaphraseProvider:
    """Minimal stdlib client for OpenAI-compatible chat completion endpoints.

    Endpoint URL, model, and API key are constructor inputs so no private URL or
    token ever needs to be hardcoded in the repository.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
        timeout: float = 60.0,
        temperature: float = 0.7,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.temperature = temperature

    def generate(self, prompt: str) -> Sequence[str]:
        """Call the configured chat endpoint and split lines into candidates."""
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
        }
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers=self._headers(),
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
        content = data["choices"][0]["message"]["content"]
        return tuple(line.strip("- \t") for line in content.splitlines() if line.strip())

    def _headers(self) -> dict[str, str]:
        """Build request headers without exposing secrets in logs."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers


def _safe_prompt_value(value: Any) -> Any:
    """Return a prompt-safe representation of a label value."""
    if not isinstance(value, str):
        return value
    if re.fullmatch(r"[A-Za-z0-9_ -]{1,64}", value):
        return value
    return "<redacted-string>"


def _unsafe_goal_suffix(value: str) -> str:
    """Return the old unsafe suffix shape for defensive goal-id redaction."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "empty"


def _safe_prompt_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively sanitize a mapping before it enters an LLM prompt."""
    return {key: _safe_prompt_payload(item) for key, item in value.items()}


def _safe_prompt_payload(value: Any) -> Any:
    """Sanitize nested prompt payload values."""
    if isinstance(value, dict):
        return _safe_prompt_mapping(value)
    if isinstance(value, (list, tuple)):
        return [_safe_prompt_payload(item) for item in value]
    return _safe_prompt_value(value)


def _safe_prompt_goal_ref(ref: GoalRef) -> dict[str, Any]:
    """Return a prompt-safe view of a GoalRef without raw captured strings."""
    data = ref.to_dict()
    sanitized_target = _safe_prompt_value(ref.target_value)
    data["target_value"] = sanitized_target
    data["arguments"] = _safe_prompt_payload(dict(ref.arguments))
    data["constraints"] = _safe_prompt_payload(dict(ref.constraints))
    if sanitized_target != ref.target_value:
        unsafe_suffix = _unsafe_goal_suffix(str(ref.target_value))
        if str(ref.goal_id).endswith(f".{unsafe_suffix}"):
            data["goal_id"] = ".".join((
                ref.family,
                ref.resource_type,
                ref.property_path,
                ref.operator,
                "<redacted>",
            ))
    return data


def build_paraphrase_prompt(
    goal_refs: Sequence[GoalRef],
    dependencies: Sequence[GoalDependency] = (),
    count: int = 8,
) -> str:
    """Build a public-safe prompt for candidate operator text.

    :param goal_refs: true atomic sub-goals.
    :param dependencies: true text-level partial-order hints.
    :param count: requested number of paraphrases.
    :return: prompt string.
    """
    refs_json = json.dumps(
        [_safe_prompt_goal_ref(ref) for ref in goal_refs],
        sort_keys=True,
        indent=2,
    )
    deps_json = json.dumps([dep.to_dict() for dep in dependencies], sort_keys=True, indent=2)
    ordering_rule = (
        "Use the ordering implied by the dependency hints."
        if dependencies
        else "Do not add ordering words such as then, before, after, first, or second."
    )
    return (
        "Generate natural-language operator requests for a server-management agent.\n"
        "Each request must mean exactly the atomic sub-goals below.\n"
        f"{ordering_rule}\n"
        "Do not add, remove, weaken, strengthen, or reorder goals beyond the dependency hints.\n"
        "Return one request per line, no numbering, no markdown.\n\n"
        f"Requested count: {count}\n"
        f"Atomic sub-goals:\n{refs_json}\n\n"
        f"Dependency hints:\n{deps_json}\n"
    )


def _goal_id_set(refs: Iterable[GoalRef]) -> frozenset[str]:
    """Return stable goal-id set for semantic equality checks."""
    return frozenset(ref.goal_id for ref in refs)


def _relation_set(relations: Iterable[GoalDependency]) -> frozenset[tuple[str, str, str]]:
    """Return stable dependency-relation set."""
    return frozenset(
        (rel.before_goal_id, rel.after_goal_id, rel.relation) for rel in relations
    )


def validate_paraphrase_texts(
    texts: Sequence[str],
    extractor: Any,
    expected_goal_refs: Sequence[GoalRef],
    expected_relations: Sequence[GoalDependency] = (),
    text_source: str = "llm_paraphrase",
    split: str = "train",
) -> tuple[GoalTextExample, ...]:
    """Accept only generated text that extracts to exactly the target schema.

    :param texts: candidate operator requests.
    :param extractor: deterministic or learned extractor used as validator.
    :param expected_goal_refs: true atomic sub-goals.
    :param expected_relations: true text-level relations.
    :param text_source: source label for accepted rows.
    :param split: dataset split for accepted rows.
    :return: accepted examples.
    """
    expected_ids = _goal_id_set(expected_goal_refs)
    expected_edges = _relation_set(expected_relations)
    accepted: list[GoalTextExample] = []
    seen: set[str] = set()
    for text in texts:
        normalized = " ".join(text.split())
        if not normalized or normalized.lower() in seen:
            continue
        seen.add(normalized.lower())
        extraction = extractor.extract(normalized)
        if _goal_id_set(extraction.atomic_goal_refs) != expected_ids:
            continue
        if _relation_set(extraction.relations) != expected_edges:
            continue
        accepted.append(make_goal_text_example(
            text=normalized,
            goal_refs=tuple(expected_goal_refs),
            dependencies=tuple(expected_relations),
            text_source=text_source,
            split=split,
            metadata={"validated_by": extractor.__class__.__name__},
        ))
    return tuple(accepted)


def generate_goal_text_drafts(
    provider: ParaphraseProvider,
    goal_refs: Sequence[GoalRef],
    dependencies: Sequence[GoalDependency] = (),
    count: int = 8,
    text_source: str = "llm_paraphrase",
    split: str = "train",
) -> tuple[GoalTextExample, ...]:
    """Generate text rows from an LLM provider without extractor validation.

    This is the bootstrapping path before the trained GoalExtractor exists:
    deterministic JSON-derived ``GoalRef`` rows remain the labels, while the LLM
    contributes only candidate operator wording. Downstream review/training can
    filter or score these rows, but no model output changes ``true_y``.

    :param provider: paraphrase provider.
    :param goal_refs: deterministic labels for every generated text.
    :param dependencies: optional text-level relation labels.
    :param count: requested number of drafts.
    :param text_source: source label.
    :param split: dataset split.
    :return: generated draft examples.
    """
    prompt = build_paraphrase_prompt(goal_refs, dependencies=dependencies, count=count)
    rows: list[GoalTextExample] = []
    seen: set[str] = set()
    for text in provider.generate(prompt):
        normalized = " ".join(text.split())
        if not normalized or normalized.lower() in seen:
            continue
        seen.add(normalized.lower())
        rows.append(make_goal_text_example(
            text=normalized,
            goal_refs=tuple(goal_refs),
            dependencies=tuple(dependencies),
            text_source=text_source,
            split=split,
            metadata={
                "validation": "llm_generated_unvalidated",
                "requested_count": count,
            },
        ))
    return tuple(rows)


def generate_goal_text_examples(
    provider: ParaphraseProvider,
    extractor: Any,
    goal_refs: Sequence[GoalRef],
    dependencies: Sequence[GoalDependency] = (),
    count: int = 8,
    text_source: str = "llm_paraphrase",
    split: str = "train",
) -> tuple[GoalTextExample, ...]:
    """Generate and validate text rows for one target goal set."""
    prompt = build_paraphrase_prompt(goal_refs, dependencies=dependencies, count=count)
    return validate_paraphrase_texts(
        provider.generate(prompt),
        extractor=extractor,
        expected_goal_refs=goal_refs,
        expected_relations=dependencies,
        text_source=text_source,
        split=split,
    )


# Author: Mus mbayramo@stanford.edu
