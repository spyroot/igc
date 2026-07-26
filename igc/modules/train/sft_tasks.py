"""Strict YAML task registry for the shared Phase 1/2/3 SFT engine."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from string import Template
from typing import Any, Mapping

import yaml


TASK_SPEC_PATH = (
    Path(__file__).resolve().parents[3]
    / "configs"
    / "training"
    / "sft_tasks.yaml"
)


@dataclass(frozen=True)
class SFTTaskSpec:
    """One phase's data and lineage boundary for the generic SFT engine."""

    name: str
    phase: int
    contract_version: str
    parent_role: str
    output_role: str
    dataset_format: str
    metric_namespace: str
    renderer: str
    order_sensitive: bool
    prompt_template: str
    sources: tuple[str, ...]
    input_policy: Mapping[str, Any]
    spec_sha256: str

    def render_prompt(self, values: Mapping[str, str]) -> str:
        """Render the YAML-owned prompt and reject missing placeholders."""
        try:
            return Template(self.prompt_template).substitute(values)
        except KeyError as exc:
            raise ValueError(
                f"task {self.name!r} prompt is missing value {exc.args[0]!r}"
            ) from exc


_TASK_FIELDS = {
    "phase",
    "contract_version",
    "parent_role",
    "output_role",
    "dataset_format",
    "metric_namespace",
    "renderer",
    "order_sensitive",
    "prompt_template",
    "sources",
    "input_policy",
}


def load_sft_tasks(path: str | Path = TASK_SPEC_PATH) -> dict[str, SFTTaskSpec]:
    """Load and strictly validate all SFT task specifications."""
    spec_path = Path(path)
    raw_bytes = spec_path.read_bytes()
    try:
        payload = yaml.safe_load(raw_bytes)
    except yaml.YAMLError as exc:
        raise ValueError(f"cannot parse SFT task spec {spec_path}: {exc}") from exc
    if not isinstance(payload, Mapping) or payload.get("version") != 1:
        raise ValueError("SFT task spec version must be 1")
    raw_tasks = payload.get("tasks")
    if not isinstance(raw_tasks, Mapping) or not raw_tasks:
        raise ValueError("SFT task spec must contain a non-empty tasks mapping")

    digest = hashlib.sha256(raw_bytes).hexdigest()
    from igc.ds.source_registry import load_source_registry

    known_sources = set(load_source_registry().sources)
    tasks: dict[str, SFTTaskSpec] = {}
    for name, raw in raw_tasks.items():
        if not isinstance(name, str) or not isinstance(raw, Mapping):
            raise ValueError("SFT task names must map to objects")
        missing = sorted(_TASK_FIELDS - set(raw))
        unknown = sorted(set(raw) - _TASK_FIELDS)
        if missing:
            raise ValueError(f"task {name!r} missing keys: {missing}")
        if unknown:
            raise ValueError(f"task {name!r} has unknown keys: {unknown}")
        sources = raw["sources"]
        if not isinstance(sources, list) or not all(
            isinstance(source, str) and source for source in sources
        ):
            raise ValueError(f"task {name!r} sources must be list[str]")
        unknown_sources = sorted(set(sources) - known_sources - {"accepted_labelled_requests"})
        if unknown_sources:
            raise ValueError(f"task {name!r} has unknown sources: {unknown_sources}")
        phase = int(raw["phase"])
        if phase not in (1, 2, 3):
            raise ValueError(f"task {name!r} phase must be 1, 2, or 3")
        if bool(raw["order_sensitive"]):
            raise ValueError(f"task {name!r} must be order-insensitive")
        input_policy = raw["input_policy"]
        if not isinstance(input_policy, Mapping):
            raise ValueError(f"task {name!r} input_policy must be an object")
        tasks[name] = SFTTaskSpec(
            name=name,
            phase=phase,
            contract_version=str(raw["contract_version"]),
            parent_role=str(raw["parent_role"]),
            output_role=str(raw["output_role"]),
            dataset_format=str(raw["dataset_format"]),
            metric_namespace=str(raw["metric_namespace"]),
            renderer=str(raw["renderer"]),
            order_sensitive=False,
            prompt_template=str(raw["prompt_template"]),
            sources=tuple(sources),
            input_policy=dict(input_policy),
            spec_sha256=f"sha256:{digest}",
        )
    return tasks


def resolve_sft_task(
    name: str,
    path: str | Path = TASK_SPEC_PATH,
) -> SFTTaskSpec:
    """Resolve one task by name from the YAML registry."""
    return _resolve_sft_task_cached(name, str(Path(path).expanduser().resolve()))


@lru_cache(maxsize=32)
def _resolve_sft_task_cached(name: str, path: str) -> SFTTaskSpec:
    """Resolve one immutable task once per process and exact spec path."""
    tasks = load_sft_tasks(path)
    try:
        return tasks[name]
    except KeyError as exc:
        raise KeyError(f"unknown SFT task {name!r}; choose from {sorted(tasks)}") from exc
