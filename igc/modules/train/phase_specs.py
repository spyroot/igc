"""Shared Phase 1/2/3 training spec loader.

The YAML file under ``configs/phase_training/`` is the run-contract gate for
Phase 1 Redfish JSON pretraining, Phase 2 ordered REST-goal extraction, and
Phase 3 ordered method/argument extraction. It keeps model ids, optimizer
knobs, PEFT settings, trainer defaults, distributed hints, and W&B metric names
in one schema instead of letting each phase grow a separate trainer dialect.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any


DEFAULT_PHASE_TRAINING_SPEC = (
    Path(__file__).resolve().parents[3] / "configs" / "phase_training" / "profiles.yaml"
)


def load_phase_training_spec(spec_file: str | Path | None = None) -> dict[str, Any]:
    """Load the shared Phase 1/2/3 training spec.

    :param spec_file: optional path to a compatible YAML spec.
    :return: parsed YAML mapping.
    :raises ValueError: if the spec schema is missing or not the expected phase schema.
    """
    spec = _read_yaml(spec_file or DEFAULT_PHASE_TRAINING_SPEC)
    schema = spec.get("schema_version")
    if schema != "igc.phase_training.v1":
        raise ValueError(
            f"phase training spec requires schema_version 'igc.phase_training.v1'; got {schema!r}"
        )
    return spec


def phase_names(spec_file: str | Path | None = None) -> list[str]:
    """Return phase names in the order declared by the spec."""
    spec = load_phase_training_spec(spec_file)
    return list(spec.get("phases") or [])


def profile_names(phase: str | None = None, spec_file: str | Path | None = None) -> list[str]:
    """Return profile names, optionally filtered to one phase.

    :param phase: one phase name from :func:`phase_names`.
    :param spec_file: optional YAML spec path.
    :return: profile names in spec order.
    """
    spec = load_phase_training_spec(spec_file)
    profiles = spec.get("profiles") or {}
    if phase is None:
        return list(profiles)
    if phase not in phase_names(spec_file):
        raise KeyError(f"unknown phase {phase!r}; choose from {phase_names(spec_file)}")
    return [name for name, profile in profiles.items() if profile.get("phase") == phase]


def default_profile_name(phase: str, spec_file: str | Path | None = None) -> str:
    """Return the default profile for a phase."""
    spec = load_phase_training_spec(spec_file)
    defaults = spec.get("default_profiles") or {}
    if phase not in defaults:
        raise KeyError(f"no default profile for phase {phase!r}")
    return str(defaults[phase])


def load_phase_profile(
    profile_name: str | None = None,
    *,
    phase: str | None = None,
    spec_file: str | Path | None = None,
) -> dict[str, Any]:
    """Resolve one phase profile, following shared refs.

    :param profile_name: profile key under ``profiles``. If omitted, ``phase``
        must be provided and its default profile is used.
    :param phase: optional phase filter/default selector.
    :param spec_file: optional YAML spec path.
    :return: resolved profile mapping with ``dataset``, ``model``,
        ``optimizer``, ``trainer``, ``distributed``, optional ``peft``, and
        tracking metrics attached.
    """
    spec = load_phase_training_spec(spec_file)
    selected = profile_name or default_profile_name(_require_phase(phase), spec_file)
    profiles = spec.get("profiles") or {}
    if selected not in profiles:
        available = ", ".join(sorted(profiles))
        raise KeyError(f"unknown phase profile {selected!r}; available: {available}")

    profile = copy.deepcopy(profiles[selected])
    profile["name"] = selected
    if phase is not None and profile.get("phase") != phase:
        raise ValueError(f"profile {selected!r} belongs to phase {profile.get('phase')!r}, not {phase!r}")

    profile["dataset"] = _resolve_ref(spec, "datasets", profile.get("dataset") or {})
    profile["model"] = _resolve_ref(spec, "models", profile.get("model") or {})
    profile["optimizer"] = _resolve_ref(spec, "optimizers", profile.get("optimizer") or {})
    profile["trainer"] = _expand_values(profile.get("trainer") or {})
    profile["distributed"] = _expand_values(profile.get("distributed") or {})
    if profile.get("peft"):
        profile["peft"] = _resolve_ref(spec, "peft", profile.get("peft") or {})
    profile["tracking"] = _resolve_metric_set(spec, profile.get("tracking") or {})
    return profile


def _require_phase(phase: str | None) -> str:
    """Return ``phase`` or raise for missing phase defaults."""
    if phase is None:
        raise ValueError("phase is required when profile_name is omitted")
    return phase


def _read_yaml(path: str | Path) -> dict[str, Any]:
    """Read one YAML file into a mapping."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("PyYAML is required to read phase training specs") from exc

    spec_path = Path(path).expanduser().resolve()
    with spec_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"phase training spec {spec_path} must contain a mapping")
    return data


def _resolve_ref(
    spec: dict[str, Any],
    section_name: str,
    value: dict[str, Any],
) -> dict[str, Any]:
    """Resolve a ``ref`` in one spec section and merge local overrides."""
    value = copy.deepcopy(value)
    ref = value.pop("ref", None)
    if ref is None:
        return _expand_values(value)

    section = spec.get(section_name) or {}
    if ref not in section:
        available = ", ".join(sorted(section))
        raise ValueError(f"unknown {section_name} ref {ref!r}; available: {available}")
    merged = copy.deepcopy(section[ref])
    merged.update(value)
    return _expand_values(merged)


def _resolve_metric_set(spec: dict[str, Any], tracking: dict[str, Any]) -> dict[str, Any]:
    """Attach metric fields/plot names requested by a tracking spec."""
    tracking = _expand_values(copy.deepcopy(tracking))
    metric_ref = tracking.get("metric_set")
    if metric_ref is None:
        return tracking

    metric_sets = spec.get("metric_sets") or {}
    if metric_ref not in metric_sets:
        available = ", ".join(sorted(metric_sets))
        raise ValueError(f"unknown metric_set {metric_ref!r}; available: {available}")
    tracking["metrics"] = copy.deepcopy(metric_sets[metric_ref])
    return tracking


def _expand_values(value: Any) -> Any:
    """Expand ``~`` and environment variables in string leaves."""
    if isinstance(value, dict):
        return {key: _expand_values(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_values(item) for item in value]
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    return value


# Author: Mus mbayramo@stanford.edu
