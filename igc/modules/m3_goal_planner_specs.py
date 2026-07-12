"""Spec loader for M3 goal-planner training profiles.

The M3 trainer is configured from YAML profiles under
``configs/m3_goal_planner/``. Profiles keep model ids, optimizer knobs,
precision, distributed launcher hints, dataset paths, and output paths together
so GPU runs are reproducible and do not depend on scattered shell flags.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Optional


DEFAULT_M3_GOAL_PLANNER_SPEC = (
    Path(__file__).resolve().parents[2] / "configs" / "m3_goal_planner" / "profiles.yaml"
)


def load_m3_goal_planner_profile(
    profile_name: Optional[str] = None,
    spec_file: str | Path | None = None,
) -> dict[str, Any]:
    """Load and resolve one M3 training profile.

    :param profile_name: Profile name under ``profiles``. If omitted, the
        spec's ``default_profile`` is used.
    :param spec_file: YAML spec path. Defaults to
        ``configs/m3_goal_planner/profiles.yaml``.
    :return: Resolved profile dictionary with ``dataset``, ``model``,
        ``optimizer``, ``trainer``, ``distributed``, and optional ``peft``.
    """
    spec = _read_yaml(spec_file or DEFAULT_M3_GOAL_PLANNER_SPEC)
    profiles = spec.get("profiles") or {}
    selected = profile_name or spec.get("default_profile")
    if not selected:
        raise ValueError("M3 spec requires default_profile or --profile")
    if selected not in profiles:
        available = ", ".join(sorted(profiles))
        raise ValueError(f"unknown M3 profile {selected!r}; available: {available}")

    profile = copy.deepcopy(profiles[selected])
    profile["name"] = selected
    profile["dataset"] = _resolve_ref(spec, "datasets", profile.get("dataset") or {})
    profile["model"] = _resolve_ref(spec, "models", profile.get("model") or {})
    profile["optimizer"] = _resolve_ref(spec, "optimizers", profile.get("optimizer") or {})
    profile["trainer"] = _expand_values(profile.get("trainer") or {})
    profile["distributed"] = _expand_values(profile.get("distributed") or {})
    if profile.get("peft"):
        profile["peft"] = _resolve_ref(spec, "peft", profile.get("peft") or {})
    profile["tracking"] = _resolve_metric_set(spec, profile.get("tracking") or {})
    return profile


def _read_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML spec file."""
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read M3 training specs") from exc
    spec_path = Path(path).expanduser().resolve()
    with spec_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"M3 spec {spec_path} must contain a mapping")
    return data


def _resolve_ref(
    spec: dict[str, Any],
    section_name: str,
    value: dict[str, Any],
) -> dict[str, Any]:
    """Resolve ``ref`` inside a spec section and merge local overrides."""
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


def _resolve_metric_set(
    spec: dict[str, Any],
    tracking: dict[str, Any],
) -> dict[str, Any]:
    """Attach metric field/plot names requested by a tracking spec."""
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
