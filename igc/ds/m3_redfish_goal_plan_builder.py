"""Build M3 goal-planner JSONL from ``redfish_ctl`` captures.

This builder is the bridge from the existing Redfish crawl/capture corpus
to M3 supervised fine-tuning. It reads the offline ``~/.json_responses`` tree
written by ``redfish_ctl`` (and by legacy ``idrac_ctl`` runs that preserved the
same output contract), recursively merges ``rest_api_map.npy`` files, extracts
discovered Redfish resources and action targets, then emits
:class:`M3GoalPlanRecord` JSONL examples.

It never crawls a live controller and never writes captured response bodies to
the training JSONL. The output contains only action names, target paths,
methods, argument schemas, and M3 goal-plan targets.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from igc.ds.m3_goal_plan_dataset import M3GoalPlanJsonlDataset


@dataclass
class M3RedfishActionCatalogBuilder:
    """Extract an M3 action catalog from offline ``redfish_ctl`` captures."""

    json_root: str | Path | Iterable[str | Path]
    rest_api_map_dir: Optional[str | Path | Iterable[str | Path]] = None
    source: str = "redfish_ctl_capture"
    vendor: Optional[str] = None

    def action_catalog(self) -> list[dict[str, Any]]:
        """Return deduplicated action records for every discovered Redfish action."""
        allowed_methods = self._load_allowed_methods()
        records = []
        seen = set()
        for path, body in self._iter_json_bodies():
            url = _resource_url(body, path)
            methods = _allowed_for_url(url, allowed_methods)
            for record in _resource_method_records(url, body, methods, self.source, self.vendor):
                key = _record_key(record)
                if key not in seen:
                    seen.add(key)
                    records.append(record)
            for record in _redfish_action_records(url, body, self.source, self.vendor):
                key = _record_key(record)
                if key not in seen:
                    seen.add(key)
                    records.append(record)
            for record in _patchable_allowable_value_records(url, body, methods, self.source, self.vendor):
                key = _record_key(record)
                if key not in seen:
                    seen.add(key)
                    records.append(record)
        records.sort(key=lambda item: (str(item.get("target")), str(item.get("action"))))
        return records

    def build_dataset(self, templates_per_action: int = 3) -> M3GoalPlanJsonlDataset:
        """Build an M3 JSONL dataset from the extracted catalog."""
        return M3GoalPlanJsonlDataset.from_redfish_action_catalog(
            self.action_catalog(),
            templates_per_action=templates_per_action,
        )

    def write_jsonl(self, output_jsonl: str | Path, templates_per_action: int = 3) -> None:
        """Write M3 goal-plan JSONL records to ``output_jsonl``."""
        self.build_dataset(templates_per_action=templates_per_action).write_jsonl(output_jsonl)

    def _iter_json_bodies(self) -> Iterable[tuple[Path, dict[str, Any]]]:
        """Yield parsed Redfish resource JSON objects."""
        for root in _roots(self.json_root):
            for path in sorted(root.rglob("*.json")):
                try:
                    text = path.read_text(encoding="utf-8", errors="replace")
                    body = json.loads(text)
                except (OSError, json.JSONDecodeError):
                    continue
                if isinstance(body, dict) and _looks_like_redfish_resource(body):
                    yield path, body

    def _load_allowed_methods(self) -> dict[str, list[str]]:
        """Load and merge ``allowed_methods_mapping`` from capture ``*.npy`` maps."""
        try:
            import numpy as np
        except ImportError:
            return {}
        merged: dict[str, list[str]] = {}
        map_roots = self.rest_api_map_dir if self.rest_api_map_dir is not None else self.json_root
        for root in _roots(map_roots):
            maps = sorted(root.rglob("*.npy"), key=lambda item: item.stat().st_mtime)
            for mapping_path in maps:
                try:
                    mapping = np.load(mapping_path, allow_pickle=True).item()
                except (OSError, ValueError, EOFError):
                    continue
                allowed = mapping.get("allowed_methods_mapping") or {}
                for url, methods in allowed.items():
                    normalized_url = _normalize_url(str(url))
                    merged[normalized_url] = sorted({str(method).upper() for method in methods})
        return merged


def _resource_url(body: dict[str, Any], path: Path) -> str:
    """Return resource URL from ``@odata.id`` or filename fallback."""
    value = body.get("@odata.id")
    if isinstance(value, str) and value:
        return _normalize_url(value)
    return "/" + path.name[:-5].strip("_").replace("_", "/")


def _roots(value: str | Path | Iterable[str | Path]) -> list[Path]:
    """Normalize one or more corpus roots into existing paths."""
    if isinstance(value, (str, Path)):
        values = [value]
    else:
        values = list(value)
    roots = []
    for item in values:
        root = Path(os.path.expanduser(str(item)))
        if root.is_dir():
            roots.append(root)
    return roots


def _looks_like_redfish_resource(body: dict[str, Any]) -> bool:
    """Return whether a JSON body is a captured Redfish resource."""
    redfish_markers = (
        "@odata.id",
        "@odata.type",
        "@odata.context",
        "RedfishVersion",
        "Actions",
    )
    return any(marker in body for marker in redfish_markers)


def _allowed_for_url(url: str, allowed_methods: dict[str, list[str]]) -> list[str]:
    """Return allowed methods for a URL with conservative GET fallback."""
    normalized_url = _normalize_url(url)
    methods = allowed_methods.get(normalized_url) or allowed_methods.get(normalized_url.rstrip("/")) or []
    if methods:
        return sorted(set(method.upper() for method in methods))
    return ["GET"]


def _resource_method_records(
    url: str,
    body: dict[str, Any],
    methods: list[str],
    source: str,
    vendor: Optional[str],
) -> Iterable[dict[str, Any]]:
    """Emit one generic action record per discovered resource method."""
    resource_type = str(body.get("@odata.type", "")).lstrip("#")
    for method in methods:
        action = f"{method}_{_resource_name(url)}"
        yield {
            "action": action,
            "target": url,
            "method": method,
            "allowed_methods": methods,
            "resource_type": resource_type,
            "source": source,
            "vendor": vendor,
            "description": f"{method} {url}",
        }


def _redfish_action_records(
    url: str,
    body: dict[str, Any],
    source: str,
    vendor: Optional[str],
) -> Iterable[dict[str, Any]]:
    """Emit Redfish ``Actions`` records with target and argument schema."""
    actions = body.get("Actions")
    if not isinstance(actions, dict):
        return
    for action_name, action_body in actions.items():
        if not isinstance(action_body, dict):
            continue
        target = action_body.get("target") or action_body.get("Target")
        if not isinstance(target, str):
            continue
        arguments = _allowable_arguments(action_body)
        yield {
            "action": str(action_name).lstrip("#"),
            "target": target,
            "method": "POST",
            "allowed_methods": ["POST"],
            "arguments": arguments,
            "resource": url,
            "source": source,
            "vendor": vendor,
            "description": f"{action_name} on {url}",
        }


def _patchable_allowable_value_records(
    url: str,
    body: dict[str, Any],
    methods: list[str],
    source: str,
    vendor: Optional[str],
) -> Iterable[dict[str, Any]]:
    """Emit PATCH records for allowable-value fields on mutable resources."""
    if "PATCH" not in methods:
        return
    arguments = _allowable_arguments(body)
    for field_name, values in sorted(arguments.items()):
        clean_field = field_name.replace("@Redfish.AllowableValues", "")
        yield {
            "action": f"Set_{_resource_name(url)}_{clean_field}",
            "target": url,
            "method": "PATCH",
            "allowed_methods": methods,
            "arguments": {clean_field: values},
            "source": source,
            "vendor": vendor,
            "description": f"PATCH {clean_field} on {url}",
        }


def _allowable_arguments(body: dict[str, Any]) -> dict[str, Any]:
    """Extract ``@Redfish.AllowableValues`` argument hints from a body."""
    out = {}
    for key, value in body.items():
        if key.endswith("@Redfish.AllowableValues"):
            out[key.replace("@Redfish.AllowableValues", "")] = value
    return out


def _resource_name(url: str) -> str:
    """Return readable resource/action name from a URL."""
    return url.rstrip("/").rsplit("/", 1)[-1] or "resource"


def _record_key(record: dict[str, Any]) -> tuple[str, str, str, str, str]:
    """Dedup key for action records."""
    return (
        str(record.get("action", "")),
        str(record.get("target", "")),
        str(record.get("method", "")),
        str(record.get("resource_type", "")),
        json.dumps(record.get("arguments") or {}, sort_keys=True),
    )


def _normalize_url(url: str) -> str:
    """Normalize an absolute or host-relative Redfish URL to a path."""
    if "://" in url:
        from urllib.parse import urlparse

        parsed = urlparse(url)
        url = parsed.path or "/"
    if not url.startswith("/"):
        url = f"/{url}"
    return url.rstrip("/") or "/"


M3RedfishCtlGoalPlanDatasetBuilder = M3RedfishActionCatalogBuilder


# Author: Mus mbayramo@stanford.edu
