"""Gate: only one canonical Phase 2/3 REST-goal contract exists.

The gate reads ``configs/contracts/rest_goal.yaml`` and scans repo source/docs for
parallel-contract fingerprints. It also verifies that the canonical symbols are
defined only in ``igc/ds/rest_goal_contract.py``. This catches the class of
mistake where a stale branch introduces a second contract module with green
isolated tests.
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = REPO_ROOT / "configs/contracts/rest_goal.yaml"


@dataclass(frozen=True)
class Registry:
    """Validated contract-authority registry."""

    canonical_module: Path
    scan_paths: tuple[Path, ...]
    file_suffixes: tuple[str, ...]
    allow_paths: frozenset[Path]
    required_symbols: tuple[str, ...]
    forbidden_regex: tuple[re.Pattern[str], ...]


def _repo_rel(path: Path, root: Path) -> Path:
    """Return a repo-relative path for diagnostics."""

    return path.resolve().relative_to(root.resolve())


def _load_registry(path: Path, root: Path) -> Registry:
    """Load and validate the YAML registry."""

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("registry must be a mapping")
    if payload.get("version") != 1:
        raise ValueError("registry.version must be 1")

    canonical = root / _required_str(payload, "canonical_module")
    if not canonical.is_file():
        raise ValueError(f"canonical module missing: {_repo_rel(canonical, root)}")

    scan_paths = tuple(root / item for item in _required_str_list(payload, "scan_paths"))
    suffixes = tuple(_required_str_list(payload, "file_suffixes"))
    allow_paths = frozenset(Path(item) for item in _required_str_list(payload, "allow_paths"))
    symbols = tuple(_required_str_list(payload, "required_symbols"))
    patterns = tuple(re.compile(item) for item in _required_str_list(payload, "forbidden_regex"))

    return Registry(
        canonical_module=_repo_rel(canonical, root),
        scan_paths=scan_paths,
        file_suffixes=suffixes,
        allow_paths=allow_paths,
        required_symbols=symbols,
        forbidden_regex=patterns,
    )


def _required_str(payload: dict[str, Any], key: str) -> str:
    """Return a required string registry field."""

    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"registry.{key} must be a non-empty string")
    return value


def _required_str_list(payload: dict[str, Any], key: str) -> list[str]:
    """Return a required string-list registry field."""

    value = payload.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"registry.{key} must be a list[str]")
    return value


def _iter_scan_files(registry: Registry, root: Path) -> Iterable[Path]:
    """Yield repo files covered by the registry scan."""

    for raw_path in registry.scan_paths:
        if not raw_path.exists():
            continue
        if raw_path.is_file():
            if raw_path.suffix in registry.file_suffixes:
                yield raw_path
            continue
        for child in sorted(raw_path.rglob("*")):
            if child.is_file() and child.suffix in registry.file_suffixes:
                rel = _repo_rel(child, root)
                if _is_generated_or_private(rel):
                    continue
                yield child


def _is_generated_or_private(rel: Path) -> bool:
    """Skip local-only or generated material that must never be public evidence."""

    parts = rel.parts
    return (
        ".git" in parts
        or ".codex" in parts
        or ".claude" in parts
        or ".internal" in parts
        or parts[:2] == ("docs", "internal")
        or parts[:1] == ("reports",)
    )


def _definition_pattern(symbol: str) -> re.Pattern[str]:
    """Return a pattern for top-level class/function definitions."""

    return re.compile(rf"^(?:class|def)\s+{re.escape(symbol)}\b", re.MULTILINE)


def check(registry_path: Path = DEFAULT_REGISTRY, root: Path = REPO_ROOT) -> list[str]:
    """Return contract-authority violations; an empty list is a pass."""

    registry = _load_registry(registry_path, root)
    violations: list[str] = []
    definitions: dict[str, list[Path]] = {symbol: [] for symbol in registry.required_symbols}

    for path in _iter_scan_files(registry, root):
        rel = _repo_rel(path, root)
        text = path.read_text(encoding="utf-8", errors="replace")
        if rel not in registry.allow_paths:
            for pattern in registry.forbidden_regex:
                if pattern.search(text):
                    violations.append(f"{rel}: forbidden token matched {pattern.pattern}")
        if path.suffix == ".py":
            for symbol in registry.required_symbols:
                if _definition_pattern(symbol).search(text):
                    definitions[symbol].append(rel)

    for symbol, paths in definitions.items():
        if paths != [registry.canonical_module]:
            rendered = ", ".join(str(path) for path in paths) or "none"
            violations.append(
                f"{symbol}: expected only {registry.canonical_module}, found {rendered}"
            )
    return violations


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--root", default=str(REPO_ROOT))
    args = parser.parse_args(argv)

    try:
        violations = check(Path(args.registry), Path(args.root))
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"CONTRACT_SINGLE_SOURCE_FAIL {exc}", file=sys.stderr)
        return 1
    if violations:
        print("CONTRACT_SINGLE_SOURCE_FAIL", file=sys.stderr)
        for violation in violations:
            print(f"- {violation}", file=sys.stderr)
        return 1
    print("CONTRACT_SINGLE_SOURCE_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
