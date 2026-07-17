"""Gate: repo.contract-single-source — one canonical Phase 2/3 REST-goal contract.

This repo-level gate reads ``configs/contracts/rest_goal.yaml`` and fails when a
change reintroduces the exact class of defect that PR #108 slipped past green
per-module tests:

* a forked namespace token (e.g. ``phase2_goal_extract`` instead of the canonical
  ``phase2_goal_extraction``) appears anywhere under the live package, or
* a canonical contract symbol (row builder / renderer / parser / schema class) is
  ``def``/``class``-defined in a module OTHER than the one canonical module, i.e. a
  parallel "contract island".

It also fails if the canonical module is missing or no longer defines a symbol the
registry claims it owns, so the registry cannot silently drift from reality.

Used by ``.github/workflows/ci.yml`` (gate job) and, later, the home-lab GitLab
k8s runner via the platform gate registry. Runs offline, imports nothing from the
package (pure source scan), so it is safe on any CI surface.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path
from typing import Any

import yaml


def _load_registry(path: Path) -> dict[str, Any]:
    """Load and minimally validate the contract registry.

    :param path: path to ``configs/contracts/rest_goal.yaml``.
    :return: parsed registry mapping.
    :raises SystemExit: if the registry is missing or malformed.
    """
    if not path.is_file():
        _fail([f"contract registry not found: {path}"])
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        _fail([f"contract registry is not a mapping: {path}"])
    for key in ("canonical_module", "search_root", "forbidden_namespace_literals", "canonical_symbols"):
        if key not in data:
            _fail([f"contract registry missing required key: {key}"])
    return data


def _iter_py_files(root: Path) -> list[Path]:
    """Return every ``*.py`` file under ``root`` (sorted, deterministic)."""
    return sorted(p for p in root.rglob("*.py") if p.is_file())


def _forbidden_hits(files: list[Path], literals: list[str]) -> list[str]:
    """Find whole-token occurrences of forbidden namespace literals.

    Whole-token (``\\bTOKEN\\b``) matching is required so a canonical value such as
    ``phase2_goal_extraction`` does NOT trip the ``phase2_goal_extract`` fork token.

    :param files: source files to scan.
    :param literals: forbidden namespace tokens from the registry.
    :return: list of ``path:line: token`` violation messages.
    """
    patterns = [(lit, re.compile(rf"\b{re.escape(lit)}\b")) for lit in literals]
    hits: list[str] = []
    for path in files:
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            for literal, pattern in patterns:
                if pattern.search(line):
                    hits.append(f"{path}:{lineno}: forbidden forked namespace token '{literal}'")
    return hits


def _symbol_definitions(files: list[Path], symbols: set[str]) -> dict[str, list[str]]:
    """Map each canonical symbol to the files that ``def``/``class``-define it.

    :param files: source files to scan.
    :param symbols: canonical symbol names owned by the one canonical module.
    :return: ``{symbol: [file, ...]}`` for symbols actually defined somewhere.
    """
    found: dict[str, list[str]] = {name: [] for name in symbols}
    for path in files:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:  # a broken source file is a separate gate's job
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.name in symbols:
                    found[node.name].append(str(path))
    return found


def check(registry_path: Path, repo_root: Path) -> list[str]:
    """Run the single-source contract checks; return violation messages (empty = pass).

    :param registry_path: path to the contract registry YAML.
    :param repo_root: repository root the registry paths are relative to.
    :return: list of human-readable violations; empty list means the gate passes.
    """
    registry = _load_registry(registry_path)
    canonical_module = (repo_root / registry["canonical_module"]).resolve()
    search_root = (repo_root / registry["search_root"]).resolve()
    forbidden = list(registry["forbidden_namespace_literals"])
    canonical_symbols = set(registry["canonical_symbols"])

    violations: list[str] = []
    if not canonical_module.is_file():
        violations.append(f"canonical contract module missing: {canonical_module}")
    if not search_root.is_dir():
        violations.append(f"search_root not found: {search_root}")
        return violations

    files = _iter_py_files(search_root)

    # (1) No forked namespace tokens anywhere in the live package.
    violations.extend(_forbidden_hits(files, forbidden))

    # (2) Every canonical symbol is defined ONLY in the canonical module.
    definitions = _symbol_definitions(files, canonical_symbols)
    for symbol, where in definitions.items():
        outside = [w for w in where if Path(w).resolve() != canonical_module]
        if outside:
            joined = ", ".join(sorted(outside))
            violations.append(
                f"canonical symbol '{symbol}' defined outside {registry['canonical_module']} "
                f"(parallel contract island): {joined}"
            )
        if canonical_module.is_file() and str(canonical_module) not in where:
            violations.append(
                f"registry claims '{symbol}' but it is not defined in the canonical module "
                f"{registry['canonical_module']} (stale registry)"
            )
    return violations


def _fail(messages: list[str]) -> None:
    """Print BLOCKER-style messages and exit non-zero."""
    for message in messages:
        print(f"BLOCKER: {message}", file=sys.stderr)
    raise SystemExit(1)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the ``repo.contract-single-source`` gate."""
    parser = argparse.ArgumentParser(description="Assert a single canonical Phase 2/3 REST-goal contract.")
    parser.add_argument(
        "--registry",
        default="configs/contracts/rest_goal.yaml",
        help="Path to the contract registry YAML (relative to repo root).",
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root the registry paths resolve against.",
    )
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve()
    violations = check(repo_root / args.registry, repo_root)
    if violations:
        _fail(violations)
    print("OK: single canonical Phase 2/3 REST-goal contract; no forked namespaces or islands.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# Author: Mus mbayramo@stanford.edu
