#!/usr/bin/env python3
"""Gate: docs.consistency — the documentation-authority contract checks.

Active documentation must describe ONE architecture: unordered Phase 2/3
contracts (`rest_api_list: list[str]`, `calls: list[Call]`), order as separate
RL-oracle evidence, two separate z_rest/z_method encoders with raw argument
values outside both, locked encoder sources with the RL freeze, and no planner
or goal ontology in the language frontend. This gate checks semantic contract
fragments — stale identifiers, example shapes, k=1/2/3 coverage, link/diagram
integrity — not just filenames. Historical terms are allowed only inside
docs/DECISIONS.md (the accepted-decision log marks superseded designs there).

Whether a change touches protected Phase 1 training/cache/profiler code is a
review-time diff property, out of scope for a repo-state gate.

Used by:
  tests/gates/test_docs_consistency.py  (offline gate; runs in `pytest -q`)
  CLI: python scripts/gates/docs_consistency.py

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS = REPO_ROOT / "docs"

# Identifiers of the superseded designs. Any hit in an ACTIVE doc is a failure;
# docs/DECISIONS.md is the only file allowed to carry them (marked historical).
STALE_IDENTIFIERS = (
    "ordered_goals",
    "text_to_ordered_rest_api_list",
    "atomic_goal_refs",
    "GoalSurface",
    "GoalRef",
    "GoalEnvelope",
    "z_sub_goal",
    "call_ordered_exact_match_rate",
    "ordered_exact_match_rate",
    "order_evidence",
    "order/kendall_tau",
    "order/edit_distance",
    "ToolCatalog",
    "ToolAction",
    "RedfishStateV0",
)

# The scalar/list-union example forms the contract forbids.
FORBIDDEN_SHAPES = ('{"call":', "{'call':")


def _active_docs() -> list[Path]:
    """All tracked active markdown docs (top-level + use_cases) plus the root README."""
    docs = sorted(DOCS.glob("*.md")) + sorted((DOCS / "use_cases").glob("*.md"))
    return [REPO_ROOT / "README.md"] + [d for d in docs if d.name != "DECISIONS.md"]


def _relative_links(text: str) -> list[str]:
    """Markdown link/image targets that are relative file paths (not URLs/anchors)."""
    targets = re.findall(r"!?\[[^\]]*\]\(([^)#\s]+)[^)]*\)", text)
    return [t for t in targets if not t.startswith(("http://", "https://", "mailto:"))]


def check_docs(root: Path = REPO_ROOT) -> list[str]:
    """Run every documentation-consistency check.

    :param root: repo root (parameterized for tests).
    :return: list of human-readable violations; empty means the docs are consistent.
    """
    docs_dir = root / "docs"
    violations: list[str] = []

    active = (
        [root / "README.md"]
        + sorted(docs_dir.glob("*.md"))
        + sorted((docs_dir / "use_cases").glob("*.md"))
    )
    active = [d for d in active if d.exists() and d.name != "DECISIONS.md"]

    # 1) Stale identifiers + forbidden example shapes in active docs.
    for doc in active:
        text = doc.read_text(encoding="utf-8")
        rel = doc.relative_to(root)
        for token in STALE_IDENTIFIERS:
            if token in text:
                violations.append(f"{rel}: stale identifier {token!r}")
        for shape in FORBIDDEN_SHAPES:
            if shape in text:
                violations.append(f"{rel}: forbidden scalar call form {shape!r}")
        # In the CONTRACT docs, 'planner' may appear only in negated statements
        # ("no planner", "not a planner"). Operational docs may legitimately name
        # unrelated planners (e.g. a token batch planner in the RL scaling plan).
        if doc.name in (
            "README.md",
            "ARCHITECTURE.md",
            "phase_2.md",
            "phase_3.md",
            "GOAL_LATENT_DESIGN.md",
        ):
            for line_no, line in enumerate(text.splitlines(), 1):
                lowered = line.lower()
                if "planner" in lowered and not any(
                    negation in lowered
                    for negation in ("no planner", "not ", "never", "without")
                ):
                    violations.append(
                        f"{rel}:{line_no}: 'planner' outside a negated statement"
                    )

    # 2) Link integrity for the two nav documents.
    for nav in (root / "README.md", docs_dir / "README.md"):
        if not nav.exists():
            violations.append(f"{nav.relative_to(root)}: nav file missing")
            continue
        for target in _relative_links(nav.read_text(encoding="utf-8")):
            if not (nav.parent / target).exists():
                violations.append(f"{nav.relative_to(root)}: dead link {target}")

    # 3) Diagram integrity: every diagram referenced somewhere active; every
    #    reference resolves (no orphan SVG, no dead diagram link).
    diagrams = sorted((docs_dir / "diagrams").glob("*.svg")) if (docs_dir / "diagrams").exists() else []
    all_text = "\n".join(d.read_text(encoding="utf-8") for d in active)
    for diagram in diagrams:
        if diagram.name not in all_text:
            violations.append(f"docs/diagrams/{diagram.name}: orphan diagram (no active doc links it)")
    for doc in active:
        for target in _relative_links(doc.read_text(encoding="utf-8")):
            if target.endswith(".svg") and not (doc.parent / target).exists():
                violations.append(f"{doc.relative_to(root)}: dead diagram link {target}")

    # 4) Phase 2/3 example shapes + k=1/2/3 coverage.
    phase2 = docs_dir / "phase_2.md"
    phase3 = docs_dir / "phase_3.md"
    if phase2.exists():
        text2 = phase2.read_text(encoding="utf-8")
        if '"rest_api_list": [' not in text2 and "rest_api_list: [" not in text2:
            violations.append("docs/phase_2.md: no rest_api_list list example")
        if '{"rest_api":' in text2:
            violations.append("docs/phase_2.md: scalar rest_api example (forbidden union form)")
        for k in ("1", "2", "3"):
            if f"k={k}" not in text2 and f"k = {k}" not in text2:
                violations.append(f"docs/phase_2.md: missing k={k} example")
    if phase3.exists():
        text3 = phase3.read_text(encoding="utf-8")
        if '"calls": [' not in text3 and "calls: [" not in text3:
            violations.append("docs/phase_3.md: no calls list example")
        for k in ("1", "2", "3"):
            if f"k={k}" not in text3 and f"k = {k}" not in text3:
                violations.append(f"docs/phase_3.md: missing k={k} example")
        if "operation_name" not in text3:
            violations.append("docs/phase_3.md: Call examples missing operation_name")

    # 5) Unordered contract + order-as-oracle-evidence stated where it matters.
    for name in ("ARCHITECTURE.md", "phase_2.md", "phase_3.md", "TRAINING.md", "GOAL_LATENT_DESIGN.md"):
        doc = docs_dir / name
        if doc.exists() and "unordered" not in doc.read_text(encoding="utf-8").lower():
            violations.append(f"docs/{name}: does not state the unordered contract")
    for name in ("ARCHITECTURE.md", "phase_3.md", "GOAL_LATENT_DESIGN.md"):
        doc = docs_dir / name
        if doc.exists() and "expert_call_order" not in doc.read_text(encoding="utf-8"):
            violations.append(f"docs/{name}: does not name expert_call_order oracle evidence")

    # 6) Latent boundary: argument values stay outside both latents; the encoder
    #    sources + RL freeze are stated.
    for name in ("ARCHITECTURE.md", "GOAL_LATENT_DESIGN.md"):
        doc = docs_dir / name
        if doc.exists() and "outside both" not in doc.read_text(encoding="utf-8"):
            violations.append(f"docs/{name}: does not keep argument values outside both latents")
    for name in ("GOAL_LATENT_DESIGN.md", "RL_SCALING_PLAN.md"):
        doc = docs_dir / name
        if not doc.exists():
            continue
        text = doc.read_text(encoding="utf-8")
        if "frozen" not in text or "nly the RL policy learns" not in text:
            violations.append(f"docs/{name}: does not state the RL encoder freeze")
        for source in ("model_x", "goal_extractor", "argument_extractor"):
            if source not in text:
                violations.append(f"docs/{name}: encoder-source table missing {source}")

    return violations


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    _ = argv
    violations = check_docs()
    for violation in violations:
        print(f"BLOCKER: {violation}", file=sys.stderr)
    if not violations:
        print(f"OK: documentation consistency checks passed ({len(_active_docs())} active docs).")
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())


# Author: Mus mbayramo@stanford.edu
