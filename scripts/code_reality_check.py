"""
Reality check: is a symbol actually used by LIVE igc code, or only by docs/tests/scripts?

The rule this enforces (see TEAM_GUIDE "Verify in code"): docs and a passing test do NOT prove a
thing exists or is wired. This tool answers the mechanical half of that — for each given symbol
(class/function name) it greps the tree and buckets every reference into live code (``igc/`` minus
tests), tests, offline scripts, or docs, then prints a verdict so a plan can lead with reality
instead of intent. It does NOT decide producer-vs-consumer (that still needs a read), but it
reliably catches the failure mode that bit us: a symbol that exists only in docs/tests/scripts and
touches nothing the running system executes.

TWO-TIER by design. This grep is the FAST first pass: NOT-FOUND / DOC-ONLY / TEST-ONLY / OFFLINE-ONLY
are DEFINITIVE ("we do not have it"). But ``LIVE-REF`` is only *necessary, not sufficient* for wired —
the hit may be a config string (e.g. an arg-parser default) or a dead chain (referenced by a file that
is itself unwired). A ``LIVE-REF`` symbol still needs a read to confirm real producer/consumer wiring;
use the deeper reachability audit (the ``code-reality-check`` workflow) for the true WIRED verdict.

Used by: run before trusting any plan/claim about igc — ``python scripts/code_reality_check.py
RedfishStateV0 Igc_PointerQNetwork q_learning_target`` — and paste the table at the top of the plan.
Consumed by humans/agents and CI-style gating; imports nothing from igc so it never drags the model
stack in. Verdicts: LIVE-REF (referenced in live code beyond its own file), SELF-ONLY (defined,
referenced only within its own definition file), OFFLINE/TEST-ONLY (only scripts/tests), DOC-ONLY,
NOT-FOUND (no definition in the tree).

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

_REPO = Path(__file__).resolve().parents[1]
_CODE_EXT = (".py",)
_DOC_EXT = (".md", ".rst", ".txt")


def _zone(path: Path) -> str:
    """Bucket a file into live | test | script | doc | other by its repo-relative path."""
    rel = path.relative_to(_REPO).as_posix()
    name = path.name
    if path.suffix in _DOC_EXT:
        return "doc"
    if name.startswith("test_") or name.endswith("_test.py") or "/tests/" in f"/{rel}":
        return "test"
    if rel.startswith("scripts/"):
        return "script"
    if rel.startswith("igc/"):
        return "live"
    return "other"


def _iter_files() -> List[Path]:
    """All searchable source/doc files under the repo, skipping .git and the submodule tree."""
    out: List[Path] = []
    for p in _REPO.rglob("*"):
        if not p.is_file() or p.suffix not in _CODE_EXT + _DOC_EXT:
            continue
        rel = p.relative_to(_REPO).as_posix()
        if rel.startswith((".git/", "idrac_ctl/", ".venv/", "build/")):
            continue
        out.append(p)
    return out


def check(symbol: str, files: List[Path]) -> Dict:
    """Find the definition and every reference of ``symbol``, bucketed by zone.

    :param symbol: a class or function name to look for (word-boundary matched).
    :param files: the pre-collected file list to scan.
    :return: dict with the definition location, per-zone reference file:line lists, and a verdict.
    """
    def_pat = re.compile(rf"^\s*(class|def)\s+{re.escape(symbol)}\b")
    ref_pat = re.compile(rf"\b{re.escape(symbol)}\b")
    definition = ""
    def_file = None
    refs: Dict[str, List[str]] = defaultdict(list)
    for path in files:
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        rel = path.relative_to(_REPO).as_posix()
        zone = _zone(path)
        for i, line in enumerate(lines, 1):
            if not ref_pat.search(line):
                continue
            if not definition and def_pat.search(line):
                definition = f"{rel}:{i}"
                def_file = rel
            refs[zone].append(f"{rel}:{i}")

    live = [r for r in refs["live"] if r.split(":")[0] != def_file]
    verdict = _verdict(definition, live, refs, def_file)
    return {
        "symbol": symbol,
        "definition": definition or "NOT-FOUND",
        "live_refs": live,
        "test_refs": refs["test"],
        "script_refs": refs["script"],
        "doc_refs": refs["doc"],
        "verdict": verdict,
    }


def _verdict(definition: str, live: List[str], refs: Dict[str, List[str]], def_file) -> str:
    """Classify from where the symbol is actually referenced."""
    if not definition and not any(refs.values()):
        return "NOT-FOUND"
    if not definition:
        # referenced but never defined in the tree (e.g. a stale/typo name)
        return "UNDEFINED-REF"
    if live:
        return "LIVE-REF"
    if refs["script"]:
        return "OFFLINE-ONLY"
    if refs["test"]:
        return "TEST-ONLY"
    if any(r.split(":")[0] == def_file for r in refs["live"]):
        return "SELF-ONLY"
    if refs["doc"]:
        return "DOC-ONLY"
    return "SELF-ONLY"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("symbols", nargs="+", help="class/function names to reality-check")
    ap.add_argument("--show", type=int, default=3, help="max live ref lines to print per symbol")
    args = ap.parse_args()

    files = _iter_files()
    width = max(len(s) for s in args.symbols)
    print(f"{'symbol':<{width}}  {'verdict':<13}  definition / where it actually lives")
    print("-" * (width + 60))
    for sym in args.symbols:
        r = check(sym, files)
        note = r["definition"]
        if r["verdict"] == "LIVE-REF":
            note += "  live: " + ", ".join(r["live_refs"][: args.show])
        elif r["verdict"] == "OFFLINE-ONLY":
            note += "  scripts-only: " + ", ".join(r["script_refs"][: args.show])
        elif r["verdict"] in ("TEST-ONLY", "SELF-ONLY", "DOC-ONLY", "UNDEFINED-REF"):
            pool = r["test_refs"] or r["doc_refs"] or r["script_refs"]
            if pool:
                note += "  only: " + ", ".join(pool[: args.show])
        print(f"{sym:<{width}}  {r['verdict']:<13}  {note}")


if __name__ == "__main__":
    main()


# Author: Mus mbayramo@stanford.edu
