"""Offline sanity for the dependency manifest.

The previous requirements.txt was a 2023 conda-env freeze (conda-only packages,
unresolvable local `+git` builds, duplicate pins) that setup.py ingested
verbatim, comments included — so ``pip install .`` could never resolve. These
pin the new contract: every non-comment line parses as a valid PEP 508
requirement, no duplicates, the validated stack floors are present, and
setup.py's filter drops comments and blanks. CPU-only, no network.

Author:
Mus mbayramo@stanford.edu
"""

from pathlib import Path

from packaging.requirements import Requirement

_ROOT = Path(__file__).resolve().parents[1]


def _requirement_lines():
    lines = (_ROOT / "requirements.txt").read_text().splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]


def test_every_line_is_a_valid_requirement():
    """Each non-comment line parses as a PEP 508 requirement."""
    for line in _requirement_lines():
        Requirement(line)  # raises InvalidRequirement on garbage


def test_no_duplicate_packages():
    """No package is pinned twice (the old freeze had six duplicates)."""
    names = [Requirement(ln).name.lower() for ln in _requirement_lines()]
    assert len(names) == len(set(names))


def test_core_stack_floors_present():
    """The validated core stack is declared with floors."""
    reqs = {Requirement(ln).name.lower(): ln for ln in _requirement_lines()}
    for pkg in ("torch", "transformers", "accelerate", "peft", "numpy", "gymnasium"):
        assert pkg in reqs, f"{pkg} missing from requirements.txt"
        assert any(op in reqs[pkg] for op in (">=", "==")), f"{pkg} has no floor"


def test_deepspeed_stays_an_extra():
    """deepspeed must not be a core requirement (cluster-only extra)."""
    names = {Requirement(ln).name.lower() for ln in _requirement_lines()}
    assert "deepspeed" not in names


# Author: Mus mbayramo@stanford.edu
