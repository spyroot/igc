#!/usr/bin/env python3
"""Gate: repo.agent-leak-guard — nothing agent-internal leaks to the outside.

The shared-code repo (GitHub + its internal mirror) must never carry agent
context: agent/instruction files stay untracked, commit and PR/MR messages
carry no agent attribution or agent-file chatter, and internal endpoints
(lab IPs, internal hostnames) never appear in tracked files. The private
context repo on the internal GitLab is where agent files live — this gate is
the outbound fence.

Five checks (binding: agent tokens appear NOWHERE outward — not in branches,
not in PR/MR titles or bodies, not in commit messages, not in tracked prose):

1. tracked files — no agent artifact is committed (the binding artifact list);
2. commit messages (a range) and an optional PR/MR body — no agent tokens at
   all, no attribution trailers, no "Generated with ..." footers, no
   agent-file mentions, no internal context paths;
3. the branch name itself — no agent tokens (branches are public);
4. tracked file contents — no agent tokens in prose. The ignore/config files
   that EXCLUDE the agent artifacts (.gitignore, .dockerignore, pytest.ini)
   are the fence itself and are exempt, as are this gate and its tests;
5. tracked file contents — no internal endpoint literals (lab IP ranges,
   internal hostnames); env-var indirection is the allowed form.

Used by:
  tests/gates/test_agent_leak_guard.py  (offline gate; runs in `pytest -q`)
  .gitlab-ci.yml cpu-gate + the GitHub Actions gate job
  CLI: python scripts/gates/agent_leak_guard.py [--commit-range A..B]
       [--message-file FILE]

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

import argparse
import fnmatch
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Agent artifacts that must never be tracked in the shared-code repo.
FORBIDDEN_TRACKED = (
    "CLAUDE.md",
    "AGENTS.md",
    "AGENT_BOOTSTRAP.md",
    "AGENT_HANDOFF.md",
    "CODEX_HANDOFF.md",
    "CODEX_TASKS.md",
    "TEAM_GUIDE.md",
    "COORDINATION.md",
    "IMPROVEMENT_PLAN.md",
    "CLAUDE_REVIEW.md*",
    "CLAUDE_PLAN.md",
    "CLAUDE_PATCH.diff",
    ".claude/*",
    ".codex/*",
    ".internal/*",
    ".agent-review/*",
)

# Message patterns that ARE leaks. The agent-token rule is absolute: the
# tokens must not appear in messages, PR/MR text, or branch names in any form
# (including branch refs inside merge-commit subjects).
AGENT_TOKEN = re.compile(r"claude|codex|anthropic", re.I)
MESSAGE_PATTERNS = (
    (AGENT_TOKEN, "agent token"),
    (re.compile(r"co-authored-by:", re.I), "attribution trailer"),
    (re.compile(r"\b(CLAUDE|AGENTS|TEAM_GUIDE|COORDINATION|AGENT_HANDOFF|CODEX_TASKS)\.md\b"), "agent file mention"),
    (re.compile(r"\.internal/"), "internal context path"),
    (re.compile(r"\bagent (handoff|session|worker)\b", re.I), "agent session chatter"),
)

# Tracked-content prose must be token-free too. The exclusion configs that keep
# the artifacts OUT are the fence itself and stay exempt.
_TOKEN_SCAN_EXEMPT = {
    ".gitignore",
    ".dockerignore",
    "pytest.ini",
    "scripts/gates/agent_leak_guard.py",
    "tests/gates/test_agent_leak_guard.py",
}

# The token scan targets PROSE AND CODE, not data assets (a tokenizer
# vocabulary legitimately contains every common word). Endpoint literals are
# scanned everywhere regardless.
_TOKEN_SCAN_SUFFIXES = {
    ".py", ".sh", ".bash", ".bats", ".md", ".rst", ".txt",
    ".yaml", ".yml", ".toml", ".cfg", ".ini",
}

# Internal endpoint literals that must never appear in tracked content.
ENDPOINT_PATTERNS = (
    (re.compile(r"\b172\.25\.230\.\d{1,3}\b"), "lab node IP"),
    (re.compile(r"\b10\.12\.2\.\d{1,3}\b"), "lab storage IP"),
    (re.compile(r"\bgitlab\.rnd\.[a-z0-9.-]+\b", re.I), "internal GitLab hostname"),
    (re.compile(r"\brnd\.embedings\.ai\b", re.I), "internal domain"),
    (re.compile(r"nv72-brain://", re.I), "internal MCP resource"),
)

# The gate and its tests carry the detection patterns/fixtures as literals —
# they are the ONLY files exempt from the endpoint scan.
_ENDPOINT_SCAN_EXEMPT = {
    "scripts/gates/agent_leak_guard.py",
    "tests/gates/test_agent_leak_guard.py",
}

# Binary-ish assets are skipped by extension instead of content sniffing.
_SKIP_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".pt", ".bin", ".safetensors", ".npy", ".tar", ".gz"}


def _git(args: list[str], cwd: Path) -> str:
    """Run git and return stdout (empty string on failure)."""
    try:
        return subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
        ).stdout
    except subprocess.CalledProcessError:
        return ""


def check_tracked_files(repo: Path = REPO_ROOT) -> list[str]:
    """No agent artifact may be tracked."""
    violations: list[str] = []
    for path in _git(["ls-files"], repo).splitlines():
        name = Path(path).name
        for pattern in FORBIDDEN_TRACKED:
            if fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(name, pattern):
                violations.append(f"tracked agent artifact: {path}")
                break
    return violations


def scan_message(message: str, label: str) -> list[str]:
    """Scan one commit/PR/MR message for agent leaks."""
    violations: list[str] = []
    for pattern, why in MESSAGE_PATTERNS:
        if pattern.search(message):
            violations.append(f"{label}: {why} ({pattern.pattern})")
    return violations


def check_commit_messages(commit_range: str, repo: Path = REPO_ROOT) -> list[str]:
    """Scan every commit message (subject+body+trailers) in the range."""
    violations: list[str] = []
    raw = _git(["log", "--format=%H%x01%B%x02", commit_range], repo)
    for entry in raw.split("\x02"):
        entry = entry.strip()
        if not entry:
            continue
        sha, _, body = entry.partition("\x01")
        violations += scan_message(body, f"commit {sha[:9]}")
    return violations


def check_branch_name(branch: str) -> list[str]:
    """The branch name itself is public and must carry no agent token."""
    if branch and AGENT_TOKEN.search(branch):
        return [f"branch name {branch!r}: agent token"]
    return []


def check_endpoint_literals(repo: Path = REPO_ROOT) -> list[str]:
    """No internal endpoint literal (and no agent token) in tracked contents."""
    violations: list[str] = []
    for path in _git(["ls-files"], repo).splitlines():
        file_path = repo / path
        if file_path.suffix.lower() in _SKIP_SUFFIXES or not file_path.is_file():
            continue
        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if path not in _ENDPOINT_SCAN_EXEMPT:
            for pattern, why in ENDPOINT_PATTERNS:
                match = pattern.search(text)
                if match:
                    violations.append(f"{path}: {why} ({match.group(0)})")
        if path not in _TOKEN_SCAN_EXEMPT and file_path.suffix.lower() in _TOKEN_SCAN_SUFFIXES:
            match = AGENT_TOKEN.search(text)
            if match:
                violations.append(f"{path}: agent token in tracked content ({match.group(0)})")
    return violations


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    :return: process exit code (0 = no leaks).
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--commit-range",
        default="",
        help="git range to scan (e.g. origin/main..HEAD); omitted/unresolvable -> skipped.",
    )
    parser.add_argument(
        "--message-file",
        default="",
        help="Optional file carrying the PR/MR title+body to scan.",
    )
    parser.add_argument(
        "--branch",
        default="",
        help="Branch name under review (source branch of the PR/MR).",
    )
    args = parser.parse_args(argv)

    violations = check_tracked_files() + check_endpoint_literals()
    violations += check_branch_name(args.branch)
    if args.commit_range:
        # Skip gracefully when the range is unresolvable (shallow clone).
        if _git(["rev-list", "-n", "1", args.commit_range], REPO_ROOT):
            violations += check_commit_messages(args.commit_range)
        else:
            print(f"note: commit range {args.commit_range!r} unresolvable — skipped")
    if args.message_file:
        text = Path(args.message_file).read_text(encoding="utf-8", errors="ignore")
        violations += scan_message(text, "PR/MR message")

    for violation in violations:
        print(f"BLOCKER: {violation}", file=sys.stderr)
    if not violations:
        print("OK: no agent leaks in tracked files, messages, or endpoints.")
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())


# Author: Mus mbayramo@stanford.edu
