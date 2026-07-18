#!/usr/bin/env python3
"""Gate: repo.agent-leak-guard — nothing agent-internal leaks to the outside.

The shared-code repo (GitHub + its internal mirror) must never carry agent
context: agent/instruction files stay untracked, commit and PR/MR messages
carry no agent attribution or agent-file chatter, and internal endpoints
(lab IPs, internal hostnames) never appear in tracked files. The private
context repo on the internal GitLab is where agent files live — this gate is
the outbound fence.

Three checks:

1. tracked files — no agent artifact is committed (the binding artifact list);
2. commit messages (a range) and an optional PR/MR body — no agent
   attribution trailers, no "Generated with ..." footers, no agent-file
   mentions, no session chatter. ``claude/*`` / ``codex/*`` BRANCH names are
   the established public naming convention and are not leaks; patterns below
   are anchored so branch refs in merge commits pass.
3. tracked file contents — no internal endpoint literals (lab IP ranges,
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

# Message patterns that ARE leaks. Deliberately anchored/specific so the
# established public branch naming (claude/<topic>, codex/<topic>) passes.
MESSAGE_PATTERNS = (
    (re.compile(r"co-authored-by:.*(claude|anthropic)", re.I), "agent attribution trailer"),
    (re.compile(r"noreply@anthropic\.com", re.I), "agent attribution email"),
    (re.compile(r"generated with.*claude", re.I), "agent generation footer"),
    (re.compile(r"\bclaude code\b", re.I), "agent tool name"),
    (re.compile(r"\b(claude|codex)\s+(session|worker|agent|pass)\b", re.I), "agent session chatter"),
    (re.compile(r"\b(CLAUDE|AGENTS|TEAM_GUIDE|COORDINATION|AGENT_HANDOFF|CODEX_TASKS)\.md\b"), "agent file mention"),
    (re.compile(r"\.internal/"), "internal context path"),
    (re.compile(r"\bagent handoff\b", re.I), "agent session chatter"),
)

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


def check_endpoint_literals(repo: Path = REPO_ROOT) -> list[str]:
    """No internal endpoint literal in tracked file contents."""
    violations: list[str] = []
    for path in _git(["ls-files"], repo).splitlines():
        if path in _ENDPOINT_SCAN_EXEMPT:
            continue
        file_path = repo / path
        if file_path.suffix.lower() in _SKIP_SUFFIXES or not file_path.is_file():
            continue
        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for pattern, why in ENDPOINT_PATTERNS:
            match = pattern.search(text)
            if match:
                violations.append(f"{path}: {why} ({match.group(0)})")
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
    args = parser.parse_args(argv)

    violations = check_tracked_files() + check_endpoint_literals()
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
