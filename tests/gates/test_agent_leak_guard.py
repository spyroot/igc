"""Offline tests for the agent-leak guard.

The real repo must be clean (no tracked agent artifacts, no internal endpoint
literals); fixture repos prove each leak class is detected; the established
public branch naming (claude/<topic>, codex/<topic>) must NOT be flagged.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

SCRIPT = Path("scripts/gates/agent_leak_guard.py")


def _load_gate() -> ModuleType:
    """Load the gate script module for direct testing."""
    spec = importlib.util.spec_from_file_location("agent_leak_guard", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _git(repo: Path, *args: str) -> None:
    """Run git in the fixture repo."""
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
        cwd=repo, check=True, capture_output=True,
    )


def _fixture_repo(tmp_path: Path) -> Path:
    """A tiny git repo with one clean tracked file."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(repo, "add", "module.py")
    _git(repo, "commit", "-q", "-m", "Add module")
    return repo


def test_real_repo_is_clean() -> None:
    """The shared-code repo tracks no agent artifacts and no endpoint literals."""
    gate = _load_gate()
    assert gate.check_tracked_files() == []
    assert gate.check_endpoint_literals() == []


def test_tracked_agent_artifact_fails(tmp_path: Path) -> None:
    """A committed CLAUDE.md (or .internal file) is a leak."""
    gate = _load_gate()
    repo = _fixture_repo(tmp_path)
    (repo / "CLAUDE.md").write_text("agent instructions", encoding="utf-8")
    (repo / ".internal").mkdir()
    (repo / ".internal" / "notes.md").write_text("internal", encoding="utf-8")
    _git(repo, "add", "-f", "CLAUDE.md", ".internal/notes.md")
    _git(repo, "commit", "-q", "-m", "oops")

    violations = gate.check_tracked_files(repo)
    assert any("CLAUDE.md" in v for v in violations)
    assert any(".internal/notes.md" in v for v in violations)


def test_endpoint_literal_fails(tmp_path: Path) -> None:
    """A lab IP or internal hostname in tracked content is a leak."""
    gate = _load_gate()
    repo = _fixture_repo(tmp_path)
    (repo / "config.py").write_text(
        'HOST = "172.25.230.42"\nGL = "gitlab.rnd.example"\n', encoding="utf-8"
    )
    _git(repo, "add", "config.py")
    _git(repo, "commit", "-q", "-m", "Add config")

    violations = gate.check_endpoint_literals(repo)
    assert any("lab node IP" in v for v in violations)
    assert any("internal GitLab hostname" in v for v in violations)


def test_agent_tokens_and_chatter_in_messages_fail(tmp_path: Path) -> None:
    """Any agent token, attribution, or agent-file mention in a message is a leak."""
    gate = _load_gate()

    for message, expected in (
        ("Fix bug\n\nCo-Authored-By: A Tool <x@y>", "attribution trailer"),
        ("Fix bug\n\nGenerated with Claude Code", "agent token"),
        ("Update per CLAUDE.md instructions", "agent file mention"),
        ("Sync notes to .internal/board", "internal context path"),
        ("Handled by the codex worker overnight", "agent token"),
        # Branch refs inside merge subjects leak the token too — no whitelist.
        ("Merge pull request #142 from spyroot/claude/phase23-semantics", "agent token"),
        ("Merge branch 'codex/p0-fix' into main", "agent token"),
        ("Reviewed by anthropic tooling", "agent token"),
    ):
        violations = gate.scan_message(message, "m")
        assert violations, message
        assert any(expected in v for v in violations), (message, violations)


def test_clean_message_passes() -> None:
    """Ordinary engineering messages carry no flags."""
    gate = _load_gate()
    for message in (
        "Make Phase 3 an unordered bound-call set with explicit methods",
        "Merge pull request #200 from spyroot/phase23-semantics",
        "Fix sample-width reporting for no-action runs",
    ):
        assert gate.scan_message(message, "m") == [], message


def test_agent_token_in_branch_name_fails() -> None:
    """Branch names are public; agent tokens in them are leaks."""
    gate = _load_gate()
    assert gate.check_branch_name("claude/phase23-semantics")
    assert gate.check_branch_name("codex/p0-fix")
    assert gate.check_branch_name("phase23-semantics") == []
    assert gate.check_branch_name("") == []


def test_agent_token_in_tracked_content_fails(tmp_path: Path) -> None:
    """Agent tokens in tracked prose are leaks; exclusion configs are exempt."""
    gate = _load_gate()
    repo = _fixture_repo(tmp_path)
    (repo / "notes.py").write_text("# reviewed by the codex lane\n", encoding="utf-8")
    (repo / ".gitignore").write_text("CLAUDE.md\n.codex/\n", encoding="utf-8")
    _git(repo, "add", "notes.py", ".gitignore")
    _git(repo, "commit", "-q", "-m", "Add notes")

    violations = gate.check_endpoint_literals(repo)
    assert any("notes.py" in v and "agent token" in v for v in violations)
    assert not any(".gitignore" in v for v in violations)


def test_commit_range_scan_detects_leaky_commit(tmp_path: Path) -> None:
    """A leaky message inside the scanned range is caught."""
    gate = _load_gate()
    repo = _fixture_repo(tmp_path)
    base = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()
    (repo / "module.py").write_text("VALUE = 2\n", encoding="utf-8")
    _git(repo, "commit", "-aq", "-m", "Tune value\n\nCo-Authored-By: Claude <noreply@anthropic.com>")

    violations = gate.check_commit_messages(f"{base}..HEAD", repo)
    assert any("attribution" in v for v in violations)


# Author: Mus mbayramo@stanford.edu
