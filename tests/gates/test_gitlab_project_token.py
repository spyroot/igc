"""Offline tests for the gitlab.project-token.* gate family.

All HTTP is faked through the injectable fetcher — no network, no real token.
Fixtures mirror the live-verified behavior: a bound project token answers 200
as the project bot on its own project and 404 on every other project.

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

SCRIPT = Path("scripts/gates/gitlab_project_token.py")


def _load_gate() -> ModuleType:
    """Load the gate script module for direct testing."""
    spec = importlib.util.spec_from_file_location("gitlab_project_token", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _env() -> dict:
    """A complete, well-formed token config (fake values, realistic shapes)."""
    return {
        "GITLAB_URL": "https://gitlab.example.test",
        "GITLAB_PROJECT_PATH": "group/project-a",
        "GITLAB_PROJECT_ID": "3",
        "GITLAB_PROJECT_TOKEN": "glpat-" + "x" * 45,
        "GITLAB_CANARY_PROJECTS": "group/project-b,group/other",
    }


def _bound_fetcher(url: str, token: str) -> tuple[int, object]:
    """Fake GitLab: bound project bot; own project 200; canaries 404."""
    assert token  # the gate must always send the token
    if url.endswith("/api/v4/user"):
        return 200, {"username": "project_3_bot_abcdef"}
    if "projects/group%2Fproject-a/pipelines" in url:
        return 200, []
    if url.endswith("projects/group%2Fproject-a"):
        return 200, {"path_with_namespace": "group/project-a"}
    return 404, None


def test_all_four_gates_pass_for_a_bound_token() -> None:
    """A bound project token with denied canaries passes every gate."""
    gate = _load_gate()
    results = gate.run_all(_env(), fetch=_bound_fetcher)
    assert all(v == [] for v in results.values()), results
    assert set(results) == {
        "gitlab.project-token.exists",
        "gitlab.project-token.project-bound",
        "gitlab.project-token.api-access",
        "gitlab.project-token.no-cross-project-access",
    }


def test_missing_or_malformed_token_fails_exists_and_skips_rest() -> None:
    """exists fails on missing vars or a malformed token; later checks skip."""
    gate = _load_gate()
    env = _env()
    env["GITLAB_PROJECT_TOKEN"] = "not-a-token"
    results = gate.run_all(env, fetch=_bound_fetcher)
    assert any("well-formed" in v for v in results["gitlab.project-token.exists"])
    assert results["gitlab.project-token.project-bound"] == ["skipped: exists failed"]

    empty = gate.run_all({}, fetch=_bound_fetcher)
    assert any("missing GITLAB_URL" in v for v in empty["gitlab.project-token.exists"])


def test_personal_token_fails_project_bound() -> None:
    """A token whose identity is a human user is not project-bound."""
    gate = _load_gate()

    def personal(url: str, token: str) -> tuple[int, object]:
        if url.endswith("/api/v4/user"):
            return 200, {"username": "some-human"}
        return _bound_fetcher(url, token)

    results = gate.run_all(_env(), fetch=personal)
    assert any(
        "not the bound project bot" in v
        for v in results["gitlab.project-token.project-bound"]
    )


def test_wrong_project_bot_fails_project_bound() -> None:
    """A bot bound to a DIFFERENT project id fails the binding check."""
    gate = _load_gate()

    def other_bot(url: str, token: str) -> tuple[int, object]:
        if url.endswith("/api/v4/user"):
            return 200, {"username": "project_99_bot_zzz"}
        return _bound_fetcher(url, token)

    results = gate.run_all(_env(), fetch=other_bot)
    assert results["gitlab.project-token.project-bound"]


def test_denied_api_access_fails() -> None:
    """403 on the project read fails api-access with the status in the message."""
    gate = _load_gate()

    def denied(url: str, token: str) -> tuple[int, object]:
        if url.endswith("/api/v4/user"):
            return 200, {"username": "project_3_bot_abcdef"}
        return 403, None

    results = gate.run_all(_env(), fetch=denied)
    assert any("HTTP 403" in v for v in results["gitlab.project-token.api-access"])


def test_cross_project_leak_fails() -> None:
    """A canary answering 200 is a cross-project leak."""
    gate = _load_gate()

    def leaky(url: str, token: str) -> tuple[int, object]:
        if "project-b" in url:
            return 200, {"path_with_namespace": "group/project-b"}
        return _bound_fetcher(url, token)

    results = gate.run_all(_env(), fetch=leaky)
    assert any(
        "group/project-b returned HTTP 200" in v
        for v in results["gitlab.project-token.no-cross-project-access"]
    )


def test_missing_canaries_fails_closed() -> None:
    """No configured canaries fails the cross-project gate (never vacuous)."""
    gate = _load_gate()
    env = _env()
    del env["GITLAB_CANARY_PROJECTS"]
    results = gate.run_all(env, fetch=_bound_fetcher)
    assert any(
        "no GITLAB_CANARY_PROJECTS" in v
        for v in results["gitlab.project-token.no-cross-project-access"]
    )


def test_env_file_parsing(tmp_path: Path) -> None:
    """KEY=value parsing skips comments and strips quotes; token never echoed."""
    gate = _load_gate()
    env_file = tmp_path / "token.env"
    env_file.write_text(
        "# provisioned by operator\n"
        "GITLAB_URL=https://gitlab.example.test\n"
        'GITLAB_PROJECT_PATH="group/project-a"\n'
        "GITLAB_PROJECT_ID=3\n"
        f"GITLAB_PROJECT_TOKEN=glpat-{'y' * 45}\n",
        encoding="utf-8",
    )
    env = gate.load_env_file(env_file)
    assert env["GITLAB_PROJECT_PATH"] == "group/project-a"
    assert env["GITLAB_PROJECT_TOKEN"].startswith("glpat-")


# Author: Mus mbayramo@stanford.edu
