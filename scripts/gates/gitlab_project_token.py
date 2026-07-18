#!/usr/bin/env python3
"""Gate: gitlab.project-token.* — the per-project token is present, bound, and contained.

Every project gets its own PROJECT access token (never a personal or group
token), provisioned by the operator into a gitignored env file. Four checks,
one gate id each:

* ``gitlab.project-token.exists`` — the env file provides GITLAB_URL,
  GITLAB_PROJECT_PATH, GITLAB_PROJECT_ID, and a well-formed
  GITLAB_PROJECT_TOKEN (never printed).
* ``gitlab.project-token.project-bound`` — the token authenticates as the
  project bot user of EXACTLY the configured project
  (``project_<id>_bot...``), proving it is a project token bound to this
  project, not a personal/group token.
* ``gitlab.project-token.api-access`` — the token reads its own project and
  pipelines (HTTP 200, path matches the configured project path).
* ``gitlab.project-token.no-cross-project-access`` — canary project paths
  (GITLAB_CANARY_PROJECTS, comma-separated) are all DENIED (404/403); a 200
  on any canary means the token leaks across projects.

This is a SHARED gate class: portable across projects — all endpoints and
identities come from the env file (no hostname or project name in this file),
so the same gate ships in every repo's gate set. Offline tests inject a fake
fetcher; the live run needs network to the configured GitLab.

Used by:
  tests/gates/test_gitlab_project_token.py  (offline; runs in `pytest -q`)
  CLI: python scripts/gates/gitlab_project_token.py --env-file <path>

Author:
Mus mbayramo@stanford.edu
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

# A fetcher returns (http_status, parsed_json_or_None). Injectable for tests.
Fetcher = Callable[[str, str], tuple[int, Any]]

_TOKEN_SHAPE = re.compile(r"^glpat-[A-Za-z0-9._-]{20,}$")
_REQUIRED_VARS = (
    "GITLAB_URL",
    "GITLAB_PROJECT_PATH",
    "GITLAB_PROJECT_ID",
    "GITLAB_PROJECT_TOKEN",
)


def load_env_file(path: str | Path) -> dict[str, str]:
    """Parse a KEY=value env file (comments/blank lines ignored)."""
    env: dict[str, str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def _default_fetcher(url: str, token: str) -> tuple[int, Any]:
    """GET with the token header, bypassing any local proxy (VPN-local host)."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    request = urllib.request.Request(url, headers={"PRIVATE-TOKEN": token})
    try:
        with opener.open(request, timeout=10) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, None
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return 0, None


def _quote_path(project_path: str) -> str:
    """URL-encode a namespace/project path for the projects API."""
    return project_path.replace("/", "%2F")


def check_exists(env: dict[str, str]) -> list[str]:
    """gitlab.project-token.exists — config complete, token well-formed."""
    violations = [f"exists: missing {v}" for v in _REQUIRED_VARS if not env.get(v)]
    token = env.get("GITLAB_PROJECT_TOKEN", "")
    if token and not _TOKEN_SHAPE.match(token):
        violations.append("exists: token is not a well-formed project access token")
    if not str(env.get("GITLAB_PROJECT_ID", "")).isdigit():
        violations.append("exists: GITLAB_PROJECT_ID must be numeric")
    return violations


def check_project_bound(env: dict[str, str], fetch: Fetcher) -> list[str]:
    """gitlab.project-token.project-bound — token is THIS project's bot."""
    status, body = fetch(f"{env['GITLAB_URL']}/api/v4/user", env["GITLAB_PROJECT_TOKEN"])
    if status != 200 or not isinstance(body, dict):
        return [f"project-bound: /user returned HTTP {status}"]
    username = str(body.get("username", ""))
    expected_prefix = f"project_{env['GITLAB_PROJECT_ID']}_bot"
    if not username.startswith(expected_prefix):
        return [
            "project-bound: token identity is not the bound project bot "
            f"(expected prefix {expected_prefix}, got {username or 'unknown'!r})"
        ]
    return []


def check_api_access(env: dict[str, str], fetch: Fetcher) -> list[str]:
    """gitlab.project-token.api-access — reads its own project + pipelines."""
    violations: list[str] = []
    base = f"{env['GITLAB_URL']}/api/v4/projects/{_quote_path(env['GITLAB_PROJECT_PATH'])}"
    status, body = fetch(base, env["GITLAB_PROJECT_TOKEN"])
    if status != 200 or not isinstance(body, dict):
        violations.append(f"api-access: project read returned HTTP {status}")
    elif body.get("path_with_namespace") != env["GITLAB_PROJECT_PATH"]:
        violations.append(
            "api-access: project path mismatch "
            f"({body.get('path_with_namespace')!r} != {env['GITLAB_PROJECT_PATH']!r})"
        )
    status, _ = fetch(f"{base}/pipelines?per_page=1", env["GITLAB_PROJECT_TOKEN"])
    if status != 200:
        violations.append(f"api-access: pipelines read returned HTTP {status}")
    return violations


def check_no_cross_project(env: dict[str, str], fetch: Fetcher) -> list[str]:
    """gitlab.project-token.no-cross-project-access — canaries all denied."""
    canaries = [c.strip() for c in env.get("GITLAB_CANARY_PROJECTS", "").split(",") if c.strip()]
    if not canaries:
        return ["no-cross-project-access: no GITLAB_CANARY_PROJECTS configured (need >= 1 canary)"]
    violations: list[str] = []
    for canary in canaries:
        url = f"{env['GITLAB_URL']}/api/v4/projects/{_quote_path(canary)}"
        status, _ = fetch(url, env["GITLAB_PROJECT_TOKEN"])
        if status not in (403, 404):
            violations.append(
                f"no-cross-project-access: canary {canary} returned HTTP {status} "
                "(must be denied with 403/404)"
            )
    return violations


def run_all(env: dict[str, str], fetch: Fetcher = _default_fetcher) -> dict[str, list[str]]:
    """Run the four checks; later checks are skipped when exists fails.

    :param env: parsed env config (token value is used, never echoed).
    :param fetch: HTTP fetcher, injectable for offline tests.
    :return: {gate_id: [violations]} for the four gate ids.
    """
    results = {"gitlab.project-token.exists": check_exists(env)}
    if results["gitlab.project-token.exists"]:
        skip = ["skipped: exists failed"]
        results["gitlab.project-token.project-bound"] = skip
        results["gitlab.project-token.api-access"] = skip
        results["gitlab.project-token.no-cross-project-access"] = skip
        return results
    results["gitlab.project-token.project-bound"] = check_project_bound(env, fetch)
    results["gitlab.project-token.api-access"] = check_api_access(env, fetch)
    results["gitlab.project-token.no-cross-project-access"] = check_no_cross_project(env, fetch)
    return results


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    :return: 0 when all four gates pass, 1 otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env-file",
        default=".internal/gitlab-igc-api-key.env",
        help="Gitignored env file with GITLAB_URL/PROJECT_PATH/PROJECT_ID/PROJECT_TOKEN"
        " (+ GITLAB_CANARY_PROJECTS). Defaults to this project's token file;"
        " other projects pass their own.",
    )
    args = parser.parse_args(argv)

    if not Path(args.env_file).exists():
        print(f"BLOCKER: gitlab.project-token.exists: env file not found: {args.env_file}", file=sys.stderr)
        return 1
    results = run_all(load_env_file(args.env_file))
    failed = 0
    for gate_id, violations in results.items():
        if violations:
            failed += 1
            for violation in violations:
                print(f"BLOCKER: {gate_id}: {violation}", file=sys.stderr)
        else:
            print(f"OK: {gate_id}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())


# Author: Mus mbayramo@stanford.edu
