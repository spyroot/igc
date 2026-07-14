#!/usr/bin/env python
"""Synchronize a Docker image according to an igc run spec.

This script is safe to test offline: command execution is isolated behind a
runner, and ``--dry-run`` prints the Docker calls without touching the daemon.
Live mode uses only the image section of the public run spec and never reads
private env files.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Protocol

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from igc.shared.run_spec import RunSpecError, load_run_spec


class Runner(Protocol):
    """Protocol for Docker command execution, mockable in tests."""

    def run(self, cmd: list[str]) -> tuple[int, str, str]:
        """Run a command and return ``(returncode, stdout, stderr)``."""


class SubprocessRunner:
    """Subprocess-backed runner for live Docker commands."""

    def run(self, cmd: list[str]) -> tuple[int, str, str]:
        """Run a Docker command and capture text output."""
        result = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            check=False,
        )
        return result.returncode, result.stdout, result.stderr


def render_sync_plan(spec_path: str | Path, *, push: bool = False) -> str:
    """Render Docker image-sync commands without executing them."""
    spec = load_run_spec(spec_path)
    image = str(spec.image["ref"])
    policy = str(spec.image.get("pull_policy", "if_missing"))
    lines = ["# igc docker image sync dry-run"]

    if policy == "if_missing":
        lines.append(
            f"docker image inspect {_shell(image)} >/dev/null 2>&1 "
            f"|| docker pull {_shell(image)}"
        )
    elif policy == "always":
        lines.append(f"docker pull {_shell(image)}")
    elif policy == "build":
        lines.append(_build_command(spec.image, image))
    elif policy in {"never", "locked_digest"}:
        lines.append(f"docker image inspect {_shell(image)} >/dev/null")
    else:
        raise RunSpecError(f"unsupported image.pull_policy: {policy}")

    if push:
        lines.append(f"docker push {_shell(image)}")
    return "\n".join(lines) + "\n"


def sync_image(
        spec_path: str | Path,
        *,
        runner: Runner | None = None,
        push: bool = False,
) -> int:
    """Synchronize the Docker image for a run spec.

    :param spec_path: YAML run spec.
    :param runner: Docker command runner, injectable for tests.
    :param push: push the image after a successful build/pull/inspect.
    :return: process exit code.
    """
    try:
        spec = load_run_spec(spec_path)
    except RunSpecError as exc:
        print(f"RUN SPEC ERROR: {exc}")
        return 2

    image = str(spec.image["ref"])
    policy = str(spec.image.get("pull_policy", "if_missing"))
    runner = runner or SubprocessRunner()

    if policy == "if_missing":
        if _inspect(runner, image):
            print(f"docker image present: {image}")
        elif not _pull(runner, image):
            return 1
    elif policy == "always":
        if not _pull(runner, image):
            return 1
    elif policy == "build":
        if not _build(runner, spec.image, image):
            return 1
    elif policy in {"never", "locked_digest"}:
        if not _inspect(runner, image):
            print(f"BLOCKER: required docker image is not present: {image}")
            return 1
    else:
        print(f"RUN SPEC ERROR: unsupported image.pull_policy: {policy}")
        return 2

    if push and not _push(runner, image):
        return 1
    return 0


def _inspect(runner: Runner, image: str) -> bool:
    rc, _, _ = runner.run(["docker", "image", "inspect", image])
    return rc == 0


def _pull(runner: Runner, image: str) -> bool:
    rc, out, err = runner.run(["docker", "pull", image])
    if rc != 0:
        print(f"BLOCKER: docker pull failed rc={rc}: {(err or out).strip()}")
        return False
    print(f"docker pull OK: {image}")
    return True


def _build(runner: Runner, image_cfg: dict, image: str) -> bool:
    dockerfile = str(image_cfg.get("dockerfile", "docker/Dockerfile.train"))
    context = str(image_cfg.get("context", "."))
    rc, out, err = runner.run(["docker", "build", "-f", dockerfile, "-t", image, context])
    if rc != 0:
        print(f"BLOCKER: docker build failed rc={rc}: {(err or out).strip()}")
        return False
    print(f"docker build OK: {image}")
    return True


def _push(runner: Runner, image: str) -> bool:
    rc, out, err = runner.run(["docker", "push", image])
    if rc != 0:
        print(f"BLOCKER: docker push failed rc={rc}: {(err or out).strip()}")
        return False
    print(f"docker push OK: {image}")
    return True


def _build_command(image_cfg: dict, image: str) -> str:
    dockerfile = str(image_cfg.get("dockerfile", "docker/Dockerfile.train"))
    context = str(image_cfg.get("context", "."))
    return (
        f"docker build -f {_shell(dockerfile)} "
        f"-t {_shell(image)} {_shell(context)}"
    )


def _shell(value: str) -> str:
    """Return a copy/paste-safe shell token for dry-run output."""
    return shlex.quote(str(value))


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, help="YAML run spec")
    parser.add_argument("--dry-run", action="store_true", help="print commands only")
    parser.add_argument("--push", action="store_true", help="push after sync")
    args = parser.parse_args(argv)

    try:
        if args.dry_run:
            sys.stdout.write(render_sync_plan(args.spec, push=args.push))
            return 0
    except RunSpecError as exc:
        print(f"RUN SPEC ERROR: {exc}", file=sys.stderr)
        return 2

    return sync_image(args.spec, push=args.push)


if __name__ == "__main__":
    raise SystemExit(main())
