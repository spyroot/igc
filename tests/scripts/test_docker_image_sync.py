"""Offline tests for Docker image synchronization decisions.

The image sync path is tested with a fake runner only. It never calls a real
Docker daemon, registry, network, or private env file.

Author:
Mus mbayramo@stanford.edu
"""

import importlib.util
import pathlib
import textwrap


ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "docker_image_sync.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("docker_image_sync", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_spec(tmp_path, pull_policy="if_missing"):
    path = tmp_path / "docker.yaml"
    image_extra = ""
    if pull_policy == "build":
        image_extra = "  dockerfile: docker/Dockerfile.train\n  context: .\n"
    path.write_text(
        textwrap.dedent(
            f"""
            version: 1
            name: docker-sync
            backend: docker
            image:
              ref: igc-train:test
              pull_policy: {pull_policy}
            """,
        )
        + image_extra
        + textwrap.dedent(
            """
            runtime:
              command: [python, igc_main.py, --help]
            paths: {}
            resources: {}
            """,
        ),
        encoding="utf-8",
    )
    return path


class FakeDocker:
    """Subprocess stand-in for Docker commands."""

    def __init__(self, inspect_rc=0, pull_rc=0, build_rc=0, push_rc=0):
        self.inspect_rc = inspect_rc
        self.pull_rc = pull_rc
        self.build_rc = build_rc
        self.push_rc = push_rc
        self.calls = []

    def run(self, cmd):
        self.calls.append(cmd)
        if cmd[:3] == ["docker", "image", "inspect"]:
            return self.inspect_rc, "", "missing"
        if cmd[:2] == ["docker", "pull"]:
            return self.pull_rc, "pulled", "pull failed"
        if cmd[:2] == ["docker", "build"]:
            return self.build_rc, "built", "build failed"
        if cmd[:2] == ["docker", "push"]:
            return self.push_rc, "pushed", "push failed"
        return 127, "", f"unexpected command: {cmd}"


def test_if_missing_uses_existing_image_without_pull(tmp_path):
    """An existing image is reused; no pull/build happens."""
    module = _load_module()
    runner = FakeDocker(inspect_rc=0)
    assert module.sync_image(_write_spec(tmp_path), runner=runner) == 0
    assert runner.calls == [["docker", "image", "inspect", "igc-train:test"]]


def test_if_missing_reports_pull_failure(tmp_path, capsys):
    """A failed pull is reported as a blocker with no build fallback."""
    module = _load_module()
    runner = FakeDocker(inspect_rc=1, pull_rc=42)
    assert module.sync_image(_write_spec(tmp_path), runner=runner) == 1
    assert "BLOCKER: docker pull failed" in capsys.readouterr().out
    assert ["docker", "build"] not in [call[:2] for call in runner.calls]


def test_docker_command_failure_is_reported(tmp_path, capsys):
    """Missing daemon/binary style failures still produce a blocker."""
    module = _load_module()
    runner = FakeDocker(inspect_rc=127, pull_rc=127)
    assert module.sync_image(_write_spec(tmp_path), runner=runner) == 1
    output = capsys.readouterr().out
    assert "BLOCKER: docker pull failed rc=127" in output


def test_build_policy_builds_and_optionally_pushes(tmp_path):
    """Build policy builds once and pushes only when requested."""
    module = _load_module()
    runner = FakeDocker()
    assert module.sync_image(_write_spec(tmp_path, pull_policy="build"), runner=runner, push=True) == 0
    assert any(call[:2] == ["docker", "build"] for call in runner.calls)
    assert any(call[:2] == ["docker", "push"] for call in runner.calls)


def test_locked_digest_fails_when_image_is_absent(tmp_path, capsys):
    """Locked images are inspected, never pulled opportunistically."""
    module = _load_module()
    runner = FakeDocker(inspect_rc=1)
    assert module.sync_image(_write_spec(tmp_path, pull_policy="locked_digest"), runner=runner) == 1
    assert "BLOCKER: required docker image is not present" in capsys.readouterr().out
    assert not any(call[:2] == ["docker", "pull"] for call in runner.calls)


# Author: Mus mbayramo@stanford.edu
