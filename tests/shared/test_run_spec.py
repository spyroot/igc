"""Offline tests for the spec-driven run launcher contract.

These tests pin the pure, non-mutating part of the run orchestration path:
YAML parsing, validation, redaction, and command rendering. They never call
Docker, Slurm, a GPU, a private endpoint, or a gitignored env file.

Author:
Mus mbayramo@stanford.edu
"""

import textwrap

import pytest

from igc.shared.run_spec import (
    RunSpecError,
    load_run_spec,
    redact_value,
    render_docker_plan,
    render_slurm_plan,
)


def _write_spec(tmp_path, body: str):
    path = tmp_path / "run.yaml"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


def _docker_spec(tmp_path):
    return _write_spec(
        tmp_path,
        """
        version: 1
        name: docker-smoke
        backend: docker
        image:
          ref: igc-train:test
          pull_policy: if_missing
        runtime:
          command:
            - python
            - igc_main.py
            - --help
          env:
            WANDB_PROJECT: igc
            WANDB_API_KEY: ${WANDB_API_KEY}
        paths:
          code: ${IGC_CODE_DIR}
          data: ${IGC_DATA_DIR}
          output: ${IGC_OUTPUT_DIR}
        resources:
          gpus: 4
        docker:
          env_files:
            - ${IGC_DOCKER_ENV_FILE}
          mounts:
            - source: ${IGC_CODE_DIR}
              target: /workspace/igc
            - source: ${IGC_DATA_DIR}
              target: /root/.json_responses
              read_only: true
        sanity:
          dry_run: true
        """,
    )


def _slurm_spec(tmp_path):
    return _write_spec(
        tmp_path,
        """
        version: 1
        name: slurm-smoke
        backend: slurm
        image:
          ref: igc-train:test
          pull_policy: if_missing
        runtime:
          command:
            - python
            - igc_main.py
            - --help
        paths:
          code: ${IGC_CODE_DIR}
          data: ${IGC_DATA_DIR}
          output: ${IGC_OUTPUT_DIR}
        resources:
          gpus: 4
          nodes: 15
          cpus_per_task: 16
          wall_time: "00:10:00"
        slurm:
          partition: ${IGC_SLURM_PARTITION}
          account: ${IGC_SLURM_ACCOUNT}
          output: ${IGC_OUTPUT_DIR}/slurm-%j.out
          extra_args:
            - --exclusive
        sanity:
          slurm:
            enabled: true
            sentinel: IGC_SLURM_SANITY_OK
            poll_seconds: 5
            timeout_seconds: 120
        """,
    )


def test_load_docker_spec_preserves_private_env_file_placeholder(tmp_path):
    """Docker env files are referenced but never read by the parser."""
    spec = load_run_spec(_docker_spec(tmp_path))
    assert spec.backend == "docker"
    assert spec.docker["env_files"] == ["${IGC_DOCKER_ENV_FILE}"]
    assert spec.resources["gpus"] == 4


def test_unknown_top_level_key_is_rejected(tmp_path):
    """Unknown keys fail early so misspelled knobs do not silently noop."""
    path = _write_spec(
        tmp_path,
        """
        version: 1
        name: bad
        backend: docker
        image: {ref: igc:test}
        runtime: {command: [python, --version]}
        paths: {}
        resources: {}
        typo_key: nope
        """,
    )
    with pytest.raises(RunSpecError, match="unknown top-level keys"):
        load_run_spec(path)


def test_literal_secret_value_is_rejected(tmp_path):
    """Secret-looking env names must use runtime placeholders, not literals."""
    path = _write_spec(
        tmp_path,
        """
        version: 1
        name: bad-secret
        backend: docker
        image: {ref: igc:test}
        runtime:
          command: [python, --version]
          env:
            WANDB_API_KEY: plain-text-token
        paths: {}
        resources: {}
        """,
    )
    with pytest.raises(RunSpecError, match="WANDB_API_KEY"):
        load_run_spec(path)


def test_unsupported_backend_is_rejected(tmp_path):
    """Backend selection is explicit and closed over docker/slurm."""
    path = _write_spec(
        tmp_path,
        """
        version: 1
        name: bad-backend
        backend: local
        image: {ref: igc:test}
        runtime: {command: [python, --version]}
        paths: {}
        resources: {}
        """,
    )
    with pytest.raises(RunSpecError, match="backend"):
        load_run_spec(path)


def test_dangerous_image_ref_is_rejected(tmp_path):
    """Image refs are names, not shell snippets."""
    path = _write_spec(
        tmp_path,
        """
        version: 1
        name: bad-image
        backend: docker
        image: {ref: "igc:latest; echo bad"}
        runtime: {command: [python, --version]}
        paths: {}
        resources: {}
        """,
    )
    with pytest.raises(RunSpecError, match="image.ref"):
        load_run_spec(path)


def test_dangerous_docker_env_file_is_rejected(tmp_path):
    """Docker env-file entries may be placeholders, but not shell snippets."""
    path = _write_spec(
        tmp_path,
        """
        version: 1
        name: bad-env-file
        backend: docker
        image: {ref: igc:test}
        runtime: {command: [python, --version]}
        paths: {}
        resources: {}
        docker:
          env_files:
            - "$(cat secret)"
        """,
    )
    with pytest.raises(RunSpecError, match="docker.env_files"):
        load_run_spec(path)


def test_dangerous_runtime_command_is_rejected(tmp_path):
    """Runtime command entries are argv tokens, not shell snippets."""
    path = _write_spec(
        tmp_path,
        """
        version: 1
        name: bad-command
        backend: slurm
        image: {ref: igc:test}
        runtime: {command: [echo, "$(whoami)"]}
        paths: {}
        resources: {}
        """,
    )
    with pytest.raises(RunSpecError, match="runtime.command"):
        load_run_spec(path)


def test_dangerous_slurm_sanity_command_is_rejected(tmp_path):
    """The scheduler sanity command is tiny and must not be a shell program."""
    path = _write_spec(
        tmp_path,
        """
        version: 1
        name: bad-sanity
        backend: slurm
        image: {ref: igc:test}
        runtime: {command: [python, --version]}
        paths: {}
        resources: {}
        sanity:
          slurm:
            enabled: true
            command: "printf ok; scancel all"
        """,
    )
    with pytest.raises(RunSpecError, match="sanity.slurm.command"):
        load_run_spec(path)


def test_redact_value_hides_secret_like_content():
    """Dry-run renderers redact secret-like values before display."""
    assert redact_value("WANDB_API_KEY", "plain-text") == "<redacted>"
    assert redact_value("WANDB_PROJECT", "igc") == "igc"
    assert redact_value("WANDB_API_KEY", "${WANDB_API_KEY}") == "${WANDB_API_KEY}"


def test_docker_render_reuses_or_pulls_image_without_build(tmp_path):
    """if_missing policy inspects first, then pulls, and never builds."""
    spec = load_run_spec(_docker_spec(tmp_path))
    rendered = render_docker_plan(spec)
    assert "docker image inspect igc-train:test" in rendered
    assert "docker pull igc-train:test" in rendered
    assert "docker build" not in rendered
    assert "--env-file ${IGC_DOCKER_ENV_FILE}" in rendered
    assert "-e WANDB_API_KEY" in rendered
    assert "plain-text-token" not in rendered


def test_slurm_render_scales_resources_from_spec(tmp_path):
    """Large GPU counts come from the spec, not hardcoded launcher defaults."""
    spec = load_run_spec(_slurm_spec(tmp_path))
    rendered = render_slurm_plan(spec)
    assert "--nodes=15" in rendered
    assert "--gres=gpu:4" in rendered
    assert "--cpus-per-task=16" in rendered
    assert "--time=00:10:00" in rendered
    assert "--exclusive" in rendered


# Author: Mus mbayramo@stanford.edu
