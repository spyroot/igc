"""Run-spec validation and dry-run rendering for training launchers.

The module is deliberately pure: it parses YAML, validates the public run
contract, redacts sensitive-looking values, and renders command plans without
calling Docker, Slurm, the network, or gitignored operator files.

Author:
Mus mbayramo@stanford.edu
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
import shlex
from typing import Any, Mapping

import yaml


class RunSpecError(ValueError):
    """Raised when a run spec is invalid or unsafe to render."""


_TOP_LEVEL_KEYS = {
    "version",
    "name",
    "backend",
    "image",
    "runtime",
    "paths",
    "resources",
    "docker",
    "slurm",
    "data",
    "checkpoint",
    "sanity",
    "observability",
}
_BACKENDS = {"docker", "slurm"}
_PULL_POLICIES = {"never", "always", "if_missing", "build", "locked_digest"}
_SECRET_NAME = re.compile(
    r"(TOKEN|SECRET|PASSWORD|PASSWD|API[_-]?KEY|PRIVATE[_-]?KEY|CREDENTIAL|AUTH)",
    re.IGNORECASE,
)
_SHELL_META = re.compile(r"[\n\r;&|`<>]|\$\(")
_PLACEHOLDER = re.compile(
    r"^\$[A-Za-z_][A-Za-z0-9_]*$|^\$\{[A-Za-z_][A-Za-z0-9_]*(?::-[^}\n]*)?\}$"
)


@dataclass(frozen=True)
class RunSpec:
    """Normalized, public-safe run specification.

    :param path: source YAML file.
    :param name: stable logical run name.
    :param backend: launch backend, ``docker`` or ``slurm``.
    :param image: image settings such as ``ref`` and ``pull_policy``.
    :param runtime: command and environment allowlist.
    :param paths: code, data, output, scratch, and artifact roots.
    :param resources: GPU, node, CPU, memory, and wall-time knobs.
    :param docker: Docker-specific mounts and env-file references.
    :param slurm: Slurm-specific resource options.
    :param data: optional data staging settings.
    :param checkpoint: optional checkpoint publish/resume settings.
    :param sanity: optional pre-run sanity settings.
    :param observability: optional logging and metric settings.
    """

    path: Path
    name: str
    backend: str
    image: dict[str, Any]
    runtime: dict[str, Any]
    paths: dict[str, Any]
    resources: dict[str, Any]
    docker: dict[str, Any] = field(default_factory=dict)
    slurm: dict[str, Any] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)
    checkpoint: dict[str, Any] = field(default_factory=dict)
    sanity: dict[str, Any] = field(default_factory=dict)
    observability: dict[str, Any] = field(default_factory=dict)


def load_run_spec(path: str | Path) -> RunSpec:
    """Load and validate a run spec from YAML.

    :param path: YAML spec path.
    :return: normalized run spec.
    :raises RunSpecError: if the spec is missing required fields or is unsafe.
    """
    spec_path = Path(path)
    try:
        raw = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RunSpecError(f"cannot read spec {spec_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise RunSpecError(f"cannot parse YAML in {spec_path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise RunSpecError("run spec must be a YAML mapping")

    unknown = sorted(set(raw) - _TOP_LEVEL_KEYS)
    if unknown:
        raise RunSpecError(f"unknown top-level keys: {', '.join(unknown)}")

    name = _required_string(raw, "name")
    backend = _required_string(raw, "backend")
    if backend not in _BACKENDS:
        raise RunSpecError("backend must be one of: docker, slurm")

    image = _mapping(raw, "image", required=True)
    runtime = _mapping(raw, "runtime", required=True)
    paths = _mapping(raw, "paths")
    resources = _mapping(raw, "resources")
    docker = _mapping(raw, "docker")
    slurm = _mapping(raw, "slurm")
    data = _mapping(raw, "data")
    checkpoint = _mapping(raw, "checkpoint")
    sanity = _mapping(raw, "sanity")
    observability = _mapping(raw, "observability")

    _validate_image(image)
    _validate_runtime(runtime)
    _validate_backend_section(backend, docker, slurm)
    _validate_sanity(sanity)

    return RunSpec(
        path=spec_path,
        name=name,
        backend=backend,
        image=image,
        runtime=runtime,
        paths=paths,
        resources=resources,
        docker=docker,
        slurm=slurm,
        data=data,
        checkpoint=checkpoint,
        sanity=sanity,
        observability=observability,
    )


def redact_value(name: str, value: Any) -> str:
    """Return a display-safe value for dry-run output.

    :param name: variable or field name.
    :param value: value to display.
    :return: original value, a placeholder, or ``<redacted>``.
    """
    text = "" if value is None else str(value)
    if _is_placeholder(text):
        return text
    if _SECRET_NAME.search(name):
        return "<redacted>"
    return text


def render_plan(spec: RunSpec) -> str:
    """Render a dry-run command plan for the spec backend."""
    if spec.backend == "docker":
        return render_docker_plan(spec)
    if spec.backend == "slurm":
        return render_slurm_plan(spec)
    raise RunSpecError(f"unsupported backend: {spec.backend}")


def render_docker_plan(spec: RunSpec) -> str:
    """Render the Docker dry-run plan without executing it."""
    image = _image_ref(spec)
    pull_policy = spec.image.get("pull_policy", "if_missing")
    command = _command_string(spec.runtime["command"])
    lines = [
        "# igc run dry-run",
        "# backend: docker",
        "set -euo pipefail",
    ]

    if pull_policy == "always":
        lines.append(f"docker pull {_shell(image)}")
    elif pull_policy == "if_missing":
        lines.append(
            f"docker image inspect {_shell(image)} >/dev/null 2>&1 "
            f"|| docker pull {_shell(image)}"
        )
    elif pull_policy == "build":
        dockerfile = spec.image.get("dockerfile", "docker/Dockerfile.train")
        context = spec.image.get("context", ".")
        lines.append(
            f"docker build -f {_shell(dockerfile)} -t {_shell(image)} {_shell(context)}"
        )
    elif pull_policy == "locked_digest":
        lines.append(f"docker image inspect {_shell(image)} >/dev/null")

    docker_args = ["docker run", "--rm"]
    gpus = spec.resources.get("gpus")
    if gpus:
        docker_args.extend(["--gpus", "all"])
    if spec.docker.get("ipc", "host"):
        docker_args.append("--ipc=host")

    for env_file in _string_list(spec.docker.get("env_files", []), "docker.env_files"):
        docker_args.extend(["--env-file", _shell(env_file)])

    for env_name in sorted(_mapping(spec.runtime, "env").keys()):
        docker_args.extend(["-e", env_name])

    for mount in _list(spec.docker.get("mounts", []), "docker.mounts"):
        if not isinstance(mount, dict):
            raise RunSpecError("docker.mounts entries must be mappings")
        source = _required_string(mount, "source", context="docker.mounts")
        target = _required_string(mount, "target", context="docker.mounts")
        suffix = ":ro" if mount.get("read_only") else ""
        docker_args.extend(["-v", f"{source}:{target}{suffix}"])

    workdir = spec.runtime.get("workdir") or spec.docker.get("workdir") or "/workspace/igc"
    docker_args.extend(["-w", _shell(str(workdir)), _shell(image), *command])
    lines.append(_join_command(docker_args))
    return "\n".join(lines) + "\n"


def render_slurm_plan(spec: RunSpec) -> str:
    """Render the Slurm dry-run plan without submitting it."""
    command = " ".join(_command_string(spec.runtime["command"]))
    resources = spec.resources
    slurm = spec.slurm

    args = [
        "sbatch",
        "--parsable",
        f"--job-name={_slurm_value(spec.name)}",
        f"--nodes={resources.get('nodes', 1)}",
        f"--gres=gpu:{resources.get('gpus', 1)}",
    ]
    if resources.get("cpus_per_task"):
        args.append(f"--cpus-per-task={resources['cpus_per_task']}")
    if resources.get("wall_time"):
        args.append(f"--time={resources['wall_time']}")
    if slurm.get("partition"):
        args.append(f"--partition={slurm['partition']}")
    if slurm.get("account"):
        args.append(f"--account={slurm['account']}")
    if slurm.get("output"):
        args.append(f"--output={slurm['output']}")
    args.extend(_string_list(slurm.get("extra_args", []), "slurm.extra_args"))
    args.append(f"--wrap={_shell(command)}")
    return "\n".join([
        "# igc run dry-run",
        "# backend: slurm",
        _join_command(args),
        "",
    ])


def slurm_sanity_settings(spec: RunSpec) -> dict[str, Any]:
    """Return normalized Slurm sanity settings from a run spec."""
    sanity = _mapping(spec.sanity, "slurm")
    return {
        "enabled": bool(sanity.get("enabled", False)),
        "command": str(sanity.get("command", "printf IGC_SLURM_SANITY_OK")),
        "sentinel": str(sanity.get("sentinel", "IGC_SLURM_SANITY_OK")),
        "timeout_seconds": int(sanity.get("timeout_seconds", 300)),
        "poll_seconds": int(sanity.get("poll_seconds", 5)),
    }


def slurm_output_path(spec: RunSpec, job_id: str | None = None) -> str:
    """Return the Slurm output path, replacing ``%j`` when a job id is known."""
    output = spec.slurm.get("output")
    if not output:
        root = str(spec.paths.get("output", "."))
        output = f"{root.rstrip('/')}/slurm-%j.out"
    if job_id is not None:
        output = str(output).replace("%j", job_id)
    return str(output)


def _validate_image(image: Mapping[str, Any]) -> None:
    image_ref = _required_string(image, "ref", context="image")
    _reject_shell_meta("image.ref", image_ref)
    pull_policy = image.get("pull_policy", "if_missing")
    if pull_policy not in _PULL_POLICIES:
        raise RunSpecError(
            "image.pull_policy must be one of: " + ", ".join(sorted(_PULL_POLICIES))
        )


def _validate_runtime(runtime: Mapping[str, Any]) -> None:
    command = runtime.get("command")
    if not isinstance(command, list) or not command:
        raise RunSpecError("runtime.command must be a non-empty list")
    if not all(isinstance(part, str) for part in command):
        raise RunSpecError("runtime.command entries must be strings")
    for part in command:
        _reject_shell_meta("runtime.command", part)
    env = _mapping(runtime, "env")
    for key, value in env.items():
        if not isinstance(key, str):
            raise RunSpecError("runtime.env keys must be strings")
        text = "" if value is None else str(value)
        if _SECRET_NAME.search(key) and not _is_placeholder(text):
            raise RunSpecError(
                f"runtime.env.{key} must reference an environment placeholder, "
                "not a literal value"
            )


def _validate_backend_section(
        backend: str,
        docker: Mapping[str, Any],
        slurm: Mapping[str, Any],
) -> None:
    if backend == "docker":
        for env_file in _string_list(docker.get("env_files", []), "docker.env_files"):
            _reject_shell_meta("docker.env_files", env_file)
    if backend == "slurm":
        for arg in _string_list(slurm.get("extra_args", []), "slurm.extra_args"):
            _reject_shell_meta("slurm.extra_args", arg)


def _validate_sanity(sanity: Mapping[str, Any]) -> None:
    slurm = _mapping(sanity, "slurm")
    command = slurm.get("command")
    if command is not None:
        if not isinstance(command, str):
            raise RunSpecError("sanity.slurm.command must be a string")
        _reject_shell_meta("sanity.slurm.command", command)


def _image_ref(spec: RunSpec) -> str:
    return _required_string(spec.image, "ref", context="image")


def _required_string(
        mapping: Mapping[str, Any],
        key: str,
        *,
        context: str = "spec",
) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or value == "":
        raise RunSpecError(f"{context}.{key} must be a non-empty string")
    return value


def _mapping(
        mapping: Mapping[str, Any],
        key: str,
        *,
        required: bool = False,
) -> dict[str, Any]:
    value = mapping.get(key)
    if value is None:
        if required:
            raise RunSpecError(f"missing required section: {key}")
        return {}
    if not isinstance(value, dict):
        raise RunSpecError(f"{key} must be a mapping")
    return dict(value)


def _list(value: Any, field: str) -> list[Any]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise RunSpecError(f"{field} must be a list")
    return list(value)


def _string_list(value: Any, field: str) -> list[str]:
    values = _list(value, field)
    if not all(isinstance(item, str) for item in values):
        raise RunSpecError(f"{field} entries must be strings")
    return values


def _is_placeholder(value: str) -> bool:
    return bool(_PLACEHOLDER.fullmatch(value))


def _reject_shell_meta(field: str, value: str) -> None:
    if _is_placeholder(value):
        return
    if _SHELL_META.search(value):
        raise RunSpecError(f"{field} contains shell metacharacters")


def _command_string(command: list[str]) -> list[str]:
    return [_shell(part) for part in command]


def _shell(value: Any) -> str:
    text = str(value)
    if _is_placeholder(text):
        return text
    return shlex.quote(text)


def _join_command(parts: list[str]) -> str:
    return " ".join(str(part) for part in parts)


def _slurm_value(value: Any) -> str:
    text = str(value)
    if re.fullmatch(r"[A-Za-z0-9_.=-]+", text) or _is_placeholder(text):
        return text
    return _shell(text)


# Author: Mus mbayramo@stanford.edu
