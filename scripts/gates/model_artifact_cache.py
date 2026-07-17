"""Materialize and verify the shared IGC model-artifact cache.

The cache keeps immutable run artifacts under one root, for example
``/Volumes/k8s-sata-stripe/igc/phases/phase1/model_x/runs/<run_id>/...``, and
stores base-model caches once under ``base_models/``. It is designed to run on an
approved remote/Kubernetes surface. By default it can plan and verify metadata;
LFS hydration requires ``--download-lfs`` so a worker cannot accidentally pull
large objects to the wrong machine.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPEC = REPO_ROOT / "configs/artifacts/model_cache.yaml"


class ArtifactGateError(RuntimeError):
    """A hard gate failure with a user-readable message."""


@dataclass(frozen=True)
class LfsPointer:
    """Parsed Git LFS pointer metadata."""

    sha256: str
    size_bytes: int


def load_spec(path: Path = DEFAULT_SPEC) -> dict[str, Any]:
    """Load the artifact-cache spec from YAML."""

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ArtifactGateError("artifact cache spec must be a mapping")
    if payload.get("version") != 1:
        raise ArtifactGateError("artifact cache spec version must be 1")
    if not isinstance(payload.get("models"), list) or not payload["models"]:
        raise ArtifactGateError("artifact cache spec must contain models")
    return payload


def resolve_root(spec: Mapping[str, Any], override: str | None = None) -> Path:
    """Resolve the cache root from CLI, env, or spec default."""

    raw = override or os.environ.get("IGC_MODEL_CACHE_ROOT") or spec.get("default_root")
    if not isinstance(raw, str) or not raw:
        raise ArtifactGateError("set --root or IGC_MODEL_CACHE_ROOT")
    return Path(raw).expanduser()


def run_dir(root: Path, model: Mapping[str, Any]) -> Path:
    """Return the immutable cache directory for one model run."""

    return (
        root
        / "phases"
        / str(model["phase"])
        / str(model["role"])
        / "runs"
        / str(model["run_id"])
    )


def base_model_dir(root: Path, model: Mapping[str, Any]) -> Path:
    """Return the shared base-model cache location for one model spec."""

    base = _mapping(model, "base_model")
    return root / "base_models" / str(base["cache_subdir"])


def planned_paths(spec: Mapping[str, Any], root: Path) -> dict[str, Any]:
    """Return a JSON-serializable layout plan without touching the filesystem."""

    planned: dict[str, Any] = {
        "root": str(root),
        "base_models": str(root / "base_models"),
        "runs": [],
    }
    for model in spec["models"]:
        rd = run_dir(root, model)
        planned["runs"].append({
            "role": model["role"],
            "phase": model["phase"],
            "run_id": model["run_id"],
            "run_dir": str(rd),
            "base_model_dir": str(base_model_dir(root, model)),
            "files": [
                {
                    "kind": file_spec["kind"],
                    "source_path": file_spec["source_path"],
                    "dest": str(rd / file_spec["dest"]),
                    "lfs": bool(file_spec.get("lfs", False)),
                }
                for file_spec in model["files"]
            ],
        })
    return planned


def init_layout(spec: Mapping[str, Any], root: Path) -> None:
    """Create the shared cache directory structure."""

    (root / "base_models").mkdir(parents=True, exist_ok=True)
    (root / "reports").mkdir(parents=True, exist_ok=True)
    for model in spec["models"]:
        rd = run_dir(root, model)
        (rd / "artifacts").mkdir(parents=True, exist_ok=True)
        (rd / "reports").mkdir(parents=True, exist_ok=True)


def materialize(
        spec: Mapping[str, Any],
        root: Path,
        *,
        download_lfs: bool = False,
        base_source_root: Path | None = None,
        copy_base_models: bool = False) -> dict[str, Any]:
    """Copy small reports and optionally hydrate LFS weights into the cache."""

    init_layout(spec, root)
    records: list[dict[str, Any]] = []
    if base_source_root is not None:
        records.extend(materialize_base_models(
            spec,
            root,
            base_source_root,
            copy_base_models=copy_base_models,
        ))
    for model in spec["models"]:
        ref = str(model["lfs_ref"])
        rd = run_dir(root, model)
        for file_spec in model["files"]:
            source = str(file_spec["source_path"])
            dest = rd / str(file_spec["dest"])
            dest.parent.mkdir(parents=True, exist_ok=True)
            if file_spec.get("lfs"):
                pointer = read_lfs_pointer(ref, source)
                _validate_pointer(file_spec, pointer)
                if dest.exists() and _sha256_file(dest) == pointer.sha256:
                    status = "exists"
                elif download_lfs:
                    _smudge_lfs_pointer(ref, source, dest)
                    _verify_file(file_spec, dest, pointer=pointer)
                    status = "materialized"
                else:
                    raise ArtifactGateError(
                        f"{source} is LFS-backed; rerun with --download-lfs on an approved surface"
                    )
            else:
                payload = _git_show_bytes(ref, source)
                dest.write_bytes(payload)
                _verify_file(file_spec, dest)
                status = "copied"
            records.append({
                "role": model["role"],
                "phase": model["phase"],
                "run_id": model["run_id"],
                "kind": file_spec["kind"],
                "dest": str(dest),
                "status": status,
            })
    manifest = {
        "schema": "igc.model_artifact_cache.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "records": records,
    }
    (root / "cache_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def materialize_base_models(
        spec: Mapping[str, Any],
        root: Path,
        source_root: Path,
        *,
        copy_base_models: bool = False) -> list[dict[str, Any]]:
    """Link or copy base-model cache directories once into the shared root."""

    records: list[dict[str, Any]] = []
    for model in spec["models"]:
        base = _mapping(model, "base_model")
        cache_subdir = str(base["cache_subdir"])
        source = source_root / cache_subdir
        dest = base_model_dir(root, model)
        if not source.exists():
            raise ArtifactGateError(f"base model source missing: {source}")
        if dest.exists():
            status = "exists"
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            if copy_base_models:
                shutil.copytree(source, dest, symlinks=True)
                status = "copied"
            else:
                dest.symlink_to(source, target_is_directory=True)
                status = "linked"
        records.append({
            "role": model["role"],
            "phase": model["phase"],
            "run_id": model["run_id"],
            "kind": "base_model",
            "source": str(source),
            "dest": str(dest),
            "status": status,
        })
    return records


def verify(spec: Mapping[str, Any], root: Path, *, require_base_model: bool = False) -> dict[str, Any]:
    """Verify materialized reports, weights, and optional base-model cache paths."""

    checks: list[dict[str, Any]] = []
    for model in spec["models"]:
        if require_base_model:
            base_dir = base_model_dir(root, model)
            if not base_dir.exists():
                raise ArtifactGateError(f"base model cache missing: {base_dir}")
            checks.append({"kind": "base_model", "path": str(base_dir), "status": "present"})
        rd = run_dir(root, model)
        for file_spec in model["files"]:
            dest = rd / str(file_spec["dest"])
            if not dest.is_file():
                raise ArtifactGateError(f"cached artifact missing: {dest}")
            pointer = None
            if file_spec.get("lfs"):
                pointer = LfsPointer(
                    sha256=str(file_spec["sha256"]),
                    size_bytes=int(file_spec["size_bytes"]),
                )
            _verify_file(file_spec, dest, pointer=pointer)
            checks.append({
                "kind": file_spec["kind"],
                "path": str(dest),
                "status": "verified",
            })
    return {
        "schema": "igc.model_artifact_cache.verify.v1",
        "root": str(root),
        "checks": checks,
    }


def read_lfs_pointer(ref: str, path: str) -> LfsPointer:
    """Read and parse an LFS pointer from ``ref:path`` without hydrating it."""

    pointer = _git_show_bytes(ref, path).decode("utf-8")
    lines = pointer.splitlines()
    if not lines or lines[0] != "version https://git-lfs.github.com/spec/v1":
        raise ArtifactGateError(f"{path} is not a Git LFS pointer at {ref}")
    oid = next((line for line in lines if line.startswith("oid sha256:")), "")
    size = next((line for line in lines if line.startswith("size ")), "")
    if not oid or not size:
        raise ArtifactGateError(f"{path} has malformed Git LFS pointer metadata")
    return LfsPointer(sha256=oid.removeprefix("oid sha256:"), size_bytes=int(size.split()[1]))


def _validate_pointer(file_spec: Mapping[str, Any], pointer: LfsPointer) -> None:
    """Validate pointer metadata against the artifact spec."""

    if str(file_spec.get("sha256", "")).lower() != pointer.sha256.lower():
        raise ArtifactGateError(f"LFS SHA mismatch for {file_spec['source_path']}")
    if int(file_spec.get("size_bytes", -1)) != pointer.size_bytes:
        raise ArtifactGateError(f"LFS size mismatch for {file_spec['source_path']}")


def _verify_file(
        file_spec: Mapping[str, Any],
        path: Path,
        *,
        pointer: LfsPointer | None = None) -> None:
    """Verify cached file size/SHA and report JSON parseability."""

    if pointer is not None:
        if path.stat().st_size != pointer.size_bytes:
            raise ArtifactGateError(f"size mismatch for {path}")
        if _sha256_file(path).lower() != pointer.sha256.lower():
            raise ArtifactGateError(f"sha256 mismatch for {path}")
    expected_sha = file_spec.get("sha256")
    if expected_sha and _sha256_file(path).lower() != str(expected_sha).lower():
        raise ArtifactGateError(f"sha256 mismatch for {path}")
    if file_spec["kind"] in {"report", "parameters"}:
        json.loads(path.read_text(encoding="utf-8"))


def _smudge_lfs_pointer(ref: str, source: str, dest: Path) -> None:
    """Hydrate one LFS pointer into ``dest`` using git-lfs smudge."""

    pointer = _git_show_bytes(ref, source)
    completed = subprocess.run(
        ["git", "lfs", "smudge"],
        cwd=REPO_ROOT,
        input=pointer,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise ArtifactGateError(
            f"git lfs smudge failed for {source}: {completed.stderr.decode('utf-8', 'replace')}"
        )
    dest.write_bytes(completed.stdout)


def _git_show_bytes(ref: str, path: str) -> bytes:
    """Return ``git show ref:path`` bytes."""

    completed = subprocess.run(
        ["git", "show", f"{ref}:{path}"],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise ArtifactGateError(
            f"git show failed for {ref}:{path}: {completed.stderr.decode('utf-8', 'replace')}"
        )
    return completed.stdout


def _sha256_file(path: Path) -> str:
    """Return the SHA256 digest of a local file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    """Return a required nested mapping."""

    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ArtifactGateError(f"{key} must be a mapping")
    return value


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "init", "materialize", "verify"))
    parser.add_argument("--spec", default=str(DEFAULT_SPEC))
    parser.add_argument("--root")
    parser.add_argument("--download-lfs", action="store_true")
    parser.add_argument("--base-source-root")
    parser.add_argument("--copy-base-models", action="store_true")
    parser.add_argument("--require-base-model", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    try:
        spec = load_spec(Path(args.spec))
        root = resolve_root(spec, args.root)
        if args.command == "plan":
            result = planned_paths(spec, root)
        elif args.command == "init":
            init_layout(spec, root)
            result = {"status": "initialized", "root": str(root)}
        elif args.command == "materialize":
            base_source_root = Path(args.base_source_root) if args.base_source_root else None
            result = materialize(
                spec,
                root,
                download_lfs=args.download_lfs,
                base_source_root=base_source_root,
                copy_base_models=args.copy_base_models,
            )
        else:
            result = verify(spec, root, require_base_model=args.require_base_model)
    except ArtifactGateError as exc:
        print(f"BLOCKER: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"MODEL_ARTIFACT_CACHE_{args.command.upper()}_OK root={result.get('root', root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
