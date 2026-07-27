#!/usr/bin/env python3
"""Verify Phase 1 dataset, source, and training-image compatibility."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_LABEL = "ai.spyroot.igc.dataset-contract-sha"
TRANSFORM_LABEL = "ai.spyroot.igc.dataset-transform-version"
CONTRACT_INPUTS = (
    "configs/contracts/phase1.yaml",
    "configs/data/redfish_sources.yaml",
    "configs/training/sft_tasks.yaml",
    "igc/ds/phase1_chunking.py",
    "igc/ds/phase1_render.py",
    "igc/ds/source_registry.py",
    "igc/ds/sources/corpus_io.py",
    "igc/ds/sources/mixer.py",
    "igc/ds/sources/redfish_fixture_source.py",
    "igc/modules/train/sft_tasks.py",
)
RAW_COPY_MARKERS = (
    ".internal",
    ".json_responses",
    "datasets",
    "full_corpus",
)


class GateError(ValueError):
    """A closed-world validation failure."""


class _GateArgumentParser(argparse.ArgumentParser):
    """Convert usage errors into the gate's machine-readable failure path."""

    def error(self, message: str) -> None:
        raise GateError(f"argument error: {message}")


def sha256_file(path: Path) -> str:
    """Return the exact file identity as ``sha256:<hex>``."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def contract_identity(repo_root: Path = REPO_ROOT) -> tuple[str, str, list[str]]:
    """Hash the files that define D0 bytes and runtime interpretation."""

    contract = _load_yaml(repo_root / CONTRACT_INPUTS[0])
    chunking = contract.get("chunking")
    transform = chunking.get("transform") if isinstance(chunking, Mapping) else None
    if not isinstance(transform, str) or not transform:
        raise GateError("configs/contracts/phase1.yaml: missing chunking.transform")
    digest = hashlib.sha256(b"igc.dataset-runtime-contract.v1\n")
    for relative in CONTRACT_INPUTS:
        path = repo_root / relative
        if not path.is_file():
            raise GateError(f"contract input is missing: {relative}")
        body = path.read_bytes()
        digest.update(f"{relative}\0{len(body)}\0".encode())
        digest.update(body)
    return f"sha256:{digest.hexdigest()}", transform, list(CONTRACT_INPUTS)


def compare_image(
    image: str,
    *,
    labels: Mapping[str, str] | None = None,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Return compatible/update-required for one exact image label set."""

    expected_sha, expected_transform, inputs = contract_identity(repo_root)
    observed = dict(labels) if labels is not None else _inspect_image_labels(image)
    reasons = []
    if observed.get(CONTRACT_LABEL) != expected_sha:
        reasons.append("dataset contract sha mismatch")
    if observed.get(TRANSFORM_LABEL) != expected_transform:
        reasons.append("dataset transform version mismatch")
    return {
        "status": "compatible" if not reasons else "update-required",
        "image": image,
        "expected_contract_sha": expected_sha,
        "observed_contract_sha": observed.get(CONTRACT_LABEL),
        "expected_transform": expected_transform,
        "observed_transform": observed.get(TRANSFORM_LABEL),
        "contract_inputs": inputs,
        "reasons": reasons,
    }


def validate_release(
    *,
    release_root: Path,
    summary_path: Path,
    source_registry_path: Path,
    source_manifests: Mapping[str, Path],
    training_profile: str,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Validate all D0 rows, exact release bytes, and source identities."""

    sys.path.insert(0, str(repo_root))
    try:
        from igc.ds.phase1_chunking import reassemble_phase1_rows
        from igc.ds.phase1_render import validate_phase1_row
        from igc.ds.sources.mixer import DataManifest
        from igc.modules.train.profiles import resolve_profile
    except (ImportError, SyntaxError) as exc:
        raise GateError(f"unable to load dataset runtime modules: {exc}") from exc

    summary = _load_json(summary_path)
    registry = _load_yaml(source_registry_path)
    expected_sources = set(_mapping(registry, "sources"))
    source_shas = _mapping(summary, "source_manifest_shas")
    if set(source_shas) != expected_sources:
        raise GateError("summary source_manifest_shas does not match registry sources")
    for name, digest in source_shas.items():
        _sha_value(digest, f"source_manifest_shas.{name}")
    if summary.get("source_registry_sha") != sha256_file(source_registry_path):
        raise GateError("source registry SHA mismatch")
    if set(source_manifests) != expected_sources:
        raise GateError("provided source manifests do not match registry sources")
    for name, path in source_manifests.items():
        if sha256_file(path) != source_shas[name]:
            raise GateError(f"source manifest SHA mismatch: {name}")

    train_path = release_root / "train" / "examples.jsonl"
    heldout_path = release_root / "heldout" / "examples.jsonl"
    train_manifest_path = release_root / "train" / "manifest.json"
    heldout_manifest_path = release_root / "heldout" / "manifest.json"
    expected_files = (train_path, heldout_path, train_manifest_path, heldout_manifest_path)
    if any(not path.is_file() for path in expected_files):
        raise GateError("release is missing a train/heldout examples or manifest member")
    if summary.get("train_artifact_sha") != sha256_file(train_path):
        raise GateError("train artifact SHA mismatch")
    if summary.get("heldout_artifact_sha") != sha256_file(heldout_path):
        raise GateError("heldout artifact SHA mismatch")
    if train_manifest_path.read_bytes() != heldout_manifest_path.read_bytes():
        raise GateError("train and heldout manifests differ")
    if summary.get("written_manifest_sha") != sha256_file(train_manifest_path):
        raise GateError("written manifest SHA mismatch")

    manifest_payload = _load_json(train_manifest_path)
    try:
        manifest = DataManifest(**manifest_payload)
    except (TypeError, ValueError) as exc:
        raise GateError(f"release manifest is invalid: {exc}") from exc
    if summary.get("manifest_sha") != manifest.content_hash():
        raise GateError("semantic manifest SHA mismatch")
    if manifest.source_registry_sha != summary["source_registry_sha"]:
        raise GateError("release manifest source registry SHA mismatch")
    if manifest.source_manifest_shas != dict(source_shas):
        raise GateError("release manifest source artifact SHAs mismatch")
    transform = _mapping(manifest_payload, "phase1_transform")
    try:
        profile = resolve_profile(training_profile)
    except (KeyError, TypeError, ValueError) as exc:
        raise GateError(f"training profile is invalid: {exc}") from exc
    if summary.get("training_profile") != profile.name:
        raise GateError("release training profile does not match requested profile")
    _validate_transform(summary, transform, profile=profile, repo_root=repo_root)

    train = _validate_split(train_path, validate_phase1_row, reassemble_phase1_rows)
    heldout = _validate_split(heldout_path, validate_phase1_row, reassemble_phase1_rows)
    if train["original_ids"] & heldout["original_ids"]:
        raise GateError("one original resource crosses train and heldout splits")
    if train["rows"] != summary.get("train_rows"):
        raise GateError("train row count mismatch")
    if heldout["rows"] != summary.get("heldout_rows"):
        raise GateError("heldout row count mismatch")
    if train["resources"] != summary.get("train_resources"):
        raise GateError("train resource count mismatch")
    if heldout["resources"] != summary.get("heldout_resources"):
        raise GateError("heldout resource count mismatch")
    return {
        "status": "passed",
        "contract_sha": contract_identity(repo_root)[0],
        "transform": transform["transform"],
        "train_rows": train["rows"],
        "heldout_rows": heldout["rows"],
        "train_resources": train["resources"],
        "heldout_resources": heldout["resources"],
        "source_manifest_shas": dict(sorted(source_shas.items())),
    }


def check_dockerfile(path: Path, *, repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Reject raw data in Docker COPY/ADD instructions."""

    resolved = path if path.is_absolute() else repo_root / path
    text = _read_text(resolved)
    violations = []
    required_markers = (
        "ARG IGC_DATASET_CONTRACT_SHA",
        "ARG IGC_DATASET_TRANSFORM_VERSION",
        CONTRACT_LABEL,
        TRANSFORM_LABEL,
    )
    for marker in required_markers:
        if marker not in text:
            violations.append(f"missing image contract marker: {marker}")
    for number, line in _dockerfile_instructions(text):
        if line.upper().startswith(("COPY ", "ADD ")):
            for marker in RAW_COPY_MARKERS:
                pattern = (
                    rf"(?:^|[/\s\"'\[\],]){re.escape(marker)}"
                    r"(?:$|[/\s\"'\[\],])"
                )
                if re.search(pattern, line, flags=re.IGNORECASE):
                    violations.append(f"line {number}: COPY/ADD references {marker}")
    return {
        "status": "passed" if not violations else "failed",
        "dockerfile": str(path),
        "violations": violations,
    }


def _dockerfile_instructions(text: str) -> list[tuple[int, str]]:
    """Return Dockerfile logical instructions with their starting line."""

    instructions: list[tuple[int, str]] = []
    parts: list[str] = []
    start_line = 0
    for number, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not parts and (not line or line.startswith("#")):
            continue
        if not parts:
            start_line = number
        continued = line.endswith("\\")
        parts.append(line[:-1].rstrip() if continued else line)
        if not continued:
            instructions.append((start_line, " ".join(parts)))
            parts = []
    if parts:
        instructions.append((start_line, " ".join(parts)))
    return instructions


def _validate_transform(
    summary: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    profile: Any,
    repo_root: Path,
) -> None:
    expected = contract_identity(repo_root)[1]
    if summary.get("phase1_transform") != manifest:
        raise GateError("summary and release phase1_transform differ")
    if manifest.get("transform") != expected:
        raise GateError("release transform does not match current contract")
    if manifest.get("overflow_policy") != "lossless_json_chunk":
        raise GateError("release overflow policy is not lossless_json_chunk")
    if manifest.get("split_before_chunk") is not True:
        raise GateError("release was not split before chunking")
    if manifest.get("exact_reassembly_verified") is not True:
        raise GateError("release lacks exact reassembly evidence")
    tokenizer_sha = _sha_value(
        manifest.get("tokenizer_sha"),
        "phase1_transform.tokenizer_sha",
    )
    profile_tokenizer_sha = _sha_value(
        getattr(profile, "tokenizer_sha", None),
        "training_profile.tokenizer_sha",
    )
    if tokenizer_sha != profile_tokenizer_sha:
        raise GateError("release tokenizer SHA does not match training profile")
    if not isinstance(manifest.get("max_tokens"), int) or manifest["max_tokens"] < 2:
        raise GateError("phase1_transform.max_tokens must be >= 2")
    profile_seq_len = getattr(profile, "seq_len", None)
    if not isinstance(profile_seq_len, int) or profile_seq_len < 2:
        raise GateError("training_profile.seq_len must be >= 2")
    if manifest["max_tokens"] != profile_seq_len:
        raise GateError("release max_tokens does not match training profile seq_len")


def _validate_split(path: Path, validate: Any, reassemble: Any) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    rows = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                validate(row)
                chunk = row["metadata"]["chunk"]
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise GateError(f"{path.name}:{line_number}: invalid D0 row: {exc}") from exc
            groups.setdefault(chunk["original_row_id"], []).append(row)
            rows += 1
    if not groups:
        raise GateError(f"{path}: split has no D0 rows")
    for original_id, chunks in groups.items():
        try:
            reassemble(chunks)
        except ValueError as exc:
            raise GateError(f"{path.name}: reassembly failed for {original_id}: {exc}") from exc
    return {"rows": rows, "resources": len(groups), "original_ids": set(groups)}


def _inspect_image_labels(image: str) -> dict[str, str]:
    result = subprocess.run(
        ["docker", "image", "inspect", image],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise GateError(f"docker image inspect failed: {result.stderr.strip()}")
    return _labels_from_json(result.stdout)


def _labels_from_json(text: str) -> dict[str, str]:
    payload = json.loads(text)
    if isinstance(payload, list):
        payload = payload[0] if payload else {}
    config = payload.get("Config") if isinstance(payload, Mapping) else None
    if config is None:
        config = {}
    if not isinstance(config, Mapping):
        raise GateError("docker image Config must be an object")
    labels = config.get("Labels")
    if labels is None:
        labels = {}
    if not isinstance(labels, Mapping):
        raise GateError("docker image labels must be an object")
    return {str(key): str(value) for key, value in labels.items()}


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeError as exc:
        raise GateError(f"{path}: expected UTF-8 text") from exc


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(_read_text(path))
    if not isinstance(payload, dict):
        raise GateError(f"{path}: expected JSON object")
    return payload


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(_read_text(path))
    if not isinstance(payload, dict):
        raise GateError(f"{path}: expected YAML object")
    return payload


def _mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise GateError(f"{key}: expected object")
    return value


def _sha_value(value: Any, field: str) -> str:
    digest = value[7:] if isinstance(value, str) and value.startswith("sha256:") else ""
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise GateError(f"{field}: expected sha256:<64 hex>")
    return str(value)


def _source_manifest_args(values: Sequence[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name or not path or name in result:
            raise GateError("--source-manifest requires unique NAME=PATH values")
        result[name] = Path(path)
    return result


def main(argv: list[str] | None = None) -> int:
    try:
        parser = _GateArgumentParser(description=__doc__)
        parser.add_argument("--dry-run", action="store_true")
        parser.add_argument("--log-format", choices=("text", "json"), default="json")
        parser.add_argument("--log-level", default="info")
        parser.add_argument("--log-file")
        parser.add_argument("--run-id", default="")
        commands = parser.add_subparsers(dest="command", required=True)
        identity = commands.add_parser("contract-sha")
        identity.add_argument("--output", choices=("json", "text"), default="json")
        transform = commands.add_parser("transform")
        transform.add_argument("--output", choices=("json", "text"), default="json")
        image = commands.add_parser("check-image")
        image.add_argument("--image", required=True)
        image.add_argument("--labels-json")
        release = commands.add_parser("check-release")
        release.add_argument("--release-root", required=True)
        release.add_argument("--summary", required=True)
        release.add_argument("--source-registry", required=True)
        release.add_argument("--training-profile", required=True)
        release.add_argument("--source-manifest", action="append", required=True)
        dockerfile = commands.add_parser("check-dockerfile")
        dockerfile.add_argument("--dockerfile", default="docker/Dockerfile.train")
        args = parser.parse_args(argv)

        digest, version, inputs = contract_identity()
        if args.command == "contract-sha":
            result: Any = digest if args.output == "text" else {
                "status": "passed", "contract_sha": digest, "inputs": inputs}
        elif args.command == "transform":
            result = version if args.output == "text" else {
                "status": "passed", "transform": version}
        elif args.command == "check-image":
            labels = None
            if args.labels_json:
                text = (
                    sys.stdin.read()
                    if args.labels_json == "-"
                    else _read_text(Path(args.labels_json))
                )
                labels = _labels_from_json(text)
            result = compare_image(args.image, labels=labels)
        elif args.command == "check-release":
            manifests = _source_manifest_args(args.source_manifest)
            result = validate_release(
                release_root=Path(args.release_root),
                summary_path=Path(args.summary),
                source_registry_path=Path(args.source_registry),
                source_manifests=manifests,
                training_profile=args.training_profile,
            )
        else:
            result = check_dockerfile(Path(args.dockerfile))
        if isinstance(result, str):
            print(result)
        else:
            print(json.dumps(result, sort_keys=True))
        ok = not isinstance(result, Mapping) or result.get("status") in {
            "passed",
            "compatible",
        }
        return 0 if ok else 1
    except (GateError, OSError, json.JSONDecodeError, yaml.YAMLError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
