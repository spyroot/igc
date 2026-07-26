"""Spec-driven held-out inference for the Phase 2 and Phase 3 SFT models."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml

from igc.ds.rest_goal_contract import render_phase2_sft, render_phase3_sft
from igc.modules.train.promotion_evidence import is_sha256
from igc.modules.train.sft_tasks import resolve_sft_task


_SPEC_FIELDS = {
    "version",
    "name",
    "phase",
    "task",
    "contract_version",
    "target_semantics",
    "base_model",
    "adapter",
    "runtime",
    "generation",
}
_BASE_FIELDS = {
    "id",
    "cache_dir",
    "tokenizer",
    "foundation_model_sha",
    "tokenizer_sha",
}
_ADAPTER_FIELDS = {
    "path",
    "artifact_sha",
    "size_bytes",
    "method",
    "rank",
    "alpha",
    "parent_role",
    "parent_artifact_sha",
    "output_role",
}
_RUNTIME_FIELDS = {
    "torch_dtype",
    "device",
    "device_map",
    "require_cuda",
    "trust_remote_code",
}
_GENERATION_FIELDS = {"seed", "max_new_tokens", "target_token_margin"}
_TARGET_SEMANTICS = {
    2: "unordered_unique_rest_api_set",
    3: "unordered_unique_call_set",
}
_RENDERERS: dict[int, Callable[[Mapping[str, Any]], tuple[str, str]]] = {
    2: render_phase2_sft,
    3: render_phase3_sft,
}


class SFTInferenceError(ValueError):
    """Raised when held-out inference inputs or evidence are invalid."""


@dataclass(frozen=True)
class SFTInferenceSpec:
    """Resolved model, lineage, runtime, and generation inputs for one phase."""

    path: Path
    name: str
    phase: int
    task: str
    contract_version: str
    target_semantics: str
    base_model: str
    cache_dir: Path
    tokenizer: str
    foundation_model_sha: str
    tokenizer_sha: str
    adapter_dir: Path
    artifact_sha: str
    adapter_size_bytes: int
    adapter_method: str
    adapter_rank: int
    adapter_alpha: int
    parent_role: str
    parent_artifact_sha: str
    output_role: str
    torch_dtype: str
    device: str
    device_map: str
    require_cuda: bool
    trust_remote_code: bool
    seed: int
    max_new_tokens: int
    target_token_margin: int


def load_sft_inference_spec(path: str | Path) -> SFTInferenceSpec:
    """Load a strict Phase 2/3 inference spec and resolve environment values."""
    spec_path = Path(path).expanduser().resolve()
    try:
        raw = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SFTInferenceError(f"cannot read inference spec {spec_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise SFTInferenceError(f"cannot parse inference spec {spec_path}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise SFTInferenceError("inference spec must be a YAML object")
    _exact_fields(raw, _SPEC_FIELDS, "inference spec")
    if raw.get("version") != 1:
        raise SFTInferenceError("inference spec version must be 1")

    base = _section(raw, "base_model", _BASE_FIELDS)
    adapter = _section(raw, "adapter", _ADAPTER_FIELDS)
    runtime = _section(raw, "runtime", _RUNTIME_FIELDS)
    generation = _section(raw, "generation", _GENERATION_FIELDS)
    phase = _integer(raw.get("phase"), "phase", minimum=2)
    if phase not in _RENDERERS:
        raise SFTInferenceError("inference phase must be 2 or 3")
    task_name = _string(raw.get("task"), "task")
    task = resolve_sft_task(task_name)
    target_semantics = _string(raw.get("target_semantics"), "target_semantics")
    if task.phase != phase:
        raise SFTInferenceError("inference phase disagrees with the SFT task")
    if raw.get("contract_version") != task.contract_version:
        raise SFTInferenceError("inference contract_version disagrees with the SFT task")
    if target_semantics != _TARGET_SEMANTICS[phase]:
        raise SFTInferenceError("inference target_semantics is invalid for the phase")

    parent_role = _resolved_string(adapter.get("parent_role"), "adapter.parent_role")
    output_role = _resolved_string(adapter.get("output_role"), "adapter.output_role")
    if parent_role != task.parent_role or output_role != task.output_role:
        raise SFTInferenceError("adapter lineage roles disagree with the SFT task")

    digests = {
        "base_model.foundation_model_sha": _resolved_string(
            base.get("foundation_model_sha"),
            "base_model.foundation_model_sha",
        ),
        "base_model.tokenizer_sha": _resolved_string(
            base.get("tokenizer_sha"),
            "base_model.tokenizer_sha",
        ),
        "adapter.artifact_sha": _resolved_string(
            adapter.get("artifact_sha"),
            "adapter.artifact_sha",
        ),
        "adapter.parent_artifact_sha": _resolved_string(
            adapter.get("parent_artifact_sha"),
            "adapter.parent_artifact_sha",
        ),
    }
    for label, value in digests.items():
        if not is_sha256(value):
            raise SFTInferenceError(f"{label} must resolve to sha256:<64 hex>")

    torch_dtype = _resolved_string(runtime.get("torch_dtype"), "runtime.torch_dtype")
    if torch_dtype not in {"auto", "bfloat16", "float16", "float32"}:
        raise SFTInferenceError(
            "runtime.torch_dtype must be auto, bfloat16, float16, or float32"
        )
    return SFTInferenceSpec(
        path=spec_path,
        name=_string(raw.get("name"), "name"),
        phase=phase,
        task=task.name,
        contract_version=task.contract_version,
        target_semantics=target_semantics,
        base_model=_resolved_string(base.get("id"), "base_model.id"),
        cache_dir=Path(
            _resolved_string(base.get("cache_dir"), "base_model.cache_dir")
        ).expanduser(),
        tokenizer=_resolved_string(base.get("tokenizer"), "base_model.tokenizer"),
        foundation_model_sha=digests["base_model.foundation_model_sha"],
        tokenizer_sha=digests["base_model.tokenizer_sha"],
        adapter_dir=Path(
            _resolved_string(adapter.get("path"), "adapter.path")
        ).expanduser(),
        artifact_sha=digests["adapter.artifact_sha"],
        adapter_size_bytes=_resolved_integer(
            adapter.get("size_bytes"),
            "adapter.size_bytes",
            minimum=1,
        ),
        adapter_method=_resolved_string(adapter.get("method"), "adapter.method"),
        adapter_rank=_resolved_integer(adapter.get("rank"), "adapter.rank", minimum=1),
        adapter_alpha=_resolved_integer(
            adapter.get("alpha"),
            "adapter.alpha",
            minimum=1,
        ),
        parent_role=parent_role,
        parent_artifact_sha=digests["adapter.parent_artifact_sha"],
        output_role=output_role,
        torch_dtype=torch_dtype,
        device=_resolved_string(runtime.get("device"), "runtime.device"),
        device_map=_resolved_string(runtime.get("device_map"), "runtime.device_map"),
        require_cuda=_boolean(runtime.get("require_cuda"), "runtime.require_cuda"),
        trust_remote_code=_boolean(
            runtime.get("trust_remote_code"),
            "runtime.trust_remote_code",
        ),
        seed=_integer(generation.get("seed"), "generation.seed", minimum=0),
        max_new_tokens=_integer(
            generation.get("max_new_tokens"),
            "generation.max_new_tokens",
            minimum=1,
        ),
        target_token_margin=_integer(
            generation.get("target_token_margin"),
            "generation.target_token_margin",
            minimum=0,
        ),
    )


def verify_dataset_release(
    *,
    artifact_path: str | Path,
    manifest_path: str | Path,
    phase: int,
    keep_rows: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Verify exact JSONL bytes, row count, immutable manifest, and row contract."""
    artifact = Path(artifact_path)
    manifest_file = Path(manifest_path)
    manifest = _json_object(manifest_file)
    if manifest.get("immutable") is not True or manifest.get("complete") is not True:
        raise SFTInferenceError(f"dataset manifest is not immutable and complete: {manifest_file}")
    if manifest.get("dataset") != "D1":
        raise SFTInferenceError(f"dataset manifest must identify D1: {manifest_file}")
    artifact_sha = sha256_file(artifact)
    if manifest.get("artifact_sha256") != artifact_sha:
        raise SFTInferenceError(f"dataset manifest artifact SHA mismatch: {manifest_file}")
    expected_view = f"phase{phase}"
    if manifest.get("view") not in {None, expected_view}:
        raise SFTInferenceError(
            f"dataset manifest view must be {expected_view!r}: {manifest_file}"
        )

    renderer = renderer_for_phase(phase)
    rows: list[dict[str, Any]] = []
    seen_cases: set[str] = set()
    row_count = 0
    with artifact.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise SFTInferenceError(
                    f"{artifact}:{line_number}: blank JSONL row"
                )
            value = json.loads(line)
            if not isinstance(value, dict):
                raise SFTInferenceError(
                    f"{artifact}:{line_number}: row must be an object"
                )
            if "y_pred" in value:
                raise SFTInferenceError(
                    f"{artifact}:{line_number}: committed y_pred is forbidden"
                )
            renderer(value)
            metadata = value.get("metadata")
            row_id = metadata.get("row_id") if isinstance(metadata, Mapping) else None
            if not isinstance(row_id, str) or not is_sha256(row_id):
                raise SFTInferenceError(
                    f"{artifact}:{line_number}: metadata.row_id must be a SHA-256 id"
                )
            inference_case_id = metadata.get("inference_case_id")
            if inference_case_id is not None:
                if not isinstance(inference_case_id, str) or not is_sha256(
                    inference_case_id
                ):
                    raise SFTInferenceError(
                        f"{artifact}:{line_number}: metadata.inference_case_id "
                        "must be a SHA-256 id"
                    )
                case_id = inference_case_id
            else:
                case_id = row_id
            if case_id in seen_cases:
                raise SFTInferenceError(
                    f"{artifact}:{line_number}: duplicate inference case identity"
                )
            seen_cases.add(case_id)
            row_count += 1
            if keep_rows:
                rows.append(value)
    if row_count == 0:
        raise SFTInferenceError(f"JSONL input is empty: {artifact}")
    if manifest.get("rows") != row_count:
        raise SFTInferenceError(f"dataset manifest row count mismatch: {manifest_file}")
    return rows, {
        "immutable": True,
        "complete": True,
        "rows": row_count,
        "artifact_sha": artifact_sha,
        "manifest_sha": sha256_file(manifest_file),
    }


def validate_parent_promotion(
    value: Mapping[str, Any],
    *,
    expected_role: str,
    expected_artifact_sha: str,
) -> None:
    """Require a passing parent promotion result tied to the configured artifact."""
    if value.get("status") != "pass":
        raise SFTInferenceError("parent checkpoint has no passing promotion result")
    if value.get("role") != expected_role:
        raise SFTInferenceError("parent promotion role disagrees with inference spec")
    if value.get("artifact_sha") != expected_artifact_sha:
        raise SFTInferenceError("parent promotion artifact SHA disagrees with inference spec")


def validate_split_release(
    *,
    release_manifest_path: str | Path,
    source_full_manifest_path: str | Path,
    train_manifest_path: str | Path,
    heldout_manifest_path: str | Path,
    phase: int,
) -> dict[str, Any]:
    """Bind exact train/held-out manifests to one disjoint atomic release."""
    release_path = Path(release_manifest_path).resolve()
    source_full_path = Path(source_full_manifest_path).resolve()
    train_path = Path(train_manifest_path).resolve()
    heldout_path = Path(heldout_manifest_path).resolve()
    release = _json_object(release_path)
    if (
        release.get("schema_version") != "d1_phase23_split_release.v1"
        or release.get("dataset") != "D1"
        or release.get("immutable") is not True
        or release.get("complete") is not True
        or release.get("disjoint") is not True
    ):
        raise SFTInferenceError("split release is not immutable, complete, and disjoint")
    if not is_sha256(release.get("split_spec_sha256")):
        raise SFTInferenceError("split release is missing its exact spec SHA")
    source_full_sha = sha256_file(source_full_path)
    if release.get(f"phase{phase}_full_manifest_sha256") != source_full_sha:
        raise SFTInferenceError(
            "split release references a different complete source manifest"
        )

    files = release.get("files")
    if not isinstance(files, Mapping):
        raise SFTInferenceError("split release files must be an object")
    train_entry = _split_entry(files, f"phase{phase}_train")
    heldout_entry = _split_entry(files, f"phase{phase}_heldout")
    _require_split_manifest_path(release_path, train_entry, train_path)
    _require_split_manifest_path(release_path, heldout_entry, heldout_path)

    children: dict[tuple[int, str], Mapping[str, Any]] = {}
    for child_phase in (2, 3):
        child_source_sha = release.get(
            f"phase{child_phase}_full_manifest_sha256"
        )
        if not is_sha256(child_source_sha):
            raise SFTInferenceError(
                f"split release phase{child_phase} source manifest SHA is invalid"
            )
        for split in ("train", "heldout"):
            entry = _split_entry(files, f"phase{child_phase}_{split}")
            child_path = _split_manifest_path(release_path, entry)
            child = _json_object(child_path)
            _validate_split_child(
                child,
                entry=entry,
                phase=child_phase,
                split=split,
                manifest_path=child_path,
                source_full_manifest_sha=child_source_sha,
            )
            children[(child_phase, split)] = child

    train = children[(phase, "train")]
    heldout = children[(phase, "heldout")]
    if _unique_source_ids(children[(2, "train")], label="phase2 train") != (
        _unique_source_ids(children[(3, "train")], label="phase3 train")
    ):
        raise SFTInferenceError("Phase 2/3 train source row IDs are not aligned")
    if _unique_source_ids(children[(2, "heldout")], label="phase2 heldout") != (
        _unique_source_ids(children[(3, "heldout")], label="phase3 heldout")
    ):
        raise SFTInferenceError("Phase 2/3 held-out source row IDs are not aligned")
    train_ids = _unique_source_ids(train, label="train")
    heldout_ids = _unique_source_ids(heldout, label="heldout")
    if train_ids & heldout_ids:
        raise SFTInferenceError("split release train and held-out source IDs overlap")
    if release.get("train_source_rows") != len(train_ids):
        raise SFTInferenceError("split release train source-row count mismatch")
    if release.get("heldout_source_rows") != len(heldout_ids):
        raise SFTInferenceError("split release held-out source-row count mismatch")
    return {
        "immutable": True,
        "complete": True,
        "disjoint": True,
        "sha256": sha256_file(release_path),
        "source_full_manifest_sha": source_full_sha,
        "train_manifest_sha": sha256_file(train_path),
        "heldout_manifest_sha": sha256_file(heldout_path),
    }


def prepare_heldout_cases(
    rows: Sequence[Mapping[str, Any]],
    *,
    phase: int,
    tokenizer: Any,
    max_new_tokens: int,
    target_token_margin: int,
) -> tuple[list[dict[str, Any]], int]:
    """Render every row and require generation budget >= target p95 plus margin."""
    renderer = renderer_for_phase(phase)
    prepared: list[dict[str, Any]] = []
    target_lengths: list[int] = []
    for index, row in enumerate(rows):
        prompt, completion = renderer(row)
        target_tokens = len(
            tokenizer(completion, add_special_tokens=False)["input_ids"]
        )
        if target_tokens <= 0:
            raise SFTInferenceError(f"row {index}: target tokenized to zero tokens")
        target_lengths.append(target_tokens)
        prepared.append({
            "row": dict(row),
            "prompt": prompt,
            "target_tokens": target_tokens,
        })
    target_p95 = _percentile(target_lengths, 95)
    required = math.ceil(target_p95 + target_token_margin)
    if max_new_tokens < required:
        raise SFTInferenceError(
            "generation.max_new_tokens does not cover target p95+margin: "
            f"configured={max_new_tokens} required={required}"
        )
    return prepared, target_p95


def generate_prediction_rows(
    model: Any,
    tokenizer: Any,
    device: Any,
    prepared: Sequence[Mapping[str, Any]],
    *,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    """Generate one raw prediction per held-out row with no missing outputs."""
    import torch

    predictions: list[dict[str, Any]] = []
    for index, item in enumerate(prepared):
        encoded = tokenizer(item["prompt"], return_tensors="pt")
        if hasattr(encoded, "to"):
            encoded = encoded.to(device)
        else:
            encoded = {key: value.to(device) for key, value in encoded.items()}
        prompt_tokens = int(encoded["input_ids"].shape[-1])
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated_ids = generated[0][prompt_tokens:]
        completion = tokenizer.decode(generated_ids, skip_special_tokens=True)
        if not completion.strip():
            raise SFTInferenceError(f"row {index}: model produced an empty prediction")
        output = dict(item["row"])
        output["y_pred"] = completion
        output["inference"] = {
            "prompt_tokens": prompt_tokens,
            "target_tokens": int(item["target_tokens"]),
            "generated_tokens": int(generated_ids.numel()),
            "max_new_tokens": max_new_tokens,
        }
        predictions.append(output)
    if len(predictions) != len(prepared):
        raise SFTInferenceError("inference produced missing prediction rows")
    return predictions


def build_artifact_evidence(
    *,
    spec: SFTInferenceSpec,
    source_full_release: Mapping[str, Any],
    train_release: Mapping[str, Any],
    heldout_release: Mapping[str, Any],
    split_release: Mapping[str, Any],
    prediction_rows: int,
    predictions_sha: str,
) -> dict[str, Any]:
    """Build promotion evidence tied to exact data, adapter bytes, and outputs."""
    return {
        "schema_version": "sft_inference_evidence.v1",
        "phase": spec.phase,
        "task": spec.task,
        "role": spec.output_role,
        "immutable_full_manifest": {
            "immutable": source_full_release.get("immutable") is True,
            "complete": source_full_release.get("complete") is True,
            "rows": source_full_release.get("rows"),
            "sha256": source_full_release.get("manifest_sha"),
            "artifact_sha": source_full_release.get("artifact_sha"),
        },
        "immutable_train_manifest": {
            "immutable": train_release.get("immutable") is True,
            "complete": train_release.get("complete") is True,
            "rows": train_release.get("rows"),
            "manifest_sha": train_release.get("manifest_sha"),
            "artifact_sha": train_release.get("artifact_sha"),
        },
        "disjoint_split_release": dict(split_release),
        "real_promoted_parent_checkpoint": {
            "role": spec.parent_role,
            "promotion_status": "pass",
            "artifact_sha": spec.parent_artifact_sha,
        },
        "real_heldout_data": {
            "real": True,
            "split": "heldout",
            "rows": heldout_release.get("rows"),
            "manifest_sha": heldout_release.get("manifest_sha"),
            "artifact_sha": heldout_release.get("artifact_sha"),
        },
        "artifact_sha": spec.artifact_sha,
        "checkpoint_reload": {
            "status": "pass",
            "artifact_sha": spec.artifact_sha,
            "adapter_dir": str(spec.adapter_dir),
            "foundation_model_sha": spec.foundation_model_sha,
            "tokenizer_sha": spec.tokenizer_sha,
        },
        "inference_smoke": {
            "status": "pass",
            "artifact_sha": spec.artifact_sha,
            "rows": prediction_rows,
            "missing_prediction_rows": 0,
            "predictions_sha": predictions_sha,
        },
    }


def renderer_for_phase(
    phase: int,
) -> Callable[[Mapping[str, Any]], tuple[str, str]]:
    """Return the canonical YAML-backed renderer for Phase 2 or Phase 3."""
    try:
        return _RENDERERS[phase]
    except KeyError as exc:
        raise SFTInferenceError("inference phase must be 2 or 3") from exc


def sha256_file(path: str | Path) -> str:
    """Return a canonical SHA-256 identifier for exact file bytes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SFTInferenceError(f"{path} must contain a JSON object")
    return value


def _section(
    source: Mapping[str, Any],
    name: str,
    fields: set[str],
) -> Mapping[str, Any]:
    value = source.get(name)
    if not isinstance(value, Mapping):
        raise SFTInferenceError(f"inference spec section {name} must be an object")
    _exact_fields(value, fields, f"inference spec {name}")
    return value


def _split_entry(files: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = files.get(key)
    if not isinstance(value, Mapping):
        raise SFTInferenceError(f"split release is missing files.{key}")
    required = {"path", "manifest", "artifact_sha256", "manifest_sha256"}
    if set(value) != required:
        raise SFTInferenceError(
            f"split release files.{key} must contain exactly {sorted(required)}"
        )
    if not is_sha256(value.get("artifact_sha256")) or not is_sha256(
        value.get("manifest_sha256")
    ):
        raise SFTInferenceError(f"split release files.{key} has invalid SHA evidence")
    return value


def _require_split_manifest_path(
    release_path: Path,
    entry: Mapping[str, Any],
    observed_path: Path,
) -> None:
    expected_path = _split_manifest_path(release_path, entry)
    if expected_path != observed_path:
        raise SFTInferenceError("split release references a different child manifest")


def _split_manifest_path(
    release_path: Path,
    entry: Mapping[str, Any],
) -> Path:
    manifest_name = entry.get("manifest")
    if not isinstance(manifest_name, str) or not manifest_name:
        raise SFTInferenceError("split release manifest path must be a non-empty string")
    return (release_path.parent / manifest_name).resolve()


def _validate_split_child(
    value: Mapping[str, Any],
    *,
    entry: Mapping[str, Any],
    phase: int,
    split: str,
    manifest_path: Path,
    source_full_manifest_sha: str,
) -> None:
    if (
        value.get("schema_version") != "d1_phase23_split_view.v1"
        or value.get("dataset") != "D1"
        or value.get("view") != f"phase{phase}"
        or value.get("split") != split
        or value.get("immutable") is not True
        or value.get("complete") is not True
    ):
        raise SFTInferenceError(f"split {split} child manifest identity is invalid")
    if sha256_file(manifest_path) != entry.get("manifest_sha256"):
        raise SFTInferenceError(f"split {split} child manifest SHA mismatch")
    if value.get("artifact_sha256") != entry.get("artifact_sha256"):
        raise SFTInferenceError(f"split {split} child artifact SHA mismatch")
    if value.get("source_full_manifest_sha256") != source_full_manifest_sha:
        raise SFTInferenceError(
            f"split {split} child source-full manifest SHA mismatch"
        )


def _unique_source_ids(value: Mapping[str, Any], *, label: str) -> set[str]:
    raw = value.get("source_row_ids")
    if (
        not isinstance(raw, list)
        or not raw
        or not all(isinstance(item, str) and is_sha256(item) for item in raw)
        or len(raw) != len(set(raw))
    ):
        raise SFTInferenceError(
            f"split {label} source_row_ids must be unique SHA-256 ids"
        )
    return set(raw)


def _exact_fields(value: Mapping[str, Any], fields: set[str], label: str) -> None:
    missing = sorted(fields - set(value))
    unknown = sorted(set(value) - fields)
    if missing:
        raise SFTInferenceError(f"{label} missing keys: {missing}")
    if unknown:
        raise SFTInferenceError(f"{label} has unknown keys: {unknown}")


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SFTInferenceError(f"{label} must be a non-empty string")
    return value.strip()


def _resolved_string(value: Any, label: str) -> str:
    resolved = os.path.expandvars(_string(value, label))
    if "$" in resolved:
        raise SFTInferenceError(f"{label} references an unset environment variable")
    return resolved


def _integer(value: Any, label: str, *, minimum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise SFTInferenceError(f"{label} must be an integer >= {minimum}")
    return value


def _resolved_integer(value: Any, label: str, *, minimum: int) -> int:
    if isinstance(value, str):
        resolved = _resolved_string(value, label)
        try:
            value = int(resolved)
        except ValueError as exc:
            raise SFTInferenceError(f"{label} must resolve to an integer") from exc
    return _integer(value, label, minimum=minimum)


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise SFTInferenceError(f"{label} must be a boolean")
    return value


def _percentile(values: Sequence[int], percentile: int) -> int:
    if not values:
        raise SFTInferenceError("cannot compute target percentile from no rows")
    ordered = sorted(values)
    index = max(0, math.ceil((percentile / 100) * len(ordered)) - 1)
    return int(ordered[index])


__all__ = (
    "SFTInferenceError",
    "SFTInferenceSpec",
    "build_artifact_evidence",
    "generate_prediction_rows",
    "load_sft_inference_spec",
    "prepare_heldout_cases",
    "renderer_for_phase",
    "sha256_file",
    "validate_parent_promotion",
    "validate_split_release",
    "verify_dataset_release",
)
