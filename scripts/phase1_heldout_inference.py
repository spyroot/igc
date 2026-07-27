#!/usr/bin/env python3
"""Generate paired foundation/model_x predictions for the full Phase 1 held-out set.

Run this only in the approved GB300 container. It loads every model from local
cache, performs deterministic batch-size-one generation, and emits the exact
prediction-row evidence consumed by ``phase1_inference_gate.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT))

from phase1_gpu_inference_gate import (  # noqa: E402
    GateError,
    Paths,
    load_gate_spec,
    load_model_and_tokenizer,
    preflight_inputs,
    set_offline_env,
    validate_adapter_config,
)
from igc.ds.phase1_render import (  # noqa: E402
    build_phase1_row,
    render_phase1_completion,
    render_phase1_prompt,
)
from igc.modules.train.phase1_golden import percentile  # noqa: E402


class HeldoutInferenceError(ValueError):
    """Raised when full held-out inference cannot produce valid evidence."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse immutable input and output artifact paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--heldout-jsonl", required=True)
    parser.add_argument("--heldout-manifest", required=True)
    parser.add_argument("--baseline-output", required=True)
    parser.add_argument("--model-output", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the paired full-manifest inference pass."""
    args = parse_args(argv)
    try:
        spec = load_gate_spec(Path(args.spec))
        input_path = Path(args.heldout_jsonl)
        manifest = _json_object(Path(args.heldout_manifest))
        raw_rows = _jsonl(input_path)
        source_sha = _sha256(input_path)
        _validate_manifest(manifest, raw_rows, source_sha)
        normalized = [_normalize_input(row, index) for index, row in enumerate(raw_rows)]
        _validate_row_ids(manifest, normalized)

        set_offline_env(spec.cache_dir)
        paths = Paths(
            base_model=spec.base_model,
            cache_dir=spec.cache_dir,
            adapter_dir=spec.adapter_dir,
            output_json=None,
        )
        preflight_inputs(paths, require_cuda=spec.require_cuda)
        validate_adapter_config(spec)
        _configure_determinism(spec.seed)
        runtime_args = SimpleNamespace(spec=spec, allow_cpu=False)

        baseline_model, tokenizer, device = load_model_and_tokenizer(
            runtime_args,
            load_adapter=False,
        )
        prepared, target_p95 = _prepare_rows(normalized, tokenizer)
        required_budget = math.ceil(target_p95 + spec.target_token_margin)
        if spec.max_new_tokens < required_budget:
            raise HeldoutInferenceError(
                "generation.max_new_tokens does not cover held-out target p95+margin: "
                f"configured={spec.max_new_tokens} required={required_budget}"
            )
        baseline_rows = _generate_rows(
            baseline_model,
            tokenizer,
            device,
            prepared,
            max_new_tokens=spec.max_new_tokens,
        )
        del baseline_model
        _release_model()

        model_x, tokenizer, device = load_model_and_tokenizer(
            runtime_args,
            load_adapter=True,
        )
        model_rows = _generate_rows(
            model_x,
            tokenizer,
            device,
            prepared,
            max_new_tokens=spec.max_new_tokens,
        )
        del model_x
        _release_model()

        _write_prediction_pair_atomic(
            baseline_path=Path(args.baseline_output),
            baseline_rows=baseline_rows,
            model_path=Path(args.model_output),
            model_rows=model_rows,
        )
        print(json.dumps({
            "status": "pass",
            "rows": len(prepared),
            "source_sha256": source_sha,
            "target_completion_tokens_p95": target_p95,
            "max_new_tokens": spec.max_new_tokens,
            "baseline_output": args.baseline_output,
            "model_output": args.model_output,
        }, sort_keys=True))
        return 0
    except (GateError, HeldoutInferenceError, OSError, TypeError, ValueError) as exc:
        print(json.dumps({
            "status": "blocked",
            "error": str(exc),
            "safe_next_step": "repair the immutable manifest/spec or approved remote model cache",
        }), file=sys.stderr)
        return 2


def _normalize_input(row: Mapping[str, Any], index: int) -> dict[str, Any]:
    source = _source_corpus(row)
    if isinstance(row.get("x"), Mapping):
        phase1 = build_phase1_row(
            rest_api=row["x"].get("rest_api"),
            allowed_methods=row["x"].get("allowed_methods"),
            input_json=row["x"].get("json"),
            target_json=row.get("y_true", {}).get("json"),
        )
    else:
        action = row.get("request_or_action")
        response = row.get("response")
        if not isinstance(action, Mapping) or not isinstance(response, Mapping):
            raise HeldoutInferenceError(f"row {index}: invalid Phase 1 source row")
        phase1 = build_phase1_row(
            rest_api=action.get("url"),
            allowed_methods=row.get("allowed_methods", []),
            input_json=response,
            target_json=response,
        )
    canonical_row_id = _row_id(source, phase1["x"]["rest_api"])
    metadata = row.get("metadata")
    metadata_row_id = metadata.get("row_id") if isinstance(metadata, Mapping) else None
    supplied_row_id = row.get("row_id") or row.get("id") or metadata_row_id
    if supplied_row_id is not None and str(supplied_row_id) != canonical_row_id:
        raise HeldoutInferenceError(
            f"row {index}: supplied row_id disagrees with source/REST API identity"
        )
    return {
        "row_id": canonical_row_id,
        "source_corpus": source,
        "phase1": phase1,
    }


def _prepare_rows(
    rows: Iterable[Mapping[str, Any]],
    tokenizer: Any,
) -> tuple[list[dict[str, Any]], float]:
    prepared: list[dict[str, Any]] = []
    target_lengths: list[int] = []
    for row in rows:
        prompt, target_json = render_phase1_prompt(row["phase1"])
        completion = render_phase1_completion(target_json)
        target_tokens = len(tokenizer(
            completion,
            add_special_tokens=False,
        )["input_ids"])
        if target_tokens <= 0:
            raise HeldoutInferenceError(f"row {row['row_id']}: empty tokenized target")
        target_lengths.append(target_tokens)
        prepared.append({
            **dict(row),
            "prompt": prompt,
            "target_tokens": target_tokens,
        })
    target_p95 = percentile(target_lengths, 95)
    if target_p95 is None:
        raise HeldoutInferenceError("held-out manifest contains no target lengths")
    return prepared, target_p95


def _generate_rows(
    model: Any,
    tokenizer: Any,
    device: Any,
    rows: Iterable[Mapping[str, Any]],
    *,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    import torch

    output: list[dict[str, Any]] = []
    for row in rows:
        encoded = tokenizer(row["prompt"], return_tensors="pt")
        if hasattr(encoded, "to"):
            encoded = encoded.to(device)
        else:
            encoded = {key: value.to(device) for key, value in encoded.items()}
        prompt_tokens = int(encoded["input_ids"].shape[-1])
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latency = time.perf_counter() - started
        new_ids = generated[0][prompt_tokens:]
        generated_tokens = int(new_ids.numel())
        prediction = tokenizer.decode(new_ids, skip_special_tokens=True)
        phase1 = row["phase1"]
        output.append({
            "row_id": row["row_id"],
            "source_corpus": row["source_corpus"],
            "x": phase1["x"],
            "y_true": phase1["y_true"],
            "y_pred": {"text": prediction},
            "target_tokens": row["target_tokens"],
            "max_new_tokens": max_new_tokens,
            "generated_tokens": generated_tokens,
            "sequence_length": prompt_tokens + generated_tokens,
            "latency_sec": latency,
            "memory_peak_mb": (
                torch.cuda.max_memory_allocated() / (1024 * 1024)
                if torch.cuda.is_available()
                else 0.0
            ),
        })
    return output


def _validate_manifest(
    manifest: Mapping[str, Any],
    rows: list[Mapping[str, Any]],
    observed_sha: str,
) -> None:
    if manifest.get("artifact_sha256") != observed_sha:
        raise HeldoutInferenceError("held-out JSONL SHA does not match its manifest")
    if manifest.get("approved_heldout_rows") != len(rows):
        raise HeldoutInferenceError("held-out row count does not match approved manifest")
    required_corpora = manifest.get("required_corpora")
    if not isinstance(required_corpora, list) or not required_corpora:
        raise HeldoutInferenceError("manifest.required_corpora must be non-empty list[str]")
    observed_corpora = {_source_corpus(row) for row in rows}
    if not set(required_corpora) <= observed_corpora:
        raise HeldoutInferenceError("held-out JSONL is missing a required corpus")


def _source_corpus(row: Mapping[str, Any]) -> str:
    metadata = row.get("metadata")
    metadata_source = (
        metadata.get("source_corpus") if isinstance(metadata, Mapping) else None
    )
    value = (
        row.get("source_corpus")
        or row.get("source")
        or row.get("vendor")
        or metadata_source
    )
    if not isinstance(value, str) or not value.strip():
        raise HeldoutInferenceError("held-out row is missing source corpus identity")
    return value.strip()


def _validate_row_ids(
    manifest: Mapping[str, Any],
    rows: list[Mapping[str, Any]],
) -> None:
    expected = manifest.get("row_ids")
    if not isinstance(expected, list) or not expected:
        raise HeldoutInferenceError("held-out manifest has no row_ids")
    observed = [row["row_id"] for row in rows]
    if observed != expected:
        raise HeldoutInferenceError(
            "normalized held-out row IDs/order disagree with the approved manifest"
        )


def _row_id(source: str, rest_api: str) -> str:
    digest = hashlib.sha256(f"{source}\0{rest_api}".encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise HeldoutInferenceError(f"{path} must contain a JSON object")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise HeldoutInferenceError(f"{path}:{line_number}: blank row")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise HeldoutInferenceError(f"{path}:{line_number}: row must be an object")
            rows.append(value)
    if not rows:
        raise HeldoutInferenceError("held-out JSONL is empty")
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _write_prediction_pair_atomic(
    *,
    baseline_path: Path,
    baseline_rows: Iterable[Mapping[str, Any]],
    model_path: Path,
    model_rows: Iterable[Mapping[str, Any]],
) -> None:
    if baseline_path == model_path:
        raise HeldoutInferenceError("baseline and model output paths must differ")
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_pending = Path(f"{baseline_path}.pending")
    model_pending = Path(f"{model_path}.pending")
    baseline_rollback = Path(f"{baseline_path}.rollback")
    model_rollback = Path(f"{model_path}.rollback")
    artifacts = (
        baseline_pending,
        model_pending,
        baseline_rollback,
        model_rollback,
    )
    if any(path.exists() for path in artifacts):
        raise HeldoutInferenceError("stale prediction pending/rollback artifact exists")
    if baseline_path.exists() != model_path.exists():
        raise HeldoutInferenceError("baseline and model predictions must exist as a pair")

    _write_jsonl_pending(baseline_pending, baseline_rows)
    try:
        _write_jsonl_pending(model_pending, model_rows)
    except Exception:
        baseline_pending.unlink(missing_ok=True)
        raise
    had_previous = baseline_path.exists()
    if had_previous:
        baseline_path.replace(baseline_rollback)
        try:
            model_path.replace(model_rollback)
        except Exception:
            baseline_rollback.replace(baseline_path)
            baseline_pending.unlink(missing_ok=True)
            model_pending.unlink(missing_ok=True)
            raise
    try:
        baseline_pending.replace(baseline_path)
        model_pending.replace(model_path)
    except Exception:
        baseline_path.unlink(missing_ok=True)
        model_path.unlink(missing_ok=True)
        baseline_pending.unlink(missing_ok=True)
        model_pending.unlink(missing_ok=True)
        if had_previous:
            baseline_rollback.replace(baseline_path)
            model_rollback.replace(model_path)
        raise
    baseline_rollback.unlink(missing_ok=True)
    model_rollback.unlink(missing_ok=True)


def _write_jsonl_pending(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True))
            handle.write("\n")


def _release_model() -> None:
    import gc
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _configure_determinism(seed: int) -> None:
    import os
    import torch

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


if __name__ == "__main__":
    raise SystemExit(main())
