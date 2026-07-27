#!/usr/bin/env python3
"""Run Phase 2 or Phase 3 held-out inference and emit promotion evidence."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT))

from phase1_gpu_inference_gate import (  # noqa: E402
    Paths,
    load_model_and_tokenizer,
    preflight_inputs,
    set_offline_env,
    validate_adapter_config,
)
from igc.modules.train.sft_inference import (  # noqa: E402
    SFTInferenceError,
    build_artifact_evidence,
    generate_prediction_rows,
    load_sft_inference_spec,
    prepare_heldout_cases,
    sha256_file,
    validate_parent_promotion,
    validate_split_release,
    verify_dataset_release,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse exact immutable inputs and paired output artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--source-full-jsonl", required=True)
    parser.add_argument("--source-full-manifest", required=True)
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--heldout-jsonl", required=True)
    parser.add_argument("--heldout-manifest", required=True)
    parser.add_argument("--split-release-manifest", required=True)
    parser.add_argument("--parent-promotion-evidence", required=True)
    parser.add_argument("--predictions-output", required=True)
    parser.add_argument("--artifact-evidence-output", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Load one promoted adapter, infer every held-out row, and publish evidence."""
    args = parse_args(argv)
    try:
        spec = load_sft_inference_spec(args.spec)
        parent_promotion = _json_object(Path(args.parent_promotion_evidence))
        validate_parent_promotion(
            parent_promotion,
            expected_role=spec.parent_role,
            expected_artifact_sha=spec.parent_artifact_sha,
        )
        _, source_full_release = verify_dataset_release(
            artifact_path=args.source_full_jsonl,
            manifest_path=args.source_full_manifest,
            phase=spec.phase,
            keep_rows=False,
        )
        _, train_release = verify_dataset_release(
            artifact_path=args.train_jsonl,
            manifest_path=args.train_manifest,
            phase=spec.phase,
            keep_rows=False,
        )
        heldout_rows, heldout_release = verify_dataset_release(
            artifact_path=args.heldout_jsonl,
            manifest_path=args.heldout_manifest,
            phase=spec.phase,
        )
        split_release = validate_split_release(
            release_manifest_path=args.split_release_manifest,
            source_full_manifest_path=args.source_full_manifest,
            train_manifest_path=args.train_manifest,
            heldout_manifest_path=args.heldout_manifest,
            phase=spec.phase,
        )

        runtime_spec = _runtime_spec(spec)
        paths = Paths(
            base_model=spec.base_model,
            cache_dir=spec.cache_dir,
            adapter_dir=spec.adapter_dir,
            output_json=None,
        )
        set_offline_env(spec.cache_dir)
        preflight_inputs(paths, require_cuda=spec.require_cuda)
        validate_adapter_config(runtime_spec)
        _validate_adapter_bytes(spec)
        _configure_determinism(spec.seed)

        model_args = SimpleNamespace(spec=runtime_spec, allow_cpu=False)
        model, tokenizer, device = load_model_and_tokenizer(model_args)
        prepared, target_p95 = prepare_heldout_cases(
            heldout_rows,
            phase=spec.phase,
            tokenizer=tokenizer,
            max_new_tokens=spec.max_new_tokens,
            target_token_margin=spec.target_token_margin,
        )
        predictions = generate_prediction_rows(
            model,
            tokenizer,
            device,
            prepared,
            max_new_tokens=spec.max_new_tokens,
        )
        predictions_path = Path(args.predictions_output)
        evidence_path = Path(args.artifact_evidence_output)
        _write_prediction_evidence_pair(
            predictions_path=predictions_path,
            predictions=predictions,
            evidence_path=evidence_path,
            evidence_factory=lambda predictions_sha: {
                **build_artifact_evidence(
                    spec=spec,
                    source_full_release=source_full_release,
                    train_release=train_release,
                    heldout_release=heldout_release,
                    split_release=split_release,
                    prediction_rows=len(predictions),
                    predictions_sha=predictions_sha,
                ),
                "spec_sha": sha256_file(spec.path),
                "target_completion_tokens_p95": target_p95,
                "max_new_tokens": spec.max_new_tokens,
                "target_token_margin": spec.target_token_margin,
            },
        )
        print(json.dumps({
            "status": "pass",
            "phase": spec.phase,
            "role": spec.output_role,
            "rows": len(predictions),
            "predictions": str(predictions_path),
            "artifact_evidence": str(evidence_path),
        }, sort_keys=True))
        return 0
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(json.dumps({
            "status": "blocked",
            "error": str(exc),
            "safe_next_step": (
                "repair the immutable data/model evidence or approved remote runtime"
            ),
        }, sort_keys=True), file=sys.stderr)
        return 2


def _runtime_spec(spec: Any) -> SimpleNamespace:
    """Adapt the shared SFT spec to the existing offline PEFT loader contract."""
    return SimpleNamespace(
        path=spec.path,
        base_model=spec.base_model,
        cache_dir=spec.cache_dir,
        tokenizer=spec.tokenizer,
        adapter_dir=spec.adapter_dir,
        torch_dtype=spec.torch_dtype,
        device=spec.device,
        device_map=spec.device_map,
        require_cuda=spec.require_cuda,
        trust_remote_code=spec.trust_remote_code,
        adapter_sha256=spec.artifact_sha.removeprefix("sha256:"),
        adapter_size_bytes=spec.adapter_size_bytes,
        adapter_method=spec.adapter_method,
        adapter_rank=spec.adapter_rank,
        adapter_alpha=spec.adapter_alpha,
    )


def _validate_adapter_bytes(spec: Any) -> None:
    """Tie the configured artifact identity to the exact adapter weight file."""
    weight = spec.adapter_dir / "adapter_model.safetensors"
    if weight.stat().st_size != spec.adapter_size_bytes:
        raise SFTInferenceError(
            "adapter weight size does not match the inference spec"
        )
    if sha256_file(weight) != spec.artifact_sha:
        raise SFTInferenceError(
            "adapter weight SHA does not match the inference spec"
        )


def _write_prediction_evidence_pair(
    *,
    predictions_path: Path,
    predictions: Sequence[Mapping[str, Any]],
    evidence_path: Path,
    evidence_factory: Callable[[str], Mapping[str, Any]],
) -> None:
    """Atomically publish predictions and their exact evidence as one pair."""
    if predictions_path == evidence_path:
        raise SFTInferenceError("prediction and evidence paths must differ")
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_pending = Path(f"{predictions_path}.pending")
    evidence_pending = Path(f"{evidence_path}.pending")
    prediction_rollback = Path(f"{predictions_path}.rollback")
    evidence_rollback = Path(f"{evidence_path}.rollback")
    scratch = (
        prediction_pending,
        evidence_pending,
        prediction_rollback,
        evidence_rollback,
    )
    if any(path.exists() for path in scratch):
        raise SFTInferenceError(
            "stale inference pending/rollback artifact requires inspection"
        )
    if predictions_path.exists() != evidence_path.exists():
        raise SFTInferenceError(
            "canonical predictions and evidence must exist as a pair"
        )

    try:
        with prediction_pending.open("x", encoding="utf-8") as handle:
            for row in predictions:
                handle.write(json.dumps(dict(row), sort_keys=True))
                handle.write("\n")
        predictions_sha = sha256_file(prediction_pending)
        evidence = evidence_factory(predictions_sha)
        evidence_pending.write_text(
            json.dumps(evidence, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if sha256_file(prediction_pending) != predictions_sha:
            raise SFTInferenceError("pending predictions changed before publication")
    except Exception:
        prediction_pending.unlink(missing_ok=True)
        evidence_pending.unlink(missing_ok=True)
        raise

    had_previous = predictions_path.exists()
    if had_previous:
        os.replace(predictions_path, prediction_rollback)
        try:
            os.replace(evidence_path, evidence_rollback)
        except Exception:
            os.replace(prediction_rollback, predictions_path)
            prediction_pending.unlink(missing_ok=True)
            evidence_pending.unlink(missing_ok=True)
            raise
    try:
        os.replace(prediction_pending, predictions_path)
        os.replace(evidence_pending, evidence_path)
    except Exception:
        predictions_path.unlink(missing_ok=True)
        evidence_path.unlink(missing_ok=True)
        prediction_pending.unlink(missing_ok=True)
        evidence_pending.unlink(missing_ok=True)
        if had_previous:
            os.replace(prediction_rollback, predictions_path)
            os.replace(evidence_rollback, evidence_path)
        raise
    prediction_rollback.unlink(missing_ok=True)
    evidence_rollback.unlink(missing_ok=True)


def _json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SFTInferenceError(f"{path} must contain a JSON object")
    return value


def _configure_determinism(seed: int) -> None:
    import torch

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


if __name__ == "__main__":
    raise SystemExit(main())
