"""Verify cached model artifacts and optionally load base+adapter on CPU.

The default mode validates reports and adapter tensor shapes without downloading
or loading a giant base model. ``--load-cpu`` is an opt-in heavyweight gate for
approved Kubernetes/remote containers; it refuses laptop execution unless the
caller sets ``--allow-local-fixture`` for tiny unit-test fixtures.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.gates.model_artifact_cache import (  # noqa: E402
    ArtifactGateError,
    base_model_dir,
    load_spec,
    resolve_root,
    run_dir,
    verify,
)


def is_approved_execution_surface() -> bool:
    """Return whether heavy model loading is allowed on this machine."""

    return bool(os.environ.get("KUBERNETES_SERVICE_HOST") or os.environ.get("IGC_APPROVED_REMOTE"))


def inspect_adapter_tensors(adapter_path: Path) -> dict[str, Any]:
    """Inspect safetensors keys, dtypes, and shapes without moving tensors."""

    try:
        from safetensors import safe_open
    except Exception as exc:  # pragma: no cover - dependency is part of remote image
        raise ArtifactGateError(f"safetensors is required for tensor shape inspection: {exc}") from exc
    tensors: list[dict[str, Any]] = []
    with safe_open(adapter_path, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            tensor = handle.get_tensor(key)
            tensors.append({
                "key": key,
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
            })
    if not tensors:
        raise ArtifactGateError(f"adapter has no tensors: {adapter_path}")
    return {
        "tensor_count": len(tensors),
        "first_tensor": tensors[0],
        "all_shapes_rank_positive": all(len(item["shape"]) > 0 for item in tensors),
    }


def load_cpu_smoke(model: Mapping[str, Any], root: Path) -> dict[str, Any]:
    """Load base model + adapter on CPU from the shared cache."""

    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # pragma: no cover - dependency is remote-image specific
        raise ArtifactGateError(f"transformers/peft are required for CPU load gate: {exc}") from exc

    base_dir = base_model_dir(root, model)
    adapter_dir = run_dir(root, model) / "artifacts"
    tokenizer = AutoTokenizer.from_pretrained(base_dir, local_files_only=True)
    base = AutoModelForCausalLM.from_pretrained(
        base_dir,
        local_files_only=True,
        device_map=None,
        torch_dtype="auto",
    )
    loaded = PeftModel.from_pretrained(base, adapter_dir, local_files_only=True)
    return {
        "base_model_dir": str(base_dir),
        "adapter_dir": str(adapter_dir),
        "tokenizer_vocab_size": int(len(tokenizer)),
        "parameter_count": int(sum(param.numel() for param in loaded.parameters())),
    }


def run_gate(
        spec_path: Path,
        root_override: str | None,
        *,
        load_cpu: bool = False,
        allow_local_fixture: bool = False) -> dict[str, Any]:
    """Run report/tensor-shape checks and optional heavyweight CPU load."""

    spec = load_spec(spec_path)
    root = resolve_root(spec, root_override)
    verify_report = verify(spec, root, require_base_model=load_cpu)
    model_reports: list[dict[str, Any]] = []
    for model in spec["models"]:
        adapter = run_dir(root, model) / "artifacts" / "adapter_model.safetensors"
        model_report: dict[str, Any] = {
            "phase": model["phase"],
            "role": model["role"],
            "run_id": model["run_id"],
            "adapter_tensors": inspect_adapter_tensors(adapter),
        }
        if load_cpu:
            if not allow_local_fixture and not is_approved_execution_surface():
                raise ArtifactGateError(
                    "--load-cpu requires Kubernetes/approved remote execution; refusing laptop load"
                )
            model_report["cpu_load"] = load_cpu_smoke(model, root)
        model_reports.append(model_report)
    return {
        "schema": "igc.model_artifact_load_gate.v1",
        "root": str(root),
        "cache": verify_report,
        "models": model_reports,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default=str(Path("configs/artifacts/model_cache.yaml")))
    parser.add_argument("--root")
    parser.add_argument("--report")
    parser.add_argument("--load-cpu", action="store_true")
    parser.add_argument("--allow-local-fixture", action="store_true")
    args = parser.parse_args(argv)

    try:
        report = run_gate(
            Path(args.spec),
            args.root,
            load_cpu=args.load_cpu,
            allow_local_fixture=args.allow_local_fixture,
        )
    except ArtifactGateError as exc:
        print(f"BLOCKER: {exc}", file=sys.stderr)
        return 1

    text = json.dumps(report, indent=2, sort_keys=True)
    if args.report:
        path = Path(args.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print("MODEL_ARTIFACT_LOAD_GATE_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
