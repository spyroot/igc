"""Tests for model artifact cache planning, verification, and tensor inspection."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from scripts.gates.model_artifact_cache import materialize_base_models, planned_paths, verify
from scripts.gates.model_artifact_load_gate import inspect_adapter_tensors


def _sha(path: Path) -> str:
    """Return the SHA256 digest for a tiny fixture file."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _spec(root: Path, report: Path, weight: Path) -> dict:
    """Build a minimal cache spec for temp-file verification."""

    return {
        "version": 1,
        "default_root": str(root),
        "models": [{
            "role": "model_x",
            "phase": "phase1",
            "run_id": "fixture-run",
            "lfs_ref": "HEAD",
            "base_model": {
                "id": "fixture/base",
                "cache_subdir": "models--fixture--base",
            },
            "files": [
                {
                    "kind": "report",
                    "source_path": "unused/report.json",
                    "dest": "reports/report.json",
                    "sha256": _sha(report),
                },
                {
                    "kind": "weight",
                    "source_path": "unused/adapter_model.safetensors",
                    "dest": "artifacts/adapter_model.safetensors",
                    "lfs": True,
                    "sha256": _sha(weight),
                    "size_bytes": weight.stat().st_size,
                },
            ],
        }],
    }


def test_model_artifact_cache_plans_phase_role_run_layout(tmp_path: Path) -> None:
    """The plan nests artifacts by phase, role, and run id."""

    report = tmp_path / "report.json"
    weight = tmp_path / "adapter_model.safetensors"
    report.write_text('{"ok": true}', encoding="utf-8")
    weight.write_bytes(b"not-real-safetensors")
    spec = _spec(tmp_path / "cache", report, weight)

    plan = planned_paths(spec, tmp_path / "cache")

    run = plan["runs"][0]
    assert run["phase"] == "phase1"
    assert run["role"] == "model_x"
    assert run["run_id"] == "fixture-run"
    assert run["run_dir"].endswith("/phases/phase1/model_x/runs/fixture-run")


def test_model_artifact_cache_verify_checks_reports_and_weight_hash(tmp_path: Path) -> None:
    """Verify accepts a materialized report and adapter with matching SHA/size."""

    cache = tmp_path / "cache"
    run = cache / "phases/phase1/model_x/runs/fixture-run"
    report = run / "reports/report.json"
    weight = run / "artifacts/adapter_model.safetensors"
    report.parent.mkdir(parents=True)
    weight.parent.mkdir(parents=True)
    report.write_text(json.dumps({"schema": "fixture"}), encoding="utf-8")
    weight.write_bytes(b"fixture-weight")

    spec = _spec(cache, report, weight)
    result = verify(spec, cache)

    assert result["schema"] == "igc.model_artifact_cache.verify.v1"
    assert len(result["checks"]) == 2


def test_model_artifact_cache_links_base_model_once(tmp_path: Path) -> None:
    """Base HF cache directories are linked into the shared cache once."""

    source_root = tmp_path / "hf-cache"
    source_base = source_root / "models--fixture--base"
    source_base.mkdir(parents=True)
    (source_base / "config.json").write_text("{}", encoding="utf-8")
    report = tmp_path / "report.json"
    weight = tmp_path / "adapter_model.safetensors"
    report.write_text('{"ok": true}', encoding="utf-8")
    weight.write_bytes(b"weight")
    cache = tmp_path / "cache"

    records = materialize_base_models(_spec(cache, report, weight), cache, source_root)

    dest = cache / "base_models/models--fixture--base"
    assert records[0]["status"] == "linked"
    assert dest.is_symlink()
    assert dest.exists()


def test_model_artifact_load_gate_inspects_safetensor_shapes(tmp_path: Path) -> None:
    """Tensor inspection reads key/dtype/shape metadata from a tiny safetensors file."""

    safetensors = pytest.importorskip("safetensors.torch")
    path = tmp_path / "adapter_model.safetensors"
    safetensors.save_file({"adapter.weight": torch.zeros(2, 3)}, str(path))

    report = inspect_adapter_tensors(path)

    assert report["tensor_count"] == 1
    assert report["first_tensor"]["key"] == "adapter.weight"
    assert report["first_tensor"]["shape"] == [2, 3]
    assert report["all_shapes_rank_positive"] is True
