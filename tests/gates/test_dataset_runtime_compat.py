"""Offline contract tests for Phase 1 data/image compatibility."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from igc.ds.phase1_render import build_phase1_row
from igc.ds.sources.mixer import DataManifest
from scripts.gates import dataset_runtime_compat as gate


TOKENIZER_SHA = "sha256:" + "1" * 64
FOUNDATION_SHA = "sha256:" + "2" * 64


def _sha_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _json_sha(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return _sha_bytes(payload)


def _contract_repo(root: Path) -> Path:
    for relative in gate.CONTRACT_INPUTS:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        body = "chunking:\n  transform: phase1.lossless-json-chunk.v1\n"
        path.write_text(body if relative.endswith("phase1.yaml") else "contract\n")
    return root


def _whole_document_row(source: str, body: dict) -> dict:
    original_id = _sha_bytes(f"{source}\0/api/{source}".encode())
    chunk = {
        "version": "phase1.lossless-json-chunk.v1",
        "original_row_id": original_id,
        "original_json_sha256": _json_sha(body),
        "tokenizer_sha": TOKENIZER_SHA,
        "max_tokens": 2048,
        "index": 0,
        "count": 1,
        "json_path": [],
        "kind": "whole_document",
        "range_start": None,
        "range_stop": None,
        "container_length": None,
        "chunk_json_sha256": _json_sha(body),
    }
    row_id = _sha_bytes(
        json.dumps(chunk, sort_keys=True, separators=(",", ":")).encode()
    )
    return build_phase1_row(
        rest_api=f"/api/{source}",
        allowed_methods=["GET"],
        input_json=body,
        target_json=body,
        metadata={
            "row_id": row_id,
            "source_corpus": source,
            "trust_level": "REAL" if source == "real" else "REPLAY",
            "vendor": source,
            "chunk": chunk,
        },
    )


def _write_release(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, Path]]:
    release = tmp_path / "release"
    registry = tmp_path / "redfish_sources.yaml"
    registry.write_text(
        yaml.safe_dump({"sources": {"real": {}, "dsp": {}}}, sort_keys=True),
        encoding="utf-8",
    )
    source_manifests = {
        "real": tmp_path / "real.manifest.json",
        "dsp": tmp_path / "dsp.manifest.json",
    }
    for name, path in source_manifests.items():
        path.write_text(json.dumps({"source": name}) + "\n", encoding="utf-8")

    train_row = _whole_document_row("real", {"Value": 1})
    heldout_row = _whole_document_row("dsp", {"Value": 2})
    transform = {
        "transform": "phase1.lossless-json-chunk.v1",
        "tokenizer_sha": TOKENIZER_SHA,
        "max_tokens": 2048,
        "padding": "max_length",
        "overflow_policy": "lossless_json_chunk",
        "split_before_chunk": True,
        "exact_reassembly_verified": True,
    }
    manifest = DataManifest(
        total=2,
        train_count=1,
        eval_count=1,
        by_source={"real": 1, "dsp": 1},
        by_trust={"REAL": 1, "REPLAY": 1},
        by_vendor={"real": 1, "dsp": 1},
        eval_trust_floor="REAL",
        eval_fraction=0.5,
        seed=7,
        sources=["dsp", "real"],
        train_row_ids=[train_row["metadata"]["row_id"]],
        heldout_row_ids=[heldout_row["metadata"]["row_id"]],
        source_registry_sha=_sha_file(registry),
        source_manifest_shas={
            name: _sha_file(path) for name, path in source_manifests.items()
        },
        phase1_transform=transform,
    )
    manifest_payload = dict(manifest.__dict__)
    for split, row in (("train", train_row), ("heldout", heldout_row)):
        split_root = release / split
        split_root.mkdir(parents=True)
        (split_root / "examples.jsonl").write_text(
            json.dumps(row, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (split_root / "manifest.json").write_text(
            json.dumps(manifest_payload, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    summary = {
        "source_registry_sha": _sha_file(registry),
        "source_manifest_shas": manifest.source_manifest_shas,
        "train_artifact_sha": _sha_file(release / "train" / "examples.jsonl"),
        "heldout_artifact_sha": _sha_file(
            release / "heldout" / "examples.jsonl"
        ),
        "written_manifest_sha": _sha_file(release / "train" / "manifest.json"),
        "manifest_sha": manifest.content_hash(),
        "train_rows": 1,
        "heldout_rows": 1,
        "train_resources": 1,
        "heldout_resources": 1,
        "phase1_transform": transform,
        "training_profile": "phase1_7b_rslora_r32",
    }
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary) + "\n", encoding="utf-8")
    return release, summary_path, registry, source_manifests


def test_contract_identity_changes_with_materializer_inputs(tmp_path: Path) -> None:
    repo = _contract_repo(tmp_path)
    first = gate.contract_identity(repo)

    target = repo / "igc" / "ds" / "sources" / "mixer.py"
    target.write_text("changed\n", encoding="utf-8")
    second = gate.contract_identity(repo)

    assert first[0] != second[0]
    assert first[1] == "phase1.lossless-json-chunk.v1"


def test_image_labels_report_compatible_or_update_required(tmp_path: Path) -> None:
    repo = _contract_repo(tmp_path)
    digest, transform, _ = gate.contract_identity(repo)

    compatible = gate.compare_image(
        "igc-train:test",
        labels={gate.CONTRACT_LABEL: digest, gate.TRANSFORM_LABEL: transform},
        repo_root=repo,
    )
    stale = gate.compare_image(
        "igc-train:test",
        labels={gate.CONTRACT_LABEL: "sha256:" + "0" * 64},
        repo_root=repo,
    )

    assert compatible["status"] == "compatible"
    assert stale["status"] == "update-required"

    stale_transform = gate.compare_image(
        "igc-train:test",
        labels={
            gate.CONTRACT_LABEL: digest,
            gate.TRANSFORM_LABEL: "phase1.stale-transform.v0",
        },
        repo_root=repo,
    )
    assert stale_transform["status"] == "update-required"
    assert stale_transform["reasons"] == ["dataset transform version mismatch"]


def test_release_gate_rechecks_both_source_manifests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release, summary, registry, manifests = _write_release(tmp_path)
    monkeypatch.setenv("IGC_FOUNDATION_MODEL_SHA", FOUNDATION_SHA)
    monkeypatch.setenv("IGC_TOKENIZER_SHA", TOKENIZER_SHA)

    result = gate.validate_release(
        release_root=release,
        summary_path=summary,
        source_registry_path=registry,
        source_manifests=manifests,
        training_profile="phase1_7b_rslora_r32",
    )
    assert result["status"] == "passed"

    manifests["dsp"].write_text('{"source":"changed"}\n', encoding="utf-8")
    with pytest.raises(gate.GateError, match="source manifest SHA mismatch: dsp"):
        gate.validate_release(
            release_root=release,
            summary_path=summary,
            source_registry_path=registry,
            source_manifests=manifests,
            training_profile="phase1_7b_rslora_r32",
        )


def test_dockerfile_gate_rejects_raw_dataset_copy(tmp_path: Path) -> None:
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text(
        "\n".join((
            "FROM scratch",
            "ARG IGC_DATASET_CONTRACT_SHA",
            "ARG IGC_DATASET_TRANSFORM_VERSION",
            f"LABEL {gate.CONTRACT_LABEL}=x",
            f"LABEL {gate.TRANSFORM_LABEL}=x",
            "COPY datasets /data",
            "",
        )),
        encoding="utf-8",
    )

    result = gate.check_dockerfile(dockerfile)

    assert result["status"] == "failed"
    assert result["violations"] == ["line 6: COPY/ADD references datasets"]


def test_dockerfile_gate_rejects_raw_copy_on_continuation_line(tmp_path: Path) -> None:
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text(
        "\n".join((
            "FROM scratch",
            "ARG IGC_DATASET_CONTRACT_SHA",
            "ARG IGC_DATASET_TRANSFORM_VERSION",
            f"LABEL {gate.CONTRACT_LABEL}=x",
            f"LABEL {gate.TRANSFORM_LABEL}=x",
            "COPY igc \\",
            "     datasets /workspace",
            "",
        )),
        encoding="utf-8",
    )

    result = gate.check_dockerfile(dockerfile)

    assert result["status"] == "failed"
    assert result["violations"] == ["line 6: COPY/ADD references datasets"]


def test_dockerfile_gate_accepts_contract_labelled_image(tmp_path: Path) -> None:
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text(
        "\n".join((
            "FROM scratch",
            "ARG IGC_DATASET_CONTRACT_SHA",
            "ARG IGC_DATASET_TRANSFORM_VERSION",
            f"LABEL {gate.CONTRACT_LABEL}=x",
            f"LABEL {gate.TRANSFORM_LABEL}=x",
            "COPY igc /workspace/igc",
            "COPY copy.internal.conf /workspace/copy.internal.conf",
            "",
        )),
        encoding="utf-8",
    )

    result = gate.check_dockerfile(dockerfile)

    assert result["status"] == "passed"
    assert result["violations"] == []


def test_text_loader_blocks_non_utf8_input(tmp_path: Path) -> None:
    path = tmp_path / "invalid.json"
    path.write_bytes(b"\xff")

    with pytest.raises(gate.GateError, match="expected UTF-8 text"):
        gate._load_json(path)


def test_cli_usage_error_is_machine_readable(capsys: pytest.CaptureFixture[str]) -> None:
    rc = gate.main(["unknown-command"])

    assert rc == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "blocked"
    assert "argument error" in payload["error"]


def test_sha_parser_requires_algorithm_prefix() -> None:
    with pytest.raises(gate.GateError, match="sha256"):
        gate._sha_value("1" * 64, "artifact")


def test_sha_parser_accepts_prefixed_digest() -> None:
    value = "sha256:" + "a" * 64

    assert gate._sha_value(value, "artifact") == value
