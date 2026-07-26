"""Offline tests for canonical Phase 2/3 SFT held-out inference contracts."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

from igc.ds.rest_goal_contract import (
    RedfishContext,
    build_d1_rest_api_list_row,
)
from igc.modules.train import sft_inference


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "sft_heldout_inference.py"
PHASE2_TASK = "text_to_rest_api_list"
PHASE3_TASK = "text_and_rest_api_list_to_calls"
SOURCE_FULL_SHA = "sha256:" + "1" * 64
TRAIN_MANIFEST_SHA = "sha256:" + "2" * 64
TRAIN_DATA_SHA = "sha256:" + "3" * 64
SOURCE_FULL_DATA_SHA = "sha256:" + "c" * 64
HELDOUT_MANIFEST_SHA = "sha256:" + "4" * 64
HELDOUT_DATA_SHA = "sha256:" + "5" * 64
SPLIT_RELEASE_SHA = "sha256:" + "6" * 64
PARENT_SHA = "sha256:" + "7" * 64
ARTIFACT_SHA = "sha256:" + "8" * 64
FOUNDATION_SHA = "sha256:" + "9" * 64
TOKENIZER_SHA = "sha256:" + "a" * 64
ROW_ID = "sha256:" + "b" * 64


def _load_script():
    spec = importlib.util.spec_from_file_location("sft_heldout_inference", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _without_flag(argv: list[str], flag: str) -> list[str]:
    """Drop one required flag and its value from a parser argv fixture."""
    index = argv.index(flag)
    return [*argv[:index], *argv[index + 2 :]]


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _digest(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _contexts() -> list[RedfishContext]:
    apis = [
        "/redfish/v1/Systems/1",
        "/redfish/v1/TaskService",
        "/redfish/v1/EventService",
        "/redfish/v1/AccountService",
        "/redfish/v1/UpdateService",
    ]
    return [
        RedfishContext(
            rest_api=api,
            allowed_methods=["GET"],
            json={"@odata.id": api, "Name": api.rsplit("/", 1)[-1]},
        )
        for api in apis
    ]


def _phase2_row(
    *,
    row_id: str = ROW_ID,
    inference_case_id: str | None = None,
    variant: str | None = None,
) -> dict[str, Any]:
    row = build_d1_rest_api_list_row(
        text="inspect the selected Redfish system",
        contexts=_contexts(),
        rest_api_list=["/redfish/v1/Systems/1"],
        validation={
            "valid_json": True,
            "accepted": True,
            "natural": True,
            "nonsense": False,
            "ambiguous": False,
            "duplicate_intent": False,
            "extra_intents": False,
            "method_semantics_valid": True,
            "covered_api_set": ["/redfish/v1/Systems/1"],
        },
    )
    metadata = {"row_id": row_id, "sample_width_k": 1}
    if inference_case_id is not None:
        metadata["inference_case_id"] = inference_case_id
    if variant is not None:
        metadata["semantic_case_id"] = row_id
        metadata["robustness_variant"] = variant
    row["metadata"] = metadata
    return row


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _write_manifest(
    path: Path,
    *,
    artifact: Path,
    rows: int,
    view: str = "phase2",
) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": "d1_phase23_split_view.v1",
                "dataset": "D1",
                "view": view,
                "immutable": True,
                "complete": True,
                "rows": rows,
                "artifact_sha256": _sha256(artifact),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _write_spec(path: Path, *, adapter_dir: Path, adapter_size: int) -> Path:
    path.write_text(
        f"""
version: 1
name: unit-phase2-inference
phase: 2
task: {PHASE2_TASK}
contract_version: phase2-rest-api-set/v1
target_semantics: unordered_unique_rest_api_set
base_model:
  id: ${{UNIT_BASE_MODEL}}
  cache_dir: ${{UNIT_CACHE_DIR}}
  tokenizer: ${{UNIT_TOKENIZER}}
  foundation_model_sha: {FOUNDATION_SHA}
  tokenizer_sha: {TOKENIZER_SHA}
adapter:
  path: {adapter_dir}
  artifact_sha: {ARTIFACT_SHA}
  size_bytes: {adapter_size}
  method: lora
  rank: 8
  alpha: 16
  parent_role: model_x
  parent_artifact_sha: {PARENT_SHA}
  output_role: goal_extractor
runtime:
  torch_dtype: float32
  device: cpu
  device_map: auto
  require_cuda: false
  trust_remote_code: false
generation:
  seed: 17
  max_new_tokens: 64
  target_token_margin: 8
""",
        encoding="utf-8",
    )
    return path


def _spec_obj(tmp_path: Path) -> sft_inference.SFTInferenceSpec:
    return sft_inference.SFTInferenceSpec(
        path=tmp_path / "spec.yaml",
        name="unit-phase2-inference",
        phase=2,
        task=PHASE2_TASK,
        contract_version="phase2-rest-api-set/v1",
        target_semantics="unordered_unique_rest_api_set",
        base_model="local/model-x",
        cache_dir=tmp_path / "cache",
        tokenizer="local/tokenizer",
        foundation_model_sha=FOUNDATION_SHA,
        tokenizer_sha=TOKENIZER_SHA,
        adapter_dir=tmp_path / "adapter",
        artifact_sha=ARTIFACT_SHA,
        adapter_size_bytes=12,
        adapter_method="lora",
        adapter_rank=8,
        adapter_alpha=16,
        parent_role="model_x",
        parent_artifact_sha=PARENT_SHA,
        output_role="goal_extractor",
        torch_dtype="float32",
        device="cpu",
        device_map="auto",
        require_cuda=False,
        trust_remote_code=False,
        seed=17,
        max_new_tokens=64,
        target_token_margin=8,
    )


def _split_child(
    path: Path,
    *,
    phase: int,
    split: str,
    source_ids: list[str],
    source_full_manifest_sha: str,
) -> dict[str, Any]:
    payload = {
        "schema_version": "d1_phase23_split_view.v1",
        "dataset": "D1",
        "view": f"phase{phase}",
        "split": split,
        "immutable": True,
        "complete": True,
        "rows": len(source_ids),
        "artifact_sha256": _digest(f"{phase}:{split}:artifact"),
        "source_full_manifest_sha256": source_full_manifest_sha,
        "source_row_ids": source_ids,
    }
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _write_split_release(tmp_path: Path, *, overlap: bool = False) -> dict[str, Path]:
    release_dir = tmp_path / "split"
    release_dir.mkdir()
    source_full_manifest = release_dir / "source-full.manifest.json"
    source_full_manifest.write_text(
        json.dumps({"dataset": "D1", "rows": 3}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    source_full_sha = _sha256(source_full_manifest)
    train_ids = [_digest("train-row")]
    heldout_ids = train_ids if overlap else [_digest("heldout-row")]
    children: dict[str, dict[str, Any]] = {}
    for phase in (2, 3):
        for split, ids in (("train", train_ids), ("heldout", heldout_ids)):
            child_path = release_dir / f"phase{phase}_{split}.jsonl.manifest.json"
            children[f"phase{phase}_{split}"] = _split_child(
                child_path,
                phase=phase,
                split=split,
                source_ids=ids,
                source_full_manifest_sha=source_full_sha,
            )
    files = {}
    for name, child in children.items():
        manifest = f"{name}.jsonl.manifest.json"
        files[name] = {
            "path": f"{name}.jsonl",
            "manifest": manifest,
            "artifact_sha256": child["artifact_sha256"],
            "manifest_sha256": _sha256(release_dir / manifest),
        }
    release = {
        "schema_version": "d1_phase23_split_release.v1",
        "dataset": "D1",
        "immutable": True,
        "complete": True,
        "disjoint": True,
        "split_spec_sha256": _digest("split-spec"),
        "phase2_full_manifest_sha256": source_full_sha,
        "phase3_full_manifest_sha256": source_full_sha,
        "train_source_rows": len(train_ids),
        "heldout_source_rows": len(heldout_ids),
        "files": files,
    }
    release_path = release_dir / "release_manifest.json"
    release_path.write_text(json.dumps(release, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "release": release_path,
        "source_full_manifest": source_full_manifest,
        "phase2_train_manifest": release_dir / "phase2_train.jsonl.manifest.json",
        "phase2_heldout_manifest": release_dir / "phase2_heldout.jsonl.manifest.json",
    }


def test_sft_inference_spec_is_strict_resolves_env_and_binds_phase_renderer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The canonical spec rejects unknown keys and resolves env placeholders."""
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    spec_path = _write_spec(tmp_path / "phase2.yaml", adapter_dir=adapter_dir, adapter_size=17)
    monkeypatch.setenv("UNIT_BASE_MODEL", "local/model-x")
    monkeypatch.setenv("UNIT_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("UNIT_TOKENIZER", "local/tokenizer")

    spec = sft_inference.load_sft_inference_spec(spec_path)

    assert spec.phase == 2
    assert spec.task == PHASE2_TASK
    assert spec.base_model == "local/model-x"
    assert sft_inference.renderer_for_phase(2).__name__ == "render_phase2_sft"
    assert sft_inference.renderer_for_phase(3).__name__ == "render_phase3_sft"
    bad_payload = spec_path.read_text(encoding="utf-8") + "unexpected: true\n"
    bad_spec = tmp_path / "bad.yaml"
    bad_spec.write_text(bad_payload, encoding="utf-8")
    with pytest.raises(sft_inference.SFTInferenceError, match="unknown keys"):
        sft_inference.load_sft_inference_spec(bad_spec)
    monkeypatch.delenv("UNIT_TOKENIZER")
    with pytest.raises(sft_inference.SFTInferenceError, match="UNIT_TOKENIZER"):
        sft_inference.load_sft_inference_spec(spec_path)


def test_verify_dataset_release_allows_shared_row_id_when_inference_case_id_is_unique(
    tmp_path: Path,
) -> None:
    """Robustness variants may share metadata.row_id but need unique inference cases."""
    rows = [
        _phase2_row(
            inference_case_id=_digest(f"{ROW_ID}:base"),
            variant="base",
        ),
        _phase2_row(
            inference_case_id=_digest(f"{ROW_ID}:api_context_shuffled"),
            variant="api_context_shuffled",
        ),
    ]
    data_path = _write_jsonl(tmp_path / "heldout.jsonl", rows)
    manifest_path = _write_manifest(
        tmp_path / "heldout.manifest.json",
        artifact=data_path,
        rows=len(rows),
    )

    loaded, release = sft_inference.verify_dataset_release(
        artifact_path=data_path,
        manifest_path=manifest_path,
        phase=2,
    )

    assert [row["metadata"]["row_id"] for row in loaded] == [ROW_ID, ROW_ID]
    assert release["artifact_sha"] == _sha256(data_path)
    rows[1]["metadata"]["inference_case_id"] = rows[0]["metadata"]["inference_case_id"]
    _write_jsonl(data_path, rows)
    _write_manifest(manifest_path, artifact=data_path, rows=len(rows))
    with pytest.raises(sft_inference.SFTInferenceError, match="duplicate inference case"):
        sft_inference.verify_dataset_release(
            artifact_path=data_path,
            manifest_path=manifest_path,
            phase=2,
        )


@pytest.mark.parametrize(
    ("manifest_update", "message"),
    [
        ({"artifact_sha256": "sha256:" + "0" * 64}, "artifact SHA"),
        ({"rows": 99}, "row count"),
        ({"dataset": "phase2_labelled_requests"}, "D1"),
        ({"immutable": False}, "immutable"),
        ({"complete": False}, "immutable"),
    ],
)
def test_verify_dataset_release_rejects_manifest_identity_or_data_mismatch(
    tmp_path: Path,
    manifest_update: dict[str, object],
    message: str,
) -> None:
    """Held-out JSONL verification checks exact bytes, row count, and D1 identity."""
    rows = [_phase2_row()]
    data_path = _write_jsonl(tmp_path / "heldout.jsonl", rows)
    manifest_path = _write_manifest(tmp_path / "heldout.manifest.json", artifact=data_path, rows=1)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(manifest_update)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(sft_inference.SFTInferenceError, match=message):
        sft_inference.verify_dataset_release(
            artifact_path=data_path,
            manifest_path=manifest_path,
            phase=2,
        )


def test_validate_split_release_binds_source_train_and_heldout_manifests(
    tmp_path: Path,
) -> None:
    """Split lineage keeps source-full, train, and heldout manifests distinct."""
    paths = _write_split_release(tmp_path)

    release = sft_inference.validate_split_release(
        release_manifest_path=paths["release"],
        source_full_manifest_path=paths["source_full_manifest"],
        train_manifest_path=paths["phase2_train_manifest"],
        heldout_manifest_path=paths["phase2_heldout_manifest"],
        phase=2,
    )

    assert release["source_full_manifest_sha"] == _sha256(paths["source_full_manifest"])
    assert release["train_manifest_sha"] == _sha256(paths["phase2_train_manifest"])
    assert release["heldout_manifest_sha"] == _sha256(paths["phase2_heldout_manifest"])
    assert release["source_full_manifest_sha"] != release["train_manifest_sha"]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        ("overlap", "overlap"),
        ("child_sha", "child manifest SHA"),
        ("source_sha", "complete source manifest"),
    ],
)
def test_validate_split_release_rejects_overlap_child_sha_and_source_sha_mismatch(
    tmp_path: Path,
    mutate: str,
    message: str,
) -> None:
    """Split release evidence fails closed on overlap and SHA mismatches."""
    paths = _write_split_release(tmp_path, overlap=mutate == "overlap")
    if mutate == "child_sha":
        release = json.loads(paths["release"].read_text(encoding="utf-8"))
        release["files"]["phase2_train"]["manifest_sha256"] = "sha256:" + "0" * 64
        paths["release"].write_text(json.dumps(release, sort_keys=True) + "\n", encoding="utf-8")
    if mutate == "source_sha":
        release = json.loads(paths["release"].read_text(encoding="utf-8"))
        release["phase2_full_manifest_sha256"] = "sha256:" + "0" * 64
        paths["release"].write_text(json.dumps(release, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(sft_inference.SFTInferenceError, match=message):
        sft_inference.validate_split_release(
            release_manifest_path=paths["release"],
            source_full_manifest_path=paths["source_full_manifest"],
            train_manifest_path=paths["phase2_train_manifest"],
            heldout_manifest_path=paths["phase2_heldout_manifest"],
            phase=2,
        )


def test_generate_prediction_rows_preserves_raw_completion_text_and_blocks_empty() -> None:
    """Generation evidence stores raw y_pred text without parsing or dropping rows."""
    import torch

    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 2

        def __call__(self, _text, return_tensors=None, add_special_tokens=True):
            return {"input_ids": torch.tensor([[10, 11]])}

        def decode(self, _ids, skip_special_tokens=True):
            return '{"rest_api_list":["/redfish/v1/Systems/1"]}'

    class EmptyTokenizer(FakeTokenizer):
        def decode(self, _ids, skip_special_tokens=True):
            return "  "

    class FakeModel:
        def generate(self, **_kwargs):
            return torch.tensor([[10, 11, 12, 13]])

    prepared = [{"row": _phase2_row(), "prompt": "prompt", "target_tokens": 4}]

    rows = sft_inference.generate_prediction_rows(
        FakeModel(),
        FakeTokenizer(),
        torch.device("cpu"),
        prepared,
        max_new_tokens=8,
    )

    assert rows[0]["y_pred"] == '{"rest_api_list":["/redfish/v1/Systems/1"]}'
    assert rows[0]["inference"]["generated_tokens"] == 2
    with pytest.raises(sft_inference.SFTInferenceError, match="empty prediction"):
        sft_inference.generate_prediction_rows(
            FakeModel(),
            EmptyTokenizer(),
            torch.device("cpu"),
            prepared,
            max_new_tokens=8,
        )


def test_build_artifact_evidence_keeps_source_train_heldout_and_split_lineage(
    tmp_path: Path,
) -> None:
    """Artifact evidence exposes exact source-full, train, heldout, and split SHAs."""
    evidence = sft_inference.build_artifact_evidence(
        spec=_spec_obj(tmp_path),
        source_full_release={
            "immutable": True,
            "complete": True,
            "rows": 12,
            "manifest_sha": SOURCE_FULL_SHA,
            "artifact_sha": SOURCE_FULL_DATA_SHA,
        },
        train_release={
            "immutable": True,
            "complete": True,
            "rows": 8,
            "manifest_sha": TRAIN_MANIFEST_SHA,
            "artifact_sha": TRAIN_DATA_SHA,
        },
        heldout_release={
            "immutable": True,
            "complete": True,
            "rows": 4,
            "manifest_sha": HELDOUT_MANIFEST_SHA,
            "artifact_sha": HELDOUT_DATA_SHA,
        },
        split_release={
            "immutable": True,
            "complete": True,
            "disjoint": True,
            "sha256": SPLIT_RELEASE_SHA,
            "source_full_manifest_sha": SOURCE_FULL_SHA,
            "train_manifest_sha": TRAIN_MANIFEST_SHA,
            "heldout_manifest_sha": HELDOUT_MANIFEST_SHA,
        },
        prediction_rows=4,
        predictions_sha=_digest("predictions"),
    )

    assert evidence["immutable_full_manifest"]["sha256"] == SOURCE_FULL_SHA
    assert evidence["immutable_full_manifest"]["artifact_sha"] == SOURCE_FULL_DATA_SHA
    assert evidence["immutable_train_manifest"]["manifest_sha"] == TRAIN_MANIFEST_SHA
    assert evidence["immutable_train_manifest"]["artifact_sha"] == TRAIN_DATA_SHA
    assert evidence["disjoint_split_release"]["train_manifest_sha"] == TRAIN_MANIFEST_SHA
    assert evidence["real_heldout_data"]["manifest_sha"] == HELDOUT_MANIFEST_SHA
    assert evidence["artifact_sha"] == ARTIFACT_SHA
    rendered = json.dumps(evidence, sort_keys=True)
    assert "api_key" not in rendered.lower()
    assert "judge_route" not in rendered.lower()


def test_script_parser_requires_source_train_heldout_split_and_outputs() -> None:
    """The CLI exposes distinct source-full, train, heldout, and split manifests."""
    script = _load_script()

    with pytest.raises(SystemExit):
        script.parse_args(
            [
                "--spec",
                "spec.yaml",
                "--source-full-jsonl",
                "source-full.jsonl",
                "--source-full-manifest",
                "source-full.manifest.json",
                "--train-jsonl",
                "train.jsonl",
                "--train-manifest",
                "train.manifest.json",
                "--heldout-jsonl",
                "heldout.jsonl",
                "--heldout-manifest",
                "heldout.manifest.json",
                "--parent-promotion-evidence",
                "parent.json",
                "--predictions-output",
                "predictions.jsonl",
                "--artifact-evidence-output",
                "evidence.json",
            ],
        )

    args = script.parse_args(
        [
            "--spec",
            "spec.yaml",
            "--source-full-jsonl",
            "source-full.jsonl",
            "--source-full-manifest",
            "source-full.manifest.json",
            "--train-jsonl",
            "train.jsonl",
            "--train-manifest",
            "train.manifest.json",
            "--heldout-jsonl",
            "heldout.jsonl",
            "--heldout-manifest",
            "heldout.manifest.json",
            "--split-release-manifest",
            "split.release.json",
            "--parent-promotion-evidence",
            "parent.json",
            "--predictions-output",
            "predictions.jsonl",
            "--artifact-evidence-output",
            "evidence.json",
        ],
    )

    assert args.source_full_jsonl == "source-full.jsonl"
    assert args.source_full_manifest == "source-full.manifest.json"
    assert args.train_jsonl == "train.jsonl"
    assert args.train_manifest == "train.manifest.json"
    assert args.heldout_manifest == "heldout.manifest.json"
    assert args.split_release_manifest == "split.release.json"


@pytest.mark.parametrize("missing_flag", ["--source-full-jsonl", "--train-jsonl"])
def test_script_parser_requires_source_and_train_jsonl_independently(
    missing_flag: str,
) -> None:
    """Held-out inference requires both full-source and train JSONL inputs."""
    script = _load_script()
    argv = [
        "--spec",
        "spec.yaml",
        "--source-full-jsonl",
        "source-full.jsonl",
        "--source-full-manifest",
        "source-full.manifest.json",
        "--train-jsonl",
        "train.jsonl",
        "--train-manifest",
        "train.manifest.json",
        "--heldout-jsonl",
        "heldout.jsonl",
        "--heldout-manifest",
        "heldout.manifest.json",
        "--split-release-manifest",
        "split.release.json",
        "--parent-promotion-evidence",
        "parent.json",
        "--predictions-output",
        "predictions.jsonl",
        "--artifact-evidence-output",
        "evidence.json",
    ]

    with pytest.raises(SystemExit):
        script.parse_args(_without_flag(argv, missing_flag))


def test_script_adapter_weight_sha_size_and_atomic_output(tmp_path: Path) -> None:
    """The script verifies adapter bytes and atomically publishes predictions/evidence."""
    script = _load_script()
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    weight = adapter_dir / "adapter_model.safetensors"
    payload = b"exact adapter bytes\n"
    weight.write_bytes(payload)
    spec = _spec_obj(tmp_path)
    spec = sft_inference.SFTInferenceSpec(
        **{
            **spec.__dict__,
            "adapter_dir": adapter_dir,
            "adapter_size_bytes": len(payload),
            "artifact_sha": _sha256(weight),
        },
    )

    script._validate_adapter_bytes(spec)
    wrong_sha = sft_inference.SFTInferenceSpec(
        **{**spec.__dict__, "artifact_sha": "sha256:" + "0" * 64},
    )
    with pytest.raises(sft_inference.SFTInferenceError, match="SHA"):
        script._validate_adapter_bytes(wrong_sha)
    wrong_size = sft_inference.SFTInferenceSpec(
        **{**spec.__dict__, "adapter_size_bytes": len(payload) + 1},
    )
    with pytest.raises(sft_inference.SFTInferenceError, match="size"):
        script._validate_adapter_bytes(wrong_size)

    predictions = [{"metadata": {"row_id": ROW_ID}, "y_pred": "{}"}]
    predictions_path = tmp_path / "predictions.jsonl"
    evidence_path = tmp_path / "evidence.json"
    script._write_prediction_evidence_pair(
        predictions_path=predictions_path,
        predictions=predictions,
        evidence_path=evidence_path,
        evidence_factory=lambda predictions_sha: {
            "status": "pass",
            "predictions_sha": predictions_sha,
            "artifact_sha": spec.artifact_sha,
        },
    )

    assert json.loads(evidence_path.read_text(encoding="utf-8"))["predictions_sha"] == _sha256(
        predictions_path,
    )
    assert not Path(f"{predictions_path}.pending").exists()
    assert not Path(f"{evidence_path}.pending").exists()
