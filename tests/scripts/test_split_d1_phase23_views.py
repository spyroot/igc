"""Offline tests for deterministic aligned D1 Phase 2/3 split releases."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

from igc.ds.rest_goal_contract import (
    RedfishContext,
    build_call_row,
    build_d1_rest_api_list_row,
)


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "split_d1_phase23_views.py"
SEED = "unit-d1-phase23-split"
HELDOUT_FRACTION = 0.5
VARIANTS = (
    "base",
    "api_context_shuffled",
    "json_key_order_shuffled",
    "target_serialization_reversed",
    "irrelevant_distractors_added",
)
ARGUMENT_CLASSES = (
    "read_only_empty",
    "patch_scalar",
    "patch_nested",
    "post_no_arguments",
    "post_one_argument",
    "post_multiple_arguments",
    "delete_no_body",
)
DISTRACTOR_APIS = (
    "/redfish/v1/TaskService",
    "/redfish/v1/EventService",
    "/redfish/v1/AccountService",
    "/redfish/v1/UpdateService",
)


def _load_script():
    spec = importlib.util.spec_from_file_location("split_d1_phase23_views", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _digest(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _row_id_for_bucket(label: str, *, heldout: bool) -> str:
    cutoff = int(HELDOUT_FRACTION * (1 << 256))
    for index in range(10_000):
        row_id = _digest(f"{label}:{index}")
        raw = hashlib.sha256(f"{SEED}\0{row_id}".encode("utf-8")).digest()
        if (int.from_bytes(raw, "big") < cutoff) is heldout:
            return row_id
    raise AssertionError(f"could not find deterministic row_id for {label}")


def _context(api: str) -> RedfishContext:
    if api.endswith("Bios/Settings"):
        return RedfishContext(
            rest_api=api,
            allowed_methods=["PATCH"],
            operation_names=["UpdateBiosSettings"],
            argument_schema={
                "properties": {
                    "Attributes": {
                        "type": "object",
                        "properties": {"BootMode": {"type": "string"}},
                        "required": ["BootMode"],
                    },
                },
                "required": ["Attributes"],
            },
            json={"@odata.id": api, "Attributes": {"BootMode": "Legacy"}},
        )
    if api.endswith("ComputerSystem.Reset"):
        return RedfishContext(
            rest_api=api,
            allowed_methods=["PATCH"],
            operation_names=["Reset"],
            argument_schema={
                "properties": {"ResetType": {"type": "string"}},
                "required": ["ResetType"],
            },
            json={"@odata.id": api, "target": api},
        )
    if api.endswith("SessionService/Sessions"):
        return RedfishContext(
            rest_api=api,
            allowed_methods=["POST"],
            operation_names=["CreateSessionWithDefaults"],
            argument_schema={},
            json={"@odata.id": api, "Members": []},
        )
    if api.endswith("Manager.Reset"):
        return RedfishContext(
            rest_api=api,
            allowed_methods=["POST"],
            operation_names=["ResetManager"],
            argument_schema={
                "properties": {"ResetType": {"type": "string"}},
                "required": ["ResetType"],
            },
            json={"@odata.id": api, "target": api},
        )
    if api.endswith("UpdateService.SimpleUpdate"):
        return RedfishContext(
            rest_api=api,
            allowed_methods=["POST"],
            operation_names=["SimpleUpdate"],
            argument_schema={
                "properties": {
                    "ImageURI": {"type": "string"},
                    "TransferProtocol": {"type": "string"},
                },
                "required": ["ImageURI", "TransferProtocol"],
            },
            json={"@odata.id": api, "target": api},
        )
    if api.endswith("Sessions/1"):
        return RedfishContext(
            rest_api=api,
            allowed_methods=["DELETE"],
            operation_names=["DeleteSession"],
            argument_schema={},
            json={"@odata.id": api, "Id": "1"},
        )
    return RedfishContext(
        rest_api=api,
        allowed_methods=["GET"],
        operation_names=[],
        argument_schema={},
        json={"@odata.id": api, "Name": api.rsplit("/", 1)[-1]},
    )


def _call_for_class(name: str) -> dict[str, Any]:
    calls: dict[str, dict[str, Any]] = {
        "read_only_empty": {
            "rest_api": "/redfish/v1/Systems/1",
            "http_method": "GET",
            "operation_name": None,
            "arguments": {},
        },
        "patch_scalar": {
            "rest_api": "/redfish/v1/Systems/1/Actions/ComputerSystem.Reset",
            "http_method": "PATCH",
            "operation_name": "Reset",
            "arguments": {"ResetType": "GracefulRestart"},
        },
        "patch_nested": {
            "rest_api": "/redfish/v1/Systems/1/Bios/Settings",
            "http_method": "PATCH",
            "operation_name": "UpdateBiosSettings",
            "arguments": {"Attributes": {"BootMode": "Uefi"}},
        },
        "post_no_arguments": {
            "rest_api": "/redfish/v1/SessionService/Sessions",
            "http_method": "POST",
            "operation_name": "CreateSessionWithDefaults",
            "arguments": {},
        },
        "post_one_argument": {
            "rest_api": "/redfish/v1/Managers/1/Actions/Manager.Reset",
            "http_method": "POST",
            "operation_name": "ResetManager",
            "arguments": {"ResetType": "ForceRestart"},
        },
        "post_multiple_arguments": {
            "rest_api": "/redfish/v1/UpdateService/Actions/UpdateService.SimpleUpdate",
            "http_method": "POST",
            "operation_name": "SimpleUpdate",
            "arguments": {
                "ImageURI": "https://updates.example.invalid/firmware.bin",
                "TransferProtocol": "HTTPS",
            },
        },
        "delete_no_body": {
            "rest_api": "/redfish/v1/SessionService/Sessions/1",
            "http_method": "DELETE",
            "operation_name": "DeleteSession",
            "arguments": {},
        },
    }
    return dict(calls[name])


def _contexts(targets: Sequence[str]) -> list[RedfishContext]:
    apis = list(dict.fromkeys([*targets, *DISTRACTOR_APIS]))
    return [_context(api) for api in apis]


def _phase_rows(class_name: str, *, row_id: str, group: str) -> tuple[dict, dict]:
    call = _call_for_class(class_name)
    targets = [call["rest_api"]]
    contexts = _contexts(targets)
    phase2 = build_d1_rest_api_list_row(
        text=f"Operate on {class_name}.",
        contexts=contexts,
        rest_api_list=targets,
        validation={
            "valid_json": True,
            "accepted": True,
            "natural": True,
            "nonsense": False,
            "ambiguous": False,
            "duplicate_intent": False,
            "extra_intents": False,
            "method_semantics_valid": True,
            "covered_api_set": targets,
        },
    )
    phase3 = build_call_row(
        text=f"Operate on {class_name}.",
        contexts=contexts,
        rest_api_list=targets,
        method_by_api={call["rest_api"]: call["http_method"]},
        operation_name_by_api={call["rest_api"]: call["operation_name"]},
        arguments_by_api={call["rest_api"]: call["arguments"]},
    )
    metadata = {
        "row_id": row_id,
        "sample_width_k": 1,
        "heldout_vendor_or_model": [group],
    }
    phase2["metadata"] = dict(metadata)
    phase3["metadata"] = dict(metadata)
    return phase2, phase3


def _empty_phase_rows(*, row_id: str, group: str) -> tuple[dict, dict]:
    contexts = _contexts([])
    phase2 = build_d1_rest_api_list_row(
        text="Ask for something outside the shown resources.",
        contexts=contexts,
        rest_api_list=[],
        validation={
            "valid_json": True,
            "accepted": True,
            "natural": True,
            "nonsense": False,
            "ambiguous": False,
            "duplicate_intent": False,
            "extra_intents": False,
            "method_semantics_valid": True,
            "covered_api_set": [],
        },
    )
    phase3 = build_call_row(
        text="Ask for something outside the shown resources.",
        contexts=contexts,
        rest_api_list=[],
        method_by_api={},
        operation_name_by_api={},
        arguments_by_api={},
    )
    metadata = {
        "row_id": row_id,
        "sample_width_k": 0,
        "heldout_vendor_or_model": [group],
    }
    phase2["metadata"] = dict(metadata)
    phase3["metadata"] = dict(metadata)
    return phase2, phase3


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_manifest(path: Path, *, artifact: Path, view: str, rows: int) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": "d1_full_view.v1",
                "dataset": "D1",
                "view": view,
                "immutable": True,
                "complete": True,
                "rows": rows,
                "artifact_sha256": _sha256(artifact),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _write_spec(path: Path, **overrides: Any) -> Path:
    values = {
        "version": 1,
        "name": "unit-d1-phase23-split",
        "seed": SEED,
        "heldout_fraction": HELDOUT_FRACTION,
        "min_train_rows": 1,
        "min_heldout_rows": 8,
        "min_heldout_rows_per_vendor_or_model": 1,
        "min_empty_set_heldout_rows": 1,
        "required_phase3_argument_classes": list(ARGUMENT_CLASSES),
        "robustness_variants": list(VARIANTS),
    }
    values.update(overrides)
    path.write_text(json.dumps(values, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _write_full_views(
    tmp_path: Path,
    *,
    classes: Sequence[str] = ARGUMENT_CLASSES,
) -> dict[str, Path]:
    phase2_rows: list[dict] = []
    phase3_rows: list[dict] = []
    for class_name in classes:
        row_id = _row_id_for_bucket(class_name, heldout=True)
        phase2, phase3 = _phase_rows(
            class_name,
            row_id=row_id,
            group=f"Vendor/{class_name}",
        )
        phase2_rows.append(phase2)
        phase3_rows.append(phase3)
    empty_id = _row_id_for_bucket("empty-set", heldout=True)
    empty2, empty3 = _empty_phase_rows(row_id=empty_id, group="Vendor/empty-set")
    phase2_rows.append(empty2)
    phase3_rows.append(empty3)
    train2, train3 = _phase_rows(
        "read_only_empty",
        row_id=_row_id_for_bucket("train", heldout=False),
        group="Vendor/train",
    )
    phase2_rows.append(train2)
    phase3_rows.append(train3)

    phase2_path = _write_jsonl(tmp_path / "phase2-full.jsonl", phase2_rows)
    phase3_path = _write_jsonl(tmp_path / "phase3-full.jsonl", phase3_rows)
    return {
        "phase2": phase2_path,
        "phase2_manifest": _write_manifest(
            tmp_path / "phase2-full.manifest.json",
            artifact=phase2_path,
            view="phase2",
            rows=len(phase2_rows),
        ),
        "phase3": phase3_path,
        "phase3_manifest": _write_manifest(
            tmp_path / "phase3-full.manifest.json",
            artifact=phase3_path,
            view="phase3",
            rows=len(phase3_rows),
        ),
    }


def test_checked_in_split_spec_declares_required_classes_and_variants() -> None:
    """The canonical split YAML owns sampling floors, argument classes, and variants."""
    script = _load_script()
    spec = script._load_spec(ROOT / "configs" / "data" / "d1_phase23_split.yaml")

    assert spec["min_empty_set_heldout_rows"] > 0
    assert set(spec["required_phase3_argument_classes"]) == set(ARGUMENT_CLASSES)
    assert tuple(spec["robustness_variants"]) == VARIANTS


def test_split_release_is_deterministic_disjoint_aligned_and_paired(
    tmp_path: Path,
) -> None:
    """A valid split publishes aligned train/heldout views and paired variants."""
    script = _load_script()
    paths = _write_full_views(tmp_path)
    spec_path = _write_spec(tmp_path / "split.json")

    first = script.build_split_release(
        spec_path=spec_path,
        phase2_path=paths["phase2"],
        phase2_manifest_path=paths["phase2_manifest"],
        phase3_path=paths["phase3"],
        phase3_manifest_path=paths["phase3_manifest"],
        output_dir=tmp_path / "release-a",
    )
    second = script.build_split_release(
        spec_path=spec_path,
        phase2_path=paths["phase2"],
        phase2_manifest_path=paths["phase2_manifest"],
        phase3_path=paths["phase3"],
        phase3_manifest_path=paths["phase3_manifest"],
        output_dir=tmp_path / "release-b",
    )

    assert first["status"] == "released"
    assert first["release_manifest_sha256"] == second["release_manifest_sha256"]
    release_a = json.loads(
        (tmp_path / "release-a" / "release_manifest.json").read_text(
            encoding="utf-8",
        ),
    )
    release_b = json.loads(
        (tmp_path / "release-b" / "release_manifest.json").read_text(
            encoding="utf-8",
        ),
    )
    assert release_a == release_b
    assert release_a["immutable"] is True
    assert release_a["complete"] is True
    assert release_a["disjoint"] is True
    assert release_a["phase2_full_manifest_sha256"] == _sha256(paths["phase2_manifest"])
    assert release_a["phase3_full_manifest_sha256"] == _sha256(paths["phase3_manifest"])

    phase2_train = json.loads(
        (tmp_path / "release-a" / "phase2_train.jsonl.manifest.json").read_text(
            encoding="utf-8",
        ),
    )
    phase2_heldout = json.loads(
        (tmp_path / "release-a" / "phase2_heldout.jsonl.manifest.json").read_text(
            encoding="utf-8",
        ),
    )
    assert set(phase2_train["source_row_ids"]).isdisjoint(
        phase2_heldout["source_row_ids"],
    )
    assert (
        phase2_train["source_full_manifest_sha256"]
        == release_a["phase2_full_manifest_sha256"]
    )
    assert (
        phase2_heldout["source_full_manifest_sha256"]
        == release_a["phase2_full_manifest_sha256"]
    )
    assert phase2_heldout["artifact_sha256"] == _sha256(
        tmp_path / "release-a" / "phase2_heldout.jsonl",
    )

    pairs = _read_jsonl(tmp_path / "release-a" / "heldout_view_pairs.jsonl")
    assert len(pairs) == len(phase2_heldout["source_row_ids"]) * len(VARIANTS)
    variants_by_row: dict[str, set[str]] = {}
    inference_case_ids: set[str] = set()
    for pair in pairs:
        phase2 = pair["phase2"]
        phase3 = pair["phase3"]
        phase2_meta = phase2["metadata"]
        phase3_meta = phase3["metadata"]
        assert phase2_meta["row_id"] == phase3_meta["row_id"]
        assert phase2_meta["robustness_variant"] == phase3_meta["robustness_variant"]
        assert phase2_meta["inference_case_id"] == phase3_meta["inference_case_id"]
        assert set(phase2["y_true"]["rest_api_list"]) == set(phase3["x"]["rest_api_list"])
        variants_by_row.setdefault(phase2_meta["row_id"], set()).add(
            phase2_meta["robustness_variant"],
        )
        inference_case_ids.add(phase2_meta["inference_case_id"])

    assert all(variants == set(VARIANTS) for variants in variants_by_row.values())
    assert len(inference_case_ids) == len(pairs)


@pytest.mark.parametrize(
    "manifest_update",
    [
        {"rows": 99},
        {"artifact_sha256": "sha256:" + "0" * 64},
    ],
)
def test_split_release_rejects_input_manifest_row_or_sha_mismatch(
    tmp_path: Path,
    manifest_update: dict[str, object],
) -> None:
    """Full-view manifests must match the exact source JSONL bytes and row count."""
    script = _load_script()
    paths = _write_full_views(tmp_path)
    spec_path = _write_spec(tmp_path / "split.json")
    manifest = json.loads(paths["phase2_manifest"].read_text(encoding="utf-8"))
    manifest.update(manifest_update)
    paths["phase2_manifest"].write_text(
        json.dumps(manifest, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="manifest data evidence mismatch"):
        script.build_split_release(
            spec_path=spec_path,
            phase2_path=paths["phase2"],
            phase2_manifest_path=paths["phase2_manifest"],
            phase3_path=paths["phase3"],
            phase3_manifest_path=paths["phase3_manifest"],
            output_dir=tmp_path / "release",
        )


@pytest.mark.parametrize(
    ("spec_overrides", "message"),
    [
        ({"min_heldout_rows_per_vendor_or_model": 2}, "vendor/model"),
        ({"min_empty_set_heldout_rows": 2}, "empty-set"),
    ],
)
def test_split_release_enforces_vendor_model_and_empty_set_heldout_floors(
    tmp_path: Path,
    spec_overrides: dict[str, object],
    message: str,
) -> None:
    """Held-out coverage requires configured vendor/model and empty-set floors."""
    script = _load_script()
    paths = _write_full_views(tmp_path)
    spec_path = _write_spec(tmp_path / "split.json", **spec_overrides)

    with pytest.raises(ValueError, match=message):
        script.build_split_release(
            spec_path=spec_path,
            phase2_path=paths["phase2"],
            phase2_manifest_path=paths["phase2_manifest"],
            phase3_path=paths["phase3"],
            phase3_manifest_path=paths["phase3_manifest"],
            output_dir=tmp_path / "release",
        )


def test_split_release_requires_all_phase3_argument_classes(tmp_path: Path) -> None:
    """The held-out split must cover every required Phase 3 argument class."""
    script = _load_script()
    paths = _write_full_views(
        tmp_path,
        classes=[name for name in ARGUMENT_CLASSES if name != "delete_no_body"],
    )
    spec_path = _write_spec(tmp_path / "split.json", min_heldout_rows=7)

    with pytest.raises(ValueError, match="Phase 3 argument classes"):
        script.build_split_release(
            spec_path=spec_path,
            phase2_path=paths["phase2"],
            phase2_manifest_path=paths["phase2_manifest"],
            phase3_path=paths["phase3"],
            phase3_manifest_path=paths["phase3_manifest"],
            output_dir=tmp_path / "release",
        )


def test_split_release_atomic_failure_leaves_no_canonical_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A publish failure cleans the pending directory and never creates canonical output."""
    script = _load_script()
    paths = _write_full_views(tmp_path)
    spec_path = _write_spec(tmp_path / "split.json")
    output_dir = tmp_path / "release"

    def fail_write(*_args, **_kwargs) -> None:
        raise OSError("simulated write failure")

    monkeypatch.setattr(script, "_write_jsonl", fail_write)

    with pytest.raises(OSError, match="simulated write failure"):
        script.build_split_release(
            spec_path=spec_path,
            phase2_path=paths["phase2"],
            phase2_manifest_path=paths["phase2_manifest"],
            phase3_path=paths["phase3"],
            phase3_manifest_path=paths["phase3_manifest"],
            output_dir=output_dir,
        )

    assert not output_dir.exists()
    assert not output_dir.with_name(f"{output_dir.name}.pending").exists()


def test_split_release_lock_preserves_pending_and_canonical_absent(
    tmp_path: Path,
) -> None:
    """A pre-existing release lock fails before touching another pending dir."""
    script = _load_script()
    paths = _write_full_views(tmp_path)
    spec_path = _write_spec(tmp_path / "split.json")
    output_dir = tmp_path / "release"
    release_lock = output_dir.with_name(f"{output_dir.name}.release.lock")
    pending_dir = output_dir.with_name(f"{output_dir.name}.pending")
    pending_dir.mkdir()
    pending_sentinel = pending_dir / "publisher-owned.txt"
    pending_sentinel.write_text("pending owned by another publisher\n", encoding="utf-8")
    release_lock.write_text("other publisher lock\n", encoding="utf-8")

    with pytest.raises(ValueError, match="release is locked"):
        script.build_split_release(
            spec_path=spec_path,
            phase2_path=paths["phase2"],
            phase2_manifest_path=paths["phase2_manifest"],
            phase3_path=paths["phase3"],
            phase3_manifest_path=paths["phase3_manifest"],
            output_dir=output_dir,
        )

    assert not output_dir.exists()
    assert release_lock.read_text(encoding="utf-8") == "other publisher lock\n"
    assert pending_dir.is_dir()
    assert (
        pending_sentinel.read_text(encoding="utf-8")
        == "pending owned by another publisher\n"
    )
