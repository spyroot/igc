"""Focused tests for atomic D1 Phase 2/3 grounded-view materialization."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from igc.ds.rest_goal_contract import (
    RedfishContext,
    build_d1_rest_api_list_row,
    d1_row_id,
)


SCRIPT = Path("scripts/build_phase3_grounded_dataset.py")
SYSTEM = "/redfish/v1/Systems/1"
BIOS = f"{SYSTEM}/Bios/Settings"
DISTRACTORS = (
    "/redfish/v1/TaskService",
    "/redfish/v1/EventService",
    "/redfish/v1/AccountService",
    "/redfish/v1/UpdateService",
)


def _load_script() -> ModuleType:
    """Load the script module for direct function testing."""
    spec = importlib.util.spec_from_file_location("build_phase3_grounded_dataset", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    """Return a canonical sha256 digest for a file."""
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    """Write deterministic JSONL rows."""
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _write_manifest(
    path: Path,
    *,
    artifact_path: Path,
    rows: int,
    immutable: bool = True,
    complete: bool = True,
    artifact_sha: str | None = None,
) -> Path:
    """Write an immutable input manifest fixture."""
    manifest = {
        "schema_version": "test_input_manifest.v1",
        "dataset": "D1",
        "immutable": immutable,
        "complete": complete,
        "rows": rows,
        "artifact_sha256": artifact_sha or _sha256(artifact_path),
    }
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _context(
    rest_api: str,
    *,
    methods: tuple[str, ...] = ("GET",),
    operation_names: tuple[str, ...] = ("get_resource",),
    argument_schema: dict | None = None,
) -> RedfishContext:
    """Build a Redfish context fixture."""
    return RedfishContext(
        rest_api=rest_api,
        allowed_methods=methods,
        operation_names=operation_names,
        argument_schema={} if argument_schema is None else argument_schema,
        json={"@odata.id": rest_api, "Name": rest_api.rsplit("/", 1)[-1]},
    )


def _contexts() -> tuple[RedfishContext, ...]:
    """Selected BIOS context plus the hidden distractors D1 requires."""
    return (
        _context(
            BIOS,
            methods=("GET", "PATCH"),
            operation_names=("set_bios_attributes",),
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
        ),
        *(_context(api) for api in DISTRACTORS),
    )


def _validation() -> dict[str, object]:
    """Accepted strict Phase 2 judge evidence."""
    return {
        "valid_json": True,
        "accepted": True,
        "natural": True,
        "nonsense": False,
        "ambiguous": False,
        "duplicate_intent": False,
        "extra_intents": False,
        "method_semantics_valid": True,
        "covered_api_set": [BIOS],
    }


def _d1_row() -> dict:
    """Return one released D1 Phase 2 row with canonical metadata.row_id."""
    row = build_d1_rest_api_list_row(
        text="set BIOS boot mode to Uefi",
        contexts=_contexts(),
        rest_api_list=(BIOS,),
        validation=_validation(),
    )
    row["metadata"] = {
        "sample_width_k": 1,
        "row_id": d1_row_id(row),
        "prompt_spec_version": "phase2-labelled-requests-v1",
        "vendor": ["unit"],
        "source_corpus": ["unit-fixture"],
    }
    return row


def _label_row(row_id: str, *, api: str = BIOS) -> dict:
    """Return one explicit call-label row."""
    return {
        "row_id": row_id,
        "method_by_api": {api: "PATCH"},
        "operation_name_by_api": {api: "set_bios_attributes"},
        "arguments_by_api": {api: {"Attributes": {"BootMode": "Uefi"}}},
        "argument_value_grounding_by_api": {
            api: {"grounded": True, "sources": ["operator_text", "argument_schema"]},
        },
    }


def _write_inputs(tmp_path: Path, *, label_row: dict | None = None) -> dict[str, Path]:
    """Write D1 and call-label JSONL inputs plus immutable manifests."""
    d1 = _d1_row()
    labels = [label_row or _label_row(d1["metadata"]["row_id"])]
    d1_path = _write_jsonl(tmp_path / "d1.jsonl", [d1])
    labels_path = _write_jsonl(tmp_path / "labels.jsonl", labels)
    return {
        "d1": d1_path,
        "d1_manifest": _write_manifest(
            tmp_path / "d1.manifest.json",
            artifact_path=d1_path,
            rows=1,
        ),
        "labels": labels_path,
        "labels_manifest": _write_manifest(
            tmp_path / "labels.manifest.json",
            artifact_path=labels_path,
            rows=len(labels),
        ),
    }


def test_build_phase3_grounded_dataset_publishes_all_views_atomically(
    tmp_path: Path,
) -> None:
    """A valid label set releases master, Phase 2, and Phase 3 views and manifests."""
    script = _load_script()
    inputs = _write_inputs(tmp_path)
    output_dir = tmp_path / "grounded-release"

    result = script.build_grounded_views_release(
        d1_path=inputs["d1"],
        d1_manifest_path=inputs["d1_manifest"],
        labels_path=inputs["labels"],
        labels_manifest_path=inputs["labels_manifest"],
        output_dir=output_dir,
    )

    assert result["status"] == "released"
    assert result["rows"] == 1
    assert output_dir.is_dir()
    assert not Path(f"{output_dir}.pending").exists()
    release_manifest = json.loads(
        (output_dir / "release_manifest.json").read_text(encoding="utf-8")
    )
    assert release_manifest["immutable"] is True
    assert release_manifest["complete"] is True
    assert set(release_manifest["files"]) == {"master", "phase2", "phase3"}
    for view in ("master", "phase2", "phase3"):
        data_path = output_dir / f"{view}.jsonl"
        manifest_path = Path(f"{data_path}.manifest.json")
        view_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert data_path.exists()
        assert view_manifest["view"] == view
        assert view_manifest["immutable"] is True
        assert view_manifest["complete"] is True
        assert view_manifest["rows"] == 1
        assert view_manifest["source_d1_manifest_sha256"] == _sha256(
            inputs["d1_manifest"]
        )
        assert view_manifest["call_labels_manifest_sha256"] == _sha256(
            inputs["labels_manifest"]
        )
    phase2 = json.loads((output_dir / "phase2.jsonl").read_text(encoding="utf-8"))
    phase3 = json.loads((output_dir / "phase3.jsonl").read_text(encoding="utf-8"))
    assert set(phase2["y_true"]["rest_api_list"]) == {
        call["rest_api"] for call in phase3["y_true"]["calls"]
    }


def test_build_phase3_grounded_dataset_rejects_label_coverage_mismatch(
    tmp_path: Path,
) -> None:
    """Call-label row IDs and API labels must exactly cover D1 input rows."""
    script = _load_script()
    inputs = _write_inputs(
        tmp_path,
        label_row=_label_row("sha256:" + "9" * 64),
    )
    output_dir = tmp_path / "grounded-release"

    with pytest.raises(ValueError, match="coverage must exactly match"):
        script.build_grounded_views_release(
            d1_path=inputs["d1"],
            d1_manifest_path=inputs["d1_manifest"],
            labels_path=inputs["labels"],
            labels_manifest_path=inputs["labels_manifest"],
            output_dir=output_dir,
        )

    assert not output_dir.exists()
    assert not Path(f"{output_dir}.pending").exists()


def test_build_phase3_grounded_dataset_rejects_label_api_set_drift(
    tmp_path: Path,
) -> None:
    """A call-label row cannot name a different API set than the D1 row."""
    script = _load_script()
    d1 = _d1_row()
    inputs = _write_inputs(
        tmp_path,
        label_row=_label_row(d1["metadata"]["row_id"], api=SYSTEM),
    )

    with pytest.raises(ValueError, match="API set differs"):
        script.build_grounded_views_release(
            d1_path=inputs["d1"],
            d1_manifest_path=inputs["d1_manifest"],
            labels_path=inputs["labels"],
            labels_manifest_path=inputs["labels_manifest"],
            output_dir=tmp_path / "grounded-release",
        )


def test_build_phase3_grounded_dataset_verifies_input_manifests_before_publish(
    tmp_path: Path,
) -> None:
    """Input manifests must be immutable, complete, and match exact JSONL bytes."""
    script = _load_script()
    inputs = _write_inputs(tmp_path)
    _write_manifest(
        inputs["d1_manifest"],
        artifact_path=inputs["d1"],
        rows=1,
        immutable=False,
    )
    output_dir = tmp_path / "grounded-release"

    with pytest.raises(ValueError, match="not immutable and complete"):
        script.build_grounded_views_release(
            d1_path=inputs["d1"],
            d1_manifest_path=inputs["d1_manifest"],
            labels_path=inputs["labels"],
            labels_manifest_path=inputs["labels_manifest"],
            output_dir=output_dir,
        )

    assert not output_dir.exists()
    assert not Path(f"{output_dir}.pending").exists()


def test_build_phase3_grounded_dataset_failure_leaves_no_canonical_output(
    tmp_path: Path,
) -> None:
    """Late validation failure removes the pending directory and never publishes output."""
    script = _load_script()
    d1 = _d1_row()
    bad_label = _label_row(d1["metadata"]["row_id"])
    bad_label["argument_value_grounding_by_api"][BIOS]["sources"] = ["current_json"]
    inputs = _write_inputs(tmp_path, label_row=bad_label)
    output_dir = tmp_path / "grounded-release"

    with pytest.raises(ValueError, match="current_json"):
        script.build_grounded_views_release(
            d1_path=inputs["d1"],
            d1_manifest_path=inputs["d1_manifest"],
            labels_path=inputs["labels"],
            labels_manifest_path=inputs["labels_manifest"],
            output_dir=output_dir,
        )

    assert not output_dir.exists()
    assert not Path(f"{output_dir}.pending").exists()


@pytest.mark.parametrize(
    ("sources", "message"),
    [
        (["operator_text", "unknown_fixture"], "unsupported sources"),
        (["operator_text"], "operator_text plus argument_schema"),
        (["argument_schema"], "operator_text plus argument_schema"),
    ],
)
def test_build_phase3_grounded_dataset_rejects_incomplete_mutation_grounding(
    tmp_path: Path,
    sources: list[str],
    message: str,
) -> None:
    """Mutation labels require operator text plus schema/action grounding evidence."""
    script = _load_script()
    d1 = _d1_row()
    bad_label = _label_row(d1["metadata"]["row_id"])
    bad_label["argument_value_grounding_by_api"][BIOS]["sources"] = sources
    inputs = _write_inputs(tmp_path, label_row=bad_label)
    output_dir = tmp_path / "grounded-release"

    with pytest.raises(ValueError, match=message):
        script.build_grounded_views_release(
            d1_path=inputs["d1"],
            d1_manifest_path=inputs["d1_manifest"],
            labels_path=inputs["labels"],
            labels_manifest_path=inputs["labels_manifest"],
            output_dir=output_dir,
        )

    assert not output_dir.exists()
    assert not Path(f"{output_dir}.pending").exists()


def test_build_phase3_grounded_dataset_release_lock_preserves_pending_and_canonical_absent(
    tmp_path: Path,
) -> None:
    """A pre-existing release lock fails closed before touching another pending dir."""
    script = _load_script()
    inputs = _write_inputs(tmp_path)
    output_dir = tmp_path / "grounded-release"
    release_lock = Path(f"{output_dir}.release.lock")
    pending_dir = Path(f"{output_dir}.pending")
    pending_dir.mkdir()
    pending_sentinel = pending_dir / "publisher-owned.txt"
    pending_sentinel.write_text("pending owned by another publisher\n", encoding="utf-8")
    release_lock.write_text("other publisher lock\n", encoding="utf-8")

    with pytest.raises(ValueError, match="release is locked"):
        script.build_grounded_views_release(
            d1_path=inputs["d1"],
            d1_manifest_path=inputs["d1_manifest"],
            labels_path=inputs["labels"],
            labels_manifest_path=inputs["labels_manifest"],
            output_dir=output_dir,
        )

    assert not output_dir.exists()
    assert release_lock.read_text(encoding="utf-8") == "other publisher lock\n"
    assert pending_dir.is_dir()
    assert (
        pending_sentinel.read_text(encoding="utf-8")
        == "pending owned by another publisher\n"
    )
