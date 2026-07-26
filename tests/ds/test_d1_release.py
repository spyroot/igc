"""Unit tests for atomic D1 JSONL release."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import igc.ds.d1_release as d1_release
from igc.ds.d1_release import D1ReleaseError, release_d1_jsonl
from igc.ds.rest_goal_contract import (
    RedfishContext,
    build_d1_rest_api_list_row,
    d1_row_id,
)


DISTRACTORS = (
    "/redfish/v1/TaskService",
    "/redfish/v1/EventService",
    "/redfish/v1/AccountService",
    "/redfish/v1/UpdateService",
)


def _contexts(targets: list[str]) -> list[RedfishContext]:
    apis = list(dict.fromkeys([*targets, *DISTRACTORS]))
    return [
        RedfishContext(
            rest_api=api,
            allowed_methods=["GET"],
            json={"@odata.id": api, "Name": api.rsplit("/", 1)[-1]},
        )
        for api in apis
    ]


def _validation(targets: list[str], *, accepted: bool = True) -> dict[str, object]:
    return {
        "valid_json": True,
        "accepted": accepted,
        "natural": True,
        "nonsense": False,
        "ambiguous": False,
        "duplicate_intent": False,
        "extra_intents": False,
        "method_semantics_valid": True,
        "covered_api_set": list(targets),
    }


def _d1_row(width: int, *, accepted: bool = True) -> dict:
    targets = [f"/redfish/v1/Systems/1/Resource{index}" for index in range(1, width + 1)]
    row = build_d1_rest_api_list_row(
        text=f"Inspect {width} selected Redfish resources.",
        contexts=_contexts(targets),
        rest_api_list=targets,
        validation=_validation(targets, accepted=accepted),
    )
    row["metadata"] = {
        "sample_width_k": width,
        "row_id": d1_row_id(row),
        "prompt_spec_version": "phase2-labelled-requests-v1",
        "vendor": ["unit"],
        "source_corpus": ["unit-fixture"],
    }
    return row


def test_release_d1_jsonl_publishes_validated_jsonl_and_manifest_atomically(
    tmp_path: Path,
) -> None:
    """A valid release writes one canonical directory with data and manifest."""
    release_dir = tmp_path / "d1-release"

    manifest = release_d1_jsonl(
        output_dir=release_dir,
        rows=[_d1_row(1), _d1_row(2), _d1_row(3)],
        expected_widths=[1, 2, 3],
        release_metadata={"build_id": "unit-test"},
    )

    rows = [
        json.loads(line)
        for line in (release_dir / "data.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    written_manifest = json.loads((release_dir / "manifest.json").read_text(encoding="utf-8"))
    assert len(rows) == 3
    assert manifest["schema_version"] == "d1_release.v1"
    assert manifest["dataset"] == "D1"
    assert manifest["immutable"] is True
    assert manifest["complete"] is True
    assert manifest["artifact_sha256"].startswith("sha256:")
    assert manifest["sample_width_counts"] == {"1": 1, "2": 1, "3": 1}
    assert written_manifest == manifest
    assert not Path(f"{release_dir}.pending").exists()
    assert not Path(f"{release_dir}.release.lock").exists()


def test_d1_release_removes_release_lock_after_success(tmp_path: Path) -> None:
    """A successful immutable release must not leave the exclusive lock behind."""
    release_dir = tmp_path / "d1-release"
    release_lock = Path(f"{release_dir}.release.lock")

    release_d1_jsonl(
        output_dir=release_dir,
        rows=[_d1_row(1)],
        expected_widths=[1],
        release_metadata={"build_id": "unit-test"},
    )

    assert (release_dir / "data.jsonl").exists()
    assert (release_dir / "manifest.json").exists()
    assert not release_lock.exists()
    assert not Path(f"{release_dir}.pending").exists()


def test_d1_release_can_require_width_zero_separately_from_positive_balance(
    tmp_path: Path,
) -> None:
    """Width 0 is required explicitly while positive balance remains k=1/2/3."""
    release_dir = tmp_path / "d1-with-empty-set"

    manifest = release_d1_jsonl(
        output_dir=release_dir,
        rows=[_d1_row(0), _d1_row(0), _d1_row(1), _d1_row(2), _d1_row(3)],
        expected_widths=[0, 1, 2, 3],
        release_metadata={"build_id": "unit-test"},
    )

    assert manifest["sample_width_counts"] == {"0": 2, "1": 1, "2": 1, "3": 1}
    rows = [
        json.loads(line)
        for line in (release_dir / "data.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    empty_rows = [row for row in rows if row["metadata"]["sample_width_k"] == 0]
    assert len(empty_rows) == 2
    assert all(row["y_true"]["rest_api_list"] == [] for row in empty_rows)


def test_d1_release_missing_width_zero_fails_when_width_zero_is_required(
    tmp_path: Path,
) -> None:
    """Requesting width 0 is a real release requirement, not implied by positives."""
    with pytest.raises(D1ReleaseError, match="missing required sample widths"):
        release_d1_jsonl(
            output_dir=tmp_path / "d1-release",
            rows=[_d1_row(1)],
            expected_widths=[0, 1],
            release_metadata={},
        )


def test_d1_release_still_balances_positive_widths_when_width_zero_is_present(
    tmp_path: Path,
) -> None:
    """Extra empty-set rows do not mask k=1/2/3 imbalance."""
    with pytest.raises(D1ReleaseError, match="sample width balance"):
        release_d1_jsonl(
            output_dir=tmp_path / "d1-release",
            rows=[
                _d1_row(0),
                _d1_row(0),
                _d1_row(1),
                _d1_row(1),
                _d1_row(1),
                _d1_row(2),
                _d1_row(3),
            ],
            expected_widths=[0, 1, 2, 3],
            release_metadata={},
        )


@pytest.mark.parametrize(
    "reserved",
    ["schema_version", "dataset", "artifact_sha256", "immutable", "complete"],
)
def test_d1_release_rejects_metadata_overriding_canonical_fields(
    tmp_path: Path,
    reserved: str,
) -> None:
    """Operator metadata cannot override canonical release manifest fields."""
    with pytest.raises(D1ReleaseError, match="release_metadata cannot override"):
        release_d1_jsonl(
            output_dir=tmp_path / "d1-release",
            rows=[_d1_row(1)],
            expected_widths=[1],
            release_metadata={reserved: "override"},
        )


def test_d1_release_rejects_unstable_metadata_row_id(tmp_path: Path) -> None:
    """Released rows require metadata.row_id to equal the canonical D1 row hash."""
    row = _d1_row(1)
    row["metadata"]["row_id"] = "sha256:" + "0" * 64

    with pytest.raises(D1ReleaseError, match="metadata.row_id"):
        release_d1_jsonl(
            output_dir=tmp_path / "d1-release",
            rows=[row],
            expected_widths=[1],
            release_metadata={},
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("valid_json", "yes"),
        ("accepted", 1),
        ("natural", None),
        ("nonsense", "false"),
        ("ambiguous", []),
        ("duplicate_intent", {}),
        ("extra_intents", 0),
        ("method_semantics_valid", "true"),
    ],
)
def test_d1_release_rejects_non_bool_judge_predicates(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    """Judge predicate evidence is strict bool-only, never truthy/falsy coercion."""
    row = _d1_row(1)
    row["validation"][field] = value

    with pytest.raises(D1ReleaseError, match="judge predicate fields must be booleans"):
        release_d1_jsonl(
            output_dir=tmp_path / "d1-release",
            rows=[row],
            expected_widths=[1],
            release_metadata={},
        )


def test_d1_release_rejects_sample_width_target_cardinality_mismatch(
    tmp_path: Path,
) -> None:
    """sample_width_k must equal the y_true rest_api_list cardinality exactly."""
    row = _d1_row(2)
    row["metadata"]["sample_width_k"] = 1

    with pytest.raises(D1ReleaseError, match="sample_width_k must equal target cardinality"):
        release_d1_jsonl(
            output_dir=tmp_path / "d1-release",
            rows=[row],
            expected_widths=[1],
            release_metadata={},
        )


def test_d1_release_removes_release_lock_after_validation_failure(
    tmp_path: Path,
) -> None:
    """Ordinary validation failures clean the lock and do not publish artifacts."""
    release_dir = tmp_path / "d1-release"
    release_lock = Path(f"{release_dir}.release.lock")

    with pytest.raises(D1ReleaseError, match="judge acceptance predicate failed"):
        release_d1_jsonl(
            output_dir=release_dir,
            rows=[_d1_row(1, accepted=False)],
            expected_widths=[1],
            release_metadata={"build_id": "bad-release"},
        )

    assert not release_lock.exists()
    assert not release_dir.exists()
    assert not Path(f"{release_dir}.pending").exists()


def test_d1_release_rejects_existing_canonical_dir_without_replacement(
    tmp_path: Path,
) -> None:
    """An immutable canonical release directory fails closed before publication."""
    release_dir = tmp_path / "d1-release"
    canonical_bytes = b'{"dataset":"D1","row":"canonical"}\n'
    manifest_bytes = b'{"schema_version":"d1_release.v1","canonical":true}\n'
    release_dir.mkdir()
    (release_dir / "data.jsonl").write_bytes(canonical_bytes)
    (release_dir / "manifest.json").write_bytes(manifest_bytes)

    with pytest.raises(
        D1ReleaseError,
        match="immutable canonical D1 release already exists",
    ):
        release_d1_jsonl(
            output_dir=release_dir,
            rows=[_d1_row(1)],
            expected_widths=[1],
            release_metadata={"build_id": "replacement-attempt"},
        )

    assert (release_dir / "data.jsonl").read_bytes() == canonical_bytes
    assert (release_dir / "manifest.json").read_bytes() == manifest_bytes
    assert not Path(f"{release_dir}.pending").exists()
    assert not Path(f"{release_dir}.release.lock").exists()


def test_d1_release_rejects_existing_release_lock_without_cleanup_or_publish(
    tmp_path: Path,
) -> None:
    """A pre-existing exclusive lock is preserved and no release artifacts are created."""
    release_dir = tmp_path / "d1-release"
    release_lock = Path(f"{release_dir}.release.lock")
    lock_bytes = b"publisher pid 12345\n"
    release_lock.write_bytes(lock_bytes)

    class RowsThatFailIfConsumed:
        def __iter__(self):
            raise AssertionError("locked release must not consume rows")

    with pytest.raises(D1ReleaseError, match="D1 release is locked"):
        release_d1_jsonl(
            output_dir=release_dir,
            rows=RowsThatFailIfConsumed(),
            expected_widths=[1],
            release_metadata={"build_id": "blocked-by-lock"},
        )

    assert release_lock.read_bytes() == lock_bytes
    assert not release_dir.exists()
    assert not Path(f"{release_dir}.pending").exists()


def test_d1_release_rejects_stale_pending_release_directory_without_cleanup(
    tmp_path: Path,
) -> None:
    """A stale pending release directory survives for operator inspection."""
    release_dir = tmp_path / "d1-release"
    pending_dir = Path(f"{release_dir}.pending")
    pending_dir.mkdir()
    data_pending = pending_dir / "data.jsonl"
    manifest_pending = pending_dir / "manifest.json"
    data_pending_bytes = b"partial data release\n"
    manifest_pending_bytes = b'{"partial": true}\n'
    data_pending.write_bytes(data_pending_bytes)
    manifest_pending.write_bytes(manifest_pending_bytes)

    with pytest.raises(D1ReleaseError, match="stale D1 pending release"):
        release_d1_jsonl(
            output_dir=release_dir,
            rows=[_d1_row(1)],
            expected_widths=[1],
            release_metadata={"build_id": "blocked-by-paired-stale-pending"},
        )

    assert data_pending.read_bytes() == data_pending_bytes
    assert manifest_pending.read_bytes() == manifest_pending_bytes
    assert not release_dir.exists()
    assert not Path(f"{release_dir}.release.lock").exists()


def test_d1_release_rejects_existing_canonical_dir_before_replace_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The immutable guard must run before any publication replace call."""
    release_dir = tmp_path / "d1-release"
    canonical_bytes = b"old canonical data\n"
    manifest_bytes = b"old canonical manifest\n"
    release_dir.mkdir()
    (release_dir / "data.jsonl").write_bytes(canonical_bytes)
    (release_dir / "manifest.json").write_bytes(manifest_bytes)
    replace_calls: list[tuple[object, object]] = []

    class RowsThatFailOnWrite:
        def __iter__(self):
            raise AssertionError("immutable release must not write pending rows")

    def forbidden_replace(src, dst) -> None:
        replace_calls.append((src, dst))
        raise AssertionError("immutable release must not call os.replace")

    monkeypatch.setattr(d1_release.os, "replace", forbidden_replace)

    with pytest.raises(
        D1ReleaseError,
        match="immutable canonical D1 release already exists",
    ):
        release_d1_jsonl(
            output_dir=release_dir,
            rows=RowsThatFailOnWrite(),
            expected_widths=[1],
            release_metadata={"build_id": "blocked-before-replace"},
        )

    assert replace_calls == []
    assert (release_dir / "data.jsonl").read_bytes() == canonical_bytes
    assert (release_dir / "manifest.json").read_bytes() == manifest_bytes
    assert not Path(f"{release_dir}.pending").exists()
    assert not Path(f"{release_dir}.release.lock").exists()


def test_d1_release_only_replaces_pending_directory_and_cleans_failed_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Publication is a single pending-dir -> canonical-dir replace."""
    release_dir = tmp_path / "d1-release"
    pending_dir = Path(f"{release_dir}.pending")
    replace_calls: list[tuple[Path, Path]] = []

    def fail_replace(src, dst) -> None:
        replace_calls.append((Path(src), Path(dst)))
        raise OSError("simulated directory rename failure")

    monkeypatch.setattr(d1_release.os, "replace", fail_replace)

    with pytest.raises(OSError, match="simulated directory rename failure"):
        release_d1_jsonl(
            output_dir=release_dir,
            rows=[_d1_row(1)],
            expected_widths=[1],
            release_metadata={"build_id": "replace-failure"},
        )

    assert replace_calls == [(pending_dir, release_dir)]
    assert not release_dir.exists()
    assert not pending_dir.exists()
    assert not Path(f"{release_dir}.release.lock").exists()
