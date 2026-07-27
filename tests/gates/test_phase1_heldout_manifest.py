"""Offline tests for ``scripts/gates/phase1_heldout_manifest.py``."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import yaml

from igc.ds.phase1_render import (
    PHASE1_DATASET,
    PHASE1_TASK,
    build_phase1_row,
)


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "gates"
    / "phase1_heldout_manifest.py"
)
PHASE1_CONTRACT = (
    Path(__file__).resolve().parents[2] / "configs" / "contracts" / "phase1.yaml"
)
SHA_A = "sha256:" + "a" * 64
SHA_B = "sha256:" + "b" * 64
SHA_C = "sha256:" + "c" * 64


def _load_script():
    """Import the gate script as a module without invoking the CLI."""
    spec = importlib.util.spec_from_file_location("phase1_heldout_manifest_gate", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _row_id(source: str, rest_api: str) -> str:
    digest = hashlib.sha256(f"{source}\0{rest_api}".encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _row(source: str, rest_api: str) -> dict:
    return build_phase1_row(
        rest_api=rest_api,
        allowed_methods=["GET"],
        input_json={"@odata.id": rest_api, "Name": rest_api.rsplit("/", 1)[-1]},
        target_json={"@odata.id": rest_api, "Name": rest_api.rsplit("/", 1)[-1]},
        metadata={
            "row_id": _row_id(source, rest_api),
            "source_corpus": source,
            "trust_level": "REAL",
            "vendor": source.removeprefix("real_"),
        },
    )


def _full_manifest(script, **overrides) -> dict:
    payload = {
        "heldout_row_ids": [
            script._record_id("real_dell", "/redfish/v1/Systems/1"),
            script._record_id("real_hpe", "/redfish/v1/Systems/2"),
        ],
        "eval_count": 2,
        "sources": ["dmtf_replay", "real_dell", "real_hpe"],
        "by_source": {"dmtf_replay": 7, "real_dell": 4, "real_hpe": 5},
        "required_heldout_sources": ["real_dell", "real_hpe"],
        "heldout_by_source": {"real_dell": 1, "real_hpe": 1},
        "source_registry_sha": SHA_A,
        "source_manifest_shas": {"real_hpe": SHA_C, "real_dell": SHA_B},
    }
    payload.update(overrides)
    return payload


def test_phase1_contract_yaml_matches_render_constants_and_invariants() -> None:
    """The machine-readable Phase 1 contract must match the renderer constants."""
    spec = yaml.safe_load(PHASE1_CONTRACT.read_text(encoding="utf-8"))
    row = _row("real_dell", "/redfish/v1/Systems/1")

    assert spec["authority"] == {
        "dataset": PHASE1_DATASET,
        "phase": 1,
        "task": PHASE1_TASK,
    }
    assert set(spec["row"]["required_top_level_fields"]) == {
        "phase",
        "dataset",
        "task",
        "x",
        "y_true",
    }
    assert spec["row"]["optional_top_level_fields"] == ["metadata"]
    assert spec["row"]["forbidden_top_level_fields"] == ["y_pred"]
    assert row["phase"] == spec["authority"]["phase"]
    assert row["dataset"] == spec["authority"]["dataset"]
    assert row["task"] == spec["authority"]["task"]
    assert set(row["x"]) == set(spec["row"]["input"])
    assert set(row["y_true"]) == set(spec["row"]["target"]) - {"required"}
    assert set(row["metadata"]) == set(spec["metadata"]["fields"])
    assert spec["objective"]["prompt_tokens_ignored"] is True
    assert spec["objective"]["completion_tokens_supervised"] == (
        "training_profile.phase1_structural_loss_profile"
    )
    assert spec["objective"]["canonical_target_immutable"] is True
    structural = spec["objective"]["historical_structural_mask_v1"]
    assert structural["selected_span_hidden_from_input"] is True
    assert spec["objective"]["overflow_policy"] == "reject"


def test_phase1_heldout_manifest_writes_exact_sanitized_manifest(
    tmp_path: Path,
    capsys,
) -> None:
    """The gate preserves exact split membership and emits corpus breakdowns."""
    script = _load_script()
    rows = [
        _row("real_dell", "/redfish/v1/Systems/1"),
        _row("real_hpe", "/redfish/v1/Systems/2"),
    ]
    heldout = _write_jsonl(tmp_path / "heldout.jsonl", rows)
    full_manifest = _write_json(
        tmp_path / "full-manifest.json",
        _full_manifest(script),
    )
    output = tmp_path / "approved-heldout-manifest.json"

    rc = script.main(
        [
            "--full-manifest",
            str(full_manifest),
            "--heldout-jsonl",
            str(heldout),
            "--output-json",
            str(output),
        ]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["status"] == "pass"
    assert payload["schema_version"] == "phase1_heldout_manifest.v1"
    assert payload["immutable"] is True
    assert payload["complete"] is True
    assert payload["artifact_sha256"] == script._sha256(heldout)
    assert payload["approved_heldout_rows"] == 2
    assert payload["required_corpora"] == ["real_dell", "real_hpe"]
    assert "dmtf_replay" not in payload["required_corpora"]
    assert payload["rows_by_corpus"] == {"real_dell": 1, "real_hpe": 1}
    assert payload["full_rows_by_corpus"] == {"real_dell": 4, "real_hpe": 5}
    assert payload["row_ids"] == [
        script._record_id("real_dell", "/redfish/v1/Systems/1"),
        script._record_id("real_hpe", "/redfish/v1/Systems/2"),
    ]
    assert payload["full_corpus_manifest_sha256"] == script._sha256(full_manifest)
    assert payload["source_registry_sha256"] == SHA_A
    assert payload["source_manifest_shas"] == {
        "real_dell": SHA_B,
        "real_hpe": SHA_C,
    }


def test_phase1_heldout_manifest_rejects_membership_or_order_mismatch(
    tmp_path: Path,
    capsys,
) -> None:
    """Held-out JSONL row IDs must exactly match the full split manifest order."""
    script = _load_script()
    heldout = _write_jsonl(
        tmp_path / "heldout.jsonl",
        [
            _row("real_dell", "/redfish/v1/Systems/1"),
            _row("real_hpe", "/redfish/v1/Systems/2"),
        ],
    )
    full_manifest = _write_json(
        tmp_path / "full-manifest.json",
        _full_manifest(
            script,
            heldout_row_ids=[
                script._record_id("real_hpe", "/redfish/v1/Systems/2"),
                script._record_id("real_dell", "/redfish/v1/Systems/1"),
            ],
        ),
    )

    rc = script.main(
        [
            "--full-manifest",
            str(full_manifest),
            "--heldout-jsonl",
            str(heldout),
            "--output-json",
            str(tmp_path / "out.json"),
        ]
    )

    assert rc == 2
    assert "row IDs/order disagree" in capsys.readouterr().err


def test_phase1_heldout_manifest_rejects_sha_shaped_but_wrong_metadata_row_id(
    tmp_path: Path,
    capsys,
) -> None:
    """metadata.row_id must equal sha256(source_corpus + NUL + x.rest_api)."""
    script = _load_script()
    row = _row("real_dell", "/redfish/v1/Systems/1")
    row["metadata"]["row_id"] = "sha256:" + "0" * 64
    heldout = _write_jsonl(tmp_path / "heldout.jsonl", [row])
    full_manifest = _write_json(
        tmp_path / "full-manifest.json",
        _full_manifest(
            script,
            heldout_row_ids=[
                script._record_id("real_dell", "/redfish/v1/Systems/1"),
            ],
            eval_count=1,
            sources=["dmtf_replay", "real_dell"],
            by_source={"dmtf_replay": 7, "real_dell": 64},
            required_heldout_sources=["real_dell"],
            heldout_by_source={"real_dell": 1},
            source_manifest_shas={"real_dell": SHA_B},
        ),
    )

    rc = script.main(
        [
            "--full-manifest",
            str(full_manifest),
            "--heldout-jsonl",
            str(heldout),
            "--output-json",
            str(tmp_path / "out.json"),
        ]
    )

    assert rc == 2
    assert "metadata.row_id disagrees" in capsys.readouterr().err


def test_phase1_heldout_manifest_rejects_invalid_registry_lineage(
    tmp_path: Path,
    capsys,
) -> None:
    """The approval manifest must carry the exact source registry SHA."""
    script = _load_script()
    heldout = _write_jsonl(
        tmp_path / "heldout.jsonl",
        [
            _row("real_dell", "/redfish/v1/Systems/1"),
            _row("real_hpe", "/redfish/v1/Systems/2"),
        ],
    )
    full_manifest = _write_json(
        tmp_path / "full-manifest.json",
        _full_manifest(
            script,
            source_registry_sha="sha256:not-a-digest",
            source_manifest_shas={"real_dell": SHA_B, "real_hpe": "missing-prefix"},
        ),
    )

    rc = script.main(
        [
            "--full-manifest",
            str(full_manifest),
            "--heldout-jsonl",
            str(heldout),
            "--output-json",
            str(tmp_path / "out.json"),
        ]
    )

    assert rc == 2
    assert "source_registry_sha" in capsys.readouterr().err


def test_phase1_heldout_manifest_rejects_invalid_source_manifest_lineage(
    tmp_path: Path,
    capsys,
) -> None:
    """The approval manifest must carry exact source manifest SHA evidence."""
    script = _load_script()
    heldout = _write_jsonl(
        tmp_path / "heldout.jsonl",
        [
            _row("real_dell", "/redfish/v1/Systems/1"),
            _row("real_hpe", "/redfish/v1/Systems/2"),
        ],
    )
    full_manifest = _write_json(
        tmp_path / "full-manifest.json",
        _full_manifest(
            script,
            source_manifest_shas={"real_dell": SHA_B, "real_hpe": "missing-prefix"},
        ),
    )

    rc = script.main(
        [
            "--full-manifest",
            str(full_manifest),
            "--heldout-jsonl",
            str(heldout),
            "--output-json",
            str(tmp_path / "out.json"),
        ]
    )

    assert rc == 2
    assert "source_manifest_shas" in capsys.readouterr().err


def test_phase1_heldout_manifest_rejects_missing_required_heldout_sources(
    tmp_path: Path,
    capsys,
) -> None:
    """The held-out gate must use required_heldout_sources, not all manifest sources."""
    script = _load_script()
    heldout = _write_jsonl(
        tmp_path / "heldout.jsonl",
        [_row("real_dell", "/redfish/v1/Systems/1")],
    )
    full_manifest = _write_json(
        tmp_path / "full-manifest.json",
        _full_manifest(
            script,
            heldout_row_ids=[
                script._record_id("real_dell", "/redfish/v1/Systems/1"),
            ],
            eval_count=1,
            sources=["dmtf_replay", "real_dell"],
            by_source={"dmtf_replay": 7, "real_dell": 1},
            heldout_by_source={"real_dell": 1},
            required_heldout_sources=None,
            source_manifest_shas={"real_dell": SHA_B},
        ),
    )

    rc = script.main(
        [
            "--full-manifest",
            str(full_manifest),
            "--heldout-jsonl",
            str(heldout),
            "--output-json",
            str(tmp_path / "out.json"),
        ]
    )

    assert rc == 2
    assert "required_heldout_sources" in capsys.readouterr().err


def test_phase1_heldout_manifest_rejects_existing_canonical_output(
    tmp_path: Path,
    capsys,
) -> None:
    """Immutable publication must fail closed instead of replacing canonical output."""
    script = _load_script()
    heldout = _write_jsonl(
        tmp_path / "heldout.jsonl",
        [
            _row("real_dell", "/redfish/v1/Systems/1"),
            _row("real_hpe", "/redfish/v1/Systems/2"),
        ],
    )
    full_manifest = _write_json(tmp_path / "full-manifest.json", _full_manifest(script))
    output = tmp_path / "approved-heldout-manifest.json"
    output.write_text('{"canonical": true}\n', encoding="utf-8")

    rc = script.main(
        [
            "--full-manifest",
            str(full_manifest),
            "--heldout-jsonl",
            str(heldout),
            "--output-json",
            str(output),
        ]
    )

    assert rc == 2
    assert json.loads(output.read_text(encoding="utf-8")) == {"canonical": True}
    assert "already exists" in capsys.readouterr().err


def test_phase1_heldout_manifest_rejects_stale_pending_output(
    tmp_path: Path,
    capsys,
) -> None:
    """A leftover pending manifest must require manual inspection before publication."""
    script = _load_script()
    heldout = _write_jsonl(
        tmp_path / "heldout.jsonl",
        [
            _row("real_dell", "/redfish/v1/Systems/1"),
            _row("real_hpe", "/redfish/v1/Systems/2"),
        ],
    )
    full_manifest = _write_json(tmp_path / "full-manifest.json", _full_manifest(script))
    output = tmp_path / "approved-heldout-manifest.json"
    pending = Path(f"{output}.pending")
    pending.write_text('{"partial": true}\n', encoding="utf-8")

    rc = script.main(
        [
            "--full-manifest",
            str(full_manifest),
            "--heldout-jsonl",
            str(heldout),
            "--output-json",
            str(output),
        ]
    )

    assert rc == 2
    assert not output.exists()
    assert json.loads(pending.read_text(encoding="utf-8")) == {"partial": True}
    assert "stale held-out manifest pending file" in capsys.readouterr().err
