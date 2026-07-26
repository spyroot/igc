"""Offline tests for ``scripts/phase1_heldout_inference.py`` using mocks only."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "phase1_heldout_inference.py"


def _load_script():
    """Import the script without invoking model loading."""
    spec = importlib.util.spec_from_file_location("phase1_heldout_inference", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _FakeTokenizer:
    """Tokenizer stub with a deterministic completion-token length."""

    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, *_args, **_kwargs):
        return {"input_ids": [1, 2, 3, 4]}


def _row_id(source: str, rest_api: str) -> str:
    digest = hashlib.sha256(f"{source}\0{rest_api}".encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _row(
    row_id: str,
    source_corpus: str,
    *,
    supplied_row_id: str | None = None,
    include_row_id: bool = True,
) -> dict:
    rest_api = f"/redfish/v1/Systems/{row_id}"
    row = {
        "source_corpus": source_corpus,
        "x": {
            "rest_api": rest_api,
            "allowed_methods": ["GET", "HEAD"],
            "json": {"@odata.id": rest_api, "PowerState": "On"},
        },
        "y_true": {"json": {"@odata.id": rest_api, "PowerState": "On", "Id": row_id}},
    }
    if include_row_id:
        row["row_id"] = supplied_row_id or _row_id(source_corpus, rest_api)
    return row


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _install_model_mocks(monkeypatch, script, *, max_new_tokens: int) -> None:
    spec = SimpleNamespace(
        base_model="local/foundation",
        cache_dir=Path("/models/cache"),
        adapter_dir=Path("/models/model_x"),
        require_cuda=False,
        seed=7,
        target_token_margin=1,
        max_new_tokens=max_new_tokens,
    )
    monkeypatch.setattr(script, "load_gate_spec", lambda _path: spec)
    monkeypatch.setattr(script, "set_offline_env", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(script, "preflight_inputs", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(script, "validate_adapter_config", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(script, "_configure_determinism", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(script, "_release_model", lambda: None)
    monkeypatch.setattr(
        script,
        "load_model_and_tokenizer",
        lambda *_args, **_kwargs: (object(), _FakeTokenizer(), "cpu"),
    )

    def fake_generate(_model, _tokenizer, _device, rows, *, max_new_tokens):
        return [
            {
                "row_id": row["row_id"],
                "source_corpus": row["source_corpus"],
                "x": row["phase1"]["x"],
                "y_true": row["phase1"]["y_true"],
                "y_pred": {"text": "{}"},
                "target_tokens": row["target_tokens"],
                "max_new_tokens": max_new_tokens,
                "generated_tokens": 1,
                "sequence_length": 10,
                "latency_sec": 0.01,
                "memory_peak_mb": 0.0,
            }
            for row in rows
        ]

    monkeypatch.setattr(script, "_generate_rows", fake_generate)


def test_phase1_heldout_inference_writes_paired_full_evidence_outputs(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    """Mocked main() writes baseline/model_x rows with corpus, token, and budget evidence."""
    script = _load_script()
    heldout = _write_jsonl(
        tmp_path / "heldout.jsonl",
        [_row("1", "real_dell"), _row("2", "real_hpe")],
    )
    expected_row_ids = [
        _row_id("real_dell", "/redfish/v1/Systems/1"),
        _row_id("real_hpe", "/redfish/v1/Systems/2"),
    ]
    manifest = tmp_path / "heldout.manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "artifact_sha256": script._sha256(heldout),
                "approved_heldout_rows": 2,
                "required_corpora": ["real_dell", "real_hpe"],
                "row_ids": expected_row_ids,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    baseline_out = tmp_path / "baseline.jsonl"
    model_out = tmp_path / "model_x.jsonl"
    _install_model_mocks(monkeypatch, script, max_new_tokens=8)

    rc = script.main(
        [
            "--spec",
            str(tmp_path / "spec.yaml"),
            "--heldout-jsonl",
            str(heldout),
            "--heldout-manifest",
            str(manifest),
            "--baseline-output",
            str(baseline_out),
            "--model-output",
            str(model_out),
        ]
    )

    assert rc == 0
    assert json.loads(capsys.readouterr().out)["status"] == "pass"
    baseline_rows = [
        json.loads(line)
        for line in baseline_out.read_text(encoding="utf-8").splitlines()
    ]
    model_rows = [
        json.loads(line)
        for line in model_out.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["source_corpus"] for row in baseline_rows] == ["real_dell", "real_hpe"]
    assert [row["source_corpus"] for row in model_rows] == ["real_dell", "real_hpe"]
    assert [row["row_id"] for row in baseline_rows] == expected_row_ids
    assert [row["row_id"] for row in model_rows] == expected_row_ids
    assert {row["target_tokens"] for row in model_rows} == {4}
    assert {row["max_new_tokens"] for row in model_rows} == {8}
    assert not Path(f"{baseline_out}.pending").exists()
    assert not Path(f"{model_out}.pending").exists()


def test_phase1_heldout_inference_rejects_generation_budget_below_p95_margin(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    """The script blocks before writing outputs when max_new_tokens is under p95+margin."""
    script = _load_script()
    heldout = _write_jsonl(tmp_path / "heldout.jsonl", [_row("1", "real_dell")])
    manifest = tmp_path / "heldout.manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "artifact_sha256": script._sha256(heldout),
                "approved_heldout_rows": 1,
                "required_corpora": ["real_dell"],
                "row_ids": [_row_id("real_dell", "/redfish/v1/Systems/1")],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _install_model_mocks(monkeypatch, script, max_new_tokens=4)

    rc = script.main(
        [
            "--spec",
            str(tmp_path / "spec.yaml"),
            "--heldout-jsonl",
            str(heldout),
            "--heldout-manifest",
            str(manifest),
            "--baseline-output",
            str(tmp_path / "baseline.jsonl"),
            "--model-output",
            str(tmp_path / "model_x.jsonl"),
        ]
    )

    assert rc == 2
    assert "max_new_tokens does not cover" in capsys.readouterr().err


def test_phase1_heldout_inference_rejects_manifest_missing_required_corpus(
    tmp_path: Path,
) -> None:
    """Manifest required_corpora must all appear in the held-out JSONL rows."""
    script = _load_script()
    rows = [_row("1", "real_dell")]

    try:
        script._validate_manifest(
            {
                "artifact_sha256": "sha256:" + "a" * 64,
                "approved_heldout_rows": 1,
                "required_corpora": ["real_dell", "real_hpe"],
            },
            rows,
            "sha256:" + "a" * 64,
        )
    except script.HeldoutInferenceError as exc:
        assert "missing a required corpus" in str(exc)
    else:  # pragma: no cover - explicit failure message is clearer than assert False
        raise AssertionError("missing required corpus should block held-out inference")


def test_phase1_heldout_inference_canonicalizes_missing_row_id() -> None:
    """Rows without row_id receive the source-qualified REST API digest."""
    script = _load_script()

    normalized = script._normalize_input(
        _row("1", "real_dell", include_row_id=False),
        index=0,
    )

    assert normalized["row_id"] == _row_id("real_dell", "/redfish/v1/Systems/1")


def test_phase1_heldout_inference_rejects_supplied_row_id_mismatch() -> None:
    """A supplied row_id cannot override the source/REST API identity."""
    script = _load_script()

    with pytest.raises(script.HeldoutInferenceError, match="supplied row_id"):
        script._normalize_input(
            _row("1", "real_dell", supplied_row_id="operator-row-1"),
            index=0,
        )


def test_phase1_heldout_inference_rejects_manifest_row_id_mismatch(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    """Normalized row IDs must exactly equal the approved held-out manifest order."""
    script = _load_script()
    heldout = _write_jsonl(tmp_path / "heldout.jsonl", [_row("1", "real_dell")])
    manifest = tmp_path / "heldout.manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "artifact_sha256": script._sha256(heldout),
                "approved_heldout_rows": 1,
                "required_corpora": ["real_dell"],
                "row_ids": ["sha256:" + "9" * 64],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _install_model_mocks(monkeypatch, script, max_new_tokens=8)

    rc = script.main(
        [
            "--spec",
            str(tmp_path / "spec.yaml"),
            "--heldout-jsonl",
            str(heldout),
            "--heldout-manifest",
            str(manifest),
            "--baseline-output",
            str(tmp_path / "baseline.jsonl"),
            "--model-output",
            str(tmp_path / "model_x.jsonl"),
        ]
    )

    assert rc == 2
    assert "row IDs/order disagree" in capsys.readouterr().err
