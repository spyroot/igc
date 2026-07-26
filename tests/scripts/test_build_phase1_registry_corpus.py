"""Parser and summary tests for ``scripts/build_phase1_registry_corpus.py``."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "build_phase1_registry_corpus.py"


def _load_script():
    """Import the script without invoking the CLI entrypoint."""
    spec = importlib.util.spec_from_file_location("build_phase1_registry_corpus", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_phase1_registry_corpus_writes_materializer_summary(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    """The CLI forwards deterministic split args and persists the returned lineage."""
    script = _load_script()
    calls: list[dict] = []
    summary = {
        "source_registry_sha": "sha256:" + "1" * 64,
        "source_manifest_shas": {"real_dell": "sha256:" + "2" * 64},
        "train_artifact_sha": "sha256:" + "3" * 64,
        "heldout_artifact_sha": "sha256:" + "4" * 64,
        "written_manifest_sha": "sha256:" + "5" * 64,
        "manifest_sha": "sha256:" + "6" * 64,
        "train_rows": 7,
        "heldout_rows": 2,
    }

    def fake_materialize(**kwargs):
        calls.append(kwargs)
        return dict(summary)

    monkeypatch.setattr(script, "materialize_phase1_registry_corpus", fake_materialize)
    output_root = tmp_path / "corpus"
    summary_json = tmp_path / "reports" / "summary.json"

    rc = script.main(
        [
            "--source-registry",
            str(tmp_path / "registry.yaml"),
            "--output-root",
            str(output_root),
            "--corpus-kind",
            "golden",
            "--eval-fraction",
            "0.25",
            "--seed",
            "19",
            "--summary-json",
            str(summary_json),
        ]
    )

    assert rc == 0
    assert calls == [
        {
            "registry_path": str(tmp_path / "registry.yaml"),
            "output_root": str(output_root),
            "corpus_kind": "golden",
            "eval_fraction": 0.25,
            "seed": 19,
        }
    ]
    assert json.loads(summary_json.read_text(encoding="utf-8")) == summary
    assert json.loads(capsys.readouterr().out) == {
        "status": "pass",
        "train_rows": 7,
        "heldout_rows": 2,
        "summary": str(summary_json),
    }


def test_build_phase1_registry_corpus_reports_blocked_materializer_errors(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    """Materializer validation errors remain a blocked CLI result, not a traceback."""
    script = _load_script()
    monkeypatch.setattr(
        script,
        "materialize_phase1_registry_corpus",
        lambda **_kwargs: (_ for _ in ()).throw(ValueError("missing required source")),
    )

    rc = script.main(
        [
            "--output-root",
            str(tmp_path / "corpus"),
            "--summary-json",
            str(tmp_path / "summary.json"),
        ]
    )

    assert rc == 2
    payload = json.loads(capsys.readouterr().err)
    assert payload["status"] == "blocked"
    assert payload["error"] == "missing required source"
