"""Offline regression tests for LLM trainer report emission."""

from argparse import Namespace
import json

import pytest

from igc.modules.llm_train_state_encoder import ValidationMetrics, emit_final_run_report


class _DatasetWithManifest:
    """Tiny dataset double that exposes the trainer's report-manifest hook."""

    @staticmethod
    def run_manifest_fields():
        return {"data_manifest": "fixture-manifest", "eval_split": "fixture-heldout"}


def _trainer_args() -> Namespace:
    return Namespace(
        model_type="fixture-model",
        profile="phase1_fixture",
        use_peft=False,
        max_train_steps=1,
        seq_len=128,
    )


def test_final_report_emission_propagates_write_failure(monkeypatch, tmp_path):
    """A report write error is a hard failure, not a rank-zero warning."""
    import igc.modules.train.emit as emit

    def fail_emit(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(emit, "emit_run_report", fail_emit)

    with pytest.raises(OSError, match="disk full"):
        emit_final_run_report(
            trainer_args=_trainer_args(),
            dataset=_DatasetWithManifest(),
            final_epoch_loss=1.25,
            epochs_done=1,
            optimizer_steps=1,
            best_eval=0.5,
            validation_result=ValidationMetrics(loss=0.25, accuracy=97.5),
            started_at="2026-07-16T01:36:00",
            ended_at="2026-07-16T01:37:00",
            wall_clock_sec=60.0,
            checkpoint_path=str(tmp_path),
            output_dir=str(tmp_path),
        )


def test_final_report_emission_preserves_accuracy_and_loss_metrics(tmp_path):
    """The hard-fail helper keeps the current Phase 1 report metric keys."""
    report_path = emit_final_run_report(
        trainer_args=_trainer_args(),
        dataset=_DatasetWithManifest(),
        final_epoch_loss=1.25,
        epochs_done=1,
        optimizer_steps=2,
        best_eval=0.25,
        validation_result=ValidationMetrics(loss=0.25, accuracy=97.5),
        started_at="2026-07-16T01:36:00",
        ended_at="2026-07-16T01:37:00",
        wall_clock_sec=60.0,
        checkpoint_path=str(tmp_path),
        output_dir=str(tmp_path),
    )

    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report_path == str(tmp_path / "report.json")
    assert report["metrics"]["eval/accuracy"] == 97.5
    assert report["metrics"]["eval/loss"] == 0.25
    assert report["manifest"]["data_manifest"] == "fixture-manifest"
    assert report["manifest"]["eval_split"] == "fixture-heldout"
