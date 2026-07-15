"""Offline regression tests for LLM trainer report hard-fail behavior."""

from argparse import Namespace

import pytest

from igc.modules.llm_train_state_encoder import emit_final_run_report


class _DatasetWithManifest:
    """Tiny dataset double that exposes the trainer's report-manifest hook."""

    @staticmethod
    def run_manifest_fields():
        return {"data_manifest": "fixture-manifest", "eval_split": "fixture-heldout"}


def test_final_report_emission_propagates_write_failure(monkeypatch, tmp_path):
    """A report write error is a hard failure, not a rank-zero warning."""
    import igc.modules.train.emit as emit

    def fail_emit(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(emit, "emit_run_report", fail_emit)

    with pytest.raises(OSError, match="disk full"):
        emit_final_run_report(
            trainer_args=Namespace(
                model_type="fixture-model",
                profile="phase1-fixture",
                use_peft=False,
                max_train_steps=1,
                seq_len=128,
            ),
            dataset=_DatasetWithManifest(),
            final_epoch_loss=1.25,
            epochs_done=1,
            optimizer_steps=1,
            best_eval=0.5,
            validation_accuracy=50.0,
            started_at="2026-07-16T01:36:00",
            ended_at="2026-07-16T01:37:00",
            wall_clock_sec=60.0,
            checkpoint_path=str(tmp_path),
            output_dir=str(tmp_path),
        )
