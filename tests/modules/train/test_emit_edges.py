"""
Broad edge tests for run-report emission.

Covers default/empty inputs and immutability boundaries around ``build_run_bundle``
and ``emit_run_report`` so end-of-run reporting stays deterministic without importing
the trainer. Pure stdlib.

Author:
Mus mbayramo@stanford.edu
"""

from pathlib import Path

from igc.modules.train.emit import build_run_bundle, emit_run_report
from igc.modules.train.report import ResultBundle


def _spec(**overrides):
    """Minimal parsed-spec dict with realistic run-report keys."""
    spec = {
        "model_type": "Qwen/Qwen2.5-7B-Instruct",
        "profile": "m1_7b_rslora_r32",
        "use_peft": True,
        "adapter_method": "rslora",
        "lora_r": "32",
        "lora_init": "pissa",
        "max_train_steps": 120,
        "seq_len": 1024,
        "gradient_accumulation_steps": 4,
        "llm_scheduler": "OneCycleLR",
        "seed": 7,
    }
    spec.update(overrides)
    return spec


def test_missing_dataset_fields_default_to_empty_manifest_ids():
    """Reports from raw-capture training keep manifest/split ids empty, not missing."""
    bundle = build_run_bundle(
        _spec(),
        training={"optimizer_steps": 120},
        metrics=None,
        dataset_fields=None,
        argv=[],
    )

    assert bundle.manifest.data_manifest == ""
    assert bundle.manifest.eval_split == ""
    assert bundle.metrics == {}


def test_run_id_is_stable_for_model_basename_and_timestamp():
    """run_id uses the model basename and strips timestamp punctuation."""
    bundle = build_run_bundle(
        _spec(model_type="/models/Qwen2.5-7B-Instruct"),
        training={},
        argv=[],
        started_at="2026-07-10T01:02:03",
    )

    assert bundle.manifest.run_id == "m1-Qwen2.5-7B-Instruct-20260710T010203"


def test_report_copies_training_and_metrics_inputs():
    """Caller mutations after bundle creation cannot change the emitted report."""
    training = {"final_epoch_loss": 3.0}
    metrics = {"eval/accuracy": 0.62}
    bundle = build_run_bundle(_spec(), training=training, metrics=metrics, argv=[])

    training["final_epoch_loss"] = 99.0
    metrics["eval/accuracy"] = 0.0

    assert bundle.manifest.training["final_epoch_loss"] == 3.0
    assert bundle.metrics["eval/accuracy"] == 0.62


def test_settings_include_only_allowlisted_run_knobs():
    """Settings omit unrelated parser values while preserving selected knobs as strings."""
    bundle = build_run_bundle(
        _spec(output_dir="/tmp/run", hf_token="redacted", password="redacted"),
        training={},
        argv=[],
    )

    assert bundle.manifest.settings == {
        "gradient_accumulation_steps": "4",
        "llm_scheduler": "OneCycleLR",
        "seed": "7",
        "use_peft": "True",
        "lora_r": "32",
    }


def test_emit_creates_nested_output_directory(tmp_path: Path):
    """emit_run_report creates the run directory and writes a readable report.json."""
    bundle = build_run_bundle(
        _spec(use_peft=False, lora_r=64),
        training={"epochs_done": 1},
        argv=["igc_main.py", "--profile", "m1_gpt2_smoke"],
        checkpoint_path="experiments/smoke",
    )
    report_path = emit_run_report(bundle, str(tmp_path / "nested" / "run"))

    assert report_path.endswith("report.json")
    back = ResultBundle.read(report_path)
    assert back.manifest.adapter_method == "none"
    assert back.manifest.adapter_rank is None
    assert back.manifest.checkpoint_path == "experiments/smoke"


# Author: Mus mbayramo@stanford.edu
