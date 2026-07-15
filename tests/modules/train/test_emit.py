"""
Offline tests for run-report emission.

Pins that build_run_bundle maps the parsed spec into a RunManifest (model, adapter
from use_peft, corpus data_manifest/eval_split, scrubbed settings), that sensitive
key names can never leak into the emitted settings, that emit_run_report writes a
report.json that ResultBundle.read round-trips, and that the bundle feeds the
existing compare() fairness check. Pure stdlib — no torch, no trainer construction.

Author:
Mus mbayramo@stanford.edu
"""

from pathlib import Path

from igc.modules.train.emit import _SENSITIVE_MARKERS, build_run_bundle, emit_run_report
from igc.modules.train.report import ResultBundle, compare


def _spec(**over):
    """A minimal parsed-spec dict as vars(spec) would produce."""
    spec = {
        "model_type": "Qwen/Qwen2.5-7B-Instruct",
        "use_peft": True, "adapter_method": "rslora", "lora_r": 32,
        "lora_alpha": 64, "lora_init": "default",
        "max_train_steps": 200, "seq_len": 1024,
        "per_device_train_batch_size": 8, "gradient_accumulation_steps": 4,
        "num_train_epochs": 3, "llm_learning_rate": 5e-5,
        "llm_scheduler": "OneCycleLR", "llm_optimizer": "AdamW",
        "early_stopping_patience": 3, "early_stopping_min_delta": 0.005,
        "mixed_precision": "bf16", "sharding": "none", "use_accelerator": False,
        "masking_type": "NO_MASK", "num_workers": 8, "seed": 42,
        "weights_role": "model_x",
    }
    spec.update(over)
    return spec


def _bundle(**over):
    return build_run_bundle(
        _spec(**over),
        training={"final_epoch_loss": 3.1, "epochs_done": 3, "optimizer_steps": 200,
                  "best_eval": 0.72},
        metrics={"eval/accuracy": 0.72},
        dataset_fields={"data_manifest": "70a2db0b0e90e194",
                        "eval_split": "floor=REAL:frac=0.15:seed=0"},
        argv=["igc_main.py", "--train", "llm", "--llm", "latent"],
        started_at="2026-07-10T01:00:00", ended_at="2026-07-10T02:00:00",
        wall_clock_sec=3600.0, checkpoint_path="experiments/run1",
    )


def test_manifest_maps_spec_and_corpus_fields():
    """Model, adapter (from use_peft), data manifest, and stats land in the manifest."""
    b = _bundle()
    m = b.manifest
    assert m.model == "Qwen/Qwen2.5-7B-Instruct"
    assert m.adapter_method == "rslora" and m.adapter_rank == 32
    assert m.data_manifest == "70a2db0b0e90e194"
    assert m.eval_split == "floor=REAL:frac=0.15:seed=0"
    assert m.max_steps == 200 and m.seq_len == 1024
    assert m.training["optimizer_steps"] == 200
    assert m.argv[0] == "igc_main.py"
    assert m.environment.get("python")  # capture_environment ran
    assert b.metrics == {"eval/accuracy": 0.72}


def test_no_peft_means_adapter_none():
    """use_peft=False reports adapter 'none' with no rank."""
    m = _bundle(use_peft=False).manifest
    assert m.adapter_method == "none" and m.adapter_rank is None


def test_settings_scrubbed_of_sensitive_keys():
    """No emitted settings key may carry a sensitive marker (token/key/secret/...)."""
    m = _bundle().manifest
    for key in m.settings:
        assert not any(mark in key.lower() for mark in _SENSITIVE_MARKERS)
    # the allowlisted run knobs made it through
    assert m.settings["gradient_accumulation_steps"] == "4"
    assert m.settings["llm_scheduler"] == "OneCycleLR"
    assert m.settings["early_stopping_patience"] == "3"
    assert m.settings["early_stopping_min_delta"] == "0.005"
    assert m.settings["weights_role"] == "model_x"


def test_emit_writes_readable_report(tmp_path: Path):
    """emit_run_report writes report.json that ResultBundle.read reconstructs."""
    path = emit_run_report(_bundle(), str(tmp_path / "run"))
    assert path.endswith("report.json")
    back = ResultBundle.read(path)
    assert back.manifest.model == "Qwen/Qwen2.5-7B-Instruct"
    assert back.manifest.training["final_epoch_loss"] == 3.1
    assert back.arm == "rslora-r32"


def test_bundle_feeds_compare_fairness():
    """Two emitted runs on the same corpus/split compare as fair."""
    rep = compare([_bundle(), _bundle(adapter_method="dora")], baseline="rslora")
    assert not rep.fairness_issues


# Author: Mus mbayramo@stanford.edu
