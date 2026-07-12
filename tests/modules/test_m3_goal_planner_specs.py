from pathlib import Path

from igc.modules.m3_goal_planner_specs import load_m3_goal_planner_profile


def test_m3_goal_planner_profile_resolves_model_and_optimizer_refs(tmp_path: Path):
    spec = tmp_path / "profiles.yaml"
    spec.write_text(
        """
schema_version: m3.goal_planner.training.v1
default_profile: m3_test_full
models:
  tiny:
    model_type: gpt2
    trust_remote_code: false
optimizers:
  adamw:
    name: adamw
    learning_rate: 0.0002
    weight_decay: 0.01
datasets:
  real_m3:
    jsonl: /models/igc/datasets/m3/m3_goal_plans_real.jsonl
    max_records: null
metric_sets:
  m3_default:
    event_fields: [record_index, train_loss]
    plot_names:
      train_loss: m3_sft/train_loss
profiles:
  m3_test_full:
    dataset:
      ref: real_m3
      max_records: 128
    model:
      ref: tiny
    optimizer:
      ref: adamw
    trainer:
      output_dir: /models/igc/checkpoints/m3_goal_planner/test
      epochs: 3
      batch_size: 4
      gradient_accumulation_steps: 2
      precision: bf16
    distributed:
      launcher: torchrun
      nproc_per_node: 4
    tracking:
      metric_set: m3_default
""",
        encoding="utf-8",
    )

    profile = load_m3_goal_planner_profile("m3_test_full", spec)

    assert profile["model"]["model_type"] == "gpt2"
    assert profile["dataset"]["jsonl"] == "/models/igc/datasets/m3/m3_goal_plans_real.jsonl"
    assert profile["dataset"]["max_records"] == 128
    assert profile["model"]["trust_remote_code"] is False
    assert profile["optimizer"]["learning_rate"] == 0.0002
    assert profile["optimizer"]["weight_decay"] == 0.01
    assert profile["trainer"]["epochs"] == 3
    assert profile["trainer"]["precision"] == "bf16"
    assert profile["distributed"]["nproc_per_node"] == 4
    assert profile["tracking"]["metrics"]["plot_names"]["train_loss"] == "m3_sft/train_loss"
