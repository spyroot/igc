"""Offline tests for _wandb_run_meta (W&B run labelling by curriculum stage).

Pins that a run's spec maps to a legible label (m1 state encoder, legacy goal extractor,
m6 RL agent) with a readable name, filterable tags, and a config snapshot — so W&B shows
what stage/model/epochs a run is instead of a random name. Pure logic — no wandb.

Author:
Mus mbayramo@stanford.edu
"""

from igc.modules.base.metric_factory import _wandb_run_meta


def test_m1_state_encoder_labels():
    """--train llm --llm latent maps to the m1 state-encoder stage with model/epoch/bs."""
    meta = _wandb_run_meta({
        "train": "llm", "llm": "latent", "model_type": "Qwen/Qwen2.5-0.5B-Instruct",
        "num_train_epochs": 5, "per_device_train_batch_size": 8, "use_peft": True,
    })
    assert meta["group"] == "m1-state-encoder"
    assert meta["job_type"] == "train"
    assert meta["name"] == "m1-state-encoder-qwen2.5-0.5b-instruct-e5-bs8"
    assert "m1-state-encoder" in meta["tags"] and "ep5" in meta["tags"] and "lora" in meta["tags"]
    assert meta["config"]["num_train_epochs"] == 5 and meta["config"]["model_type"].endswith("0.5B-Instruct")


def test_goal_extractor_and_rl_stages():
    """Other selections map to their stages (legacy goal extractor, m6 RL agent)."""
    assert _wandb_run_meta({"train": "llm", "llm": "goal"})["group"] == "goal-extractor-legacy"
    assert _wandb_run_meta({"train": "agent", "rl": "dqn"})["group"] == "m6-rl-agent"


def test_phase_training_metadata_is_first_class():
    """Phase profiles label W&B runs without using legacy m3 config keys."""
    meta = _wandb_run_meta({
        "phase": "phase2_goal_extract",
        "profile": "phase2_qwen2_5_7b_rslora",
        "model_type": "Qwen/Qwen2.5-7B-Instruct",
        "objective": "ordered_rest_goal_extraction",
        "dataset_jsonl": "/models/igc/goal-datasets/D1_ordered_rest_goals.jsonl",
        "record_count": 1024,
        "optimizer": "adamw_torch_fused",
        "scheduler": "cosine",
        "learning_rate": 0.0002,
        "weight_decay": 0.0,
        "warmup_ratio": 0.03,
        "gradient_accumulation_steps": 16,
        "max_length": 1536,
        "precision": "bf16",
        "use_peft": True,
    })

    assert meta["group"] == "phase2_goal_extract"
    assert meta["name"] == "phase2_goal_extract-qwen2.5-7b-instruct"
    assert "phase2_goal_extract" in meta["tags"]
    assert "lora" in meta["tags"]
    assert meta["config"]["phase"] == "phase2_goal_extract"
    assert meta["config"]["profile"] == "phase2_qwen2_5_7b_rslora"
    assert meta["config"]["record_count"] == 1024
    assert not any(key.startswith("m3_") for key in meta["config"])


def test_sharding_tag_only_when_set():
    """A real sharding mode is tagged; 'none' is not."""
    sharded = _wandb_run_meta({"train": "llm", "llm": "latent", "sharding": "zero3"})
    plain = _wandb_run_meta({"train": "llm", "llm": "latent", "sharding": "none"})
    assert "zero3" in sharded["tags"]
    assert "none" not in plain["tags"]


# Author: Mus mbayramo@stanford.edu
