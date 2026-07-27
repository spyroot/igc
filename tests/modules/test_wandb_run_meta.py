"""Offline tests for _wandb_run_meta (W&B run labelling by curriculum stage).

Pins that a run's spec maps to legible Phase1/2/3 labels with a readable name,
filterable tags, and a config snapshot, so W&B shows what stage/model/epochs a
run is instead of a random name. Pure logic — no wandb.

Author:
Mus mbayramo@stanford.edu
"""

from igc.modules.base.metric_factory import _wandb_run_meta


def test_phase1_profile_labels():
    """Phase 1 profile metadata maps to the Phase 1 fine-tune run group."""
    meta = _wandb_run_meta({
        "profile": "phase1_7b_rslora_r32",
        "phase": "phase1_finetune",
        "weights_role": "model_x",
        "corpus_objective": "phase1_pretrain",
        "train": "llm",
        "llm": "sft",
        "model_type": "Qwen/Qwen2.5-7B-Instruct",
        "num_train_epochs": 5, "per_device_train_batch_size": 8, "use_peft": True,
    })
    assert meta["group"] == "phase1-finetune"
    assert meta["job_type"] == "train"
    assert meta["name"] == "phase1-finetune-qwen2.5-7b-instruct-e5-bs8"
    assert "phase1-finetune" in meta["tags"]
    assert "ep5" in meta["tags"]
    assert "lora" in meta["tags"]
    assert meta["config"]["profile"] == "phase1_7b_rslora_r32"
    assert meta["config"]["weights_role"] == "model_x"


def test_phase1_profile_labels_finetune_even_when_rl_defaults_none():
    """Phase 1 fine-tuning must not be mislabeled as RL when rl='none'."""
    meta = _wandb_run_meta({
        "profile": "phase1_7b_rslora_r32",
        "weights_role": "model_x",
        "corpus_objective": "phase1_pretrain",
        "train": "llm",
        "llm": "sft",
        "rl": "none",
        "model_type": "Qwen/Qwen2.5-7B-Instruct",
        "num_train_epochs": 3,
        "per_device_train_batch_size": 8,
        "use_peft": True,
        "lora_r": 32,
    })

    assert meta["group"] == "phase1-finetune"
    assert meta["name"] == "phase1-finetune-qwen2.5-7b-instruct-e3-bs8"
    assert "phase1-finetune" in meta["tags"]
    assert "rl-agent" not in meta["tags"]
    assert meta["config"]["profile"] == "phase1_7b_rslora_r32"
    assert meta["config"]["weights_role"] == "model_x"


def test_phase2_phase3_profile_labels():
    """Phase2/3 roles map to their dedicated W&B groups."""
    phase2 = _wandb_run_meta({
        "phase": "phase2_goal_extraction",
        "weights_role": "goal_extractor",
        "profile": "phase2_7b_rslora_r32",
        "model_type": "Qwen/Qwen2.5-7B-Instruct",
    })
    phase3 = _wandb_run_meta({
        "phase": "phase3_argument_extraction",
        "weights_role": "argument_extractor",
        "profile": "phase3_7b_rslora_r32",
        "model_type": "Qwen/Qwen2.5-7B-Instruct",
    })

    assert phase2["group"] == "phase2-goal-extractor"
    assert phase2["name"] == "phase2-goal-extractor-qwen2.5-7b-instruct"
    assert "phase2-goal-extractor" in phase2["tags"]
    assert phase3["group"] == "phase3-argument-extractor"
    assert phase3["name"] == "phase3-argument-extractor-qwen2.5-7b-instruct"
    assert "phase3-argument-extractor" in phase3["tags"]


def test_rl_none_does_not_force_rl_label():
    """The argparse default rl='none' is not an active RL stage."""
    meta = _wandb_run_meta({"train": "llm", "llm": "latent", "rl": "none"})
    assert meta["group"] == "legacy-state-encoder"


def test_legacy_goal_extractor_and_rl_stages():
    """Legacy goal-extractor selection and active RL still map to distinct groups."""
    assert _wandb_run_meta({"train": "llm", "llm": "goal"})["group"] == "goal-extractor-legacy"
    assert _wandb_run_meta({"train": "agent", "rl": "dqn"})["group"] == "rl-agent"


def test_sharding_tag_only_when_set():
    """A real sharding mode is tagged; 'none' is not."""
    sharded = _wandb_run_meta({"phase": "phase1_finetune", "sharding": "zero3"})
    plain = _wandb_run_meta({"phase": "phase1_finetune", "sharding": "none"})
    assert "zero3" in sharded["tags"]
    assert "none" not in plain["tags"]


# Author: Mus mbayramo@stanford.edu
