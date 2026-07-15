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


def test_phase1_profile_labels_finetune_even_when_rl_defaults_none():
    """Phase 1 latent-route fine-tuning must not be mislabeled as M6 when rl='none'."""
    meta = _wandb_run_meta({
        "profile": "phase1_7b_rslora_r32",
        "weights_role": "model_x",
        "corpus_objective": "phase1_pretrain",
        "train": "llm",
        "llm": "latent",
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
    assert "m6-rl-agent" not in meta["tags"]
    assert meta["config"]["profile"] == "phase1_7b_rslora_r32"
    assert meta["config"]["weights_role"] == "model_x"


def test_rl_none_does_not_force_m6_label():
    """The argparse default rl='none' is not an active RL stage."""
    meta = _wandb_run_meta({"train": "llm", "llm": "latent", "rl": "none"})
    assert meta["group"] == "m1-state-encoder"


def test_goal_extractor_and_rl_stages():
    """Other selections map to their stages (legacy goal extractor, m6 RL agent)."""
    assert _wandb_run_meta({"train": "llm", "llm": "goal"})["group"] == "goal-extractor-legacy"
    assert _wandb_run_meta({"train": "agent", "rl": "dqn"})["group"] == "m6-rl-agent"


def test_sharding_tag_only_when_set():
    """A real sharding mode is tagged; 'none' is not."""
    sharded = _wandb_run_meta({"train": "llm", "llm": "latent", "sharding": "zero3"})
    plain = _wandb_run_meta({"train": "llm", "llm": "latent", "sharding": "none"})
    assert "zero3" in sharded["tags"]
    assert "none" not in plain["tags"]


# Author: Mus mbayramo@stanford.edu
