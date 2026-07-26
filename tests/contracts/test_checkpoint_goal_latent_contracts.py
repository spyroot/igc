"""Static contract tests for checkpoint lineage and goal-latent boundaries."""

from __future__ import annotations

from pathlib import Path

import yaml


CHECKPOINT_LINEAGE = Path("configs/contracts/checkpoint_lineage.yaml")
GOAL_LATENT = Path("configs/contracts/goal_latent.yaml")


def _yaml(path: Path) -> dict:
    """Load a checked-in YAML contract."""
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_checkpoint_lineage_pins_phase_order_and_parent_artifacts() -> None:
    """model_x -> goal_extractor -> argument_extractor is the only checkpoint chain."""
    contract = _yaml(CHECKPOINT_LINEAGE)
    lineage = contract["checkpoint_lineage"]

    assert lineage["model_x"] == {
        "phase": 1,
        "role": "phase1_json_backbone",
        "parent_role": "foundation_instruct",
        "immutable_after_promotion": True,
        "artifact_sha": "${MODEL_X_SHA}",
    }
    assert lineage["goal_extractor"]["phase"] == 2
    assert lineage["goal_extractor"]["parent_role"] == "model_x"
    assert lineage["goal_extractor"]["parent_sha"] == "${MODEL_X_SHA}"
    assert lineage["goal_extractor"]["artifact_sha"] == "${PHASE2_SHA}"
    assert lineage["argument_extractor"]["phase"] == 3
    assert lineage["argument_extractor"]["parent_role"] == "goal_extractor"
    assert lineage["argument_extractor"]["parent_sha"] == "${PHASE2_SHA}"
    assert lineage["argument_extractor"]["artifact_sha"] == "${PHASE3_SHA}"


def test_checkpoint_lineage_pins_encoder_sources_and_rl_freeze() -> None:
    """State/rest/method encoders come from Phase 1/2/3 and all freeze during RL."""
    contract = _yaml(CHECKPOINT_LINEAGE)
    encoders = contract["encoders"]

    assert encoders["state"]["output"] == "z_state"
    assert encoders["state"]["source_role"] == "model_x"
    assert encoders["state"]["source_sha"] == "${MODEL_X_SHA}"
    assert encoders["rest_goal"]["output"] == "z_rest"
    assert encoders["rest_goal"]["source_role"] == "goal_extractor"
    assert encoders["rest_goal"]["source_sha"] == "${PHASE2_SHA}"
    assert encoders["method_goal"]["output"] == "z_method"
    assert encoders["method_goal"]["source_role"] == "argument_extractor"
    assert encoders["method_goal"]["source_sha"] == "${PHASE3_SHA}"
    assert all(value["frozen_during_rl"] is True for value in encoders.values())
    assert contract["invariants"]["state_encoder_branches_from_model_x"] is True
    assert contract["invariants"]["phase3_initializes_from_phase2"] is True
    assert contract["invariants"]["only_rl_policy_learns_during_rl"] is True


def test_goal_latent_contract_keeps_argument_values_outside_latents() -> None:
    """z_rest/z_method exclude literal values; executor/verifier keep them separately."""
    contract = _yaml(GOAL_LATENT)
    outputs = contract["outputs"]

    assert contract["inputs"]["calls"]["semantics"] == "unordered_unique_call_set"
    assert outputs["z_rest"]["source_role"] == "goal_extractor"
    assert outputs["z_rest"]["source_phase"] == 2
    assert "literal_argument_values" in outputs["z_rest"]["excludes"]
    assert outputs["z_method"]["source_role"] == "argument_extractor"
    assert outputs["z_method"]["source_phase"] == 3
    assert "literal_argument_values" in outputs["z_method"]["excludes"]
    assert contract["literal_argument_values"] == {
        "retained_outside_latents": True,
        "available_to_executor": True,
        "available_to_success_verifier": True,
    }


def test_goal_latent_contract_freezes_goal_encoders_and_makes_no_order_claim() -> None:
    """RL consumes frozen goal encoders and Phase 2/3 make no execution-order claim."""
    contract = _yaml(GOAL_LATENT)

    assert contract["rl"]["goal_encoders_frozen"] is True
    assert contract["rl"]["execution_order_not_encoded_by_phase2_or_phase3"] is True
    assert "unified_goal_latent" in contract["not_claimed"]
    assert "shared_state_goal_latent_distribution" in contract["not_claimed"]
