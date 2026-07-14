"""Offline tests for the shared Phase 1/2/3 training spec contract.

Author:
Mus mbayramo@stanford.edu
"""

from __future__ import annotations

from igc.modules.train.phase_specs import (
    load_phase_profile,
    phase_names,
    profile_names,
)


def test_phase_spec_registers_all_phase_contracts() -> None:
    """The shared spec exposes one gate namespace for each P0 phase."""
    assert phase_names() == [
        "phase1_pretrain",
        "phase2_goal_extract",
        "phase3_argument_extract",
    ]
    assert profile_names("phase1_pretrain") == [
        "phase1_gpt2_smoke",
        "phase1_qwen2_5_7b_rslora",
    ]
    assert profile_names("phase2_goal_extract") == [
        "phase2_gpt2_smoke",
        "phase2_qwen2_5_7b_rslora",
    ]
    assert profile_names("phase3_argument_extract") == [
        "phase3_gpt2_smoke",
        "phase3_qwen2_5_7b_rslora",
    ]


def test_phase_profiles_resolve_shared_model_optimizer_and_peft_refs() -> None:
    """A profile resolves reusable model, optimizer, PEFT, and metric sections."""
    profile = load_phase_profile("phase1_qwen2_5_7b_rslora")

    assert profile["phase"] == "phase1_pretrain"
    assert profile["objective"] == "redfish_json_reconstruction"
    assert profile["model"]["model_type"] == "Qwen/Qwen2.5-7B-Instruct"
    assert profile["optimizer"]["name"] == "adamw_torch_fused"
    assert profile["peft"]["method"] == "rslora"
    assert profile["peft"]["r"] == 32
    assert profile["trainer"]["precision"] == "bf16"
    assert profile["tracking"]["metric_set"] == "phase1_pretrain"
    assert "metrics" in profile["tracking"]


def test_phase_metric_names_are_stage_scoped() -> None:
    """All W&B plot names are clearly owned by the phase that emits them."""
    expected_prefixes = {
        "phase1_pretrain": ("phase1_dataset/", "phase1_pretrain/", "phase1_eval/"),
        "phase2_goal_extract": ("phase2_dataset/", "phase2_goal_extract/", "phase2_eval/"),
        "phase3_argument_extract": ("phase3_dataset/", "phase3_argument_extract/", "phase3_eval/"),
    }

    for phase in phase_names():
        profile = load_phase_profile(profile_names(phase)[0])
        plot_names = profile["tracking"]["metrics"]["plot_names"]
        assert plot_names
        for plot_name in plot_names.values():
            assert plot_name.startswith(expected_prefixes[phase])
            assert not plot_name.startswith("m3_")


def test_phase2_and_phase3_metrics_pin_ordered_rest_contracts() -> None:
    """The spec gates the ordered REST list and ordered call contracts."""
    phase2 = load_phase_profile("phase2_qwen2_5_7b_rslora")
    phase3 = load_phase_profile("phase3_qwen2_5_7b_rslora")

    phase2_plots = phase2["tracking"]["metrics"]["plot_names"]
    assert phase2_plots["eval_rest_api_list_ordered_exact_match"] == (
        "phase2_eval/rest_api_list_ordered_exact_match"
    )
    assert phase2_plots["eval_rest_api_list_set_f1"] == "phase2_eval/rest_api_list_set_f1"
    assert phase2_plots["eval_hallucinated_endpoint_rate"] == (
        "phase2_eval/hallucinated_endpoint_rate"
    )

    phase3_plots = phase3["tracking"]["metrics"]["plot_names"]
    assert phase3_plots["eval_ordered_call_exact_match"] == (
        "phase3_eval/ordered_call_exact_match"
    )
    assert phase3_plots["eval_http_method_exact_match"] == (
        "phase3_eval/http_method_exact_match"
    )
    assert phase3_plots["eval_argument_exact_match"] == "phase3_eval/argument_exact_match"


# Author: Mus mbayramo@stanford.edu
