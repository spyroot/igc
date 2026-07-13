"""M1/M2 collator tests for structured state consumption."""

import torch

from igc.modules.igc_train_auto_state_encoder import AutoencoderTrainer
from igc.modules.llm_train_state_encoder import LlmEmbeddingsTrainer


def _sample(idx: int):
    return {
        "input_ids": torch.tensor([idx, idx + 1]),
        "attention_mask": torch.tensor([1, 1]),
        "graph_node_count": torch.tensor(2),
        "graph_edge_count": torch.tensor(3),
        "action_candidate_count": torch.tensor(4),
        "candidate_mask": torch.tensor([True, True, False]),
        "candidate_resource_type_id": torch.tensor([11, 12, 0]),
        "candidate_parent_type_id": torch.tensor([21, 22, 0]),
        "candidate_relation_name_id": torch.tensor([31, 32, 0]),
        "candidate_depth_bucket": torch.tensor([2, 2, 0]),
        "candidate_method_id": torch.tensor([1, 4, 0]),
        "candidate_has_action_target": torch.tensor([1.0, 1.0, 0.0]),
        "candidate_is_collection": torch.tensor([0.0, 0.0, 0.0]),
        "candidate_is_oem": torch.tensor([0.0, 0.0, 0.0]),
        "candidate_path_segment_hashes": torch.ones(3, 8, dtype=torch.long),
        "candidate_allowed_method_mask": torch.ones(3, 6),
        "candidate_local_state_summary": torch.ones(3, 4, dtype=torch.long),
        "scope_mask": torch.tensor([True, False]),
        "scope_resource_type_id": torch.tensor([11, 0]),
        "scope_parent_type_id": torch.tensor([21, 0]),
        "scope_relation_name_id": torch.tensor([31, 0]),
        "scope_depth_bucket": torch.tensor([2, 0]),
        "scope_method_id": torch.tensor([0, 0]),
        "scope_has_action_target": torch.tensor([1.0, 0.0]),
        "scope_is_collection": torch.tensor([0.0, 0.0]),
        "scope_is_oem": torch.tensor([0.0, 0.0]),
        "scope_path_segment_hashes": torch.ones(2, 8, dtype=torch.long),
        "scope_allowed_method_mask": torch.ones(2, 6),
        "scope_local_state_summary": torch.ones(2, 4, dtype=torch.long),
        "candidate_endpoint_scope_index": torch.tensor([0, 0, 0]),
        "state_fingerprint": f"fp{idx}",
        "state_id": f"state:{idx}",
    }


def test_m1_collator_preserves_structured_state_fields():
    """The state encoder collator consumes graph/action state fields."""
    batch = LlmEmbeddingsTrainer.custom_collate_fn([_sample(1), _sample(2)])
    assert batch["input_ids"].shape == (2, 2)
    assert batch["graph_node_count"].tolist() == [2, 2]
    assert batch["graph_edge_count"].tolist() == [3, 3]
    assert batch["action_candidate_count"].tolist() == [4, 4]
    assert batch["candidate_method_id"].shape == (2, 3)
    assert batch["candidate_path_segment_hashes"].shape == (2, 3, 8)
    assert batch["candidate_allowed_method_mask"].shape == (2, 3, 6)
    assert batch["candidate_mask"].sum().item() == 4
    assert batch["scope_mask"].shape == (2, 2)
    assert batch["candidate_endpoint_scope_index"].shape == (2, 3)
    assert batch["state_fingerprint"] == ["fp1", "fp2"]
    assert batch["state_id"] == ["state:1", "state:2"]


def test_m2_collator_preserves_structured_state_fields():
    """The autoencoder collator consumes the same State contract as M1."""
    batch = AutoencoderTrainer.custom_collate_fn([_sample(1), _sample(2)])
    assert batch["input_ids"].shape == (2, 2)
    assert batch["graph_node_count"].tolist() == [2, 2]
    assert batch["graph_edge_count"].tolist() == [3, 3]
    assert batch["action_candidate_count"].tolist() == [4, 4]
    assert batch["candidate_method_id"].shape == (2, 3)
    assert batch["candidate_path_segment_hashes"].shape == (2, 3, 8)
    assert batch["candidate_allowed_method_mask"].shape == (2, 3, 6)
    assert batch["candidate_mask"].sum().item() == 4
    assert batch["scope_mask"].shape == (2, 2)
    assert batch["candidate_endpoint_scope_index"].shape == (2, 3)
    assert batch["state_fingerprint"] == ["fp1", "fp2"]
    assert batch["state_id"] == ["state:1", "state:2"]
