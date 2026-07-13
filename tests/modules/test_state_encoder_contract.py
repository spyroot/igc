"""Tensor-contract tests for M1/M2 state pooling and M6 candidate encoding."""

import torch

from igc.core.types import ToolAction
from igc.modules.llm.llm_encoder import StubLLMEncoder
from igc.modules.policy.goal_conditioning import GoalEncoder
from igc.modules.policy.pointer_policy import Igc_PointerQNetwork
from igc.modules.policy.state_encoder import (
    CandidateEncoder,
    GraphFeatureEncoder,
    NodeFusion,
    ResourceTextPooler,
    StatePooler,
)


def _features(batch=1, nodes=2):
    return {
        "resource_type_id": torch.ones(batch, nodes, dtype=torch.long),
        "parent_type_id": torch.ones(batch, nodes, dtype=torch.long) * 2,
        "relation_name_id": torch.ones(batch, nodes, dtype=torch.long) * 3,
        "depth_bucket": torch.ones(batch, nodes, dtype=torch.long),
        "method_id": torch.ones(batch, nodes, dtype=torch.long),
        "has_action_target": torch.ones(batch, nodes),
        "is_collection": torch.zeros(batch, nodes),
        "is_oem": torch.zeros(batch, nodes),
        "path_segment_hashes": torch.ones(batch, nodes, 8, dtype=torch.long) * 5,
        "allowed_method_mask": torch.ones(batch, nodes, 6),
        "local_state_summary": torch.ones(batch, nodes, 4, dtype=torch.long) * 7,
    }


def test_resource_text_pooler_changes_when_resource_json_tokens_change():
    """Changing one resource's token hidden states changes its json embedding."""
    pooler = ResourceTextPooler()
    hidden = torch.zeros(1, 3, 4)
    mask = torch.tensor([[1, 1, 0]])
    base = pooler(hidden, mask)
    changed = hidden.clone()
    changed[0, 1, 0] = 4.0
    assert not torch.allclose(pooler(changed, mask), base)


def test_graph_feature_encoder_and_node_fusion_change_on_typed_feature_change():
    """Changing a typed graph field changes feat_emb and fused node_emb."""
    torch.manual_seed(0)
    encoder = GraphFeatureEncoder(hidden_size=16)
    fusion = NodeFusion(json_dim=16, feature_dim=16, node_dim=16)
    features = _features(nodes=1)
    feat = encoder(features)
    changed = _features(nodes=1)
    changed["resource_type_id"][0, 0] = 99
    changed_feat = encoder(changed)
    json_emb = torch.ones(1, 1, 16)

    assert not torch.allclose(changed_feat, feat)
    assert not torch.allclose(fusion(json_emb, changed_feat), fusion(json_emb, feat))


def test_state_pooler_ignores_nodes_outside_scope_and_uses_nodes_inside_scope():
    """Unrelated resources outside S_t do not perturb state_latent; scoped ones do."""
    torch.manual_seed(0)
    pooler = StatePooler(node_dim=4, state_dim=4)
    nodes = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [99.0, 99.0, 99.0, 99.0]]])
    mask = torch.tensor([[1, 0]], dtype=torch.bool)
    base = pooler(nodes, mask)

    outside_changed = nodes.clone()
    outside_changed[0, 1] = torch.tensor([-100.0, -100.0, -100.0, -100.0])
    inside_changed = nodes.clone()
    inside_changed[0, 0] = torch.tensor([2.0, 0.0, 0.0, 0.0])

    assert torch.allclose(pooler(outside_changed, mask), base)
    assert not torch.allclose(pooler(inside_changed, mask), base)


def test_candidate_encoder_consumes_endpoint_node_features_not_just_text():
    """Same action text and graph features score differently with another endpoint node."""
    torch.manual_seed(0)
    encoder = CandidateEncoder(text_dim=8, node_dim=8, graph_dim=8, candidate_dim=8)
    action_text_h = torch.ones(1, 2, 8)
    graph_h = torch.ones(1, 2, 8)
    node_h = torch.zeros(1, 2, 8)
    base = encoder(action_text_h, node_h, graph_h)
    node_h[:, 1, :] = 3.0
    changed = encoder(action_text_h, node_h, graph_h)
    assert torch.allclose(changed[:, 0], base[:, 0])
    assert not torch.allclose(changed[:, 1], base[:, 1])


def test_pointer_receives_state_latent_and_goal_latent():
    """GoalEncoder feeds goal_h while StatePooler feeds state_h into pointer scoring."""
    text_encoder = StubLLMEncoder(hidden_size=8)
    goal_encoder = GoalEncoder(text_encoder)
    pointer = Igc_PointerQNetwork(h_dim=8, q_dim=4)
    state_h = torch.randn(1, 8)
    goal_h = goal_encoder.encode([{
        "instruction": "set boot to PXE",
        "spec": {"BootSourceOverrideTarget": "Pxe"},
        "constraints": [],
        "plan": [ToolAction(tool_name="redfish", op="PATCH", target="/redfish/v1/Systems/1").to_dict()],
    }])
    cand_h = torch.randn(1, 3, 8)
    mask = torch.tensor([[1, 1, 0]], dtype=torch.bool)
    scores = pointer(state_h, goal_h, cand_h, mask)
    assert scores.shape == (1, 3)
    assert torch.isfinite(scores[0, :2]).all()
    assert torch.isinf(scores[0, 2])
