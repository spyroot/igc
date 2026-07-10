"""Offline tests for the legacy fixed-width Q-network head."""

import torch

from igc.modules.igc_q_network import Igc_QNetwork


def test_forward_returns_one_score_per_action_for_batch():
    """A batch of observations projects to one Q-score per configured action."""
    net = Igc_QNetwork(input_dim=5, num_actions=7, hidden_dim=11)

    output = net(torch.randn(3, 5))

    assert output.shape == (3, 7)


def test_forward_accepts_single_observation_tensor():
    """A single observation tensor keeps the fixed action width."""
    net = Igc_QNetwork(input_dim=4, num_actions=2, hidden_dim=6)

    output = net(torch.randn(4))

    assert output.shape == (2,)


def test_constructor_uses_configured_dimensions():
    """The legacy head derives layer sizes from constructor arguments."""
    net = Igc_QNetwork(input_dim=3, num_actions=9, hidden_dim=13)

    assert net.dense1.in_features == 3
    assert net.dense1.out_features == 13
    assert net.dense2.in_features == 13
    assert net.dense3.out_features == 13
    assert net.out.in_features == 13
    assert net.out.out_features == 9


def test_backward_updates_input_and_all_layers():
    """Gradients flow from selected Q-scores back through the full MLP."""
    torch.manual_seed(0)
    net = Igc_QNetwork(input_dim=6, num_actions=4, hidden_dim=8)
    inputs = torch.randn(5, 6, requires_grad=True)

    selected_scores = net(inputs)[:, [0, 2]].sum()
    selected_scores.backward()

    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()
    for parameter in net.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()

